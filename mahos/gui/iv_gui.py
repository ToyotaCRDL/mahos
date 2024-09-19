#!/usr/bin/env python3

import os

import pyqtgraph as pg

from ..node.node import local_conf
from ..gui.client import QBasicMeasClient
from ..gui.Qt import QtWidgets
from ..gui.param import apply_widgets
from ..gui.gui_node import GUINode
from ..gui.common_widget import ClientTopWidget, SpinBox
from ..gui.dialog import save_dialog, load_dialog, export_dialog
from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..node.global_params import GlobalParamsClient

from ..msgs.iv_msgs import IVData

Policy = QtWidgets.QSizePolicy.Policy


class IVWidget(ClientTopWidget):
    """Widget for IV."""

    def __init__(self, gconf: dict, name, gparams_name, context, parent=None):
        ClientTopWidget.__init__(self, parent)
        self.init_ui()

        self.conf = local_conf(gconf, name)
        self.cli = QBasicMeasClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)
        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli)

        self.setEnabled(False)

    def init_ui(self):
        self.setWindowTitle("IV")

        # layout input UIs in hl
        hl0 = QtWidgets.QHBoxLayout()
        hl1 = QtWidgets.QHBoxLayout()
        self.startButton = QtWidgets.QPushButton("Start")
        self.stopButton = QtWidgets.QPushButton("Stop")
        self.saveButton = QtWidgets.QPushButton("Save")
        self.exportButton = QtWidgets.QPushButton("Export")
        self.loadButton = QtWidgets.QPushButton("Load")
        self.vstartBox = QtWidgets.QDoubleSpinBox()
        self.vstartBox.setPrefix("start: ")
        self.vstartBox.setSuffix(" V")
        self.vstopBox = QtWidgets.QDoubleSpinBox()
        self.vstopBox.setPrefix("stop: ")
        self.vstopBox.setSuffix(" V")
        self.numBox = QtWidgets.QSpinBox()
        self.numBox.setPrefix("point: ")
        self.sweepsBox = QtWidgets.QSpinBox()
        self.sweepsBox.setPrefix("sweeps: ")
        self.delayBox = QtWidgets.QDoubleSpinBox()
        self.delayBox.setPrefix("delay: ")
        self.delayBox.setSuffix(" ms")
        self.nplcBox = QtWidgets.QDoubleSpinBox()
        self.nplcBox.setPrefix("nplc: ")
        self.complianceBox = SpinBox(siPrefix=True, prefix="comp.: ", suffix="A")
        self.logBox = QtWidgets.QCheckBox("log")

        for w in (
            self.vstartBox,
            self.vstopBox,
            self.numBox,
            self.sweepsBox,
            self.delayBox,
            self.nplcBox,
            self.complianceBox,
        ):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)

        for w in (
            self.startButton,
            self.stopButton,
            self.saveButton,
            self.exportButton,
            self.loadButton,
        ):
            hl0.addWidget(w)
        spacer0 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        hl0.addItem(spacer0)
        for w in (
            self.vstartBox,
            self.vstopBox,
            self.numBox,
            self.sweepsBox,
            self.delayBox,
            self.nplcBox,
            self.complianceBox,
            self.logBox,
        ):
            hl1.addWidget(w)
        spacer1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        hl1.addItem(spacer1)

        pw = pg.PlotWidget(labels={"left": ("Current", "A"), "bottom": ("Voltage", "V")})

        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl0)
        vl.addLayout(hl1)
        vl.addWidget(pw)
        self.setLayout(vl)

        self.plot_item = pw.getPlotItem().plot()

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # call this method only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        params = self.cli.get_param_dict()
        if params is not None:
            apply_widgets(
                params,
                [
                    ("start", self.vstartBox),
                    ("stop", self.vstopBox),
                    ("num", self.numBox),
                    ("sweeps", self.sweepsBox),
                    ("delay", self.delayBox, 1e3),  # sec to ms
                    ("nplc", self.nplcBox),
                    ("compliance", self.complianceBox),
                    ("logx", self.logBox),
                ],
            )
        else:
            print("[ERROR] failed to get param dict")

        # update initial GUI state
        self.update_state(status.state, last_state=BinaryState.IDLE)
        self.cli.stateUpdated.connect(self.update_state)
        self.cli.dataUpdated.connect(self.update_data)
        self.init_connection()

        self.setEnabled(True)

    def init_connection(self):
        self.startButton.clicked.connect(self.request_start)
        self.stopButton.clicked.connect(self.request_stop)
        self.saveButton.clicked.connect(self.save_data)
        self.exportButton.clicked.connect(self.export_data)
        self.loadButton.clicked.connect(self.load_data)

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.startButton,
            self.saveButton,
            self.vstartBox,
            self.vstopBox,
            self.numBox,
            self.sweepsBox,
            self.delayBox,
            self.nplcBox,
            self.complianceBox,
            self.logBox,
        ):
            w.setEnabled(state == BinaryState.IDLE)
        self.stopButton.setEnabled(state == BinaryState.ACTIVE)

    def update_data(self, data: IVData):
        xdata = data.get_xdata()
        ydata = data.get_ydata()
        self.plot_item.setData(xdata, ydata)

    def request_start(self):
        params = {
            "start": self.vstartBox.value(),
            "stop": self.vstopBox.value(),
            "num": self.numBox.value(),
            "sweeps": self.sweepsBox.value(),
            "delay": self.delayBox.value() * 1e-3,  # ms to sec
            "nplc": self.nplcBox.value(),
            "compliance": self.complianceBox.value(),
            "logx": self.logBox.isChecked(),
        }

        self.cli.start(params)

    def request_stop(self):
        self.cli.stop()

    def save_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "IV", ".iv")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.gparams_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)
        n = os.path.splitext(fn)[0] + ".png"
        self.cli.export_data(n)

        return fn

    def load_data(self):
        """TODO"""

        QtWidgets.QMessageBox.warning(self, "Not implemented", "Load is not implemented")
        return
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "IV", "iv")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.gparams_cli.set_param("loaded_note", data.note())

        self.refresh_plot()
        self.apply_widgets(data, True)

    def export_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "", (".png", ".pdf", ".eps", ".csv"))
        if not fn:
            return

        self.cli.export_data(fn)

    def set_range(self):
        self.cli.set_range_mode()


class IVGUI(GUINode):
    """GUINode for IV using IVWidget."""

    def init_widget(self, gconf: dict, name, context):
        target = local_conf(gconf, name)["target"]
        return IVWidget(gconf, target["iv"], target["gparams"], context)
