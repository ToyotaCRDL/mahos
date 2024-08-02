#!/usr/bin/env python3

import numpy as np
import pyqtgraph as pg

from mahos.node.node import local_conf
from mahos.gui.client import QBasicMeasClient
from mahos.gui.Qt import QtWidgets
from mahos.gui.param import apply_widgets
from mahos.gui.gui_node import GUINode
from mahos.gui.common_widget import ClientTopWidget
from mahos.msgs.common_msgs import BinaryState, BinaryStatus

from ivcurve_msgs import IVCurveData


class IVCurveWidget(ClientTopWidget):
    """Widget for IVCurve."""

    def __init__(self, gconf: dict, name, context, parent=None):
        ClientTopWidget.__init__(self, parent)
        self.init_ui()

        self.conf = local_conf(gconf, name)
        self.cli = QBasicMeasClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)
        self.add_clients(self.cli)

        self.setEnabled(False)

    def init_ui(self):
        self.setWindowTitle("IVCurve")

        # layout input UIs in hl
        hl = QtWidgets.QHBoxLayout()
        self.startButton = QtWidgets.QPushButton("Start")
        self.stopButton = QtWidgets.QPushButton("Stop")
        self.vstartBox = QtWidgets.QDoubleSpinBox()
        self.vstartBox.setPrefix("start: ")
        self.vstartBox.setSuffix(" V")
        self.vstopBox = QtWidgets.QDoubleSpinBox()
        self.vstopBox.setPrefix("stop: ")
        self.vstopBox.setSuffix(" V")
        self.numBox = QtWidgets.QSpinBox()
        self.numBox.setPrefix("num: ")
        self.sweepsBox = QtWidgets.QSpinBox()
        self.sweepsBox.setPrefix("sweeps: ")
        self.delayBox = QtWidgets.QDoubleSpinBox()
        self.delayBox.setPrefix("delay: ")
        self.delayBox.setSuffix(" ms")
        for w in (
            self.startButton,
            self.stopButton,
            self.vstartBox,
            self.vstopBox,
            self.numBox,
            self.sweepsBox,
            self.delayBox,
        ):
            hl.addWidget(w)
        spacer = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        hl.addItem(spacer)

        pw = pg.PlotWidget(labels={"left": ("Current", "A"), "bottom": ("Voltage", "V")})

        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl)
        vl.addWidget(pw)
        self.setLayout(vl)

        self.plot_item = pw.getPlotItem().plot()

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # call this method only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.startButton.clicked.connect(self.request_start)
        self.stopButton.clicked.connect(self.request_stop)

        params = self.cli.get_param_dict()
        if params is not None:
            apply_widgets(
                params,
                [
                    ("start", self.vstartBox),
                    ("stop", self.vstopBox),
                    ("num", self.numBox),
                    ("sweeps", self.sweepsBox),
                    ("delay_sec", self.delayBox, 1e3),  # sec to ms
                ],
            )
        else:
            print("[ERROR] failed to get param dict")

        # update initial GUI state
        self.update_state(status.state, last_state=BinaryState.IDLE)

        self.cli.stateUpdated.connect(self.update_state)
        self.cli.dataUpdated.connect(self.update_data)

        self.setEnabled(True)

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.startButton,
            self.vstartBox,
            self.vstopBox,
            self.numBox,
            self.sweepsBox,
            self.delayBox,
        ):
            w.setEnabled(state == BinaryState.IDLE)
        self.stopButton.setEnabled(state == BinaryState.ACTIVE)

    def update_data(self, data: IVCurveData):
        xdata = np.linspace(data.params["start"], data.params["stop"], data.params["num"])
        ydata = np.mean(data.data, axis=1)
        self.plot_item.setData(xdata, ydata)

    def request_start(self):
        params = {
            "start": self.vstartBox.value(),
            "stop": self.vstopBox.value(),
            "num": self.numBox.value(),
            "sweeps": self.sweepsBox.value(),
            "delay_sec": self.delayBox.value() * 1e-3,  # ms to sec
        }

        self.cli.start(params)

    def request_stop(self):
        self.cli.stop()


class IVCurveGUI(GUINode):
    """GUINode for IVCurve using IVCurveWidget."""

    def init_widget(self, gconf: dict, name, context):
        target = local_conf(gconf, name)["target"]
        return IVCurveWidget(gconf, target["ivcurve"], context)
