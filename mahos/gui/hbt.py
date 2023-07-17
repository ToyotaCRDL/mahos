#!/usr/bin/env python3

"""
GUI frontend of HBT Interferometer.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import uuid
import os

from . import Qt
from .Qt import QtCore, QtWidgets

import pyqtgraph as pg

from .ui.hbt import Ui_HBT
from .hbt_client import QHBTClient

from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..msgs.common_meas_msgs import Buffer
from ..msgs.hbt_msgs import HBTData
from ..node.param_server import ParamClient
from ..meas.confocal import ConfocalIORequester
from .gui_node import GUINode
from .common_widget import ClientWidget
from .fit_widget import FitWidget
from .param import apply_widgets
from .dialog import save_dialog, load_dialog, export_dialog
from ..node.node import local_conf, join_name


Policy = QtWidgets.QSizePolicy.Policy


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self._normalize_toggled = False
        self.init_ui()
        self.init_view()

    def sizeHint(self):
        return QtCore.QSize(1600, 1000)

    def init_ui(self):
        hl = QtWidgets.QHBoxLayout()
        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        self.indicatorBox = QtWidgets.QCheckBox("indicator", parent=self)
        self.indicatorBox.setChecked(True)
        self.normalizeBox = QtWidgets.QCheckBox("normalize", parent=self)

        self.xfixBox = QtWidgets.QCheckBox("fix x range", parent=self)
        self.xminBox = QtWidgets.QDoubleSpinBox(parent=self)
        self.xminBox.setPrefix("xmin: ")
        self.xmaxBox = QtWidgets.QDoubleSpinBox(parent=self)
        self.xmaxBox.setPrefix("xmax: ")
        for w in (self.xminBox, self.xmaxBox):
            w.setSuffix(" ns")
            w.setMinimum(-1e6)
            w.setMaximum(+1e6)
            w.setSingleStep(10.0)
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)
        self.xminBox.setValue(-100.0)
        self.xmaxBox.setValue(+100.0)

        hl.addWidget(self.indicatorBox)
        hl.addWidget(self.normalizeBox)
        hl.addWidget(self.xfixBox)
        hl.addWidget(self.xminBox)
        hl.addWidget(self.xmaxBox)
        hl.addItem(spacer)

        self.graphicsView = pg.GraphicsView(parent=self)

        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl)
        vl.addWidget(self.graphicsView)
        self.setLayout(vl)

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)

        self.plot = self.layout.addPlot(row=0, col=0, lockAspect=False)

        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.setLabel("left", "Events")

        self.refline = self.plot.addLine(y=20, movable=False)
        self.halfline = self.plot.addLine(y=10, movable=False)
        self.bgline = self.plot.addLine(y=0, movable=False)
        self.startline = self.plot.addLine(x=0, movable=False)
        self.stopline = self.plot.addLine(x=10, movable=False)

        self.indicatorBox.toggled.connect(self.update_indicator)
        self.normalizeBox.toggled.connect(self.update_normalize)
        self.xfixBox.toggled.connect(self.update_xrange)
        self.xminBox.valueChanged.connect(self.update_xrange)
        self.xmaxBox.valueChanged.connect(self.update_xrange)

    def refresh(self, data_list: list[tuple[HBTData, bool, str]], data: HBTData):
        self.update_reference(data)

        self.plot.clearPlots()

        for data, show_fit, color in data_list:
            normalize = self.normalizeBox.isChecked()
            x = data.get_xdata(normalize=normalize)
            xfit = data.get_fit_xdata(normalize=normalize)
            y = data.get_ydata(normalize=normalize)
            yfit = data.get_fit_ydata(normalize=normalize)

            if show_fit and xfit is not None and yfit is not None:
                self.plot.plot(
                    x, y, pen=None, symbolPen=None, symbol="o", symbolSize=4, symbolBrush=color
                )
                self.plot.plot(xfit, yfit, pen=color, width=1)
            else:
                self.plot.plot(
                    x,
                    y,
                    pen=color,
                    width=1,
                    symbolPen=None,
                    symbol="o",
                    symbolSize=8,
                    symbolBrush=color,
                )

        if self._normalize_toggled:
            self.plot.autoRange()
            self._normalize_toggled = False

    def update_reference(self, data: HBTData):
        if not data.has_data():
            return

        start, stop = data.get_reference_window()
        if self.normalizeBox.isChecked():
            ref = 1.0
            bg = 0.0
            start -= data.get_t0()
            stop -= data.get_t0()
        else:
            ref = data.get_reference()
            bg = ref * data.get_bg_ratio()
        half = (ref - bg) / 2.0 + bg

        self.startline.setValue(start)
        self.stopline.setValue(stop)
        self.bgline.setValue(bg)
        self.refline.setValue(ref)
        self.halfline.setValue(half)

    def update_indicator(self, enabled):
        if enabled:
            self.plot.addItem(self.startline)
            self.plot.addItem(self.stopline)
            self.plot.addItem(self.bgline)
            self.plot.addItem(self.refline)
            self.plot.addItem(self.halfline)
        else:
            self.plot.removeItem(self.startline)
            self.plot.removeItem(self.stopline)
            self.plot.removeItem(self.bgline)
            self.plot.removeItem(self.refline)
            self.plot.removeItem(self.halfline)

    def update_normalize(self, normalize):
        if normalize:
            self.plot.setLabel("left", "g2")
        else:
            self.plot.setLabel("left", "Events")
        self._normalize_toggled = True

    def get_xrange(self, ns=False):
        # ns to s
        c = 1.0 if ns else 1e-9
        return sorted((self.xminBox.value() * c, self.xmaxBox.value() * c))

    def update_xrange(self):
        fix_enable = self.xfixBox.isChecked()
        for w in (self.xminBox, self.xmaxBox):
            w.setEnabled(fix_enable)
        self.plot.setMouseEnabled(not fix_enable, True)
        if not fix_enable:
            return
        self.plot.setXRange(*self.get_xrange())


class HBTFitWidget(FitWidget):
    def load_dialog(self, default_path: str) -> str:
        return load_dialog(self, default_path, "HBT", ".hbt")


class HBTWidget(ClientWidget, Ui_HBT):
    def __init__(
        self,
        gconf: dict,
        name,
        param_server_name,
        confocal_name,
        plot: PlotWidget,
        context,
        parent=None,
    ):
        ClientWidget.__init__(self, parent)
        self.setupUi(self)

        self.conf = local_conf(gconf, name)

        self.plot = plot
        self.data = HBTData()

        self.cli = QHBTClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.param_cli = ParamClient(gconf, param_server_name, context=context)
        if confocal_name:
            self.confocal_cli = ConfocalIORequester(gconf, confocal_name, context=context)
        else:
            self.confocal_cli = None

        self.add_clients(self.cli, self.param_cli, self.confocal_cli)

        self._finalizing = False

        self._fiTab_layout = QtWidgets.QVBoxLayout(self.fiTab)
        self.fit = HBTFitWidget(self.cli, self.param_cli, parent=self.fiTab)
        self._fiTab_layout.addWidget(self.fit)

        self.setEnabled(False)

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.init_connection()
        self.init_widgets()
        self.fit.init_with_status()

        # update initial GUI state
        self.update_state(status.state, last_state=BinaryState.IDLE)

        self.cli.stateUpdated.connect(self.update_state)
        self.cli.dataUpdated.connect(self.update_data)
        self.cli.bufferUpdated.connect(self.update_buffer)
        self.cli.stopped.connect(self.finalize)

        self.setEnabled(True)

    def init_connection(self):
        self.startButton.clicked.connect(self.request_start)
        self.stopButton.clicked.connect(self.request_stop)
        self.saveButton.clicked.connect(self.save_data)
        self.exportButton.clicked.connect(self.export_data)
        self.loadButton.clicked.connect(self.load_data)

        self.plotenableBox.toggled.connect(self.update_plot_enable)
        self.plotenableBox.setChecked(True)

        self.t0Box.editingFinished.connect(self.update_plot_params)
        self.refstartBox.editingFinished.connect(self.update_plot_params)
        self.refstopBox.editingFinished.connect(self.update_plot_params)
        self.bgratioBox.editingFinished.connect(self.update_plot_params)

    def init_widgets(self):
        params = self.cli.get_param_dict("hbt")
        apply_widgets(
            params,
            [
                ("bin", self.binBox, 1e9),  # s to ns
                ("range", self.windowBox, 1e9),
            ],
        )
        apply_widgets(
            params["plot"],
            [
                ("t0", self.t0Box, 1e9),  # s to ns
                ("ref_start", self.refstartBox, 1e9),
                ("ref_stop", self.refstopBox, 1e9),
                ("bg_ratio", self.bgratioBox, 100),  # to %
            ],
        )
        self.saveconfocalBox.setEnabled(self.confocal_cli is not None)
        self.saveconfocalBox.setChecked(self.confocal_cli is not None)

    def save_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "HBT", ".hbt")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.param_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)
        n = os.path.splitext(fn)[0] + ".png"
        self.cli.export_data(n)

        if self.confocal_cli is not None and self.saveconfocalBox.isChecked():
            n = os.path.splitext(fn)[0] + ".pos.png"
            self.confocal_cli.export_view(n)

        return fn

    def load_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "HBT", ".hbt")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.param_cli.set_param("loaded_note", data.note())

        self.refresh_plot()
        self.apply_widgets(data, True)

    def update_data(self, data: HBTData):
        self.data = data
        self.refresh_plot()
        self.apply_widgets(self.data, False)

    def update_buffer(self, buffer: Buffer[tuple[str, HBTData]]):
        self.fit.update_buffer(buffer)
        self.refresh_plot()

    def refresh_plot(self):
        self.plot.refresh(self.get_plottable_data(), self.data)

    def get_plottable_data(self) -> list[tuple[HBTData, bool, str]]:
        return self.fit.get_plottable_data(self.data)

    def export_data(self):
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.param_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "HBT", (".png", ".pdf", ".eps", ".txt"))
        if not fn:
            return

        data = [d for (d, _, _) in data_list]
        params = {}
        params["color"] = params["color_fit"] = [color for (_, _, color) in data_list]
        params["normalize"] = self.plot.normalizeBox.isChecked()
        if self.plot.xfixBox.isChecked():
            params["xmin"], params["xmax"] = self.plot.get_xrange(ns=True)
        self.cli.export_data(fn, data=data, params=params)

    # State managements
    def request_stop(self):
        self.cli.stop()

    def get_plot_params(self):
        params = {}
        params["t0"] = self.t0Box.value() * 1e-9  # ns to s
        params["ref_start"] = self.refstartBox.value() * 1e-9  # ns to s
        params["ref_stop"] = self.refstopBox.value() * 1e-9  # ns to s
        params["bg_ratio"] = self.bgratioBox.value() * 0.01  # %
        return params

    def request_start(self):
        params = {}
        params["bin"] = self.binBox.value() * 1e-9  # ns to s
        params["range"] = self.windowBox.value() * 1e-9  # ns to s
        params["plot"] = self.get_plot_params()

        title = "Continue?"
        body = (
            "Continue with current data?"
            + " Press No to clear current data and start a new measurement."
        )
        if self.data.can_resume(params) and Qt.question_yn(self, title, body):
            params["resume"] = True
            params["ident"] = self.data.ident
        else:
            params["resume"] = False
            params["ident"] = uuid.uuid4()

        self.cli.start(params)

    def apply_widgets(self, data: HBTData, loaded: bool):
        if not data.has_data():
            return

        tbin = data.get_bin()
        t = data.get_xdata()

        if loaded or not self.binBox.isEnabled():
            # if binBox is disabled, state is ACTIVE
            # Don't update these when the state is IDLE
            self.binBox.setValue(tbin * 1e9)  # s to ns
            self.windowBox.setValue(round(t[-1] * 1e9))
            self.rangeLabel.setText(f"range: {data.get_range()}")

        if loaded and self.plotenableBox.isChecked():
            self.t0Box.setValue(data.get_t0() * 1e9)
            # ref_start and ref_stop ?
            self.bgratioBox.setValue(data.get_bg_ratio() * 100)

    def update_plot_params(self):
        self.cli.update_plot_params(self.get_plot_params())

    def finalize(self, data: HBTData):
        if self._finalizing:
            return
        self._finalizing = True
        # Auto save is temporarily disabled.
        self._finalizing = False

    def update_plot_enable(self, enable: bool):
        for w in (self.t0Box, self.refstartBox, self.refstopBox, self.bgratioBox):
            w.setEnabled(enable)

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.startButton,
            self.saveButton,
            self.exportButton,
            self.loadButton,
            self.binBox,
            self.windowBox,
        ):
            w.setEnabled(state == BinaryState.IDLE)

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)


class HBTMainWindow(QtWidgets.QMainWindow):
    """MainWindow with HBTWidget and PlotWidget."""

    def __init__(self, gconf: dict, name, context, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        lconf = local_conf(gconf, name)
        target = lconf["target"]

        self.plot = PlotWidget(parent=self)
        self.hbt = HBTWidget(
            gconf,
            target["hbt"],
            target["param_server"],
            target.get("confocal"),
            self.plot,
            context,
            parent=self,
        )

        self.setWindowTitle(f"MAHOS.HBTGUI ({join_name(target['hbt'])})")
        self.setAnimated(False)
        self.setCentralWidget(self.hbt)
        self.d_plot = QtWidgets.QDockWidget("Plot", parent=self)
        self.d_plot.setWidget(self.plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_plot)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_plot.toggleViewAction())

    def closeEvent(self, event):
        """Close the clients on closeEvent."""

        self.hbt.close_clients()
        # self.plot.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)


class HBTGUI(GUINode):
    """GUINode for HBT using HBTWidget."""

    def init_widget(self, gconf: dict, name, context):
        return HBTMainWindow(gconf, name, context)
