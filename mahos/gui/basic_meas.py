#!/usr/bin/env python3

"""
Common GUI for BasicMeasNode.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import os

import pyqtgraph as pg

from .Qt import QtCore, QtWidgets

from .ui.basic_meas import Ui_BasicMeas
from .client import QBasicMeasClient

from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..msgs.common_meas_msgs import Buffer
from ..msgs.common_meas_msgs import BasicMeasData
from ..node.param_server import ParamClient
from .gui_node import GUINode
from .common_widget import ClientWidget
from .fit_widget import FitWidget
from .dialog import save_dialog, load_dialog, export_dialog
from ..node.node import local_conf, join_name


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self._axes_set = False
        self.init_ui()
        self.init_view()

    def sizeHint(self):
        return QtCore.QSize(1600, 1200)

    def init_ui(self):
        self.graphicsView = pg.GraphicsView(parent=self)

        vl = QtWidgets.QVBoxLayout()
        vl.addWidget(self.graphicsView)
        self.setLayout(vl)

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)

        self.plot = self.layout.addPlot(row=0, col=0, lockAspect=False)
        self.plot.showGrid(x=True, y=True)

    def set_axes(self, data: BasicMeasData):
        if self._axes_set:
            return
        if not data.has_params():
            return

        self.plot.setLabel("bottom", data.xlabel, data.xunit)
        self.plot.setLabel("left", data.ylabel, data.yunit)
        self.plot.setLogMode(x=data.xscale == "log", y=data.yscale == "log")
        self._axes_set = True

    def refresh(self, data_list: list[tuple[BasicMeasData, bool, str]], d: BasicMeasData):
        self.plot.clearPlots()

        self.set_axes(d)

        for d, show_fit, color in data_list:
            x = d.get_xdata()
            xfit = d.get_fit_xdata()
            y = d.get_ydata()
            yfit = d.get_fit_ydata()
            if isinstance(y, tuple):
                y = y[0]

            if show_fit and xfit is not None and yfit is not None:
                self.plot.plot(
                    x, y, pen=None, symbolPen=None, symbol="o", symbolSize=4, symbolBrush=color
                )
                self.plot.plot(xfit, yfit, pen=color, width=1)
            elif x is not None and y is not None:
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


class BasicMeasWidget(ClientWidget, Ui_BasicMeas):
    def __init__(
        self, gconf: dict, name, param_server_name, plot: PlotWidget, context, parent=None
    ):
        ClientWidget.__init__(self, parent)
        self.setupUi(self)

        self.conf = local_conf(gconf, name)

        self.plot = plot
        self.data = BasicMeasData()

        self.cli = QBasicMeasClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.param_cli = ParamClient(gconf, param_server_name, context=context)

        self.add_clients(self.cli, self.param_cli)

        self._finalizing = False

        self._fiTab_layout = QtWidgets.QVBoxLayout(self.fiTab)
        self.fit = FitWidget(self.cli, self.param_cli, parent=self.fiTab)
        self._fiTab_layout.addWidget(self.fit)

        self.setEnabled(False)

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        names = self.cli.get_param_dict_names()
        self.methodBox.addItems(names)
        self.methodBox.currentIndexChanged.connect(self.update_param_table)
        if names:
            self.update_param_table()

        self.init_connection()
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

    def save_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "" "")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.param_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)
        n = os.path.splitext(fn)[0] + ".png"
        self.cli.export_data(n)

        return fn

    def load_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "", "")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.param_cli.set_param("loaded_note", data.note())

        self.refresh_plot()
        self.apply_widgets(data)

    def update_data(self, data: BasicMeasData):
        self.data = data
        self.refresh_plot()
        self.apply_widgets(self.data)

    def update_buffer(self, buffer: Buffer[tuple[str, BasicMeasData]]):
        self.fit.update_buffer(buffer)
        self.refresh_plot()

    def update_param_table(self):
        method = self.methodBox.currentText()
        d = self.cli.get_param_dict(method)
        self.paramTable.update_contents(d)

    def refresh_plot(self):
        self.plot.refresh(self.get_plottable_data(), self.data)

    def get_plottable_data(self) -> list[tuple[BasicMeasData, bool, str]]:
        return self.fit.get_plottable_data(self.data)

    def export_data(self):
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.param_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "", (".png", ".pdf", ".eps"))
        if not fn:
            return

        data = [d for (d, _, _) in data_list]
        params = {}
        self.cli.export_data(fn, data=data, params=params)

    # State managements
    def request_stop(self):
        self.cli.stop()

    def request_start(self):
        self.cli.start(self.paramTable.params())

    def apply_widgets(self, data: BasicMeasData):
        if not data.has_data():
            return

    def finalize(self, data: BasicMeasData):
        if self._finalizing:
            return
        self._finalizing = True
        # Auto save is temporarily disabled.
        self._finalizing = False

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.methodBox,
            self.startButton,
            self.saveButton,
            self.exportButton,
            self.loadButton,
        ):
            w.setEnabled(state == BinaryState.IDLE)

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)


class BasicMeasMainWindow(QtWidgets.QMainWindow):
    """MainWindow with BasicMeasWidget and PlotWidget."""

    def __init__(self, gconf: dict, name, context, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        lconf = local_conf(gconf, name)
        target = lconf["target"]

        self.plot = PlotWidget(parent=self)
        self.meas = BasicMeasWidget(
            gconf, target["meas"], target["param_server"], self.plot, context, parent=self
        )

        self.setWindowTitle(f"MAHOS.BasicMeasGUI ({join_name(target['meas'])})")
        self.setAnimated(False)
        self.setCentralWidget(self.meas)
        self.d_plot = QtWidgets.QDockWidget("Plot", parent=self)
        self.d_plot.setWidget(self.plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_plot)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_plot.toggleViewAction())

    def closeEvent(self, event):
        """Close the clients on closeEvent."""

        self.meas.close_clients()
        # self.plot.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)


class BasicMeasGUI(GUINode):
    """GUINode for BasicMeasNode using BasicMeasWidget."""

    def init_widget(self, gconf: dict, name, context):
        return BasicMeasMainWindow(gconf, name, context)
