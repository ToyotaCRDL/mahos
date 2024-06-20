#!/usr/bin/env python3

"""
Common GUI for Recorder.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os
from itertools import cycle

import pyqtgraph as pg

from .Qt import QtCore, QtWidgets

from .ui.recorder import Ui_Recorder
from .client import QBasicMeasClient

from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..msgs.recorder_msgs import RecorderData
from ..node.global_params import GlobalParamsClient
from .gui_node import GUINode
from .common_widget import ClientWidget
from .dialog import save_dialog, load_dialog, export_dialog
from ..node.node import local_conf, join_name
from ..util.plot import colors_tab10
from ..util.unit import SI_scale


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self._units = []
        self._plots = []
        self.init_ui()
        self.init_view()

    def sizeHint(self):
        return QtCore.QSize(1600, 1200)

    def init_ui(self):
        self.graphicsView = pg.GraphicsView(parent=self)
        self.label = QtWidgets.QLabel("")
        self.label.setStyleSheet("QLabel {font: bold 32pt;}")
        vl = QtWidgets.QVBoxLayout()
        vl.addWidget(self.label)
        vl.addWidget(self.graphicsView)
        self.setLayout(vl)

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)

    def set_axes(self, data: RecorderData):
        unit_to_insts = data.get_unit_to_insts()

        if self._units == list(unit_to_insts.keys()):
            return

        self._units = list(unit_to_insts.keys())
        self._plots = []
        self.layout.clear()
        for i, unit in enumerate(unit_to_insts):
            plot = self.layout.addPlot(row=i, col=0, lockAspect=False)
            plot.showGrid(x=True, y=True)

            ai = pg.DateAxisItem(text="Date time")
            plot.setAxisItems({"bottom": ai})
            plot.setLabel("left", unit, unit)
            plot.addLegend()
            self._plots.append(plot)

    def refresh(self, data: RecorderData):
        self.set_axes(data)

        unit_to_insts = data.get_unit_to_insts()
        x = data.get_xdata()
        for plot, insts in zip(self._plots, unit_to_insts.values()):
            plot.clearPlots()
            for inst, color in zip(insts, cycle(colors_tab10())):
                y = data.get_ydata(inst)
                plot.plot(x, y, name=inst, pen=color, width=1)

        latest_data = []
        for inst in data.get_insts():
            y = data.get_ydata(inst)
            unit = data.get_unit(inst)
            if len(y):
                scale, prefix = SI_scale(y[-1])
                latest_data.append(f"{inst}: {y[-1]*scale:.3f} {prefix}{unit}")
        self.label.setText(" ".join(latest_data))


class RecorderWidget(ClientWidget, Ui_Recorder):
    def __init__(self, gconf: dict, name, gparams_name, plot: PlotWidget, context, parent=None):
        ClientWidget.__init__(self, parent)
        self.setupUi(self)

        self.conf = local_conf(gconf, name)

        self.plot = plot
        self.data = RecorderData()

        self.cli = QBasicMeasClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli, self.gparams_cli)

        self._finalizing = False

        self.setEnabled(False)

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        labels = self.cli.get_param_dict_labels()
        self.labelBox.addItems(labels)
        self.labelBox.currentIndexChanged.connect(self.update_param_table)
        if labels:
            self.update_param_table()

        self.init_connection()

        # update initial GUI state
        self.update_state(status.state, last_state=BinaryState.IDLE)

        self.cli.stateUpdated.connect(self.update_state)
        self.cli.dataUpdated.connect(self.update_data)
        self.cli.stopped.connect(self.finalize)

        self.setEnabled(True)

    def init_connection(self):
        self.startButton.clicked.connect(self.request_start)
        self.stopButton.clicked.connect(self.request_stop)
        self.saveButton.clicked.connect(self.save_data)
        self.exportButton.clicked.connect(self.export_data)
        self.loadButton.clicked.connect(self.load_data)

    def save_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Recorder", ".record")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.gparams_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)
        n = os.path.splitext(fn)[0] + ".png"
        self.cli.export_data(n)

        return fn

    def load_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "Recorder", ".record")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.gparams_cli.set_param("loaded_note", data.note())

        self.refresh_plot()
        self.apply_widgets(data)

    def update_data(self, data: RecorderData):
        self.data = data
        self.refresh_plot()
        self.apply_widgets(self.data)

    def update_param_table(self):
        label = self.labelBox.currentText()
        d = self.cli.get_param_dict(label)
        self.paramTable.update_contents(d)

    def refresh_plot(self):
        self.plot.refresh(self.data)

    def export_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "", (".png", ".pdf", ".eps"))
        if not fn:
            return

        self.cli.export_data(fn)

    # State managements
    def request_stop(self):
        self.cli.stop()

    def request_start(self):
        self.cli.start(self.paramTable.params(), self.labelBox.currentText())

    def apply_widgets(self, data: RecorderData):
        if not data.has_data():
            return

    def finalize(self, data: RecorderData):
        if self._finalizing:
            return
        self._finalizing = True
        # Auto save is temporarily disabled.
        self._finalizing = False

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.labelBox,
            self.startButton,
            self.saveButton,
            self.exportButton,
            self.loadButton,
        ):
            w.setEnabled(state == BinaryState.IDLE)

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)


class RecorderMainWindow(QtWidgets.QMainWindow):
    """MainWindow with RecorderWidget and PlotWidget."""

    def __init__(self, gconf: dict, name, context, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        lconf = local_conf(gconf, name)
        target = lconf["target"]

        self.plot = PlotWidget(parent=self)
        self.meas = RecorderWidget(
            gconf, target["recorder"], target["gparams"], self.plot, context, parent=self
        )

        self.setWindowTitle(f"MAHOS.RecorderGUI ({join_name(target['recorder'])})")
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


class RecorderGUI(GUINode):
    """GUINode for Recorder using RecorderMainWindow."""

    def init_widget(self, gconf: dict, name, context):
        return RecorderMainWindow(gconf, name, context)
