#!/usr/bin/env python3

"""
GUI frontend of Spectroscopy.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import uuid
import os

import numpy as np
import pyqtgraph as pg

from . import Qt
from .Qt import QtCore, QtWidgets, QtGui

from .ui.spectroscopy import Ui_Spectroscopy
from .spectroscopy_client import QSpectroscopyClient

from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..msgs.common_meas_msgs import Buffer
from ..msgs.spectroscopy_msgs import SpectroscopyData
from ..node.param_server import ParamClient
from ..meas.confocal import ConfocalIORequester
from .gui_node import GUINode
from .common_widget import ClientWidget
from .fit_widget import FitWidget
from .dialog import save_dialog, load_dialog, export_dialog
from .param import apply_widgets
from ..node.node import local_conf, join_name


Policy = QtWidgets.QSizePolicy.Policy


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.init_ui()
        self.init_view()

    def sizeHint(self):
        return QtCore.QSize(1600, 1200)

    def init_ui(self):
        hl0 = QtWidgets.QHBoxLayout()
        self.showimgBox = QtWidgets.QCheckBox("Show image")
        self.showimgBox.setChecked(True)
        self.symbolsizeBox = QtWidgets.QSpinBox(parent=self)
        self.symbolsizeBox.setPrefix("symbol size: ")
        self.symbolsizeBox.setValue(0)
        self.symbolsizeBox.setMinimum(0)
        self.symbolsizeBox.setMaximum(10)
        self.lastnBox = QtWidgets.QSpinBox(parent=self)
        self.lastnBox.setPrefix("last_n: ")
        self.lastnBox.setMinimum(0)
        self.lastnBox.setMaximum(10000)
        self.filternBox = QtWidgets.QDoubleSpinBox(parent=self)
        self.filternBox.setPrefix("filter_n: ")
        self.filternBox.setSuffix(" σ")
        self.filternBox.setMinimum(0.0)
        self.filternBox.setMaximum(10.0)
        for w in (self.symbolsizeBox, self.lastnBox, self.filternBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)
        self.outlierLabel = QtWidgets.QLabel("Removed outliers: ")
        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        hl0.addWidget(self.showimgBox)
        hl0.addWidget(self.symbolsizeBox)
        hl0.addWidget(self.lastnBox)
        hl0.addWidget(self.filternBox)
        hl0.addWidget(self.outlierLabel)
        hl0.addItem(spacer)

        hl = QtWidgets.QHBoxLayout()
        self.graphicsView = pg.GraphicsView(parent=self)
        self.histo = pg.HistogramLUTWidget(parent=self)
        hl.addWidget(self.graphicsView)
        hl.addWidget(self.histo)

        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl0)
        vl.addLayout(hl)
        self.setLayout(vl)

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)
        self.histo.gradient.loadPreset("inferno")

        self.plot = self.layout.addPlot(row=0, col=0, lockAspect=False)

        self.img = pg.ImageItem()
        self.img_plot = self.layout.addPlot(row=1, col=0, lockAspect=False)
        self.img_plot.addItem(self.img)
        self.histo.setImageItem(self.img)

        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("bottom", "Wavelength (nm)")
        self.plot.setLabel("left", "Intensity", "counts")
        self.img_plot.setLabel("bottom", "Number of Pixels")
        self.img_plot.setLabel("left", "Number of accumulation")

        self.showimgBox.toggled.connect(self.toggle_image)

    def refresh(self, data_list: list[tuple[SpectroscopyData, bool, str]], data: SpectroscopyData):
        if data.has_data():
            self.update_image(data)
        self.update_plot(data_list)

    def update_image(self, data: SpectroscopyData, setlevel=True):
        self.img.updateImage(data.data)
        if setlevel:
            mn, mx = np.nanmin(data.data), np.nanmax(data.data)
            if mn == mx:
                mn, mx = mn - 0.1, mx + 0.1
            self.histo.setLevels(mn, mx)
        self.img.resetTransform()
        self.img.setPos(0.0, 0.0)
        self.img.setTransform(QtGui.QTransform.fromScale(1.0, 1.0))

    def update_plot(self, data_list: list[tuple[SpectroscopyData, bool, str]]):
        self.plot.clearPlots()

        n_outliers = []
        for data, show_fit, color in data_list:
            x = data.get_xdata()
            y = data.get_ydata(last_n=self.lastnBox.value(), filter_n=self.filternBox.value())
            n_outliers.append(
                str(
                    data.n_outliers(last_n=self.lastnBox.value(), filter_n=self.filternBox.value())
                )
            )
            xfit = data.get_fit_xdata()
            yfit = data.get_fit_ydata()

            if show_fit and (xfit is not None) and (yfit is not None):
                self.plot.plot(
                    x,
                    y,
                    pen=None,
                    symbolPen=None,
                    symbol="o",
                    symbolSize=self.symbolsizeBox.value(),
                    symbolBrush=color,
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
                    symbolSize=self.symbolsizeBox.value(),
                    symbolBrush=color,
                )
        self.outlierLabel.setText("Removed outliers: " + ", ".join(n_outliers))

    def toggle_image(self, show):
        if show:
            self.layout.addItem(self.img_plot, row=1, col=0)
        else:
            self.layout.removeItem(self.img_plot)


class SpectroscopyFitWidget(FitWidget):
    def load_dialog(self, default_path: str) -> str:
        return load_dialog(self, default_path, "Spectroscopy", ".spec")


class SpectroscopyWidget(ClientWidget, Ui_Spectroscopy):
    """Control widget for Spectroscopy."""

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
        self.data = SpectroscopyData()

        self.cli = QSpectroscopyClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.param_cli = ParamClient(gconf, param_server_name, context=context)
        if confocal_name:
            self.confocal_cli = ConfocalIORequester(gconf, confocal_name, context=context)
        else:
            self.confocal_cli = None

        self.add_clients(self.cli, self.param_cli, self.confocal_cli)

        self._finalizing = False

        self._fiTab_layout = QtWidgets.QVBoxLayout(self.fiTab)
        self.fit = SpectroscopyFitWidget(self.cli, self.param_cli, parent=self.fiTab)
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

    def init_widgets(self):
        params = self.cli.get_param_dict()
        self.baseconfBox.clear()
        if params is not None:
            self.baseconfBox.addItems(params["base_config"].options())
        else:
            print("[ERROR] Failed to get param_dict")

        apply_widgets(
            params,
            [
                ("exposure_time", self.exposuretimeBox),
                ("exposures", self.exposuresBox),
                ("center_wavelength", self.centerBox),
            ],
        )
        self.saveconfocalBox.setEnabled(self.confocal_cli is not None)
        self.saveconfocalBox.setChecked(self.confocal_cli is not None)

    def apply_widgets(self, data: SpectroscopyData):
        params = data.params
        i = self.baseconfBox.findText(params["base_config"])
        if i >= 0:
            self.baseconfBox.setCurrentIndex(i)
        self.exposuretimeBox.setValue(params["exposure_time"])
        self.exposuresBox.setValue(params["exposures"])
        self.centerBox.setValue(params["center_wavelength"])

    def save_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Spectroscopy", ".spec")
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
        fn = load_dialog(self, default_path, "Spectroscopy", ".spec")
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

    def update_data(self, data: SpectroscopyData):
        self.data = data
        self.refresh_plot()

    def update_buffer(self, buffer: Buffer[tuple[str, SpectroscopyData]]):
        self.fit.update_buffer(buffer)
        self.refresh_plot()

    def refresh_plot(self):
        self.plot.refresh(self.get_plottable_data(), self.data)

    def get_plottable_data(self) -> list[tuple[SpectroscopyData, bool, str]]:
        return self.fit.get_plottable_data(self.data)

    def export_data(self):
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.param_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "Spectroscopy", (".png", ".pdf", ".eps", ".txt"))
        if not fn:
            return

        data = [d for (d, _, _) in data_list]
        params = {}
        params["color"] = params["color_fit"] = [color for (_, _, color) in data_list]
        self.cli.export_data(fn, data=data, params=params)

    # State managements
    def request_stop(self):
        self.cli.stop()

    def request_start(self):
        valid, msg = self.validate()

        if valid:
            self.start_acquisition()
        else:
            QtWidgets.QMessageBox.warning(self, "Parameter invalid.", msg)

    def validate(self):
        return (True, "")

    def start_acquisition(self):
        params = {}
        params["base_config"] = self.baseconfBox.currentText()
        params["exposure_time"] = self.exposuretimeBox.value()
        params["exposures"] = self.exposuresBox.value()
        params["center_wavelength"] = self.centerBox.value()
        params["acquisitions"] = self.acquisitionsBox.value()

        title = "Continue acquisition?"
        body = (
            "Continue acquisition with current data?"
            + " Press No to clear current data and start a new acquisition."
        )
        if self.data.can_resume(params) and Qt.question_yn(self, title, body):
            params["resume"] = True
            params["ident"] = self.data.ident
        else:
            params["resume"] = False
            params["ident"] = uuid.uuid4()

        self.cli.start(params)

    def finalize(self, data: SpectroscopyData):
        if self._finalizing:
            return
        self._finalizing = True
        # Auto save is temporarily disabled.
        self._finalizing = False

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.startButton,
            self.saveButton,
            self.exportButton,
            self.loadButton,
            self.baseconfBox,
            self.exposuretimeBox,
            self.exposuresBox,
            self.centerBox,
            self.acquisitionsBox,
        ):
            w.setEnabled(state == BinaryState.IDLE)

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)


class SpectroscopyMainWindow(QtWidgets.QMainWindow):
    """MainWindow with SpectroscopyWidget and PlotWidget."""

    def __init__(self, gconf: dict, name, context, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        lconf = local_conf(gconf, name)
        target = lconf["target"]

        self.plot = PlotWidget(parent=self)
        self.spec = SpectroscopyWidget(
            gconf,
            target["spectroscopy"],
            target["param_server"],
            target.get("confocal"),
            self.plot,
            context,
            parent=self,
        )

        self.setWindowTitle(f"MAHOS.SpectroscopyGUI ({join_name(target['spectroscopy'])})")
        self.setAnimated(False)
        self.setCentralWidget(self.spec)
        self.d_plot = QtWidgets.QDockWidget("Plot", parent=self)
        self.d_plot.setWidget(self.plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_plot)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_plot.toggleViewAction())

    def closeEvent(self, event):
        """Close the clients on closeEvent."""

        self.spec.close_clients()
        # self.plot.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)


class SpectroscopyGUI(GUINode):
    """GUINode for Spectroscopy using SpectroscopyWidget."""

    def init_widget(self, gconf: dict, name, context):
        return SpectroscopyMainWindow(gconf, name, context)
