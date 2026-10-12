#!/usr/bin/env python3

"""
GUI widget for GridSweeper measurement.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg

from mahos.gui.Qt import QtCore, QtGui, QtWidgets
from mahos.gui.client import QBasicMeasClient
from mahos.gui.common_widget import ClientTopWidget, SpinBox
from mahos.gui.dialog import export_dialog, load_dialog, save_dialog
from mahos.gui.gui_node import GUINode, get_gui_logger
from mahos.gui.param import apply_widgets
from mahos.msgs.common_msgs import BinaryState, BinaryStatus
from mahos.msgs.grid_sweeper_msgs import GridSweeperData
from mahos.node.global_params import GlobalParamsClient
from mahos.node.node import join_name, local_conf

Policy = QtWidgets.QSizePolicy.Policy


class QGridSweeperClient(QBasicMeasClient):
    """Qt-based client for GridSweeper."""

    dataUpdated = QtCore.pyqtSignal(GridSweeperData)
    stopped = QtCore.pyqtSignal(GridSweeperData)


class PlotWidget(QtWidgets.QWidget):
    """Widget containing 2D image plot for GridSweeper data."""

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.init_ui()
        self.init_view()

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    def init_ui(self):
        hl0 = QtWidgets.QHBoxLayout()

        self.meanBox = QtWidgets.QCheckBox("Mean")
        self.meanBox.setToolTip("Mean image over sweeps")
        self.meanBox.setChecked(False)

        self.meanincompBox = QtWidgets.QCheckBox("Use incomp")
        self.meanincompBox.setChecked(True)
        self.meanincompBox.setToolTip("Include incompleted latest image when taking mean")

        self.lastnBox = QtWidgets.QSpinBox(parent=self)
        self.lastnBox.setPrefix("last_n: ")
        self.lastnBox.setMinimum(-1_000_000)
        self.lastnBox.setMaximum(1_000_000)
        self.lastnBox.setValue(0)
        self.lastnBox.setToolTip("Average last N sweeps (0 for all)")

        self.sweepidxBox = QtWidgets.QSpinBox(parent=self)
        self.sweepidxBox.setPrefix("index: ")
        self.sweepidxBox.setMinimum(-1_000_000)
        self.sweepidxBox.setMaximum(1_000_000)
        self.sweepidxBox.setValue(-1)
        self.sweepidxBox.setToolTip("Displayed sweep index when mean mode is off")

        self.autolevelBox = QtWidgets.QCheckBox("Auto level")
        self.autolevelBox.setChecked(True)

        for w in (self.lastnBox, self.sweepidxBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(180)

        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)

        hl0.addWidget(self.meanBox)
        hl0.addWidget(self.meanincompBox)
        hl0.addWidget(self.lastnBox)
        hl0.addWidget(self.sweepidxBox)
        hl0.addWidget(self.autolevelBox)
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

        self.meanBox.toggled.connect(self.update_widget_state)
        self.update_widget_state()

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)
        self.histo.gradient.loadPreset("inferno")

        self.img = pg.ImageItem()
        self.img_plot = self.layout.addPlot(row=0, col=0, lockAspect=False)
        self.img_plot.addItem(self.img)
        self.histo.setImageItem(self.img)
        self.img_plot.setLabel("bottom", "X")
        self.img_plot.setLabel("left", "Y")

    def update_widget_state(self):
        mean = self.meanBox.isChecked()
        self.meanincompBox.setEnabled(mean)
        self.lastnBox.setEnabled(mean)
        self.sweepidxBox.setEnabled(not mean)

    def update_labels(self, data: GridSweeperData):
        self.img_plot.setLabel("bottom", data.xlabel, data.xunit)
        self.img_plot.setLabel("left", data.ylabel, data.yunit)

    def update_sweep_index(self, data: GridSweeperData):
        # avoid n being 0
        n = max(data.image_num(), 1)
        self.sweepidxBox.setMinimum(-n)
        self.sweepidxBox.setMaximum(n - 1)

    def _select_image(self, data: GridSweeperData) -> np.ndarray | None:
        if self.meanBox.isChecked():
            return data.get_mean_image(
                last_n=self.lastnBox.value(), include_incomplete=self.meanincompBox.isChecked()
            )
        return data.get_image(self.sweepidxBox.value())

    def update_image(self, data: GridSweeperData):
        if not data.has_data():
            return

        img = self._select_image(data)
        if img is None or img.size == 0 or np.isnan(img).all():
            return

        self.img.updateImage(img)

        if self.autolevelBox.isChecked() and np.isfinite(img).any():
            mn, mx = np.nanmin(img), np.nanmax(img)
            if mn == mx:
                mn, mx = mn - 0.1, mx + 0.1
            self.histo.setLevels(mn, mx)

        xdata = data.get_xdata()
        ydata = data.get_ydata()

        if xdata.size == 0 or ydata.size == 0:
            return

        if xdata.size >= 2:
            x_scale = (xdata[-1] - xdata[0]) / (xdata.size - 1)
        else:
            x_scale = 1.0
        if ydata.size >= 2:
            y_scale = (ydata[-1] - ydata[0]) / (ydata.size - 1)
        else:
            y_scale = 1.0

        self.img.resetTransform()
        self.img.setPos(float(xdata[0]), float(ydata[0]))
        self.img.setTransform(QtGui.QTransform.fromScale(float(x_scale), float(y_scale)))

    def refresh(self, data: GridSweeperData):
        self.update_labels(data)
        self.update_sweep_index(data)
        self.update_image(data)

    def export_params(self):
        return {
            "mean": self.meanBox.isChecked(),
            "include_incomplete": self.meanincompBox.isChecked(),
            "last_n": self.lastnBox.value(),
            "sweep_index": self.sweepidxBox.value(),
        }


class GridSweeperWidget(ClientTopWidget):
    """Main widget for GridSweeper measurement."""

    def __init__(
        self,
        gconf: dict,
        name,
        gparams_name,
        context,
        parent=None,
        logger=None,
        file_transport_dir=None,
    ):
        ClientTopWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)

        self.conf = local_conf(gconf, name)
        self.init_ui()
        self.setWindowTitle(f"MAHOS.GridSweeperGUI ({join_name(name)})")

        self.cli = QGridSweeperClient(
            gconf, name, context=context, parent=self, file_transport_dir=file_transport_dir
        )
        self.cli.statusUpdated.connect(self.init_with_status)
        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli, self.gparams_cli)
        self.setEnabled(False)

    def init_ui(self):
        hl = QtWidgets.QHBoxLayout()
        self.startButton = QtWidgets.QPushButton("Start")
        self.stopButton = QtWidgets.QPushButton("Stop")
        self.saveButton = QtWidgets.QPushButton("Save")
        self.exportButton = QtWidgets.QPushButton("Export")
        self.loadButton = QtWidgets.QPushButton("Load")

        for w in (
            self.startButton,
            self.stopButton,
            self.saveButton,
            self.exportButton,
            self.loadButton,
        ):
            hl.addWidget(w)

        spacer0 = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        hl.addItem(spacer0)

        gl = QtWidgets.QGridLayout()

        xlabel = QtWidgets.QLabel("x (inner)")
        self.xstartBox = SpinBox(parent=self, prefix="start: ")
        self.xstopBox = SpinBox(parent=self, prefix="stop: ")
        self.xnumBox = QtWidgets.QSpinBox()
        self.xnumBox.setPrefix("num: ")
        self.xdelayBox = SpinBox(parent=self, prefix="delay: ")
        self.xlogBox = QtWidgets.QCheckBox("log")

        ylabel = QtWidgets.QLabel("y (outer)")
        self.ystartBox = SpinBox(parent=self, prefix="start: ")
        self.ystopBox = SpinBox(parent=self, prefix="stop: ")
        self.ynumBox = QtWidgets.QSpinBox()
        self.ynumBox.setPrefix("num: ")
        self.ydelayBox = SpinBox(parent=self, prefix="delay: ")
        self.ylogBox = QtWidgets.QCheckBox("log")

        clabel = QtWidgets.QLabel("common")
        self.sweepsBox = QtWidgets.QSpinBox()
        self.sweepsBox.setPrefix("sweeps: ")
        self.sweepsBox.setSuffix(" (0 for inf)")

        for w in (
            self.xstartBox,
            self.xstopBox,
            self.xnumBox,
            self.ystartBox,
            self.ystopBox,
            self.ynumBox,
            self.xdelayBox,
            self.ydelayBox,
            self.sweepsBox,
        ):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(160)
        for i, w in enumerate(
            (
                xlabel,
                self.xstartBox,
                self.xstopBox,
                self.xnumBox,
                self.xdelayBox,
                self.xlogBox,
            )
        ):
            gl.addWidget(w, 0, i)
        for i, w in enumerate(
            (
                ylabel,
                self.ystartBox,
                self.ystopBox,
                self.ynumBox,
                self.ydelayBox,
                self.ylogBox,
            )
        ):
            gl.addWidget(w, 1, i)

        gl.addWidget(clabel, 2, 0)
        gl.addWidget(self.sweepsBox, 2, 1)

        self.plot = PlotWidget(parent=self)

        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl)
        vl.addLayout(gl)
        vl.addWidget(self.plot)
        self.setLayout(vl)

    def init_with_status(self, status: BinaryStatus):
        """Initialize after receiving first status."""

        self.cli.statusUpdated.disconnect(self.init_with_status)

        params = self.cli.get_param_dict()
        if params is None:
            self.logger.error("Failed to get param dict.")
            return

        flat = params.flatten()
        apply_widgets(
            flat,
            [
                ("x.start", self.xstartBox),
                ("x.stop", self.xstopBox),
                ("x.num", self.xnumBox),
                ("x.delay", self.xdelayBox),
                ("x.log", self.xlogBox),
                ("y.start", self.ystartBox),
                ("y.stop", self.ystopBox),
                ("y.num", self.ynumBox),
                ("y.delay", self.ydelayBox),
                ("y.log", self.ylogBox),
                ("sweeps", self.sweepsBox),
            ],
        )
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

    def update_state(self, state: BinaryState):
        idle = state == BinaryState.IDLE

        for w in (
            self.startButton,
            self.saveButton,
            self.loadButton,
            self.xstartBox,
            self.xstopBox,
            self.xnumBox,
            self.xdelayBox,
            self.xlogBox,
            self.ystartBox,
            self.ystopBox,
            self.ynumBox,
            self.ydelayBox,
            self.ylogBox,
            self.sweepsBox,
        ):
            w.setEnabled(idle)

        self.stopButton.setEnabled(not idle)

    def update_data(self, data: GridSweeperData):
        self.plot.refresh(data)

    def request_start(self):
        params = {
            "x": {
                "start": self.xstartBox.value(),
                "stop": self.xstopBox.value(),
                "num": self.xnumBox.value(),
                "delay": self.xdelayBox.value(),
                "log": self.xlogBox.isChecked(),
            },
            "y": {
                "start": self.ystartBox.value(),
                "stop": self.ystopBox.value(),
                "num": self.ynumBox.value(),
                "delay": self.ydelayBox.value(),
                "log": self.ylogBox.isChecked(),
            },
            "sweeps": self.sweepsBox.value(),
        }
        self.cli.start(params)

    def request_stop(self):
        self.cli.stop()

    def save_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "GridSweeper", ".gsweep")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.gparams_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)

        png_fn = os.path.splitext(fn)[0] + ".png"
        self.cli.export_data(png_fn, params=self.plot.export_params())
        return fn

    def load_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "GridSweeper", ".gsweep")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.gparams_cli.set_param("loaded_note", data.note())

    def export_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "", (".png", ".pdf", ".eps", ".csv"))
        if not fn:
            return

        self.cli.export_data(fn, params=self.plot.export_params())


class GridSweeperGUI(GUINode):
    """GUINode for GridSweeperWidget.

    :param target.log: Optional LogBroker target for formal GUI logging.
    :type target.log: tuple[str, str] | str
    :param log_level: (default: ``"INFO"``) Optional logging level.
    :type log_level: str
    :param target.grid_sweeper: Target GridSweeper node full name.
    :type target.grid_sweeper: tuple[str, str] | str
    :param target.gparams: Target GlobalParams node full name.
    :type target.gparams: tuple[str, str] | str
    :param file_transport_dir: Optional GUI-side directory for shared file transport.
    :type file_transport_dir: str

    """

    def init_widget(self, gconf: dict, name, context):
        lconf = local_conf(gconf, name)
        target = lconf["target"]
        return GridSweeperWidget(
            gconf,
            target["grid_sweeper"],
            target["gparams"],
            context,
            logger=self.logger,
            file_transport_dir=lconf.get("file_transport_dir"),
        )
