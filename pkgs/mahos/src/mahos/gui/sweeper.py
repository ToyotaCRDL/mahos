#!/usr/bin/env python3

"""
GUI widget for Sweeper measurement.

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
from mahos.msgs.sweeper_msgs import SweeperData
from mahos.node.global_params import GlobalParamsClient
from mahos.node.node import local_conf, join_name

Policy = QtWidgets.QSizePolicy.Policy


class PlotWidget(QtWidgets.QWidget):
    """Widget containing line plot and image plot for 2D sweep data."""

    def __init__(self, xy_labels, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.init_ui()
        self.init_view(xy_labels)

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    def init_ui(self):
        # Plot options
        hl0 = QtWidgets.QHBoxLayout()

        self.showimgBox = QtWidgets.QCheckBox("Show image")
        self.showimgBox.setChecked(True)

        self.lastnBox = QtWidgets.QSpinBox(parent=self)
        self.lastnBox.setPrefix("last_n: ")
        self.lastnBox.setMinimum(-1_000_000)
        self.lastnBox.setMaximum(1_000_000)
        self.lastnBox.setValue(0)
        self.lastnBox.setToolTip("Average last N sweeps (0 for all)")

        self.lastnimgBox = QtWidgets.QSpinBox(parent=self)
        self.lastnimgBox.setPrefix("last_n (img): ")
        self.lastnimgBox.setMinimum(-1_000_000)
        self.lastnimgBox.setMaximum(1_000_000)
        self.lastnimgBox.setValue(0)
        self.lastnimgBox.setToolTip("Show last N sweeps in image (0 for all)")

        for w in (self.lastnBox, self.lastnimgBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(180)

        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)

        hl0.addWidget(self.showimgBox)
        hl0.addWidget(self.lastnBox)
        hl0.addWidget(self.lastnimgBox)
        hl0.addItem(spacer)

        # Graphics view with histogram
        hl = QtWidgets.QHBoxLayout()
        self.graphicsView = pg.GraphicsView(parent=self)
        self.histo = pg.HistogramLUTWidget(parent=self)
        hl.addWidget(self.graphicsView)
        hl.addWidget(self.histo)

        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl0)
        vl.addLayout(hl)
        self.setLayout(vl)

    def init_view(self, xy_labels):
        xlabel, xunit, ylabel, yunit = xy_labels
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)
        self.histo.gradient.loadPreset("inferno")

        # Line plot (top)
        self.plot = self.layout.addPlot(row=0, col=0, lockAspect=False)
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("bottom", xlabel, xunit)
        self.plot.setLabel("left", ylabel, yunit)
        self.plot_item = self.plot.plot(pen=pg.mkPen("w", width=1.0))

        # Image plot (bottom)
        self.img = pg.ImageItem()
        self.img_plot = self.layout.addPlot(row=1, col=0, lockAspect=False)
        self.img_plot.addItem(self.img)
        self.histo.setImageItem(self.img)
        self.img_plot.setLabel("bottom", xlabel, xunit)
        self.img_plot.setLabel("left", "Number of Sweeps")

        self.showimgBox.toggled.connect(self.toggle_image)

    def toggle_image(self, show: bool):
        if show:
            self.layout.addItem(self.img_plot, row=1, col=0)
            self.histo.show()
        else:
            self.layout.removeItem(self.img_plot)
            self.histo.hide()

    def update_labels(self, data: SweeperData):
        self.plot.setLabel("bottom", data.xlabel, data.xunit)
        self.plot.setLabel("left", data.ylabel, data.yunit)
        self.img_plot.setLabel("bottom", data.xlabel, data.xunit)

    def update_plot(self, data: SweeperData):
        """Update line plot with averaged data."""

        if not data.has_data():
            return

        xdata = data.get_xdata()
        ydata = data.get_ydata(last_n=self.lastnBox.value())

        self.plot_item.setData(xdata, ydata)

        # Update log scale
        if data.params.get("log", False):
            self.plot.setLogMode(x=True, y=False)
        else:
            self.plot.setLogMode(x=False, y=False)

    def update_image(self, data: SweeperData, setlevel: bool = True):
        """Update image plot with 2D data."""

        if not data.has_data():
            return

        img = data.get_image(last_n=self.lastnimgBox.value())

        if img.size == 0:
            return
        self.img.updateImage(img)

        if setlevel:
            mn, mx = np.nanmin(img), np.nanmax(img)
            if mn == mx:
                mn, mx = mn - 0.1, mx + 0.1
            self.histo.setLevels(mn, mx)

        # Set transform to map image coordinates to data coordinates
        xdata = data.get_xdata()
        if len(xdata) >= 2:
            x_scale = (xdata[-1] - xdata[0]) / (len(xdata) - 1)
        else:
            x_scale = 1.0

        self.img.resetTransform()
        self.img.setPos(data.params["start"], 0)
        self.img.setTransform(QtGui.QTransform.fromScale(x_scale, 1.0))

    def refresh(self, data: SweeperData):
        """Refresh both plots."""

        self.update_labels(data)
        self.update_plot(data)
        if self.showimgBox.isChecked():
            self.update_image(data)


class SweeperWidget(ClientTopWidget):
    """Main widget for Sweeper measurement."""

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
        self.init_ui(self._xy_labels_from_conf())
        self.setWindowTitle(f"MAHOS.SweeperGUI ({join_name(name)})")

        self.cli = QBasicMeasClient(
            gconf, name, context=context, parent=self, file_transport_dir=file_transport_dir
        )
        self.cli.statusUpdated.connect(self.init_with_status)
        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli)
        self.setEnabled(False)

    def _xy_labels_from_conf(self):
        """Read axis labels from configuration."""

        x_conf = self.conf.get("x", {})
        meas_conf = self.conf.get("measure", {})

        return (
            x_conf.get("key", "X"),
            x_conf.get("unit", ""),
            meas_conf.get("key", "Measurement"),
            meas_conf.get("unit", ""),
        )

    def init_ui(self, xy_labels):
        """Initialize UI elements."""

        # Control buttons
        hl0 = QtWidgets.QHBoxLayout()
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
            hl0.addWidget(w)

        spacer0 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        hl0.addItem(spacer0)

        # Parameter widgets
        hl1 = QtWidgets.QHBoxLayout()

        self.startBox = SpinBox(parent=self, prefix="start: ")
        self.stopBox = SpinBox(parent=self, prefix="stop: ")

        self.numBox = QtWidgets.QSpinBox()
        self.numBox.setPrefix("num: ")

        self.delayBox = SpinBox(parent=self, prefix="delay: ")

        self.sweepsBox = QtWidgets.QSpinBox()
        self.sweepsBox.setPrefix("sweeps: ")
        self.sweepsBox.setSuffix(" (0 for inf)")

        self.logBox = QtWidgets.QCheckBox("log")

        for w in (
            self.startBox,
            self.stopBox,
            self.numBox,
            self.delayBox,
            self.sweepsBox,
        ):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)
            hl1.addWidget(w)

        hl1.addWidget(self.logBox)

        spacer1 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        hl1.addItem(spacer1)

        self.plot = PlotWidget(xy_labels, parent=self)

        # Layout
        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl0)
        vl.addLayout(hl1)
        vl.addWidget(self.plot)
        self.setLayout(vl)

    def init_with_status(self, status: BinaryStatus):
        """Initialize after receiving first status."""

        self.cli.statusUpdated.disconnect(self.init_with_status)

        params = self.cli.get_param_dict()
        if params is not None:
            apply_widgets(
                params,
                [
                    ("start", self.startBox),
                    ("stop", self.stopBox),
                    ("num", self.numBox),
                    ("delay", self.delayBox),
                    ("sweeps", self.sweepsBox),
                    ("log", self.logBox),
                ],
            )
        else:
            self.logger.error("Failed to get param dict.")

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
        idle = state == BinaryState.IDLE

        for w in (
            self.startButton,
            self.saveButton,
            self.loadButton,
            self.startBox,
            self.stopBox,
            self.numBox,
            self.delayBox,
            self.sweepsBox,
            self.logBox,
        ):
            w.setEnabled(idle)

        self.stopButton.setEnabled(not idle)

    def update_data(self, data: SweeperData):
        self.plot.refresh(data)

    def request_start(self):
        params = {
            "start": self.startBox.value(),
            "stop": self.stopBox.value(),
            "num": self.numBox.value(),
            "delay": self.delayBox.value(),
            "sweeps": self.sweepsBox.value(),
            "log": self.logBox.isChecked(),
        }
        self.cli.start(params)

    def request_stop(self):
        self.cli.stop()

    def save_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Sweeper", ".sweep")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.gparams_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)

        # Auto-export PNG
        png_fn = os.path.splitext(fn)[0] + ".png"
        self.cli.export_data(png_fn)
        return fn

    def load_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "Sweeper", ".sweep")
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

        self.cli.export_data(fn)


class SweeperGUI(GUINode):
    """GUINode for SweeperWidget.

    :param target.log: Optional LogBroker target for formal GUI logging.
    :type target.log: tuple[str, str] | str
    :param log_level: (default: ``"INFO"``) Optional logging level.
    :type log_level: str
    :param target.sweeper: Target Sweeper node full name.
    :type target.sweeper: tuple[str, str] | str
    :param target.gparams: Target GlobalParams node full name.
    :type target.gparams: tuple[str, str] | str
    :param file_transport_dir: Optional GUI-side directory for shared file transport.
    :type file_transport_dir: str

    """

    def init_widget(self, gconf: dict, name, context):
        lconf = local_conf(gconf, name)
        target = lconf["target"]
        return SweeperWidget(
            gconf,
            target["sweeper"],
            target["gparams"],
            context,
            logger=self.logger,
            file_transport_dir=lconf.get("file_transport_dir"),
        )
