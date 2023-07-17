#!/usr/bin/env python3

"""
GUI frontend of Imaging ODMR.

.. This file is a part of MAHOS project.

"""

import uuid
import os
import typing as T

import numpy as np
import pyqtgraph as pg

from . import Qt
from .Qt import QtCore, QtWidgets, QtGui

from .ui.iodmr import Ui_IODMR
from .iodmr_client import QIODMRClient

from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..msgs.param_msgs import FloatParam, IntParam
from ..msgs.iodmr_msgs import IODMRData
from ..node.param_server import ParamClient
from ..util import conv
from .gui_node import GUINode
from .common_widget import ClientMainWindow
from .dialog import save_dialog, load_dialog
from ..node.node import local_conf


Policy = QtWidgets.QSizePolicy.Policy


class IODMRWidget(QtWidgets.QWidget, Ui_IODMR):
    """Central widget (parameter setting and commanding) for IODMR."""

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

    def init_widget(self, params: dict):
        # fire the event
        self.roiBox.setChecked(True)
        self.roiBox.setChecked(False)

        if params is None:
            print("[ERROR] Failed to get params.")
            return

        power: FloatParam = params["power"]
        self.powerBox.setMinimum(power.minimum())
        self.powerBox.setMaximum(power.maximum())
        self.powerBox.setValue(power.value())

        start: FloatParam = params["start"]
        self.startBox.setMinimum(start.minimum() * 1e-6)  # Hz -> MHz
        self.startBox.setMaximum(start.maximum() * 1e-6)
        self.startBox.setValue(start.value() * 1e-6)

        stop: FloatParam = params["stop"]
        self.stopBox.setMinimum(stop.minimum() * 1e-6)  # Hz -> MHz
        self.stopBox.setMaximum(stop.maximum() * 1e-6)
        self.stopBox.setValue(stop.value() * 1e-6)

        num: IntParam = params["num"]
        self.numBox.setMinimum(num.minimum())
        self.numBox.setMaximum(num.maximum())
        self.numBox.setValue(num.value())

    def init_connection(self):
        self.roiBox.stateChanged.connect(self.update_roi_boxes)

        for w in (self.startBox, self.stopBox, self.numBox):
            w.valueChanged.connect(self.update_step)

    def update_step(self):
        self.stepLabel.setText(
            "step: {:.4f} MHz".format(
                conv.num_to_step(self.startBox.value(), self.stopBox.value(), self.numBox.value())
            )
        )

    def update_roi_boxes(self):
        for b in (self.woffsetBox, self.widthBox, self.hoffsetBox, self.heightBox):
            b.setEnabled(self.roiBox.isChecked())

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.startButton,
            self.saveButton,
            self.compression,
            self.loadButton,
            self.startBox,
            self.stopBox,
            self.numBox,
            self.powerBox,
            self.exposuredelayBox,
            self.exposuretimeBox,
            self.burstBox,
            self.sweepsBox,
            self.binningBox,
        ):
            w.setEnabled(state == BinaryState.IDLE)

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)

    def validate_sweep(self):
        if self.startBox.value() >= self.stopBox.value():
            return (False, "Stop value must be greater than Start value.")

        return (True, "")

    def get_params(self) -> dict:
        params = {}
        params["start"] = self.startBox.value() * 1.0e6  # MHz to Hz
        params["stop"] = self.stopBox.value() * 1.0e6  # MHz to Hz
        params["num"] = self.numBox.value()
        params["power"] = self.powerBox.value()
        params["sweeps"] = self.sweepsBox.value()
        if self.roiBox.isChecked():
            params["roi"] = {
                "width": self.widthBox.value(),
                "height": self.heightBox.value(),
                "woffset": self.woffsetBox.value(),
                "hoffset": self.hoffsetBox.value(),
            }
        else:
            params["roi"] = None
        params["binning"] = self.binningBox.value()

        params["exposure_delay"] = self.exposuredelayBox.value() * 1.0e-3  # ms to sec
        params["exposure_time"] = self.exposuretimeBox.value() * 1.0e-3  # ms to sec

        params["burst_num"] = self.burstBox.value()

        return params

    def set_widget_values(self, data: IODMRData):
        p = data.params
        start, stop = p["start"], p["stop"]
        self.startBox.setValue(start * 1.0e-6)
        self.stopBox.setValue(stop * 1.0e-6)
        self.numBox.setValue(p["num"])
        self.powerBox.setValue(p["power"])
        self.exposuredelayBox.setValue(p["exposure_delay"] * 1.0e3)
        self.exposuretimeBox.setValue(p["exposure_time"] * 1.0e3)
        if "binning" in p:
            self.binningBox.setValue(p["binning"])
        if p.get("roi"):
            roi = p["roi"]
            self.roiBox.setChecked(True)
            self.widthBox.setValue(roi["width"])
            self.heightBox.setValue(roi["height"])
            self.woffsetBox.setValue(roi["woffset"])
            self.hoffsetBox.setValue(roi["hoffset"])
        else:
            self.roiBox.setChecked(False)
        if "burst_num" in p:
            self.burstBox.setValue(p["burst_num"])


class DataStore(object):
    """IODMRData with some convenient methods"""

    def __init__(self):
        self.data: T.Optional[IODMRData] = None

    def set_data(self, data: IODMRData):
        self.data = data

    def has_data(self):
        return self.data is not None

    def no_data(self):
        return self.data is None

    def has_same_ident(self, other: IODMRData):
        """Take another data and check if self and other has same identifier."""

        return self.data.ident == other.ident

    def can_resume(self, new_params: dict) -> bool:
        """Check if the measurement can be continued with given new_params."""

        p = new_params
        return (
            self.data.params["start"] == p["start"]
            and self.data.params["stop"] == p["stop"]
            and self.data.params["num"] == p["num"]
        )

    def ident(self):
        return self.data.ident

    def mean_data(self) -> np.ndarray:
        return self.data.data_sum / self.data.sweeps

    def latest_data(self) -> np.ndarray:
        return self.data.data_latest

    def history_data(self) -> np.ndarray:
        return self.data.data_history

    def freqs(self) -> np.ndarray:
        p = self.data.params
        return np.linspace(p["start"], p["stop"], p["num"])

    def step(self) -> float:
        return conv.num_to_step(
            self.data.params["start"], self.data.params["stop"], self.data.params["num"]
        )


class ImageView(pg.ImageView):
    def sizeHint(self):
        return QtCore.QSize(1400, 1400)


class IVROI(pg.ROI):
    def __init__(self, size, handle_size, **args):
        pg.ROI.__init__(self, pos=[0, 0], size=size, **args)
        self.handleSize = handle_size
        self.addScaleHandle([1, 1], [0, 0])
        self.addRotateHandle([0, 0], [0.5, 0.5])


class ODMRHistoryImageView(QtWidgets.QWidget):
    # TODO: backport to gui/odmr

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.init_widget()
        self.init_view()

    def init_widget(self):
        self.graphicsView = pg.GraphicsView(self)
        self.histo = pg.HistogramLUTWidget(self)
        self.lastNBox = QtWidgets.QSpinBox(self)
        self.lastNBox.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
        self.lastNBox.setMaximumWidth(200)
        self.lastNBox.setMaximum(999)
        self.lastNBox.setMinimum(0)
        self.lastNBox.setValue(0)
        self.lastNlabel = QtWidgets.QLabel(self)
        self.lastNlabel.setText("Show mean of last N accumulations (set 0 to show all)")

        spacerItem = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setSpacing(0)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)

        self.horizontalLayout.addWidget(self.graphicsView)
        self.horizontalLayout.addWidget(self.histo)

        self.horizontalLayout_2.addWidget(self.lastNlabel)
        self.horizontalLayout_2.addWidget(self.lastNBox)
        self.horizontalLayout_2.addItem(spacerItem)

        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout)

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)
        self.plot = self.layout.addPlot(row=0, col=0, lockAspect=False)
        self.plot_item = pg.PlotDataItem(np.zeros(101), pen="#FFFFFF")
        self.plot.addItem(self.plot_item)
        self.img_plot = self.layout.addPlot(row=1, col=0, lockAspect=False)
        self.img_item = self.xdata = None

        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("bottom", "MW Frequency", "Hz")
        self.plot.setLabel("left", "Intensity", "counts")

        self.img_plot.setLabel("bottom", "MW Frequency", "Hz")
        self.img_plot.setLabel("left", "Number of accumulation")

        self.histo.gradient.loadPreset("inferno")

    def update(self, data: DataStore):
        h = data.history_data()
        self._update_image(h)
        self._update_plot(h)

    def _update_image(self, img_history: np.ndarray, setlevel=True):
        self.img_item.updateImage(img_history)
        if setlevel:
            mn, mx = np.nanmin(img_history), np.nanmax(img_history)
            if mn == mx:
                mn, mx = mn - 0.1, mx + 0.1  # tekitou
            self.histo.setLevels(mn, mx)
        self.img_item.resetTransform()
        self.img_item.setPos(*self.img_pos)
        self.img_item.setTransform(QtGui.QTransform.fromScale(*self.img_steps))

    def _update_plot(self, img_history: np.ndarray):
        if self.lastNBox.value():
            ydata = np.mean(img_history[:, -self.lastNBox.value() :], axis=1)
        else:
            ydata = np.mean(img_history, axis=1)

        self.plot_item.setData(self.xdata, ydata)

    def new_data(self, data: DataStore):
        self.img_plot.clear()
        self.img_item = pg.ImageItem(data.history_data())
        self.img_plot.addItem(self.img_item)
        self.histo.setImageItem(self.img_item)

        self.xdata = data.freqs()
        self.img_pos = (self.xdata[0], 0.0)
        self.img_steps = (data.step(), 1.0)


class IODMRMainWindow(ClientMainWindow):
    """MainWindow with ConfocalWidget and traceView."""

    def __init__(self, gconf: dict, name, param_server_name, context, parent=None):
        ClientMainWindow.__init__(self, parent)

        self.conf = local_conf(gconf, name)

        self._finalizing = False

        self.data = DataStore()
        self.cw = IODMRWidget(parent=self)
        self.setWindowTitle(f"MAHOS.IODMRGUI ({name})")
        self.setAnimated(False)
        self.setCentralWidget(self.cw)

        self.init_view()

        self.init_menu()

        self.cli = QIODMRClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.param_cli = ParamClient(gconf, param_server_name, context=context)

        self.add_clients(self.cli, self.param_cli)

        self.setEnabled(False)

    def init_view(self, roi_style: T.Optional[dict] = None):
        s = roi_style or {}
        z = s.get("size", 100)
        w = s.get("width", 2.0)
        p = pg.mkPen(s.get("color", (30, 144, 255)), width=w)
        hp = pg.mkPen(s.get("handle_color", (0, 255, 255)), width=w)
        vp = pg.mkPen(s.get("hover_color", (255, 255, 0)), width=w)

        mv_roi = IVROI(
            size=pg.Point(z),
            handle_size=s.get("handle_size", 8),
            pen=p,
            hoverPen=vp,
            handlePen=hp,
            handleHoverPen=vp,
        )
        lv_roi = IVROI(
            size=pg.Point(z),
            handle_size=s.get("handle_size", 8),
            pen=p,
            hoverPen=vp,
            handlePen=hp,
            handleHoverPen=vp,
        )

        self.meanView = ImageView(roi=mv_roi, view=pg.PlotItem())
        self.latestView = ImageView(roi=lv_roi, view=pg.PlotItem())
        self.historyView = ODMRHistoryImageView()

        self.meanView.ui.histogram.gradient.loadPreset("inferno")
        self.latestView.ui.histogram.gradient.loadPreset("inferno")

        self.d_meanView = QtWidgets.QDockWidget("Mean", parent=self)
        self.d_meanView.setWidget(self.meanView)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_meanView)

        self.d_latestView = QtWidgets.QDockWidget("Latest", parent=self)
        self.d_latestView.setWidget(self.latestView)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_latestView)

        self.d_historyView = QtWidgets.QDockWidget("History", parent=self)
        self.d_historyView.setWidget(self.historyView)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_historyView)

        self.tabifyDockWidget(self.d_historyView, self.d_latestView)
        self.tabifyDockWidget(self.d_latestView, self.d_meanView)

    def init_menu(self):
        self.file_menu = self.menuBar().addMenu("File")
        self.action_close = QtGui.QAction("Close")
        self.action_close.triggered.connect(self.close)
        self.file_menu.addAction(self.action_close)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_meanView.toggleViewAction())
        self.view_menu.addAction(self.d_latestView.toggleViewAction())
        self.view_menu.addAction(self.d_historyView.toggleViewAction())

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.init_connection()
        self.cw.init_connection()
        self.cw.init_widget(self.cli.get_param_dict("bounds"))
        self.cw.update_step()

        # update initial GUI state
        self.cw.update_state(status.state, last_state=BinaryState.IDLE)

        self.cli.stateUpdated.connect(self.cw.update_state)
        self.cli.dataUpdated.connect(self.update_data)
        self.cli.stopped.connect(self.finalize_sweep)

        self.setEnabled(True)

    def init_connection(self):
        self.cw.startButton.clicked.connect(self.request_start)
        self.cw.stopButton.clicked.connect(self.request_stop)
        self.cw.saveButton.clicked.connect(self.save_data)
        self.cw.loadButton.clicked.connect(self.load_data)

    def save_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "IODMR", ".iodmr", enable_bz2=True)
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.param_cli.get_param("note", "")
        params = self.cw.compression.get_params()
        self.cli.save_data(fn, params=params, note=note)

        # TODO: export
        # n = os.path.splitext(fn)[0] + ".png"
        # self.cli.export_data(n)

        return fn

    def load_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "IODMR", ".iodmr", enable_bz2=True)
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.param_cli.set_param("loaded_note", data.note())

        self.update_data(data)
        self.cw.set_widget_values(data)

    def update_data(self, data: IODMRData):
        new = self.data.no_data() or not self.data.has_same_ident(data)
        self.data.set_data(data)
        if new:
            self.historyView.new_data(self.data)

        self.update_view()

    def update_view(self):
        freqs = self.data.freqs()
        self.meanView.setImage(self.data.mean_data().swapaxes(1, 2), xvals=freqs, autoRange=False)
        self.latestView.setImage(
            self.data.latest_data().swapaxes(1, 2), xvals=freqs, autoRange=False
        )
        self.historyView.update(self.data)

    def request_stop(self):
        self.cli.stop()

    def request_start(self):
        valid, msg = self.cw.validate_sweep()

        if valid:
            self.start_sweep()
        else:
            QtWidgets.QMessageBox.warning(self, "Sweep parameter invalid.", msg)

    def start_sweep(self, force_cont=False):
        """start the sweep.

        "Continue sweep?" dialog can be supressed by setting force_cont=True."""

        params = self.cw.get_params()

        title = "Continue sweep?"
        body = (
            "Continue sweep with current data?"
            + " Press No to clear current data and start a new sweep."
        )
        if (
            self.data.has_data()
            and self.data.can_resume(params)
            and (force_cont or Qt.question_yn(self, title, body))
        ):
            params["resume"] = True
            params["ident"] = self.data.ident()
        else:
            params["resume"] = False
            params["ident"] = uuid.uuid4()

        self.cli.start(params)

    def finalize_sweep(self, data: IODMRData):
        if self._finalizing:
            return
        self._finalizing = True
        # Auto save is temporarily disabled.
        self._finalizing = False


class IODMRGUI(GUINode):
    """GUINode for IODMR using IODMRWidget."""

    def init_widget(self, gconf: dict, name, context):
        target = local_conf(gconf, name)["target"]
        return IODMRMainWindow(gconf, target["iodmr"], target["param_server"], context)
