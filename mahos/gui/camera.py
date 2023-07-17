#!/usr/bin/env python3

"""
GUI frontend of Camera.

.. This file is a part of MAHOS project.

"""

import os

import pyqtgraph as pg

from .Qt import QtCore

from .ui.camera import Ui_Camera
from .camera_client import QCameraClient

from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..msgs.camera_msgs import Image
from ..node.param_server import ParamClient
from .gui_node import GUINode
from .common_widget import ClientTopWidget
from .dialog import save_dialog, load_dialog
from ..node.node import local_conf, join_name
from ..util.timer import FPSCounter


class CameraWidget(ClientTopWidget, Ui_Camera):
    """Top widget for Camera."""

    def __init__(self, gconf: dict, name, param_server_name, context, parent=None):
        ClientTopWidget.__init__(self, parent)
        self.setupUi(self)
        self.setWindowTitle(f"MAHOS.CameraGUI ({join_name(name)})")

        self.iv = pg.ImageView(view=pg.PlotItem())
        self.verticalLayout.addWidget(self.iv)

        self.conf = local_conf(gconf, name)

        self._finalizing = False
        self.fps_counter = FPSCounter()

        self.cli = QCameraClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.param_cli = ParamClient(gconf, param_server_name, context=context)

        self.add_clients(self.cli, self.param_cli)

        self.setEnabled(False)

    def sizeHint(self):
        return QtCore.QSize(1600, 900)

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.init_connection()
        self.init_widget()

        # update initial GUI state
        self.update_state(status.state, last_state=BinaryState.IDLE)

        self.cli.stateUpdated.connect(self.update_state)
        self.cli.dataUpdated.connect(self.update_image)
        self.cli.stopped.connect(self.finalize)

        self.setEnabled(True)

    def init_connection(self):
        self.startButton.clicked.connect(self.request_start)
        self.stopButton.clicked.connect(self.request_stop)
        self.saveButton.clicked.connect(self.save_data)
        self.loadButton.clicked.connect(self.load_data)
        self.roiBox.stateChanged.connect(self.update_roi_boxes)

    def init_widget(self):
        # fire the event
        self.roiBox.setChecked(True)
        self.roiBox.setChecked(False)

    def update_roi_boxes(self):
        for b in (self.woffsetBox, self.widthBox, self.hoffsetBox, self.heightBox):
            b.setEnabled(self.roiBox.isChecked())

    def save_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Camera", ".camera")
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
        fn = load_dialog(self, default_path, "Camera", ".camera")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        image = self.cli.load_data(fn)
        if image is None:
            return
        if image.note():
            self.param_cli.set_param("loaded_note", image.note())
        self.iv.setImage(image.image.T)
        self.set_widget_values(image)

    def set_widget_values(self, image: Image):
        p = image.params
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

    def update_image(self, img: Image):
        self.iv.setImage(img.image.T, autoRange=False)
        self.fpsLabel.setText("{:.1f} fps".format(self.fps_counter.tick()))

    # State managements
    def request_stop(self):
        self.cli.stop()

    def request_start(self):
        """start the live stream."""

        params = {}
        params["exposure_time"] = self.exposuretimeBox.value() * 1e-3  # ms to sec
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
        self.fps_counter = FPSCounter()

        self.cli.start(params)

    def finalize(self, img: Image):
        if self._finalizing:
            return
        self._finalizing = True
        # Auto save is temporarily disabled.
        self._finalizing = False

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (self.startButton, self.saveButton, self.exposuretimeBox):
            w.setEnabled(state == BinaryState.IDLE)

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)

    def update_acquisitions_box(self):
        self.acquisitionsBox.setEnabled(self.stopafterBox.isChecked())


class CameraGUI(GUINode):
    """GUINode for Camera using CameraWidget."""

    def init_widget(self, gconf: dict, name, context):
        target = local_conf(gconf, name)["target"]
        return CameraWidget(gconf, target["camera"], target["param_server"], context)
