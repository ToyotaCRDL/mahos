#!/usr/bin/env python3

"""
GUI frontend of Camera.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import os

import pyqtgraph as pg

from mahos.gui.Qt import QtCore

from mahos.gui.ui.camera import Ui_Camera
from mahos.gui.camera_client import QCameraClient

from mahos.msgs.common_msgs import BinaryState, BinaryStatus
from mahos.msgs.camera_msgs import Image
from mahos.node.global_params import GlobalParamsClient
from mahos.gui.gui_node import GUINode, get_gui_logger
from mahos.gui.common_widget import ClientTopWidget
from mahos.gui.dialog import save_dialog, load_dialog
from mahos.gui.param import apply_widgets
from mahos.node.node import local_conf, join_name
from mahos.util.timer import FPSCounter


class CameraWidget(ClientTopWidget, Ui_Camera):
    """Top widget for Camera."""

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
        self.setupUi(self)
        self.setWindowTitle(f"MAHOS.CameraGUI ({join_name(name)})")

        self.iv = pg.ImageView(view=pg.PlotItem())
        self.verticalLayout.addWidget(self.iv)

        self.conf = local_conf(gconf, name)

        self._finalizing = False
        self.fps_counter = FPSCounter()

        self.cli = QCameraClient(
            gconf, name, context=context, parent=self, file_transport_dir=file_transport_dir
        )
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli, self.gparams_cli)

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

    def init_widget(self):
        # fire the event
        self.roiBox.setChecked(True)
        self.roiBox.setChecked(False)

        params = self.cli.get_param_dict()
        if params is None:
            return

        apply_widgets(
            params,
            [
                ("exposure_time", self.exposuretimeBox, 1e3),  # s to ms
                ("frame_rate", self.framerateBox),
            ],
        )

    def save_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Camera", ".camera")
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
        fn = load_dialog(self, default_path, "Camera", ".camera")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        image = self.cli.load_data(fn)
        if image is None:
            return
        if image.note():
            self.gparams_cli.set_param("loaded_note", image.note())
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
        if self.framerateenableBox.isChecked():
            params["frame_rate"] = self.framerateBox.value()
        else:
            params["frame_rate"] = None
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
        for w in (
            self.startButton,
            self.saveButton,
            self.exposuretimeBox,
            self.binningBox,
            self.framerateenableBox,
            self.roiBox,
        ):
            w.setEnabled(state == BinaryState.IDLE)
        for w in (
            self.widthBox,
            self.heightBox,
            self.woffsetBox,
            self.hoffsetBox,
        ):
            w.setEnabled(state == BinaryState.IDLE and self.roiBox.isChecked())

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)
        self.framerateBox.setEnabled(
            state == BinaryState.IDLE and self.framerateenableBox.isChecked()
        )

    def update_acquisitions_box(self):
        self.acquisitionsBox.setEnabled(self.stopafterBox.isChecked())


class CameraGUI(GUINode):
    """GUINode for Camera using CameraWidget.

    :param target.log: Optional LogBroker target for formal GUI logging.
    :type target.log: tuple[str, str] | str
    :param log_level: (default: ``"INFO"``) Optional logging level.
    :type log_level: str
    :param target.camera: Target Camera node full name.
    :type target.camera: tuple[str, str] | str
    :param target.gparams: Target GlobalParams node full name.
    :type target.gparams: tuple[str, str] | str
    :param file_transport_dir: Optional GUI-side directory for shared file transport.
    :type file_transport_dir: str

    """

    def init_widget(self, gconf: dict, name, context):
        lconf = local_conf(gconf, name)
        target = lconf["target"]
        return CameraWidget(
            gconf,
            target["camera"],
            target["gparams"],
            context,
            logger=self.logger,
            file_transport_dir=lconf.get("file_transport_dir"),
        )
