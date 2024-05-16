#!/usr/bin/env python3

"""
Logic and instrument control part of Camera stream.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from ..msgs.common_msgs import Reply, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.param_msgs import GetParamDictReq, GetParamDictLabelsReq
from ..msgs import camera_msgs
from .common_meas import BasicMeasClient, BasicMeasNode
from .common_worker import DummyWorker, PulseGen_CW, Switch
from .camera_worker import Poller
from .camera_io import CameraIO


class CameraClient(BasicMeasClient):
    """Node Client for Camera."""

    #: Message types for Camera.
    M = camera_msgs


class Camera(BasicMeasNode):
    CLIENT = CameraClient

    def __init__(self, gconf: dict, name, context=None):
        BasicMeasNode.__init__(self, gconf, name, context=context)

        if "switch" in self.conf["target"]["servers"]:
            self.switch = Switch(self.cli, self.logger, "camera")
        else:
            self.switch = DummyWorker()
        if "pg" in self.conf["target"]["servers"]:
            self.pg = PulseGen_CW(self.cli, self.logger)
        else:
            self.pg = DummyWorker()

        self.worker = Poller(self.cli, self.logger, self.conf.get("transform", {}))
        self.io = CameraIO(self.logger)

    def close_resources(self):
        if hasattr(self, "switch"):
            self.switch.stop()
        if hasattr(self, "pg"):
            self.pg.stop()
        if hasattr(self, "worker"):
            self.worker.stop()

    def change_state(self, msg: StateReq) -> Reply:
        if self.state == msg.state:
            return Reply(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            success = self.switch.stop() and self.pg.stop() and self.worker.stop()
            if not success:
                return Reply(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            success = self.switch.start() and self.pg.start() and self.worker.start(msg.params)
            if not success:
                return Reply(False, "Failed to start internal worker.", ret=self.state)

        self.state = msg.state
        return Reply(True)

    def save_data(self, msg: SaveDataReq) -> Reply:
        success = self.io.save_data(msg.file_name, self.worker.image_msg(), msg.note)
        if success:
            for tweaker_name, cli in self.tweaker_clis.items():
                success &= cli.save(msg.file_name, "__" + tweaker_name + "__")
        return Reply(success)

    def export_data(self, msg: ExportDataReq) -> Reply:
        success = self.io.export_data(msg.file_name, self.worker.image_msg(), msg.params)
        return Reply(success)

    def load_data(self, msg: LoadDataReq) -> Reply:
        image = self.io.load_data(msg.file_name)
        if image is None:
            return Reply(False)
        else:
            return Reply(True, ret=image)

    def get_param_dict_labels(self, msg: GetParamDictLabelsReq) -> Reply:
        return Reply(True, ret=["camera"])

    def get_param_dict(self, msg: GetParamDictReq) -> Reply:
        d = self.worker.get_param_dict(msg.label)

        if d is None:
            return Reply(False, "Failed to generate param dict.")
        else:
            return Reply(True, ret=d)

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        self.cli.wait("camera")
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        if self.state == BinaryState.ACTIVE and self.worker.work():
            self.data_pub.publish(self.worker.image_msg())
        self.status_pub.publish(BinaryStatus(state=self.state))
