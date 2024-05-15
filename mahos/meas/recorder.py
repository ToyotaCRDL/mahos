#!/usr/bin/env python3

"""
Generic data-logging measurement for time-series data.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from ..msgs.common_msgs import Reply, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.param_msgs import GetParamDictReq, GetParamDictLabelsReq
from ..msgs import recorder_msgs
from ..msgs.recorder_msgs import RecorderData
from .common_meas import BasicMeasClient, BasicMeasNode
from ..util.timer import IntervalTimer
from .tweaker import TweakerClient
from .recorder_worker import Collector
from .recorder_io import RecorderIO


class RecorderClient(BasicMeasClient):
    """Node Client for Recorder."""

    #: Message types for Recorder.
    M = recorder_msgs


class Recorder(BasicMeasNode):
    """Generic data-logging measurement for time-series data."""

    CLIENT = RecorderClient
    DATA = RecorderData

    def __init__(self, gconf: dict, name, context=None):
        BasicMeasNode.__init__(self, gconf, name, context=context)

        if "tweaker" in self.conf["target"]:
            self.tweaker_cli = TweakerClient(
                gconf, self.conf["target"]["tweaker"], context=self.ctx, prefix=self.joined_name()
            )
            self.add_client(self.tweaker_cli)
        else:
            self.tweaker_cli = None

        self.worker = Collector(
            self.cli, self.logger, self.conf.get("collector", {}), self.conf["mode"]
        )
        self.io = RecorderIO(self.logger)
        self.pub_timer = IntervalTimer(self.conf.get("pub_interval_sec", 0.5))

    def close_resources(self):
        if hasattr(self, "worker"):
            self.worker.stop()

    def change_state(self, msg: StateReq) -> Reply:
        if self.state == msg.state:
            return Reply(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            if not self.worker.stop():
                return Reply(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.worker.start(msg.params):
                return Reply(False, "Failed to start worker.", ret=self.state)

        self.state = msg.state
        return Reply(True)

    def get_param_dict_labels(self, msg: GetParamDictLabelsReq) -> Reply:
        return Reply(True, ret=self.worker.get_param_dict_labels())

    def get_param_dict(self, msg: GetParamDictReq) -> Reply:
        d = self.worker.get_param_dict(msg.label)
        if d is None:
            return Reply(False, "Failed to generate param dict")
        return Reply(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Reply:
        success = self.io.save_data(msg.file_name, self.worker.data_msg(), msg.note)
        if success and self.tweaker_cli is not None:
            success &= self.tweaker_cli.save(msg.file_name, "_inst_params")
        return Reply(success)

    def export_data(self, msg: ExportDataReq) -> Reply:
        success = self.io.export_data(
            msg.file_name, msg.data if msg.data else self.worker.data_msg(), msg.params
        )
        return Reply(success)

    def load_data(self, msg: LoadDataReq) -> Reply:
        data = self.io.load_data(msg.file_name)
        if data is None:
            return Reply(False)

        if msg.to_buffer:
            msg = "Cannot load data to buffer (buffer is not supported)."
            self.logger.error(msg)
            return Reply(False, msg)
        if self.state == BinaryState.ACTIVE:
            return Reply(False, "Cannot load data when active.")
        self.worker.data = data
        return Reply(True, ret=data)

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        for inst in self.cli.insts():
            self.cli.wait(inst)
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        updated = self._work()
        time_to_pub = self.pub_timer.check()
        self._publish(updated or time_to_pub)

    def _work(self) -> bool:
        if self.state == BinaryState.ACTIVE:
            return self.worker.work()
        return False

    def _publish(self, publish_data: bool):
        self.status_pub.publish(BinaryStatus(state=self.state))
        if publish_data:
            self.data_pub.publish(self.worker.data_msg())
