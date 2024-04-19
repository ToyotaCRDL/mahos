#!/usr/bin/env python3

"""
Generic Data-logging measurement for Time-series data.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from ..msgs.common_msgs import Reply, StateReq, BinaryState, BinaryStatus
from ..msgs.param_msgs import GetParamDictReq, GetParamDictLabelsReq
from ..msgs import chrono_msgs
from ..msgs.chrono_msgs import ChronoData
from .common_meas import BasicMeasNode
from ..node.client import StatusClient
from ..util.timer import IntervalTimer
from .chrono_worker import Collector


class ChronoClient(StatusClient):
    """Simple Chrono Client."""

    M = chrono_msgs


class Chrono(BasicMeasNode):
    """Generic Data-logging measurement for Time-series data."""

    CLIENT = ChronoClient
    DATA = ChronoData

    def __init__(self, gconf: dict, name, context=None):
        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.worker = Collector(self.cli, self.logger, self.conf["collector"])
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

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        for inst in self.cli.insts():
            self.cli.wait(inst)
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        updated = self._work()
        time_to_pub = self.pub_timer.check()
        self._publish(updated or time_to_pub, time_to_pub)

    def _work(self) -> bool:
        if self.state == BinaryState.ACTIVE:
            return self.worker.work()
        return False

    def _publish(self, publish_data: bool, publish_buffer: bool):
        self.status_pub.publish(BinaryStatus(state=self.state))
        if publish_data:
            self.data_pub.publish(self.worker.data_msg())
        if publish_buffer:
            self.buffer_pub.publish(self.buffer)
