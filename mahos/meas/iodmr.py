#!/usr/bin/env python3

"""
Logic and instrument control part of Imaging ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from ..msgs.common_msgs import Reply, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.param_msgs import GetParamDictReq
from ..msgs import iodmr_msgs
from .common_meas import BasicMeasClient, BasicMeasNode
from .common_worker import DummyWorker, Switch
from .iodmr_worker import ISweeperOverlay, ISweeperDirect, WorkStatus
from .iodmr_io import IODMRIO


class IODMRClient(BasicMeasClient):
    """Node Client for IODMR."""

    #: Message types for IODMR.
    M = iodmr_msgs


class IODMR(BasicMeasNode):
    CLIENT = IODMRClient

    def __init__(self, gconf: dict, name, context=None):
        BasicMeasNode.__init__(self, gconf, name, context=context)

        _default_sw_names = ["switch"] if "switch" in self.conf["target"]["servers"] else []
        sw_names = self.conf.get("switch_names", _default_sw_names)
        if sw_names:
            self.switch = Switch(
                self.cli, self.logger, sw_names, self.conf.get("switch_command", "iodmr")
            )
        else:
            self.switch = DummyWorker()

        self._direct = "isweeper" not in self.conf["target"]["servers"]
        if self._direct:
            self.isweeper = ISweeperDirect(self.cli, self.logger)
        else:
            self.isweeper = ISweeperOverlay(self.cli, self.logger)

        self.io = IODMRIO(self.logger)
        self._manually_stopped = False

    def close_resources(self):
        if hasattr(self, "switch"):
            self.switch.stop()
        if hasattr(self, "isweeper"):
            self.isweeper.stop()
        self.io.close()

    def change_state(self, msg: StateReq) -> Reply:
        if self.state == msg.state:
            return Reply(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            success = self.switch.stop() and self.isweeper.stop()
            self._manually_stopped = True
            if not success:
                return Reply(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.switch.start():
                return Reply(False, "Failed to start switch.", ret=self.state)
            if not self.isweeper.start(msg.params):
                self.switch.stop()
                return Reply(False, "Failed to start worker.", ret=self.state)

        self.state = msg.state
        # publish changed state immediately to prevent StateManager from missing the change
        self.status_pub.publish(BinaryStatus(state=self.state))
        return Reply(True)

    def get_param_dict(self, msg: GetParamDictReq) -> Reply:
        b = self.isweeper.get_param_dict(msg.label)
        if b is None:
            return Reply(False, "Failed to generate param dict.")
        else:
            return Reply(True, ret=b)

    def save_data(self, msg: SaveDataReq) -> Reply:
        success = self.io.save_data_async(
            msg.file_name, self.isweeper.data_msg(), msg.params, msg.note
        )
        return Reply(success)

    def export_data(self, msg: ExportDataReq) -> Reply:
        success = self.io.export_data(msg.file_name, self.isweeper.data_msg())
        return Reply(success)

    def load_data(self, msg: LoadDataReq) -> Reply:
        data = self.io.load_data(msg.file_name)
        if data is None:
            return Reply(False)
        else:
            return Reply(True, ret=data)

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        if self._direct:
            self.cli.wait("sg")
        else:
            self.cli.wait("isweeper")
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        status = self._work()
        finished = self._check_finished(status == WorkStatus.Error)
        # self._publish(status == WorkStatus.SweepDone)
        self._publish(finished)

    def _work(self) -> WorkStatus:
        if self.state == BinaryState.ACTIVE:
            return self.isweeper.work()
        else:
            return WorkStatus.Normal

    def _publish(self, publish_data: bool):
        self.status_pub.publish(BinaryStatus(state=self.state))
        if publish_data:
            self.logger.info("Publishing the data.")
            self.data_pub.publish(self.isweeper.data_msg())

    def _check_finished(self, stop: bool) -> bool:
        if self._manually_stopped:
            self._manually_stopped = False
            return True
        if self.state == BinaryState.ACTIVE and (stop or self.isweeper.is_finished()):
            self.change_state(StateReq(BinaryState.IDLE))
            return True
        return False
