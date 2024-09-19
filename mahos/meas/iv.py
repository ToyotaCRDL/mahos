#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..meas.common_meas import BasicMeasClient, BasicMeasNode
from .common_worker import Worker
from ..msgs.common_msgs import Reply, StateReq, BinaryState, BinaryStatus
from ..msgs.param_msgs import GetParamDictReq, GetParamDictLabelsReq
from ..msgs import param_msgs as P
from .iv_io import IVIO

from ..msgs import iv_msgs
from ..msgs.iv_msgs import IVData

from mahos.inst.smu_interface import SMUInterface


class IVClient(BasicMeasClient):
    """Node Client for IV."""

    #: Message types for IV.
    M = iv_msgs

    # Override for type hint.
    def get_data(self) -> IVData:
        return self._get_data()


class Sweeper(Worker):
    def __init__(self, cli, logger):
        Worker.__init__(self, cli, logger)
        self.smu = SMUInterface(cli, "smu")
        self.add_instruments(self.smu)

        self.data = self.new_data({})

    def new_data(self, params: dict):
        return IVData(None, params, False)

    def get_param_dict_labels(self) -> list[str]:
        return ["iv_sweep"]

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        pd = P.ParamDict()

        bounds = self.smu.get_bounds()
        if bounds is None:
            self.logger.error("Could not get SMU bounds.")
            return None
        pd["start"] = P.FloatParam(
            0, bounds["voltage"][0], bounds["voltage"][1], doc="start voltage"
        )
        pd["stop"] = P.FloatParam(
            0.01, bounds["voltage"][0], bounds["voltage"][1], doc="stop voltage"
        )
        pd["num"] = P.IntParam(10, 2, 1000, doc="number of points")
        pd["delay"] = P.FloatParam(0.01, 0, 10, doc="delay after measurement")
        pd["sweeps"] = P.IntParam(1, 0, 10000, doc="number of sweeps")
        pd["nplc"] = P.FloatParam(10, 0.01, 10, doc="measurement time in nplc")
        pd["compliance"] = P.FloatParam(
            bounds["current"][1],
            bounds["current"][0],
            bounds["current"][1],
            doc="current compliance",
        )
        pd["logx"] = P.BoolParam(False, doc="set True for log-space sweep")
        return pd

    def start(self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is None or any((k not in params for k in ["start", "stop", "num", "delay"])):
            self.logger.error("Invalid params.")
            return False
        params = P.unwrap(params)

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        bounds = self.smu.get_bounds()
        if bounds is None:
            return self.fail_with_release("Could not get SMU bounds.")
        if not self.smu.configure_IV_sweep(
            params["start"],
            params["stop"],
            params["num"],
            params["delay"],
            params.get("nplc", 1.0),
            params.get("compliance", bounds["current"][1]),
            params.get("logx", False),
        ):
            return self.fail_with_release("Error configuring instruments.")

        if not self.smu.start():
            return self.fail_with_release("Error starting instruments.")

        self.data = self.new_data(params)
        self.logger.info("Started sweeper.")

        self.data.start()

        return True

    def append_sweep(self, line):
        line = np.array(line, ndmin=2).T
        if self.data.data is None:
            self.data.data = line
        else:
            self.data.data = np.append(self.data.data, line, axis=1)

    def work(self) -> bool:
        """Perform sweep once and append result.

        :returns: True on success (data should be published).

        """

        line = self.smu.get_data()
        if line is None:
            return False
        self.append_sweep(line)
        return True

    def is_finished(self) -> bool:
        if self.data.params is None or self.data.data is None:
            return False
        if self.data.params.get("sweeps", 0) <= 0:
            return False  # no sweep limit defined.
        return self.data.sweeps() >= self.data.params["sweeps"]

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.smu.stop() and self.smu.release()

        self.data.running = False
        if success:
            self.logger.info("Stopped sweeper.")
        else:
            self.logger.error("Error stopping sweeper.")
        return success

    def data_msg(self) -> IVData:
        return self.data


class IV(BasicMeasNode):
    CLIENT = IVClient
    DATA = IVData

    def __init__(self, gconf: dict, name, context=None):
        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.sweeper = Sweeper(self.cli, self.logger)
        self.io = IVIO(self.logger)

    def close_resources(self):
        if hasattr(self, "sweeper"):
            self.sweeper.stop()

    def change_state(self, msg: StateReq) -> Reply:
        if self.state == msg.state:
            return Reply(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            if not self.sweeper.stop():
                return Reply(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.sweeper.start(msg.params):
                return Reply(False, "Failed to start internal worker.", ret=self.state)

        self.state = msg.state
        return Reply(True)

    def get_param_dict_labels(self, msg: GetParamDictLabelsReq) -> Reply:
        return Reply(True, ret=self.sweeper.get_param_dict_labels())

    def get_param_dict(self, msg: GetParamDictReq) -> Reply:
        d = self.sweeper.get_param_dict(msg.label)
        if d is None:
            return Reply(False, "Failed to generate param dict")
        return Reply(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Reply:
        success = self.io.save_data(msg.file_name, self.sweeper.data_msg(), msg.note)
        if success:
            for tweaker_name, cli in self.tweaker_clis.items():
                success &= cli.save(msg.file_name, "__" + tweaker_name + "__")
        return Reply(success)

    def export_data(self, msg: ExportDataReq) -> Reply:
        success = self.io.export_data(
            msg.file_name, msg.data if msg.data else self.sweeper.data_msg(), msg.params
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
        self.sweeper.data = data
        return Reply(True, ret=data)

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        self.cli.wait("smu")
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        publish_data = self._work()
        self._check_finished()
        self._publish(publish_data)

    def _work(self) -> bool:
        if self.state == BinaryState.ACTIVE:
            return self.sweeper.work()
        return False

    def _publish(self, publish_data: bool):
        self.status_pub.publish(BinaryStatus(state=self.state))
        if publish_data:
            self.data_pub.publish(self.sweeper.data_msg())

    def _check_finished(self):
        if self.state == BinaryState.ACTIVE and self.sweeper.is_finished():
            self.change_state(StateReq(BinaryState.IDLE))
