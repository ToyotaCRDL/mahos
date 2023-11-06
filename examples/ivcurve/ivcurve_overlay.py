#!/usr/bin/env python3

from __future__ import annotations
import numpy as np

from mahos.meas.common_meas import BasicMeasClient, BasicMeasNode
from mahos.meas.common_worker import Worker
from mahos.msgs.common_msgs import Reply, StateReq, BinaryState, BinaryStatus
import mahos.msgs.param_msgs as P
from mahos.util.typing import NodeName

import ivcurve_msgs
from ivcurve_msgs import IVCurveData

from overlays import IVSweeperInterface


class IVCurveClient(BasicMeasClient):
    """Node Client for IVCurve."""

    #: Message types for IVCurve.
    M = ivcurve_msgs

    # Override for type hint.
    def get_data(self) -> IVCurveData:
        return self._get_data()


class Sweeper(Worker):
    def __init__(self, cli, logger):
        Worker.__init__(self, cli, logger)
        self.sweeper = IVSweeperInterface(cli, "sweeper")
        self.add_instrument(self.sweeper)

        self.data = self.new_data({})

    def new_data(self, params: dict):
        return IVCurveData(None, params, False)

    def start(self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is None or any((k not in params for k in ["start", "stop", "num"])):
            self.logger.error("Invalid params.")
            return False
        params = P.unwrap(params)

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not self.sweeper.configure(params):
            return self.fail_with_release("Error configuring instruments.")

        if not self.sweeper.start():
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

        line = self.sweeper.sweep_once()
        if line is None:
            return False
        self.append_sweep(line)
        return True

    def is_finished(self) -> bool:
        if self.data.params is None or self.data.data is None:
            return False
        if self.data.params.get("sweeps", 0) <= 0:
            return False  # no sweep limit defined.
        return self.data.data.shape[1] >= self.data.params["sweeps"]

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.sweeper.stop() and self.sweeper.release()

        self.data.running = False
        if success:
            self.logger.info("Stopped sweeper.")
        else:
            self.logger.error("Error stopping sweeper.")
        return success

    def data_msg(self) -> IVCurveData:
        return self.data


class IVCurve(BasicMeasNode):
    CLIENT = IVCurveClient

    def __init__(self, gconf: dict, name: NodeName, context=None):
        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.sweeper = Sweeper(self.cli, self.logger)

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

    def get_param_dict(self, msg: P.GetParamDictReq) -> Reply:
        params = P.ParamDict(
            start=P.FloatParam(0.0, -10.0, 10.0),
            stop=P.FloatParam(5.0, -10.0, 10.0),
            num=P.IntParam(51, 2, 1001),
            sweeps=P.IntParam(0, 0, 100000),
            delay_sec=P.FloatParam(0.0, 0.0, 0.5),
        )
        return Reply(True, ret=params)

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        self.cli.wait("sweeper")
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
