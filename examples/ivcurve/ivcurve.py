#!/usr/bin/env python3

from __future__ import annotations
import time

import numpy as np
from numpy.typing import NDArray

from mahos.meas.common_meas import BasicMeasClient, BasicMeasNode
from mahos.meas.common_worker import Worker
from mahos.msgs.common_msgs import Resp, StateReq, BinaryState, BinaryStatus
import mahos.msgs.param_msgs as P
from mahos.util.typing import NodeName

import ivcurve_msgs
from ivcurve_msgs import IVCurveData

from instruments import VoltageSourceInterface, MultimeterInterface


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
        self.source = VoltageSourceInterface(cli, "source")
        self.meter = MultimeterInterface(cli, "meter")
        self.add_instruments(self.source, self.meter)

        self.data = self.new_data({})

    def new_data(self, params: dict):
        return IVCurveData(None, params, False)

    def start(self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is None or any((k not in params for k in ["start", "stop", "num"])):
            self.logger.error("Invalid params.")
            return False
        params = P.unwrap(params)

        self._voltages = np.linspace(params["start"], params["stop"], params["num"])

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not self.meter.configure_current_meas(params.get("range", 0), params.get("navg", 1)):
            return self.fail_with_release("Error configuring instruments.")

        if not self.source.start():
            return self.fail_with_release("Error starting instruments.")

        self.data = self.new_data(params)
        self.logger.info("Started sweeper.")

        self.data.start()

        return True

    def sweep_once(self) -> NDArray | None:
        currents = []
        for v in self._voltages:
            self.source.set_voltage(v)
            time.sleep(self.data.params.get("delay_sec", 0.0))
            i = self.meter.get_meas(v)
            if i is None:
                self.logger.error("Got invalid current.")
                return None
            currents.append(i)
        return np.array(currents)

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

        line = self.sweep_once()
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

        success = (
            self.source.set_voltage(0.0)
            and self.source.stop()
            and self.source.release()
            and self.meter.release()
        )

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

    def change_state(self, msg: StateReq) -> Resp:
        if self.state == msg.state:
            return Resp(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            if not self.sweeper.stop():
                return Resp(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.sweeper.start(msg.params):
                return Resp(False, "Failed to start internal worker.", ret=self.state)

        self.state = msg.state
        return Resp(True)

    def get_param_dict(self, msg: P.GetParamDictReq) -> Resp:
        params = P.ParamDict(
            start=P.FloatParam(0.0, -10.0, 10.0),
            stop=P.FloatParam(5.0, -10.0, 10.0),
            num=P.IntParam(51, 2, 1001),
            sweeps=P.IntParam(0, 0, 100000),
            delay_sec=P.FloatParam(0.0, 0.0, 0.5),
        )
        return Resp(True, ret=params)

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        self.cli.wait("source")
        self.cli.wait("meter")
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
