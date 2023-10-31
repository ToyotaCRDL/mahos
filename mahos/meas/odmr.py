#!/usr/bin/env python3

"""
Logic and instrument control part of ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from ..msgs.common_msgs import Request, Resp, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import GetParamDictReq, GetParamDictLabelsReq
from ..msgs import odmr_msgs
from ..msgs.odmr_msgs import ODMRData, ValidateReq
from ..util.timer import IntervalTimer
from .tweaker import TweakerClient
from .common_meas import BasicMeasClient, BasicMeasNode
from .common_worker import DummyWorker, Switch
from .odmr_worker import Sweeper
from .odmr_fitter import ODMRFitter
from .odmr_io import ODMRIO


class ODMRClient(BasicMeasClient):
    """Node Client for ODMR."""

    #: Message types for ODMR.
    M = odmr_msgs

    def validate(self, params: dict) -> bool:
        resp = self.req.request(ValidateReq(params))
        return resp.success


class ODMR(BasicMeasNode):
    CLIENT = ODMRClient
    DATA = ODMRData

    def __init__(self, gconf: dict, name, context=None):
        """ODMR Sweep measurement.

        :param sweeper.pd_clock: DAQ terminal name for PD's clock (gate)
        :type sweeper.pd_clock: str
        :param sweeper.pd_names: (default: ["pd0", "pd1"]) PD names in target.servers.
        :type sweeper.pd_names: list[str]
        :param sweeper.pd_analog: (default: False) set True if PD is AnalogIn-based.
        :type sweeper.pd_analog: bool

        :param sweeper.start_delay: (default: 0.0) delay time (sec.) before starting SG/PG output.
        :type sweeper.start_delay: float
        :param sweeper.drop_first: (default: 0) drop first N freq points.
        :type sweeper.drop_first: int
        :param sweeper.sg_first: (default: False) if True, turn on SG first and PG second.
        :type sweeper.sg_first: bool

        :param pulser.pg_freq_cw: (has preset) PG frequency for CW mode.
        :type pulser.pg_freq_cw: float
        :param pulser.pg_freq_pulse: (has preset) PG frequency for Pulse mode.
        :type pulser.pg_freq_pulse: float
        :param pulser.minimum_block_length: (has preset) minimum block length in generated blocks
        :type pulser.minimum_block_length: int
        :param sweeper.block_base: (has preset) block base granularity of pulse generator.
        :type sweeper.block_base: int

        :param sweeper.start: (default param) start frequency in Hz.
        :type sweeper.start: float
        :param sweeper.stop: (default param) stop frequency in Hz.
        :type sweeper.stop: float
        :param sweeper.num: (default param) number of frequency points.
        :type sweeper.num: int
        :param sweeper.power: (default param) SG output power in dBm.
        :type sweeper.power: float
        :param sweeper.time_window: (default param) time window for cw mode.
        :type sweeper.time_window: float
        :param sweeper.pd_rate: (default param) analog PD sampling rate.
        :type sweeper.pd_rate: float
        :param sweeper.sg_modulation: (default param) enable external IQ modulation for SG.
        :type sweeper.sg_modulation: bool

        """

        BasicMeasNode.__init__(self, gconf, name, context=context)

        if "switch" in self.conf["target"]["servers"]:
            self.switch = Switch(self.cli, self.logger, "odmr")
        else:
            self.switch = DummyWorker()
        if "tweaker" in self.conf["target"]:
            self.tweaker_cli = TweakerClient(
                gconf, self.conf["target"]["tweaker"], context=self.ctx, prefix=self.joined_name()
            )
            self.add_client(self.tweaker_cli)
        else:
            self.tweaker_cli = None

        self.worker = Sweeper(self.cli, self.logger, self.conf.get("sweeper", {}))
        self.fitter = ODMRFitter(self.logger)
        self.io = ODMRIO(self.logger)
        self.buffer = Buffer()
        self.pub_timer = IntervalTimer(self.conf.get("pub_interval_sec", 0.5))

    def close_resources(self):
        if hasattr(self, "switch"):
            self.switch.stop()
        if hasattr(self, "worker"):
            self.worker.stop()

    def change_state(self, msg: StateReq) -> Resp:
        if self.state == msg.state:
            return Resp(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            success = self.switch.stop() and self.worker.stop()
            if not success:
                return Resp(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.switch.start():
                return Resp(False, "Failed to start switch.", ret=self.state)
            if not self.worker.start(msg.params):
                self.switch.stop()
                return Resp(False, "Failed to start worker.", ret=self.state)

        self.state = msg.state
        return Resp(True)

    def get_param_dict_labels(self, msg: GetParamDictLabelsReq) -> Resp:
        if msg.group == "fit":
            return Resp(True, ret=self.fitter.get_param_dict_labels())
        else:
            return Resp(True, ret=self.worker.get_param_dict_labels())

    def get_param_dict(self, msg: GetParamDictReq) -> Resp:
        if msg.group == "fit":
            d = self.fitter.get_param_dict(msg.label)
        else:
            d = self.worker.get_param_dict(msg.label)

        if d is None:
            return Resp(False, "Failed to generate param dict.")
        else:
            return Resp(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Resp:
        success = self.io.save_data(msg.file_name, self.worker.data_msg(), msg.note)
        if success and self.tweaker_cli is not None:
            success &= self.tweaker_cli.save(msg.file_name, "_inst_params")
        return Resp(success)

    def export_data(self, msg: ExportDataReq) -> Resp:
        success = self.io.export_data(
            msg.file_name, msg.data if msg.data else self.worker.data_msg(), msg.params
        )
        return Resp(success)

    def load_data(self, msg: LoadDataReq) -> Resp:
        data = self.io.load_data(msg.file_name)
        if data is None:
            return Resp(False)
        else:
            if msg.to_buffer:
                self.buffer.append((msg.file_name, data))
            else:
                if self.state == BinaryState.ACTIVE:
                    return Resp(False, "Cannot load data when active.")
                self.worker.data = data
            return Resp(True, ret=data)

    def validate(self, msg: ValidateReq) -> Resp:
        """Validate the measurement params."""

        return Resp(self.worker.validate_params(msg.params))

    def handle_req(self, msg: Request) -> Resp:
        if isinstance(msg, ValidateReq):
            return self.validate(msg)
        else:
            return Resp(False, "Invalid message type")

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        for inst in ["sg", "pg"] + self.worker.pd_names:
            self.cli.wait(inst)
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        self._work()
        finished = self._check_finished()
        time_to_pub = self.pub_timer.check()
        self._publish((self.state == BinaryState.ACTIVE) or finished or time_to_pub, time_to_pub)

    def _work(self):
        if self.state == BinaryState.ACTIVE:
            self.worker.work()

    def _publish(self, publish_data: bool, publish_buffer: bool):
        self.status_pub.publish(BinaryStatus(state=self.state))
        if publish_data:
            self.data_pub.publish(self.worker.data_msg())
        if publish_buffer:
            self.buffer_pub.publish(self.buffer)

    def _check_finished(self) -> bool:
        if self.state == BinaryState.ACTIVE and self.worker.is_finished():
            self.change_state(StateReq(BinaryState.IDLE))
            return True
        return False
