#!/usr/bin/env python3

"""
Logic and instrument control part of Pulse ODMR with Slow detectors.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from ..msgs.common_msgs import Resp, Request, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import GetParamDictLabelsReq, GetParamDictReq
from ..msgs import spodmr_msgs
from ..msgs.spodmr_msgs import SPODMRData, UpdatePlotParamsReq, ValidateReq
from ..util.timer import IntervalTimer
from .tweaker import TweakerClient
from .common_meas import BasicMeasClient, BasicMeasNode
from .common_worker import DummyWorker, Switch
from .podmr_fitter import PODMRFitter
from .spodmr_worker import Pulser, DebugPulser, SPODMRDataOperator
from .spodmr_io import SPODMRIO


class SPODMRClient(BasicMeasClient):
    """Node Client for SPODMR."""

    #: Message types for SPODMR.
    M = spodmr_msgs

    # override for annotation
    def get_data(self) -> SPODMRData:
        return self._get_data()

    def get_buffer(self) -> Buffer[tuple[str, SPODMRData]]:
        return self._get_buffer()

    def update_plot_params(self, params: dict) -> bool:
        resp = self.req.request(UpdatePlotParamsReq(params))
        return resp.success

    def validate(self, params: dict) -> bool:
        resp = self.req.request(ValidateReq(params))
        return resp.success


class SPODMR(BasicMeasNode):
    CLIENT = SPODMRClient
    DATA = SPODMRData

    def __init__(self, gconf: dict, name, context=None):
        """Pulse ODMR measurement with Slow detectors.

        :param pulser.pd_trigger: DAQ terminal name for PD trigger.
        :type pulser.pd_trigger: str
        :param pulser.trigger_channel: (default: gate) PG channel name for PD trigger.
            default value may look a bit strange, but considering shared physical line
            with PD gate for ODMR (Analog PD mode).
        :type pulser.trigger_channel: str
        :param pulser.trigger_width: (default: 1e-6) pulse width for PD trigger
        :type pulser.trigger_width: float
        :param pulser.split_fraction: (default: 4) fraction factor (F) to split the free period
            for MW phase modulation. the period (T) is split into (T // F, T - T // F) and MW phase
            is switched at T // F. Thus, larger F results in "quicker start" of the phase
            modulation (depending on hardware, but its response may be a bit slow).
        :type pulser.split_fraction: int
        :param pulser.pg_freq: pulse generator frequency (has preset)
        :type pulser.pg_freq: float
        :param pulser.reduce_start_divisor: (has preset) the divisor on start of reducing frequency
            reduce is done first by this value, and then repeated by 10.
        :type pulser.reduce_start_divisor: int
        :param pulser.minimum_block_length: (has preset) minimum block length in generated blocks
        :type pulser.minimum_block_length: int
        :param pulser.block_base: (has preset) block base granularity of pulse generator.
        :type pulser.block_base: int
        :param pulser.nest_blockseq: (has preset, default: False) allow nested BlockSeq.
        :type pulser.nest_blockseq: bool

        """

        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.pulse_pub = self.add_pub(b"pulse")

        if "switch" in self.conf["target"]["servers"]:
            self.switch = Switch(self.cli, self.logger, "spodmr")
        else:
            self.switch = DummyWorker()
        if "tweaker" in self.conf["target"]:
            self.tweaker_cli = TweakerClient(
                gconf, self.conf["target"]["tweaker"], context=self.ctx, prefix=self.joined_name()
            )
            self.add_client(self.tweaker_cli)
        else:
            self.tweaker_cli = None

        if self.conf.get("debug", False):
            self.worker = DebugPulser(self.cli, self.logger, self.conf.get("pulser", {}))
        else:
            self.worker = Pulser(self.cli, self.logger, self.conf.get("pulser", {}))
        self.fitter = PODMRFitter(self.logger)
        self.io = SPODMRIO(self.logger)
        self.buffer: Buffer[tuple[str, SPODMRData]] = Buffer()
        self.op = SPODMRDataOperator()
        self._pub_interval = self.conf.get("pub_interval_sec", 1.0)
        self.pub_timer = IntervalTimer(self._pub_interval)

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

    def update_plot_params(self, msg: UpdatePlotParamsReq) -> Resp:
        """Update the plot params."""

        success = self.worker.update_plot_params(msg.params)
        for data in self.buffer.data_list():
            if self.op.update_plot_params(data, msg.params):
                data.clear_fit_data()
        return Resp(success)

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
        success = self.io.save_data(msg.file_name, self.worker.data_msg(), msg.params, msg.note)
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
        if isinstance(msg, UpdatePlotParamsReq):
            return self.update_plot_params(msg)
        elif isinstance(msg, ValidateReq):
            return self.validate(msg)
        else:
            return Resp(False, "Invalid message type")

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        self.cli.wait("pg")
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        self._work()
        finished = self._check_finished()
        time_to_pub = self.pub_timer.check()
        self._publish(time_to_pub or finished, time_to_pub)

    def _work(self):
        if self.state == BinaryState.ACTIVE:
            self.worker.work()

    def _publish(self, publish_data: bool, publish_other: bool):
        self.status_pub.publish(BinaryStatus(state=self.state))
        if publish_data:
            self.data_pub.publish(self.worker.data_msg())
        if publish_other:
            pulse = self.worker.pulse_msg()
            if pulse is not None:
                self.pulse_pub.publish(pulse)
            self.buffer_pub.publish(self.buffer)

    def _check_finished(self) -> bool:
        if self.state == BinaryState.ACTIVE and self.worker.is_finished():
            self.change_state(StateReq(BinaryState.IDLE))
            return True
        return False
