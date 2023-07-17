#!/usr/bin/env python3

"""
Logic and instrument control part of Pulse ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from ..msgs.common_msgs import Resp, Request, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import GetParamDictNamesReq, GetParamDictReq
from ..msgs import podmr_msgs
from ..msgs.podmr_msgs import PODMRData, UpdatePlotParamsReq, ValidateReq, DiscardReq
from ..util.timer import IntervalTimer
from .common_meas import BasicMeasClient, BasicMeasNode
from .common_worker import DummyWorker, Switch
from .podmr_worker import Pulser, PODMRDataOperator
from .podmr_fitter import PODMRFitter
from .podmr_io import PODMRIO


class PODMRClient(BasicMeasClient):
    """Node Client for PODMR."""

    #: Message types for PODMR.
    M = podmr_msgs

    # override for annotation
    def get_data(self) -> PODMRData:
        return self._get_data()

    def get_buffer(self) -> Buffer[tuple[str, PODMRData]]:
        return self._get_buffer()

    def update_plot_params(self, params: dict) -> bool:
        resp = self.req.request(UpdatePlotParamsReq(params))
        return resp.success

    def validate(self, params: dict) -> bool:
        resp = self.req.request(ValidateReq(params))
        return resp.success

    def discard(self) -> bool:
        resp = self.req.request(DiscardReq())
        return resp.success


class PODMR(BasicMeasNode):
    CLIENT = PODMRClient
    DATA = PODMRData

    def __init__(self, gconf: dict, name, context=None):
        """Pulse ODMR measurement.

        :param pulser.freq: pulse generator frequency (default: 2.0E9)
        :type pulser.freq: float
        :param pulser.reduce_start_divisor: the divisor on start of reducing frequency (default: 2)
            reduce is done first by this value, and then repeated by 10.
        :type pulser.reduce_start_divisor: int
        :param pulser.split_fraction: fraction factor (F) to split the free period for MW phase
            modulation (default: 4). the period (T) is split into (T // F, T - T // F) and MW phase
            is switched at T // F. Thus, larger F results in "quicker start" of the phase
            modulation (depending on hardware, but its response may be a bit slow).
        :type pulser.split_fraction: int
        :param pulser.minimum_block_length: minimum block length in generated blocks
                                            (default: 1000)
        :type pulser.minimum_block_length: int
        :param pulser.block_base: block base granularity (default: 4)
        :type pulser.block_base: int

        """

        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.pulse_pub = self.add_pub(b"pulse")

        if "switch" in self.conf["target"]["servers"]:
            self.switch = Switch(self.cli, self.logger, "podmr")
        else:
            self.switch = DummyWorker()

        has_fg = "fg" in self.conf["target"]["servers"]
        self.worker = Pulser(self.cli, self.logger, has_fg, self.conf.get("pulser", {}))
        self.fitter = PODMRFitter(self.logger)
        self.io = PODMRIO(self.logger)
        self.buffer: Buffer[tuple[str, PODMRData]] = Buffer()
        self.op = PODMRDataOperator()
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
            if success:
                self.pub_timer = IntervalTimer(self._pub_interval)
            else:
                return Resp(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.switch.start():
                return Resp(False, "Failed to start switch.", ret=self.state)
            if not self.worker.start(msg.params):
                self.switch.stop()
                return Resp(False, "Failed to start worker.", ret=self.state)
            self.pub_timer = self.worker.timer.clone()

        self.state = msg.state
        return Resp(True)

    def update_plot_params(self, msg: UpdatePlotParamsReq) -> Resp:
        """Update the plot params."""

        success = self.worker.update_plot_params(msg.params)
        for data in self.buffer.data_list():
            if self.op.update_plot_params(data, msg.params):
                data.clear_fit_data()
                self.op.get_marker_indices(data)
                self.op.analyze(data)
        return Resp(success)

    def get_param_dict_names(self, msg: GetParamDictNamesReq) -> Resp:
        if msg.group == "fit":
            return Resp(True, ret=self.fitter.get_param_dict_names())
        else:
            return Resp(True, ret=self.worker.get_param_dict_names())

    def get_param_dict(self, msg: GetParamDictReq) -> Resp:
        if msg.group == "fit":
            d = self.fitter.get_param_dict(msg.name)
        else:
            d = self.worker.get_param_dict(msg.name)

        if d is None:
            return Resp(False, "Failed to generate param dict.")
        else:
            return Resp(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Resp:
        success = self.io.save_data(msg.file_name, self.worker.data_msg(), msg.params, msg.note)
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

    def discard(self, msg: DiscardReq) -> Resp:
        """Discard the data."""

        return Resp(self.worker.discard())

    def handle_req(self, msg: Request) -> Resp:
        if isinstance(msg, UpdatePlotParamsReq):
            return self.update_plot_params(msg)
        elif isinstance(msg, ValidateReq):
            return self.validate(msg)
        elif isinstance(msg, DiscardReq):
            return self.discard(msg)
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
