#!/usr/bin/env python3

"""
Logic and instrument control part of Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from ..msgs.common_msgs import Reply, Request, StateReq, BinaryState
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import GetParamDictLabelsReq, GetParamDictReq
from ..msgs.param_msgs import prefix_labels, remove_label_prefix
from ..msgs import podmr_msgs
from ..msgs.podmr_msgs import PODMRStatus, PODMRData, UpdatePlotParamsReq, ValidateReq, DiscardReq
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
        rep = self.req.request(UpdatePlotParamsReq(params))
        return rep.success

    def validate(self, params: dict, label: str) -> bool:
        rep = self.req.request(ValidateReq(params, label))
        return rep.success

    def discard(self) -> bool:
        rep = self.req.request(DiscardReq())
        return rep.success


class PODMR(BasicMeasNode):
    CLIENT = PODMRClient
    DATA = PODMRData

    def __init__(self, gconf: dict, name, context=None):
        """Pulse ODMR measurement.

        Default Worker (Pulser) implements Pulse ODMR using
        a PG as timing source, and SGs as MW sources.

        :param pulser.quick_resume: default value of quick_resume.
            If True, it skips instrument configurations on resume.
        :type pulser.quick_resume: bool
        :param pulser.mw_modes: mw phase control modes for each channel.
            0 is 4-phase control using IQ modulation at SG and a switch.
            1 is 2-phase control using external 90-deg splitter and two switches.
            2 is arbitral phase control using IQ modulation at SG
            (Analog output (AWG) is required for PG).
        :type pulser.mw_modes: tuple[int]
        :param pulser.iq_amplitude: (only for mw_mode 2) amplitude of analog IQ signal in V.
        :type pulser.iq_amplitude: float
        :param pulser.split_fraction: (default: 4) fraction factor (F) to split the free period
            for MW phase modulation. the period (T) is split into (T // F, T - T // F) and MW phase
            is switched at T // F. Thus, larger F results in "quicker start" of the phase
            modulation (depending on hardware, but its response may be a bit slow).
        :type pulser.split_fraction: int
        :param pulser.pg_freq: (has preset) pulse generator frequency
        :type pulser.pg_freq: float
        :param pulser.reduce_start_divisor: (has preset) the divisor on start of reducing frequency
            reduce is done first by this value, and then repeated by 10.
        :type pulser.reduce_start_divisor: int
        :param pulser.minimum_block_length: (has preset) minimum block length in generated blocks
        :type pulser.minimum_block_length: int
        :param pulser.block_base: (has preset) block base granularity of pulse generator.
        :type pulser.block_base: int
        :param pulser.divide_block: (has preset) Default value of divide_block.
        :type pulser.divide_block: bool
        :param pulser.sg_freq: default value of sg frequency
        :type pulser.sg_freq: float
        :param pulser.channel_remap: mapping to fix default channel names.
        :type pulser.channel_remap: dict[str | int, str | int]

        :param fitter.rabi.c: default value of param "c" (base line) in RabiFitter.
            You can set the bounds using "c_min" and "c_max" too.
        :type fitter.rabi.c: float
        :param fitter.rabi.A: default value of param "A" (amplitude) in RabiFitter.
            You can set the bounds using "A_min" and "A_max" too.
        :type fitter.rabi.A: float

        """

        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.pulse_pub = self.add_pub(b"pulse")

        _default_sw_names = ["switch"] if "switch" in self.conf["target"]["servers"] else []
        sw_names = self.conf.get("switch_names", _default_sw_names)
        if sw_names:
            self.switch = Switch(
                self.cli, self.logger, sw_names, self.conf.get("switch_command", "podmr")
            )
        else:
            self.switch = DummyWorker()

        self.worker = Pulser(self.cli, self.logger, self.conf.get("pulser", {}))
        self.fitter = PODMRFitter(self.logger, conf=self.conf.get("fitter"))
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

    def change_state(self, msg: StateReq) -> Reply:
        if self.state == msg.state:
            return Reply(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            success = self.switch.stop() and self.worker.stop()
            if success:
                self.pub_timer = IntervalTimer(self._pub_interval)
            else:
                return Reply(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.switch.start():
                return Reply(False, "Failed to start switch.", ret=self.state)
            if not self.worker.start(msg.params, msg.label):
                self.switch.stop()
                return Reply(False, "Failed to start worker.", ret=self.state)
            self.pub_timer = self.worker.timer.clone()

        self.state = msg.state
        # publish changed state immediately to prevent StateManager from missing the change
        self.status_pub.publish(PODMRStatus(state=self.state, pg_freq=self.worker.conf["pg_freq"]))
        return Reply(True)

    def update_plot_params(self, msg: UpdatePlotParamsReq) -> Reply:
        """Update the plot params."""

        success = self.worker.update_plot_params(msg.params)
        for data in self.buffer.data_list():
            if self.op.update_plot_params(data, msg.params):
                data.remove_fit_data()
                self.op.get_marker_indices(data)
                self.op.analyze(data)
        return Reply(success)

    def get_param_dict_labels(self, msg: GetParamDictLabelsReq) -> Reply:
        labels = (
            prefix_labels("fit", self.fitter.get_param_dict_labels())
            + self.worker.get_param_dict_labels()
        )
        return Reply(True, ret=labels)

    def get_param_dict(self, msg: GetParamDictReq) -> Reply:
        is_fit, label = remove_label_prefix("fit", msg.label)
        if is_fit:
            d = self.fitter.get_param_dict(label)
        else:
            d = self.worker.get_param_dict(label)

        if d is None:
            return Reply(False, "Failed to generate param dict.")
        else:
            return Reply(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Reply:
        success = self.io.save_data(msg.file_name, self.worker.data_msg(), msg.params, msg.note)
        if success:
            for tweaker_name, cli in self.tweaker_clis.items():
                success &= cli.save(msg.file_name, "__" + tweaker_name + "__")
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
        else:
            if msg.to_buffer:
                self.buffer.append((msg.file_name, data))
            else:
                if self.state == BinaryState.ACTIVE:
                    return Reply(False, "Cannot load data when active.")
                self.worker.data = data
            return Reply(True, ret=data)

    def validate(self, msg: ValidateReq) -> Reply:
        """Validate the measurement params."""

        return Reply(self.worker.validate_params(msg.params, msg.label))

    def discard(self, msg: DiscardReq) -> Reply:
        """Discard the data."""

        return Reply(self.worker.discard())

    def handle_req(self, msg: Request) -> Reply:
        if isinstance(msg, UpdatePlotParamsReq):
            return self.update_plot_params(msg)
        elif isinstance(msg, ValidateReq):
            return self.validate(msg)
        elif isinstance(msg, DiscardReq):
            return self.discard(msg)
        else:
            return Reply(False, "Invalid message type")

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
        self.status_pub.publish(PODMRStatus(state=self.state, pg_freq=self.worker.conf["pg_freq"]))
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
