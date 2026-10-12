#!/usr/bin/env python3

"""
Logic and instrument control part of Analog-PD Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from mahos.msgs.common_msgs import Reply, Request, StateReq, BinaryStatus, BinaryState
from mahos.msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from mahos.msgs.common_meas_msgs import Buffer
from mahos.msgs.param_msgs import GetParamDictLabelsReq, GetParamDictReq
from mahos.msgs.param_msgs import prefix_labels, remove_label_prefix
from mahos.util.timer import IntervalTimer
from mahos.meas.common_meas import BasicMeasClient, BasicMeasNode
from mahos.meas.common_worker import DummyWorker, Switch
from mahos_dq.msgs import apodmr_msgs
from mahos_dq.msgs.apodmr_msgs import (
    APODMRData,
    ValidateReq,
    GetTimingInfoReq,
    TimingInfo,
    UpdatePlotParamsReq,
)
from mahos_dq.meas.apodmr_worker import Pulser, AWGPulser, APODMRDataOperator
from mahos_dq.meas.apodmr_io import APODMRIO
from mahos_dq.meas.podmr_fitter import PODMRFitter


class APODMRClient(BasicMeasClient):
    """Node Client for APODMR."""

    M = apodmr_msgs

    def get_data(self) -> APODMRData:
        return self._get_data()

    def get_buffer(self) -> Buffer[tuple[str, APODMRData]]:
        return self._get_buffer()

    def update_plot_params(self, params: dict) -> bool:
        rep = self.req.request(UpdatePlotParamsReq(params))
        return rep.success

    def validate(self, params: dict, label: str) -> bool:
        rep = self.req.request(ValidateReq(params, label))
        return rep.success

    def get_timing_info(self, params: dict, label: str) -> TimingInfo | None:
        rep = self.req.request(GetTimingInfoReq(params, label))
        return rep.ret


class APODMR(BasicMeasNode):
    """Analog-PD pulse ODMR measurement node.

    Major difference from :class:`PODMR <mahos_dq.meas.podmr.PODMR>` is that the detector backend
    is an AnalogPD triggered before each laser pulse.

    There are two options for the worker (pulser, measurement logic).
    See docs of pulser below for pulser parameters.

    - :class:`Pulser <mahos_dq.meas.apodmr_worker.Pulser>` : pulser using SG and PG.
    - :class:`AWGPulser <mahos_dq.meas.apodmr_worker.AWGPulser>` : pulser using AWG.

    :param target.servers: InstrumentServer targets (instrument name, server full name).
        Required keys: names in ``pulser.pd_names`` and either ``awg`` or ``sg`` + ``pg``.
        Optional keys: the key named by ``pulser.clock_name`` (if pulser.pd_spectrum is False).
    :type target.servers: dict[str, str]
    :param target.tweakers: Tweaker targets saved alongside measurement data.
    :type target.tweakers: list[str]
    :param target.log: LogBroker target full name.
    :type target.log: str

    :param switch_names: Optional switch instrument names.
    :type switch_names: list[str]
    :param switch_command: Switch command label passed to Switch worker.
    :type switch_command: str
    :param pub_interval_sec: Maximum interval between periodic status/data publications.
    :type pub_interval_sec: float

    :param fitter.rabi.c: default value of param "c" (base line) in RabiFitter.
        You can set the bounds using "c_min" and "c_max" too.
    :type fitter.rabi.c: float
    :param fitter.rabi.A: default value of param "A" (amplitude) in RabiFitter.
        You can set the bounds using "A_min" and "A_max" too.
    :type fitter.rabi.A: float

    """

    CLIENT = APODMRClient
    DATA = APODMRData

    def __init__(self, gconf: dict, name, context=None):
        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.pulse_pub = self.add_pub(b"pulse")
        self.wave_pub = self.add_pub(b"wave")
        self._published_wave_ident = None

        _default_sw_names = ["switch"] if "switch" in self.conf["target"]["servers"] else []
        sw_names = self.conf.get("switch_names", _default_sw_names)
        if sw_names:
            self.switch = Switch(
                self.cli, self.logger, sw_names, self.conf.get("switch_command", "apodmr")
            )
        else:
            self.switch = DummyWorker()

        self._awg_worker = "awg" in self.conf["target"]["servers"]
        PulserClass = AWGPulser if self._awg_worker else Pulser
        self.worker = PulserClass(self.cli, self.logger, self.conf.get("pulser", {}))
        self.fitter = PODMRFitter(self.logger, conf=self.conf.get("fitter"))
        self.io = APODMRIO(self.logger)
        self.buffer: Buffer[tuple[str, APODMRData]] = Buffer()
        self.op = APODMRDataOperator()
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

        self.state = msg.state
        self.status_pub.publish(BinaryStatus(state=self.state))
        return Reply(True)

    def update_plot_params(self, msg: UpdatePlotParamsReq) -> Reply:
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
        return Reply(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Reply:
        if msg.params is not None and (msg.params.get("tmp") or msg.params.get("temp")):
            data = self.worker.data_msg().snapshot_for_save(finalize=True)
        else:
            data = self.worker.data_msg()
        success = self.io.save_data(msg.file_name, data, msg.params, msg.note)
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
        if msg.to_buffer:
            self.buffer.append((msg.buffer_name, data))
        else:
            if self.state == BinaryState.ACTIVE:
                return Reply(False, "Cannot load data when active.")
            self.worker.data = data
        return Reply(True, ret=data)

    def validate(self, msg: ValidateReq) -> Reply:
        return Reply(self.worker.validate_params(msg.params, msg.label))

    def get_timing_info(self, msg: GetTimingInfoReq) -> Reply:
        """Get timing info for given measurement params."""

        ret = self.worker.get_timing_info(msg.params, msg.label)
        success = ret is not None
        return Reply(success=success, ret=ret)

    def handle_req(self, msg: Request) -> Reply:
        if isinstance(msg, UpdatePlotParamsReq):
            return self.update_plot_params(msg)
        elif isinstance(msg, ValidateReq):
            return self.validate(msg)
        elif isinstance(msg, GetTimingInfoReq):
            return self.get_timing_info(msg)
        else:
            return Reply(False, "Invalid message type")

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        insts = ["awg"] if self._awg_worker else ["sg", "pg"]
        for inst in insts + self.worker.pd_names:
            self.cli.wait(inst)
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
        # publishing wave is one-shot because it can be huge
        wave = self.worker.wave_msg()
        if wave is not None and wave.ident != self._published_wave_ident:
            self.wave_pub.publish(wave)
            self._published_wave_ident = wave.ident

    def _check_finished(self) -> bool:
        if self.state == BinaryState.ACTIVE and self.worker.is_finished():
            self.change_state(StateReq(BinaryState.IDLE))
            return True
        return False
