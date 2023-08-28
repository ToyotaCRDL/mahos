#!/usr/bin/env python3

"""
Logic and instrument control part of Qdyne.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from ..msgs.common_msgs import Resp, Request, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import GetParamDictNamesReq, GetParamDictReq
from ..msgs import qdyne_msgs
from ..msgs.qdyne_msgs import QdyneData, ValidateReq, DiscardReq
from .qdyne_io import QdyneIO
from ..util.timer import IntervalTimer
from .common_meas import BasicMeasClient, BasicMeasNode
from .common_worker import DummyWorker, Switch
from .qdyne_worker import Pulser


class QdyneClient(BasicMeasClient):
    """Node Client for Qdyne."""

    #: Message types for Qdyne.
    M = qdyne_msgs

    # override for annotation
    def get_data(self) -> QdyneData:
        return self._get_data()

    def get_buffer(self) -> Buffer[tuple[str, QdyneData]]:
        return self._get_buffer()

    def validate(self, params: dict) -> Resp:
        return self.req.request(ValidateReq(params))

    def discard(self) -> bool:
        resp = self.req.request(DiscardReq())
        return resp.success


class Qdyne(BasicMeasNode):
    CLIENT = QdyneClient
    DATA = QdyneData

    def __init__(self, gconf: dict, name, context=None):
        """Pulse ODMR measurement with Qdyne protocol.

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
        :param pulser.start_delay: (sec.) delay time before starting PG output. (default: 0.0)
        :type pulser.start_delay: float

        :param pulser.raw_events_dir: (default: "") The directory to load RawEvents file.
        :type pulser.raw_events_dir: str
        :param pulser.remove_raw_events: (default: True) Remove RawEvents file after loading it.
        :type pulser.remove_raw_events: bool

        """

        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.pulse_pub = self.add_pub(b"pulse")

        if "switch" in self.conf["target"]["servers"]:
            self.switch = Switch(self.cli, self.logger, "podmr")
        else:
            self.switch = DummyWorker()

        has_fg = "fg" in self.conf["target"]["servers"]
        self.worker = Pulser(self.cli, self.logger, has_fg, self.conf.get("pulser", {}))
        self.io = QdyneIO(self.logger)
        self.buffer: Buffer[tuple[str, QdyneData]] = Buffer()
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
        self.logger.debug(f"Changing state from {self.state} to {msg.state}")

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

    def get_param_dict_names(self, msg: GetParamDictNamesReq) -> Resp:
        if msg.group == "fit":
            return Resp(False, "fit is not implemented.")
        else:
            return Resp(True, ret=self.worker.get_param_dict_names())

    def get_param_dict(self, msg: GetParamDictReq) -> Resp:
        if msg.group == "fit":
            return Resp(False, "fit is not implemented.")
        else:
            d = self.worker.get_param_dict(msg.name)

        if d is None:
            return Resp(False, "Failed to generate param dict.")
        else:
            return Resp(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Resp:
        success = self.io.save_data(msg.file_name, self.worker.data, msg.params, msg.note)
        return Resp(success)

    def export_data(self, msg: ExportDataReq) -> Resp:
        success = self.io.export_data(
            msg.file_name, msg.data if msg.data else self.worker.data, msg.params
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

        valid, blocks, freq, laser_timing, offsets = self.worker.validate_params(msg.params)
        return Resp(valid, ret=(blocks, freq, laser_timing, offsets))

    def discard(self, msg: DiscardReq) -> Resp:
        """Discard the data."""

        return Resp(self.worker.discard())

    def handle_req(self, msg: Request) -> Resp:
        if isinstance(msg, ValidateReq):
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

        # Publishing buffer is disabled because the buffer can be massive if containing raw_data.
        # TODO: consider eliminating raw_data in buffer to publish ?
        finished = self._check_finished()
        time_to_pub = self.pub_timer.check()
        self._publish(time_to_pub or finished, time_to_pub, False)

    def _work(self):
        if self.state == BinaryState.ACTIVE:
            self.worker.work()

    def _publish(self, publish_data: bool, publish_pulse: bool, publish_buffer: bool):
        self.status_pub.publish(BinaryStatus(state=self.state))
        if publish_data:
            if (data := self.worker.data_msg()) is not None:
                self.data_pub.publish(data)
        if publish_pulse:
            if (pulse := self.worker.pulse_msg()) is not None:
                self.pulse_pub.publish(pulse)
        if publish_buffer:
            self.buffer_pub.publish(self.buffer)

    def _check_finished(self) -> bool:
        if self.state == BinaryState.ACTIVE and self.worker.is_finished():
            self.change_state(StateReq(BinaryState.IDLE))
            return True
        return False
