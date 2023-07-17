#!/usr/bin/env python3

"""
Logic and instrument control part of HBT Interferometer.

.. This file is a part of MAHOS project.

"""

from ..msgs.common_msgs import Request, Resp, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import GetParamDictReq, GetParamDictNamesReq
from ..msgs import hbt_msgs
from ..msgs.hbt_msgs import HBTData, UpdatePlotParamsReq
from ..util.timer import IntervalTimer
from .common_meas import BasicMeasClient, BasicMeasNode
from .common_worker import DummyWorker, PulseGen_CW, Switch
from .hbt_worker import Listener
from .hbt_fitter import HBTFitter
from .hbt_io import HBTIO


class HBTClient(BasicMeasClient):
    """Node Client for HBT."""

    #: Message types for HBT.
    M = hbt_msgs

    def update_plot_params(self, params: dict) -> bool:
        resp = self.req.request(UpdatePlotParamsReq(params))
        return resp.success


class HBT(BasicMeasNode):
    CLIENT = HBTClient
    DATA = HBTData

    def __init__(self, gconf: dict, name, context=None):
        """HBT measurement.

        :param listener.interval_sec: polling interval.
        :type listener.interval_sec: float
        :param listener.default_t0_ns: default value of t0 (time delay).
        :type listener.default_t0_ns: float

        """

        BasicMeasNode.__init__(self, gconf, name, context=context)

        if "switch" in self.conf["target"]["servers"]:
            self.switch = Switch(self.cli, self.logger, "hbt")
        else:
            self.switch = DummyWorker()
        if "pg" in self.conf["target"]["servers"]:
            self.pg = PulseGen_CW(self.cli, self.logger)
        else:
            self.pg = DummyWorker()

        self.worker = Listener(self.cli, self.logger, self.conf.get("listener", {}))
        self.fitter = HBTFitter(self.logger)
        self.io = HBTIO(self.logger)
        self.buffer = Buffer()
        self.pub_timer = IntervalTimer(self.conf.get("pub_interval_sec", 0.5))

    def close_resources(self):
        if hasattr(self, "switch"):
            self.switch.stop()
        if hasattr(self, "pg"):
            self.pg.stop()
        if hasattr(self, "worker"):
            self.worker.stop()

    def change_state(self, msg: StateReq) -> Resp:
        if self.state == msg.state:
            return Resp(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            success = self.switch.stop() and self.pg.stop() and self.worker.stop()
            if not success:
                return Resp(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.switch.start():
                return Resp(False, "Failed to start switch.", ret=self.state)
            if not self.pg.start():
                self.switch.stop()
                return Resp(False, "Failed to start pg.", ret=self.state)
            if not self.worker.start(msg.params):
                self.pg.stop()
                self.switch.stop()
                return Resp(False, "Failed to start worker.", ret=self.state)

        self.state = msg.state
        return Resp(True)

    def update_plot_params(self, msg: UpdatePlotParamsReq) -> Resp:
        """Update the plot params."""

        success = self.worker.update_plot_params(msg.params)
        for data in self.buffer.data_list():
            data.update_plot_params(msg.params)
            data.clear_fit_data()
        return Resp(success)

    def get_param_dict_names(self, msg: GetParamDictNamesReq) -> Resp:
        if msg.group == "fit":
            return Resp(True, ret=self.fitter.get_param_dict_names())
        else:
            return Resp(True, ret=["hbt"])

    def get_param_dict(self, msg: GetParamDictReq) -> Resp:
        if msg.group == "fit":
            d = self.fitter.get_param_dict(msg.name)
        else:
            d = self.worker.get_param_dict()

        if d is None:
            return Resp(False, "Failed to generate param dict.")
        else:
            return Resp(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Resp:
        success = self.io.save_data(msg.file_name, self.worker.data_msg(), msg.note)
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

        if msg.to_buffer:
            self.buffer.append((msg.file_name, data))
        else:
            if self.state == BinaryState.ACTIVE:
                return Resp(False, "Cannot load data when active.")
            self.worker.data = data
        return Resp(True, ret=data)

    def handle_req(self, msg: Request) -> Resp:
        if isinstance(msg, UpdatePlotParamsReq):
            return self.update_plot_params(msg)
        else:
            return Resp(False, "Invalid message type")

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        self.cli.wait("tdc")
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        updated = self._work()
        time_to_pub = self.pub_timer.check()
        self._publish(updated or time_to_pub, time_to_pub)

    def _work(self) -> bool:
        if self.state == BinaryState.ACTIVE:
            return self.worker.work()
        return False

    def _publish(self, publish_data: bool, publish_buffer: bool):
        self.status_pub.publish(BinaryStatus(state=self.state))
        if publish_data:
            self.data_pub.publish(self.worker.data_msg())
        if publish_buffer:
            self.buffer_pub.publish(self.buffer)
