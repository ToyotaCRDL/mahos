#!/usr/bin/env python3

"""
Logic and instrument control part of HBT Interferometer.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from ..msgs.common_msgs import Request, Reply, StateReq, BinaryState, BinaryStatus
from ..msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import GetParamDictReq, GetParamDictLabelsReq
from ..msgs.param_msgs import prefix_labels, remove_label_prefix
from ..msgs import hbt_msgs
from ..msgs.hbt_msgs import HBTData, UpdatePlotParamsReq
from ..util.timer import IntervalTimer
from .tweaker import TweakerClient
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
        rep = self.req.request(UpdatePlotParamsReq(params))
        return rep.success


class HBT(BasicMeasNode):
    CLIENT = HBTClient
    DATA = HBTData

    def __init__(self, gconf: dict, name, context=None):
        """HBT measurement.

        :param listener.interval_sec: polling interval.
        :type listener.interval_sec: float
        :param listener.tdc_correlation: (has preset) enable correlation measurement mode at TDC.
        :type listener.tdc_correlation: bool
        :param listener.tdc_normalize: (has preset) enable normalization at TDC.
        :type listener.tdc_normalize: bool
        :param listener.tdc_channel: (default: 1) channel index for getting data from TDC.
        :type listener.tdc_channel: int
        :param listener.t0_ns: default value of t0 (time delay).
        :type listener.t0_ns: float
        :param listener.range_ns: default value of measurement range (time window).
        :type listener.range_ns: float

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
        if "tweaker" in self.conf["target"]:
            self.tweaker_cli = TweakerClient(
                gconf, self.conf["target"]["tweaker"], context=self.ctx, prefix=self.joined_name()
            )
            self.add_client(self.tweaker_cli)
        else:
            self.tweaker_cli = None

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

    def change_state(self, msg: StateReq) -> Reply:
        if self.state == msg.state:
            return Reply(True, "Already in that state")

        if msg.state == BinaryState.IDLE:
            success = self.switch.stop() and self.pg.stop() and self.worker.stop()
            if not success:
                return Reply(False, "Failed to stop internal worker.", ret=self.state)
        elif msg.state == BinaryState.ACTIVE:
            if not self.switch.start():
                return Reply(False, "Failed to start switch.", ret=self.state)
            if not self.pg.start():
                self.switch.stop()
                return Reply(False, "Failed to start pg.", ret=self.state)
            if not self.worker.start(msg.params):
                self.pg.stop()
                self.switch.stop()
                return Reply(False, "Failed to start worker.", ret=self.state)

        self.state = msg.state
        return Reply(True)

    def update_plot_params(self, msg: UpdatePlotParamsReq) -> Reply:
        """Update the plot params."""

        success = self.worker.update_plot_params(msg.params)
        for data in self.buffer.data_list():
            data.update_plot_params(msg.params)
            data.clear_fit_data()
        return Reply(success)

    def get_param_dict_labels(self, msg: GetParamDictLabelsReq) -> Reply:
        labels = prefix_labels("fit", self.fitter.get_param_dict_labels()) + ["hbt"]
        return Reply(True, ret=labels)

    def get_param_dict(self, msg: GetParamDictReq) -> Reply:
        is_fit, label = remove_label_prefix("fit", msg.label)
        if is_fit:
            d = self.fitter.get_param_dict(label)
        else:
            d = self.worker.get_param_dict()

        if d is None:
            return Reply(False, "Failed to generate param dict.")
        else:
            return Reply(True, ret=d)

    def save_data(self, msg: SaveDataReq) -> Reply:
        success = self.io.save_data(msg.file_name, self.worker.data_msg(), msg.note)
        if success and self.tweaker_cli is not None:
            success &= self.tweaker_cli.save(msg.file_name, "_inst_params")
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
            self.buffer.append((msg.file_name, data))
        else:
            if self.state == BinaryState.ACTIVE:
                return Reply(False, "Cannot load data when active.")
            self.worker.data = data
        return Reply(True, ret=data)

    def handle_req(self, msg: Request) -> Reply:
        if isinstance(msg, UpdatePlotParamsReq):
            return self.update_plot_params(msg)
        else:
            return Reply(False, "Invalid message type")

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
