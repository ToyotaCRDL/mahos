#!/usr/bin/env python3

"""
Logic and instrument control part of ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.msgs.common_msgs import Request, Reply, StateReq, BinaryState, BinaryStatus
from mahos.msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from mahos.msgs.common_meas_msgs import Buffer
from mahos.msgs.param_msgs import GetParamDictReq, GetParamDictLabelsReq
from mahos.msgs.param_msgs import prefix_labels, remove_label_prefix
from mahos_dq.msgs import odmr_msgs
from mahos_dq.msgs.odmr_msgs import ODMRData, ValidateReq
from mahos.util.timer import IntervalTimer
from mahos.meas.common_meas import BasicMeasClient, BasicMeasNode
from mahos.meas.common_worker import DummyWorker, Switch, PulseGen_CW
from mahos_dq.meas.odmr_worker import Sweeper, SweeperOverlay
from mahos_dq.meas.odmr_fitter import ODMRFitter
from mahos_dq.meas.odmr_io import ODMRIO


class ODMRClient(BasicMeasClient):
    """Node Client for ODMR."""

    #: Message types for ODMR.
    M = odmr_msgs

    def validate(self, params: dict, label: str) -> tuple[bool, str, str]:
        """validate the ODMR measurement parameter.

        :returns: (success, error_message, pulse_summary)

        """

        rep = self.req.request(ValidateReq(params, label))
        return rep.success, rep.message, rep.ret


class ODMR(BasicMeasNode):
    """ODMR (Optically Detected Magnetic Resonance) measurement.

    This Node is for frequency-sweep, CW and Pulse ODMR measurements for defect qubits
    using non-imaging, single-pixel photodetectors.
    See :class:`PODMR <mahos_dq.meas.podmr.PODMR>` or :class:`SPODMR <mahos_dq.meas.spodmr.SPODMR>`
    for pulse ODMR measurements (Rabi, FID, SpinEcho, etc.) with fixed frequency.
    See :class:`IODMR <mahos_dq.meas.iodmr.IODMR>` for frequency-sweep, imaging CW ODMR
    with camera.

    There are two options for the worker (sweeper, measurement logic).
    See docs of sweeper below for sweeper parameters.

    - :class:`Sweeper <mahos_dq.meas.odmr_worker.Sweeper>` : fast logic with hardware trigger.
    - :class:`SweeperOverlay <mahos_dq.meas.odmr_worker.SweeperOverlay>` : interface to overlay.

    :param target.servers: InstrumentServer targets (instrument name, server full name).
        Required keys (overlay mode): ``sweeper``.
        Required keys (direct mode): ``sg``, ``pg``, and names in ``sweeper.pd_names``.
        In direct mode, if ``sweeper.pd_analog`` is True, the key named by
        ``sweeper.clock_name`` is also required.
        Optional keys: ``pg`` in overlay mode (for non-PG overlays) and switch keys listed in
        ``switch_names``.
    :type target.servers: dict[str, str]
    :param target.tweakers: The Tweaker targets (list of tweaker full name).
    :type target.tweakers: list[str]
    :param target.log: The LogBroker target (broker full name).
    :type target.log: str

    :param switch_names: Optional switch instrument names to route signal/optical paths.
    :type switch_names: list[str]
    :param switch_command: Switch command label passed to Switch worker.
    :type switch_command: str
    :param pub_interval_sec: Maximum interval between periodic status/data publications.
    :type pub_interval_sec: float

    :param fitter.dip: (default: True) True if ODMR shape is dip instead of peak.
    :type fitter.dip: bool
    :param fitter.n_guess: (default: 20) Number of data points in peak center guess.
    :type fitter.n_guess: int
    :param fitter.n_guess_bg: (default: 40) Number of histogram bins in background guess.
    :type fitter.n_guess_bg: int

    """

    CLIENT = ODMRClient
    DATA = ODMRData

    def __init__(self, gconf: dict, name, context=None):
        BasicMeasNode.__init__(self, gconf, name, context=context)

        self.pulse_pub = self.add_pub(b"pulse")

        _default_sw_names = ["switch"] if "switch" in self.conf["target"]["servers"] else []
        sw_names = self.conf.get("switch_names", _default_sw_names)
        if sw_names:
            self.switch = Switch(
                self.cli, self.logger, sw_names, self.conf.get("switch_command", "odmr")
            )
        else:
            self.switch = DummyWorker()

        self._direct = not (
            "sweeper" in self.conf["target"]["servers"]
            or "sweeper_name" in self.conf.get("sweeper", {})
        )
        if self._direct:
            self.worker = Sweeper(self.cli, self.logger, self.conf.get("sweeper", {}))
            self.pg = DummyWorker()
        else:
            self.worker = SweeperOverlay(self.cli, self.logger, self.conf.get("sweeper", {}))
            if (
                "pg" in self.conf["target"]["servers"]
                and self.worker._class_name != "ODMRSweeperPG"
            ):
                # PG is not used in worker, so use PulseGen_CW.
                self.pg = PulseGen_CW(self.cli, self.logger, channels=("laser", "mw"))
            else:
                self.pg = DummyWorker()
        self.fitter = ODMRFitter(self.logger, silent=False, conf=self.conf.get("fitter"))
        self.io = ODMRIO(self.logger)
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
            if not self.worker.start(msg.params, msg.label):
                self.pg.stop()
                self.switch.stop()
                return Reply(False, "Failed to start worker.", ret=self.state)

        self.state = msg.state
        # publish changed state immediately to prevent StateManager from missing the change
        self.status_pub.publish(BinaryStatus(state=self.state))
        return Reply(True)

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
        success = self.io.save_data(msg.file_name, self.worker.data_msg(), msg.note)
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
                self.buffer.append((msg.buffer_name, data))
            else:
                if self.state == BinaryState.ACTIVE:
                    return Reply(False, "Cannot load data when active.")
                self.worker.data = data
            return Reply(True, ret=data)

    def validate(self, msg: ValidateReq) -> Reply:
        """Validate the measurement params."""

        success, error, summary = self.worker.validate_params(msg.params, msg.label)
        return Reply(success, message=error, ret=summary)

    def handle_req(self, msg: Request) -> Reply:
        if isinstance(msg, ValidateReq):
            return self.validate(msg)
        else:
            return Reply(False, "Invalid message type")

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        if self._direct:
            for inst in ["sg", "pg"] + self.worker.pd_names:
                self.cli.wait(inst)
        else:
            self.cli.wait(self.worker.sweeper_name)
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
