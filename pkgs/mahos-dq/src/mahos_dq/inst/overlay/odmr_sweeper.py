#!/usr/bin/env python3

"""
InstrumentOverlay for sweeping ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import time
import threading
import math

import numpy as np

from mahos.inst.overlay.overlay import InstrumentOverlay
from mahos.msgs import param_msgs as P
from mahos.msgs.inst.pg_msgs import TriggerType
from mahos.util.queue import RollingQueue
from mahos.util.conf import ConfAccessorMixin, PresetLoader
from mahos.util.param import ParamAccessor, ParamError
from mahos_dq.meas.odmr_pg import ODMRPGMixin
from mahos_dq.meas.odmr_sg import configure_modulation
from mahos_dq.meas.odmr_pd import (
    configure_analog_pds,
    configure_apds,
    configure_trace_pds,
    make_pd_buffer_size_coeff_param,
    make_pd_param_dict,
    reduce_traces,
    sum_pd_blocks,
    sum_pd_channels,
)


class ODMRSweeperCommandBase(InstrumentOverlay):
    """ODMRSweeperCommandBase provides primitive operations for ODMR sweep.

    This class performs the sweep by issuing SG / PD commands every step.
    Thus, sweep speed will not be very good.

    :param sg: The reference to SG Instrument.
    :param pd: The reference to PD Instrument.
    :param queue_size: (default: 8) Size of queue of scanned line data.
    :type queue_size: int

    """

    def __init__(self, name, conf, prefix=None):
        InstrumentOverlay.__init__(self, name, conf=conf, prefix=prefix)
        self.sg = self.conf.get("sg")
        self.pd = self.conf.get("pd")
        self.add_instruments(self.sg, self.pd)
        self._pd_analog = self.conf.get("pd_analog", False)

        self._queue_size = self.conf.get("queue_size", 8)
        self._queue = RollingQueue(self._queue_size)
        self._stop_ev = self._thread = None
        self.running = False
        self.pulse_pattern = None

    def _set_attrs(self, params):
        self.start_f, self.stop_f = params["start"], params["stop"]
        self.freqs = np.linspace(self.start_f, self.stop_f, params["num"])
        self.power = params["power"]
        self.delay = params["delay"]
        self.background = params.get("background", False)
        self.bg_delay = params.get("background_delay", 0.0)
        self._continue_mw = params.get("continue_mw", False)

    def get_line(self):
        return self._queue.pop_block()

    def sweep_loop(self, ev: threading.Event):
        while True:
            line = []
            for f in self.freqs:
                self.sg.set_freq_CW(f)
                time.sleep(self.delay)
                res = self.get_pd_data()
                if ev.is_set():
                    self.logger.info("Quitting sweep loop.")
                    return
                line.append(res)

                if self.background:
                    self.sg.set_output(False, silent=True)
                    time.sleep(self.bg_delay)
                    res = self.get_pd_data()
                    if ev.is_set():
                        self.logger.info("Quitting sweep loop.")
                        return
                    line.append(res)
                    self.sg.set_output(True, silent=True)

            self._queue.append(np.array(line))

    def configure_pd(self):
        raise NotImplementedError("configure_pd is not implemented.")

    def get_pd_data(self):
        raise NotImplementedError("get_pd_data is not implemented.")

    def get_pd_param_dict(self) -> P.ParamDict[str, P.PDValue] | None:
        raise NotImplementedError("get_pd_param_dict is not implemented.")

    def get_capability(self) -> dict[str, bool]:
        """Return detector and pulse-generation capabilities."""

        return {"pd_analog": self._pd_analog, "pd_trace": False, "pd_chop": False}

    # Standard API

    def get_param_dict_labels(self) -> list[str]:
        return ["pd"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        if label == "pd":
            return self.get_pd_param_dict()

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "pulse":
            return self.fail_with("label='pulse' is not supported.")
        if not self.check_required_params(params, ("start", "stop", "num", "power", "delay")):
            return False

        self._set_attrs(params)
        self.params = params
        self._queue = RollingQueue(self._queue_size)

        if not self.configure_pd():
            return self.fail_with("failed to configure PD.")

        success = self.sg.configure_cw(self.start_f, self.power)
        success &= configure_modulation(self.sg, label, params.get("mod", {}))
        if not success:
            return self.fail_with("failed to configure SG.")

        return True

    def start(self, label: str = "") -> bool:
        if self.running:
            self.logger.warn("start() is called while running.")
            return True

        if not self.sg.set_output(True):
            return self.fail_with("Failed to start sg.")
        if not self.pd.start():
            return self.fail_with("Failed to start pd.")

        self._stop_ev = threading.Event()
        self._thread = threading.Thread(target=self.sweep_loop, args=(self._stop_ev,))
        self._thread.start()

        self.running = True

        return True

    def stop(self, label: str = "") -> bool:
        if not self.running:
            return True
        self.running = False

        self.logger.info("Stopping sweeper.")

        self._stop_ev.set()
        self._thread.join()

        if self._continue_mw:
            self.logger.warn("Skipping to turn off MW output")
            success = True
        else:
            success = self.sg.set_output(False)
        success &= self.pd.stop()

        if not success:
            return self.fail_with("failed to stop SG or PD.")

        return True

    def get(self, key: str, args=None, label: str = ""):
        if key == "line":
            return self.get_line()
        elif key == "validate":
            return True, "", ""
        elif key == "bounds":
            return self.sg.get_bounds()
        elif key == "unit":
            return self.pd.get("unit")
        elif key == "pulse_pattern":
            return self.pulse_pattern
        elif key == "capability":
            return self.get_capability()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class ODMRSweeperCommandAnalogPD(ODMRSweeperCommandBase):
    """ODMRSweeperCommand for AnalogPD (Photo Diode read with NI-DAQ AnalogIn)."""

    def get_pd_param_dict(self) -> P.ParamDict[str, P.PDValue] | None:
        d = P.ParamDict()
        d["bounds"] = [
            P.FloatParam(-10.0, -10.0, 10.0, unit="V", doc="lower bound of expected voltage"),
            P.FloatParam(10.0, -10.0, 10.0, unit="V", doc="upper bound of expected voltage"),
        ]
        return d

    def configure_pd(self):
        t = self.params["timing"]["time_window"]
        # TODO we are not sure if AnalogIn samples at max rate for on demand readout.
        rate = self.pd.get_max_rate()
        self._oversample = int(round(t * rate))
        self.logger.info(f"AnalogPD oversample: {self._oversample}")

        return self.pd.configure_on_demand(self.params["pd"])

    def get_pd_data(self):
        return self.pd.read_on_demand(self._oversample)


class ODMRSweeperCommandAnalogPDMM(ODMRSweeperCommandBase):
    """ODMRSweeperCommand for AnalogPDMM (Photo Diode read with DMM)."""

    def get_pd_param_dict(self) -> P.ParamDict[str, P.PDValue] | None:
        return self.pd.get_param_dict("pd")

    def configure_pd(self):
        return self.pd.configure(self.params["pd"])

    def get_pd_data(self):
        return self.pd.get_data()


class ODMRSweeperPG(InstrumentOverlay, ODMRPGMixin, ConfAccessorMixin):
    """ODMRSweeperPG provides primitive operations for ODMR sweep.

    This class changes the SG frequency and software-triggers the PG for each point. With
    ``pd_trace`` enabled, its ``pulse`` method averages laser-resolved AnalogPD traces and
    reduces each averaged trace before returning the point to the ODMR worker.

    :param sg: The reference to SG Instrument.
    :type sg: Instrument
    :param pg: The reference to PG Instrument.
    :type pg: Instrument
    :param pd_names: (default: ["pd0", "pd1"]) names of PD Instruments in this overlay.
    :type pd_names: list[str]
    :param pd0: Reference to the first PD Instrument when named in ``pd_names``.
    :type pd0: Instrument
    :param pd1: Reference to the second PD Instrument when named in ``pd_names``.
    :type pd1: Instrument
    :param clock: Reference to the retriggerable clock for DAQ-based AnalogPDs.
    :type clock: Instrument
    :param pd_clock: Detector trigger terminal used by the PG ``gate`` channel.
    :type pd_clock: str
    :param pd_analog: (default: False) set True for an AnalogIn-based PD.
    :type pd_analog: bool
    :param pd_spectrum: (default: False) set True for a Spectrum_AnalogIn-based PD.
    :type pd_spectrum: bool
    :param pd_trace: (default: False) enable laser-resolved trace acquisition for the ``pulse``
        method. Requires ``pd_analog`` or ``pd_spectrum``.
    :type pd_trace: bool
    :param pd_chop: (default: False) enable the active-high SinglePhotonCounter chop output.
        This is valid only when ``pd_analog`` and ``pd_spectrum`` are False.
    :type pd_chop: bool
    :param pd_async: (default: False) acquire points asynchronously while triggering the PG.
    :type pd_async: bool
    :param pd_rate: (default param: 250e6 for trace pulse acquisition, 400e3 otherwise)
        AnalogPD sampling rate.
    :type pd_rate: float | int | list[int]
    :param pd_bounds: (default param: [-10.0, 10.0]) AnalogPD voltage bounds.
    :type pd_bounds: tuple[float, float]
    :param pd_data_transfer: data transfer mode for DAQ-based AnalogPDs.
    :type pd_data_transfer: str
    :param pd_segment_granularity: (default: 16) Spectrum trace segment granularity.
    :type pd_segment_granularity: int
    :param pd_segment_offset: (default: 0) sample offset subtracted after Spectrum segment
        rounding.
    :type pd_segment_offset: int
    :param buffer_size_coeff: (default: 20) initial and fallback value for the
        ``pd.buffer_size_coeff`` parameter.
    :type buffer_size_coeff: int
    :param queue_size: (default: 8) Size of queue of scanned line data.
    :type queue_size: int
    :param max_inflight_coeff: (default: 4) asynchronous acquisition inflight multiplier.
    :type max_inflight_coeff: int
    :param start_delay: (default: 0.0) delay before starting the sweep thread.
    :type start_delay: float
    :param channel_remap: Mapping to replace PG channel names.
    :type channel_remap: dict[str | int, str | int]
    :param pg_freq_cw: (has preset) PG frequency for CW mode.
    :type pg_freq_cw: float
    :param pg_freq_pulse: (has preset) PG frequency for pulse mode.
    :type pg_freq_pulse: float
    :param minimum_block_length: (has preset) minimum PG block length.
    :type minimum_block_length: int
    :param block_base: (has preset) PG block granularity.
    :type block_base: int
    :param pg_wait_timeout_sec: (default: 10.0) timeout for PG completion.
    :type pg_wait_timeout_sec: float
    :param pg_wait_interval_sec: (default: 0.001) PG completion polling interval.
    :type pg_wait_interval_sec: float

    """

    def __init__(self, name, conf, prefix=None):
        InstrumentOverlay.__init__(self, name, conf=conf, prefix=prefix)

        self.sg = self.conf.get("sg")
        self.pg = self.conf.get("pg")
        self.pd_names = self.conf.get("pd_names", ["pd0", "pd1"])
        self.pds = [self.conf.get(n) for n in self.pd_names]
        self._pd_spectrum = self.conf.get("pd_spectrum", False)
        self._pd_analog = self.conf.get("pd_analog", False) or self._pd_spectrum
        self._pd_trace = self.conf.get("pd_trace", False)
        if self._pd_trace and not self._pd_analog:
            raise ValueError("pd_trace requires pd_analog or pd_spectrum")
        self._pd_chop = self.conf.get("pd_chop", False)
        if self._pd_chop and self._pd_analog:
            raise ValueError("pd_chop requires a SinglePhotonCounter (pd_analog = False)")
        self._pd_async = self.conf.get("pd_async", False)
        self._max_inflight_coeff = self.conf.get("max_inflight_coeff", 4)
        self._max_inflight = 1
        self._inflight = 0
        self._inflight_cond = threading.Condition()
        if self._pd_analog and not self._pd_spectrum:
            self.clock = self.conf.get("clock")
        else:
            self.clock = None
        self.add_instruments(self.sg, self.pg, self.clock, *self.pds)

        self.load_pg_conf_preset()

        self._queue_size = self.conf.get("queue_size", 8)
        self._queue = RollingQueue(self._queue_size)
        self._stop_ev = self._thread = None
        self.running = False

        self.check_required_conf(
            ["pd_clock", "block_base", "pg_freq_cw", "pg_freq_pulse", "minimum_block_length"]
        )
        self._pd_clock = self.conf["pd_clock"]
        self._pd_data_transfer = self.conf.get("pd_data_transfer")
        self._minimum_block_length = self.conf["minimum_block_length"]
        self._block_base = self.conf["block_base"]
        self._start_delay = self.conf.get("start_delay", 0.0)
        self._channel_remap = self.conf.get("channel_remap")
        self._continue_mw = False
        self.pulse_pattern = None
        self._samples_per_trace = None

        self._pg_wait_timeout = self._conf_pos_num("pg_wait_timeout_sec", 10.0)
        self._pg_wait_interval = self._conf_pos_num("pg_wait_interval_sec", 0.001)

    def load_pg_conf_preset(self):
        loader = PresetLoader(self.logger, PresetLoader.Mode.FORWARD)
        loader.add_preset(
            "DTG",
            [
                ("block_base", 4),
                ("pg_freq_cw", 1.0e6),
                ("pg_freq_pulse", 2.0e9),
                ("minimum_block_length", 1000),
            ],
        )
        loader.add_preset(
            "PulseStreamer",
            [
                ("block_base", 8),
                ("pg_freq_cw", 1.0e9),
                ("pg_freq_pulse", 1.0e9),
                ("minimum_block_length", 1),
            ],
        )
        loader.load_preset(self.conf, self.pg.__class__.__name__)

    def _set_attrs(self, params):
        self.start_f, self.stop_f = params["start"], params["stop"]
        self.freqs = np.linspace(self.start_f, self.stop_f, params["num"])
        self.power = params["power"]
        self._continue_mw = params.get("continue_mw", False)

    def get_point(self):
        if self._pd_async:
            try:
                return self.get_pd_data()
            finally:
                self._decrement_inflight()
        else:
            return self._queue.pop_block()

    def _reserve_inflight(self, ev: threading.Event) -> bool:
        with self._inflight_cond:
            if ev.is_set():
                return False
            while self._inflight >= self._max_inflight:
                if ev.is_set():
                    return False
                self._inflight_cond.wait(0.01)
            self._inflight += 1
            return True

    def _decrement_inflight(self):
        with self._inflight_cond:
            if self._inflight > 0:
                self._inflight -= 1
            else:
                self.logger.warn("get_point() completed with no inflight acquisition.")
            self._inflight_cond.notify()

    def _wait_pg(self, ev: threading.Event) -> bool:
        deadline = time.monotonic() + self._pg_wait_timeout
        while time.monotonic() < deadline:
            if ev.is_set():
                return False
            if self.pg.get_finished():
                return True
            time.sleep(self._pg_wait_interval)
        return self.fail_with("PG hasn't finished operation.")

    def sweep_loop_async(self, ev: threading.Event, dummy_points: int):
        for i in range(dummy_points):
            self.sg.set_freq_CW(self.start_f)
            self.pg.trigger()
            if not self._wait_pg(ev):
                self.logger.info("Quitting sweep loop.")
                return
            self.logger.info(f"Dummy excitation #{i + 1}")

        while True:
            for f in self.freqs:
                if not self._reserve_inflight(ev):
                    self.logger.info("Quitting sweep loop.")
                    return
                try:
                    self.sg.set_freq_CW(f)
                    self.pg.trigger()
                except Exception:
                    self._decrement_inflight()
                    raise
                if not self._wait_pg(ev):
                    self.logger.info("Quitting sweep loop.")
                    return

    def sweep_loop_sync(self, ev: threading.Event, dummy_points: int):
        for i in range(dummy_points):
            self.sg.set_freq_CW(self.start_f)
            self.pg.trigger()
            if not self._wait_pg(ev):
                self.logger.info("Quitting sweep loop.")
                return
            self.logger.info(f"Dummy excitation #{i + 1}")

        while True:
            for f in self.freqs:
                self.sg.set_freq_CW(f)
                self.pg.trigger()
                self._queue.append(self.get_pd_data())
                if not self._wait_pg(ev):
                    self.logger.info("Quitting sweep loop.")
                    return

    def get_pd_data(self):
        """Return reduced point data and, in trace mode, its summed-channel raw traces."""

        blocks = [pd.pop_block() for pd in self.pds]
        point_count = 2 if self.params.get("background", False) else 1
        if self._label == "pulse" and self._pd_trace:
            traces = sum_pd_blocks(blocks, point_count, self._samples_per_trace)
            point = reduce_traces(traces, self.params["timing"], self.params["pd"]["rate"])
            return point, traces

        return sum_pd_channels(blocks)

    def configure_sg(self, params: dict, label: str):
        success = self.sg.configure_cw(self.start_f, self.power)
        success &= configure_modulation(self.sg, label, params.get("mod", {}))
        success &= self.sg.query_opc()
        return success

    # Standard API

    def get_param_dict_labels(self) -> list[str]:
        return ["pd", "pd_trace"] if self._pd_trace else ["pd"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        if label == "pd":
            return self.get_pd_param_dict(False)
        elif label == "pd_trace" and self._pd_trace:
            return self.get_pd_param_dict(True)
        self.logger.error(f"Unknown param dict label: {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        if not self.check_required_params(params, ("start", "stop", "num", "power", "delay")):
            return False

        success, message = self.validate_pd_params(params)
        if not success:
            return self.fail_with(message)

        self._set_attrs(params)
        self.params = params
        self._label = label
        self._queue = RollingQueue(self._queue_size)

        if not self.configure_sg(params, label):
            return self.fail_with("failed to configure SG.")
        if not self.configure_pg(params, label, TriggerType.SOFTWARE):
            return self.fail_with("failed to configure PG.")
        if not self.configure_pd(params, label):
            return self.fail_with("failed to configure PD.")

        if self._pd_async:
            if self._pd_spectrum:
                status = self.pds[0].get_fifo_status()
                segments_per_trigger = 2 if params.get("background", False) else 1
                if (
                    label == "pulse"
                    and self._pd_trace
                    and not params["pd"].get("hardware_average", True)
                ):
                    segments_per_trigger *= params["timing"]["burst_num"]
                notify_segments = status["notify_samples"] // status["segment_samples"]
                required_inflight = math.ceil(notify_segments / segments_per_trigger)
                max_inflight = self._max_inflight_coeff * required_inflight
                self.logger.debug(
                    f"max inflight: {max_inflight}; notify/segments: {notify_segments}"
                )
            else:
                # for DAQ required_inflight would be always 1.
                max_inflight = self._max_inflight_coeff
            with self._inflight_cond:
                self._inflight = 0
                self._max_inflight = max_inflight

        return True

    def start(self, label: str = "") -> bool:
        if self.running:
            self.logger.warn("start() is called while running.")
            return True

        if not self.start_pd():
            return self.fail_with("Failed to start pd.")
        if not self.sg.set_output(True):
            return self.fail_with("Failed to start sg.")
        if not self.pg.start():
            return self.fail_with("Failed to start pg.")

        time.sleep(self._start_delay)

        self._stop_ev = threading.Event()
        dummy_points = ParamAccessor(self.params).child("pd", {}).nonneg_int("dummy_points", 0)
        if self._pd_async:
            self._thread = threading.Thread(
                target=self.sweep_loop_async, args=(self._stop_ev, dummy_points)
            )
        else:
            self._thread = threading.Thread(
                target=self.sweep_loop_sync, args=(self._stop_ev, dummy_points)
            )
        self._thread.start()

        self.running = True

        return True

    def stop(self, label: str = "") -> bool:
        if not self.running:
            return True
        self.running = False

        self.logger.info("Stopping sweeper.")

        self._stop_ev.set()
        with self._inflight_cond:
            self._inflight_cond.notify_all()
        self._thread.join()

        if self._continue_mw:
            self.logger.warn("Skipping to turn off MW output")
            success = True
        else:
            success = self.sg.set_output(False)
        success &= all([pd.stop() for pd in self.pds])
        if self.clock is not None:
            success &= self.clock.stop()
        success &= self.pg.stop()

        if not success:
            return self.fail_with("failed to stop SG, PG, or PD.")

        return True

    def validate(self, params: dict, label: str) -> tuple[bool, str, str]:
        """Validate parameters using this overlay's hardware configuration.

        Pulse-generator parameters are validated only in pulse mode.

        """

        success, message = self.validate_pd_params(params)
        if not success:
            return False, message, ""
        if label == "pulse":
            return self.validate_pulse_params(params)
        return True, "", ""

    def validate_pd_params(self, params: dict) -> tuple[bool, str]:
        """Validate detector parameters shared by all measurement modes."""

        try:
            pd = ParamAccessor(params).child("pd", {})
            dummy_points = pd.nonneg_int("dummy_points", 0)
            buffer_size_coeff = pd.pos_int(
                "buffer_size_coeff", self._conf_pos_int("buffer_size_coeff", 20)
            )
            if dummy_points > buffer_size_coeff:
                raise ParamError(
                    f"pd.dummy_points ({dummy_points}) must not exceed "
                    f"pd.buffer_size_coeff ({buffer_size_coeff})."
                )
            return True, ""
        except (KeyError, TypeError, ValueError) as e:
            return False, str(e)

    def get(self, key: str, args=None, label: str = ""):
        if key == "point":
            return self.get_point()
        elif key == "validate":
            if not isinstance(args, dict):
                self.logger.error(f"Invalid args for get(validate): {args}")
                return False
            return self.validate(args, label)
        elif key == "bounds":
            return self.sg.get_bounds()
        elif key == "unit":
            return self.pds[0].get("unit")
        elif key == "pulse_pattern":
            return self.pulse_pattern
        elif key == "capability":
            return self.get_capability()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get_pd_param_dict(self, pd_trace: bool = False) -> P.ParamDict[str, P.PDValue]:
        dummy_points = P.IntParam(0, 0, 100, doc="number of dummy excitations to stabilize")
        if not self._pd_analog:
            return P.ParamDict(
                buffer_size_coeff=make_pd_buffer_size_coeff_param(self.conf),
                dummy_points=dummy_points,
            )
        pd = make_pd_param_dict(
            self.conf,
            pd_trace=pd_trace,
            has_hardware_average=pd_trace and self._pd_spectrum,
        )
        pd["dummy_points"] = dummy_points
        return pd

    def get_capability(self) -> dict[str, bool]:
        """Return detector and pulse-generation capabilities."""

        return {
            "pd_analog": self._pd_analog,
            "pd_trace": self._pd_trace,
            "pd_chop": self._pd_chop,
        }

    def configure_pd(self, params, label):
        if self._pd_analog:
            return self.configure_analog_pd(params, label)
        else:
            return self.configure_apd(params, label)

    def start_pd(self):
        if self.clock is not None:
            return self.clock.start() and all([pd.start() for pd in self.pds])
        else:
            return all([pd.start() for pd in self.pds])

    def configure_apd(self, params: dict, label: str) -> bool:
        p = ParamAccessor(params)
        pd = p.child("pd", {})
        buffer_size_coeff = pd.pos_int("buffer_size_coeff", self.conf.get("buffer_size_coeff", 20))
        return configure_apds(
            self.pds,
            self._pd_clock,
            self.apd_time_window(params, label),
            2 if p.bool("background", False) else 1,
            buffer_size_coeff,
            pd.nonneg_int("dummy_points", 0),
        )

    def configure_analog_pd(self, params: dict, label: str) -> bool:
        p = ParamAccessor(params)
        pd = p.child("pd", {})
        point_count = 2 if p.bool("background", False) else 1
        drop_first = pd.nonneg_int("dummy_points", 0)
        if label == "pulse" and self._pd_trace:
            self._samples_per_trace = configure_trace_pds(
                self.clock,
                self.pds,
                self._pd_clock,
                params,
                self.conf,
                self._pd_spectrum,
                point_count,
                drop_first,
                self._pd_data_transfer,
                self.logger,
            )
            return self._samples_per_trace is not None

        return configure_analog_pds(
            self.clock,
            self.pds,
            self._pd_clock,
            params,
            self.conf,
            self._pd_spectrum,
            point_count,
            drop_first,
            self._pd_data_transfer,
            self.logger,
        )
