#!/usr/bin/env python3

"""
Worker for ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import time

import numpy as np

from mahos_dq.msgs.odmr_msgs import ODMRData
from mahos.msgs import param_msgs as P
from mahos.msgs.inst.pg_msgs import TriggerType, PulsePattern
from mahos.inst.sg_interface import SGInterface
from mahos.inst.pg_interface import PGInterface
from mahos.inst.pd_interface import PDInterface
from mahos.inst.daq_interface import ClockSourceInterface
from mahos_dq.inst.overlay.odmr_sweeper_interface import ODMRSweeperInterface
from mahos.util.conf import PresetLoader
from mahos.util.param import ParamAccessor, ParamError
from mahos_dq.meas.odmr_pg import ODMRPGMixin
from mahos_dq.meas.odmr_sg import MOD_LABELS, configure_modulation
from mahos_dq.meas.odmr_pd import (
    configure_apds,
    configure_analog_pds,
    configure_trace_pds,
    make_pd_param_dict,
    make_trace_timing_param_dict,
    reduce_traces,
    result_unit,
    sum_pd_blocks,
    sum_pd_channels,
)
from mahos.meas.common_worker import Worker


class SweeperBase(Worker):
    def data_msg(self) -> ODMRData:
        return self.data

    def pulse_msg(self) -> PulsePattern | None:
        if hasattr(self, "pulse_pattern"):
            return self.pulse_pattern
        else:
            return None

    def _normalize_line(self, line):
        """Normalize one detector line before storing it in frequency order."""

        return line

    def _append_line_nobg(self, data, line):
        line = self._normalize_line(line)
        if data is None:
            return np.array(line, ndmin=2).T
        else:
            return np.append(data, np.array(line, ndmin=2).T, axis=1)

    def _append_line_bg(self, data, bg_data, line):
        l_data = self._normalize_line(line[0::2])
        l_bg = self._normalize_line(line[1::2])
        if data is None:
            return np.array(l_data, ndmin=2).T, np.array(l_bg, ndmin=2).T
        else:
            return (
                np.append(data, np.array(l_data, ndmin=2).T, axis=1),
                np.append(bg_data, np.array(l_bg, ndmin=2).T, axis=1),
            )

    def append_line(self, line):
        """Append one detector line, splitting interleaved background data when enabled."""

        if not self.data.measure_background():
            self.data.data = self._append_line_nobg(self.data.data, line)
        else:
            self.data.data, self.data.bg_data = self._append_line_bg(
                self.data.data, self.data.bg_data, line
            )

    def append_raw_line(self, traces: np.ndarray):
        """Add one frequency-ordered trace line to the cumulative raw trace sum."""

        traces = np.asarray(traces)
        if self.data.raw_data_sum is None:
            self.data.raw_data_sum = traces.copy()
        elif self.data.raw_data_sum.shape != traces.shape:
            self.logger.error(
                "Cannot append raw trace line with shape "
                f"{traces.shape} to raw_data_sum with shape {self.data.raw_data_sum.shape}."
            )
        else:
            self.data.raw_data_sum += traces

    def append_raw_point(self, traces: np.ndarray):
        """Add one point's traces to the matching row in the cumulative raw trace sum."""

        traces = np.asarray(traces)
        point_count = 2 if self.data.measure_background() else 1
        if traces.ndim != 2 or traces.shape[0] != point_count:
            self.logger.error(
                f"Invalid raw point trace shape {traces.shape}; expected ({point_count}, sample)."
            )
            return

        if self.data.data is None or not np.isnan(self.data.data).any():
            index = 0
        else:
            index = int(np.where(np.isnan(self.data.data[:, -1]))[0][0])

        shape = (self.data.params["num"] * point_count, traces.shape[1])
        if self.data.raw_data_sum is None:
            self.data.raw_data_sum = np.zeros(shape, dtype=traces.dtype)
        elif self.data.raw_data_sum.shape != shape:
            self.logger.error(
                f"Invalid raw_data_sum shape {self.data.raw_data_sum.shape}; expected {shape}."
            )
            return

        start = index * point_count
        self.data.raw_data_sum[start : start + point_count] += traces

    def is_finished(self) -> bool:
        if not self.data.has_params() or not self.data.has_data():
            return False
        if self.data.params.get("sweeps", 0) <= 0:
            return False  # no sweeps limit defined.
        return (
            not np.isnan(self.data.data).any() and self.data.sweeps() >= self.data.params["sweeps"]
        )

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> tuple[bool, str, str]:
        params = P.unwrap(params)

        try:
            if label not in self.get_param_dict_labels():
                raise ParamError(f"Unknown measurement label: {label}")

            p = ParamAccessor(params)
            start = p.num("start")
            stop = p.num("stop")
            if start >= stop:
                raise ParamError("stop must be greater than start")
            num = p.pos_int("num")
            if num < 2:
                raise ParamError(f"num must be at least 2. Got {num}")
            p.num("power")
            p.nonneg_int("sweeps", 0)

            for key in ("delay", "background_delay", "final_delay"):
                p.nonneg_num(key, 0.0)
            for key in ("background", "resume", "continue_mw"):
                p.bool(key, False)

            timing = p.child("timing")
            if label != "pulse" and "time_window" in timing:
                timing.pos_num("time_window")
                timing.nonneg_num("gate_delay", 0.0)
                timing.nonneg_num("post_gate_delay", 0.0)

            mod = p.child("mod", {})
            if label in ("am_ext", "am_int"):
                mod.num("am_depth")
                mod.bool("am_log")
                if label == "am_int":
                    mod.pos_num("am_rate")
            elif label in ("fm_ext", "fm_int"):
                mod.num("fm_deviation")
                if label == "fm_int":
                    mod.pos_num("fm_rate")
        except ParamError as e:
            self.logger.error(f"Invalid ODMR parameters: {e}")
            return False, str(e), ""

        if label == "pulse":
            return self._validate_pulse_params(params)
        return True, "", ""

    def _validate_pulse_params(self, params: dict) -> tuple[bool, str, str]:
        return False, "validation method for pulse is not implemented", ""

    def get_param_dict_labels(self) -> list:
        return ["cw", "pulse"] + MOD_LABELS

    def _make_param_dict(
        self, label, bounds, pd_analog, pd_trace=False, pd_chop=False
    ) -> P.ParamDict[str, P.PDValue] | None:
        if label in ["cw"] + MOD_LABELS:
            timing = P.ParamDict(
                time_window=P.FloatParam(
                    self.conf.get("time_window", 10e-3),
                    0.1e-3,
                    10.0,
                    unit="s",
                    SI_prefix=True,
                    doc="duration of the data acquisition window",
                ),
                gate_delay=P.FloatParam(
                    self.conf.get("gate_delay", 0.0),
                    0.0,
                    10.0,
                    unit="s",
                    SI_prefix=True,
                    doc="delay from excitation onset to the start of data acquisition",
                ),
                post_gate_delay=P.FloatParam(
                    self.conf.get("post_gate_delay", 0.0),
                    0.0,
                    10.0,
                    unit="s",
                    SI_prefix=True,
                    doc="delay from the end of data acquisition to excitation end",
                ),
            )
        elif label == "pulse" and pd_trace:
            timing = make_trace_timing_param_dict(self.conf)
        elif label == "pulse":
            timing = P.ParamDict(
                laser_delay=P.FloatParam(
                    100e-9, 0.0, 1e-3, unit="s", SI_prefix=True, doc="delay before laser"
                ),
                laser_width=P.FloatParam(
                    300e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="width of laser"
                ),
                mw_delay=P.FloatParam(
                    1e-6,
                    0.0,
                    1e-3,
                    unit="s",
                    SI_prefix=True,
                    doc="delay for microwave (>= trigger_width)",
                ),
                mw_width=P.FloatParam(
                    1e-6, 0.0, 1e-3, unit="s", SI_prefix=True, doc="width of microwave"
                ),
                trigger_width=P.FloatParam(
                    100e-9,
                    0.0,
                    10e-6,
                    unit="s",
                    SI_prefix=True,
                    doc="width of trigger (<= mw_delay)",
                ),
                mw_offset=P.FloatParam(
                    0.0, -1e-4, 1e-4, unit="s", SI_prefix=True, doc="global mw offset"
                ),
            )
            if pd_analog:
                timing["time_window"] = P.FloatParam(
                    self.conf.get("time_window", 10e-3),
                    0.1e-3,
                    10.0,
                    unit="s",
                    SI_prefix=True,
                    doc="duration of the data acquisition window",
                )
                timing["gate_delay"] = P.FloatParam(
                    self.conf.get("gate_delay", 0.0),
                    0.0,
                    10.0,
                    unit="s",
                    SI_prefix=True,
                    doc="delay from excitation onset to the start of data acquisition",
                )
                timing["post_gate_delay"] = P.FloatParam(
                    self.conf.get("post_gate_delay", 0.0),
                    0.0,
                    10.0,
                    unit="s",
                    SI_prefix=True,
                    doc="delay from the end of data acquisition to excitation end",
                )
            else:
                timing["burst_num"] = P.IntParam(
                    100, 1, 100_000, doc="number of bursts at each freq"
                )
                if pd_chop:
                    timing["chop_delay"] = P.FloatParam(
                        self.conf.get("chop_delay", 0.0),
                        0.0,
                        1e-3,
                        unit="s",
                        SI_prefix=True,
                        doc="delay from laser onset to counter chop onset",
                    )
                    timing["chop_width"] = P.FloatParam(
                        self.conf.get("chop_width", 100e-9),
                        1e-9,
                        1e-3,
                        unit="s",
                        SI_prefix=True,
                        doc="width of counter chop window",
                    )

        else:
            self.logger.error(f"Unknown param dict label: {label}")
            return None

        if bounds is None:
            self.logger.error("Could not get SG bounds.")
            return None
        f_min, f_max = bounds["freq"]
        p_min, p_max = bounds["power"]
        f_start = max(min(self.conf.get("start", 2.74e9), f_max), f_min)
        f_stop = max(min(self.conf.get("stop", 3.00e9), f_max), f_min)
        d = P.ParamDict(
            start=P.FloatParam(
                f_start, f_min, f_max, unit="Hz", SI_prefix=True, doc="sweep start frequency"
            ),
            stop=P.FloatParam(
                f_stop, f_min, f_max, unit="Hz", SI_prefix=True, doc="sweep stop frequency"
            ),
            num=P.IntParam(
                self.conf.get("num", 101), 2, 10000, doc="number of frequency sweep points"
            ),
            power=P.FloatParam(
                self.conf.get("power", p_min), p_min, p_max, unit="dBm", doc="MW power at SG"
            ),
            sweeps=P.IntParam(0, 0, 1_000_000_000, doc="number of sweeps (0 for infinite)"),
            timing=timing,
            background=P.BoolParam(False, doc="take background data"),
            delay=P.FloatParam(
                0.0,
                0.0,
                10.0,
                unit="s",
                SI_prefix=True,
                doc="delay after PG trigger before the measurement",
            ),
            background_delay=P.FloatParam(
                0.0,
                0.0,
                10.0,
                unit="s",
                SI_prefix=True,
                doc="delay between normal and background (reference) measurements",
            ),
            final_delay=P.FloatParam(
                0.0,
                0.0,
                10.0,
                unit="s",
                SI_prefix=True,
                doc="delay after measurement before triggering next frequency",
            ),
            resume=P.BoolParam(False),
            continue_mw=P.BoolParam(False),
            ident=P.UUIDParam(optional=True, enable=False),
        )

        mod = P.ParamDict()
        if label in ("am_ext", "am_int"):
            mod["am_depth"] = P.FloatParam(self.conf.get("am_depth", 0.1), doc="depth of AM")
            mod["am_log"] = P.BoolParam(
                self.conf.get("am_log", False), doc="True indicates log scale AM depth"
            )
            if label == "am_int":
                mod["am_rate"] = P.FloatParam(
                    self.conf.get("am_rate", 400.0),
                    unit="Hz",
                    SI_prefix=True,
                    doc="rate (baseband frequency) of AM",
                )
        elif label in ("fm_ext", "fm_int"):
            mod["fm_deviation"] = P.FloatParam(
                self.conf.get("fm_deviation", 1e3),
                unit="Hz",
                SI_prefix=True,
                doc="deviation of FM",
            )
            if label == "fm_int":
                mod["fm_rate"] = P.FloatParam(
                    self.conf.get("fm_rate", 400.0),
                    unit="Hz",
                    SI_prefix=True,
                    doc="rate (baseband frequency) of FM",
                )
        # TODO: more additional params for iq_int, am_int, and fm_int?
        d["mod"] = mod

        return d


class SweeperOverlay(SweeperBase):
    """Sweeper using Overlay.

    Refer to :mod:`mahos_dq.inst.overlay.odmr_sweeper` for docs of target overlay.
    If the target overlay has ``pd_trace`` enabled, the ``pulse`` parameter dictionary uses
    laser-resolved AnalogPD trace acquisition and exposes the trace-analysis timings below.

    :param sweeper.sweeper_name: (default: "sweeper") target overlay name in target.servers.
    :type sweeper.sweeper_name: str
    :param sweeper.point: (default: False) set True to publish data per point acquisition.
    :type sweeper.point: bool
    :param sweeper.chop_delay: (default: 0.0) delay from commanded laser onset to the
        SinglePhotonCounter chop window. Exposed for pulse mode when the target overlay has
        ``pd_chop`` enabled.
    :type sweeper.chop_delay: float
    :param sweeper.chop_width: (default: 100e-9) width of the SinglePhotonCounter chop window.
    :type sweeper.chop_width: float

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
    :param sweeper.gate_delay: (default param) gate delay before counting.
    :type sweeper.gate_delay: float
    :param sweeper.post_gate_delay: (default param) extra excitation after measurement window.
    :type sweeper.post_gate_delay: float
    :param sweeper.trigger_width: (default: 20e-9) detector-trigger width for trace mode.
    :type sweeper.trigger_width: float
    :param sweeper.burst_num: (default: 100) traces averaged at each frequency in trace mode.
    :type sweeper.burst_num: int
    :param sweeper.roi_head: (default: 20e-9) detector-trigger to laser delay in trace mode.
    :type sweeper.roi_head: float
    :param sweeper.roi_tail: (default: 100e-9) trace margin after the laser pulse.
    :type sweeper.roi_tail: float
    :param sweeper.sig_delay: (default: 0.0) signal-window delay after the laser.
    :type sweeper.sig_delay: float
    :param sweeper.sig_width: (default: 100e-9) signal-window width.
    :type sweeper.sig_width: float
    :param sweeper.ref_delay: (default: 0.0) reference delay after the signal window.
    :type sweeper.ref_delay: float
    :param sweeper.ref_width: (default: 100e-9) reference-window width.
    :type sweeper.ref_width: float
    :param sweeper.refmode: (default: "divide") trace reduction mode: ``subtract``, ``divide``,
        or ``ignore``.
    :type sweeper.refmode: str
    :param sweeper.pd.buffer_size_coeff: (default param: 20) detector buffer size relative to
        one callback block.
    :type sweeper.pd.buffer_size_coeff: int
    :param sweeper.pd.eos_deadtime: (default param: 200e-9) required deadtime after each
        realized trace before the next detector trigger.
    :type sweeper.pd.eos_deadtime: float

    :param sweeper.am_depth: (default param) depth of AM modulation.
    :type sweeper.am_depth: float
    :param sweeper.am_log: (default param) True indicates log-scale AM depth.
    :type sweeper.am_log: bool
    :param sweeper.am_rate: (default param) rate (baseband frequency) of AM in Hz.
    :type sweeper.am_rate: float
    :param sweeper.fm_deviation: (default param) FM deviation in Hz.
    :type sweeper.fm_deviation: float
    :param sweeper.fm_rate: (default param) rate (baseband frequency) of FM in Hz.
    :type sweeper.fm_rate: float

    """

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger, conf)
        self.sweeper_name = conf.get("sweeper_name", "sweeper")
        self.sweeper = ODMRSweeperInterface(cli, self.sweeper_name)
        self.add_instruments(self.sweeper)

        self._class_name = cli.class_name(self.sweeper_name)
        capability = self.sweeper.get_capability()
        self._pd_analog = bool(capability["pd_analog"])
        self._pd_trace = bool(capability.get("pd_trace", False))
        self._pd_chop = bool(capability.get("pd_chop", False))
        self._pd_spectrum = False
        self.point = self.conf.get("point", False)
        self.data = ODMRData()

    def _validate_pulse_params(self, params: dict) -> tuple[bool, str, str]:
        """Delegate pulse param validation to the overlay that owns the hardware configuration."""

        return self.sweeper.validate(params, "pulse")

    def get_param_dict_labels(self) -> list[str]:
        if self._class_name.startswith("ODMRSweeperCommand"):
            return ["cw"] + MOD_LABELS
        else:
            return ["cw", "pulse"] + MOD_LABELS

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        if self._class_name.startswith("ODMRSweeperCommand") and label == "pulse":
            return None

        bounds = self.sweeper.get_bounds()
        if bounds is None:
            self.logger.error("Failed to get bounds from sweeper.")
            return None

        d = self._make_param_dict(label, bounds, self._pd_analog, self._pd_trace, self._pd_chop)
        pd_label = "pd_trace" if label == "pulse" and self._pd_trace else "pd"
        pd = self.sweeper.get_param_dict(pd_label)
        if pd is not None:
            d["pd"] = pd

        # For reduced classes
        if label != "pulse" and self._class_name.endswith("AnalogPD"):
            d["timing"] = P.ParamDict(
                time_window=P.FloatParam(
                    self.conf.get("time_window", 10e-3), 0.1e-3, 1.0, unit="s", SI_prefix=True
                ),
            )
        elif label != "pulse" and self._class_name.endswith("AnalogPDMM"):
            d["timing"] = P.ParamDict()

        return d

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str = ""
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)
        success = self.sweeper.lock()

        if params is not None:
            success &= self.sweeper.configure(params, label)
        if not success:
            return self.fail_with_release("Error configuring sweeper.")

        self.pulse_pattern = self.sweeper.get_pulse_pattern()

        success &= self.sweeper.start()
        if not success:
            return self.fail_with_release("Error starting sweeper.")

        if params is not None and not params["resume"]:
            # new measurement.
            self.data = ODMRData(params, label)
            self.data.start()
            self.logger.info("Started sweeper.")
        else:
            # resume.
            self.data.update_params(params)
            self.data.resume()
            self.logger.info("Resuming sweeper.")
        self.data.yunit = result_unit(
            self.sweeper.get_unit(), self.data.params, self.data.label, self._pd_trace
        )

        return True

    def _new_line(self, dtype):
        return np.array([np.nan] * self.data.params["num"], dtype=dtype)

    def _append_point_nobg(self, data, point):
        if data is None:
            # the very first point
            line = self._new_line(point.dtype)
            line[0] = point[0]
            return np.array(line, ndmin=2).T
        elif not np.isnan(data).any():
            # line is finished, append new line
            line = self._new_line(point.dtype)
            line[0] = point[0]
            return np.append(data, np.array(line, ndmin=2).T, axis=1)
        else:
            # new point in latest line
            idx = np.where(np.isnan(data[:, -1]))[0][0]
            data[idx, -1] = point[0]
            return data

    def _append_point_bg(self, data, bg_data, point):
        p_data = point[0]
        p_bg = point[1]
        if data is None:
            # the very first point
            l_data = self._new_line(point.dtype)
            l_bg = self._new_line(point.dtype)
            l_data[0] = p_data
            l_bg[0] = p_bg
            return np.array(l_data, ndmin=2).T, np.array(l_bg, ndmin=2).T
        elif not np.isnan(data).any():
            # line is finished, append new line
            l_data = self._new_line(point.dtype)
            l_bg = self._new_line(point.dtype)
            l_data[0] = p_data
            l_bg[0] = p_bg
            return (
                np.append(data, np.array(l_data, ndmin=2).T, axis=1),
                np.append(bg_data, np.array(l_bg, ndmin=2).T, axis=1),
            )
        else:
            # new point in latest line
            idx = np.where(np.isnan(data[:, -1]))[0][0]
            data[idx, -1] = p_data
            bg_data[idx, -1] = p_bg
            return data, bg_data

    def append_point(self, point):
        if not self.data.measure_background():
            self.data.data = self._append_point_nobg(self.data.data, point)
        else:
            self.data.data, self.data.bg_data = self._append_point_bg(
                self.data.data, self.data.bg_data, point
            )

    def work(self):
        if self.point:
            self._work_point()
        else:
            self._work_line()

    def _work_line(self):
        if not self.data.running:
            return  # or raise Error?

        line = self.sweeper.get_line()
        if line is None:
            self.logger.error("Got None from sweeper.get_line()")
            return

        self.append_line(line)

    def _work_point(self):
        if not self.data.running:
            return  # or raise Error?

        point = self.sweeper.get_point()
        if point is None:
            self.logger.error("Got None from sweeper.get_point()")
            return

        if self.data.label == "pulse" and self._pd_trace:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                self.logger.error("Got invalid point and raw-trace pair from sweeper.")
                return
            point, traces = point
            self.append_raw_point(traces)
        self.append_point(point)

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.sweeper.stop() and self.sweeper.release()

        self.data.finalize()
        if success:
            self.logger.info("Stopped sweeper.")
        else:
            self.logger.error("Error stopping sweeper.")
        return success


class Sweeper(SweeperBase, ODMRPGMixin):
    """Worker for fast ODMR sweep using hardware triggered SG and/or PG.

    For some SGs with "frequency settled" output signal, SG and PG can be mutually triggered;
    PG is triggered by "frequency settled" of SG,
    and SG's freq sweep step is triggered at last of PG sequence.

    For SGs without such output, PG can be configured to run continuously (pg_immediate = True);
    SG's freq sweep step is continuously triggered by PG.

    :param sweeper.start_delay: (default: 0.0) delay time (sec.) before starting SG/PG output.
    :type sweeper.start_delay: float
    :param sweeper.sg_first: (has preset) if True, turn on SG first and PG second.
        This mode is for a SG which cannot start the point_freq_sweep mode without affecting PG.
    :type sweeper.sg_first: bool
    :param sweeper.pg_immediate: (has preset) if True, PG runs IMMEDIATE mode without trigger.
        This mode is for a SG which doesn't have "frequency settled" output signal.
        pg_immediate takes precedence over sg_first; when this is True, sg_first has no effect.
    :type sweeper.pg_immediate: bool

    :param sweeper.pd_clock: DAQ terminal name for PD's clock (gate)
    :type sweeper.pd_clock: str
    :param sweeper.pd_names: (default: ["pd0", "pd1"]) PD names in target.servers.
    :type sweeper.pd_names: list[str]
    :param sweeper.pd_analog: (default: False) set True if PD is AnalogIn-based.
    :type sweeper.pd_analog: bool
    :param sweeper.pd_spectrum: (default: False) set True if PD is Spectrum_AnalogIn-based.
    :type sweeper.pd_spectrum: bool
    :param sweeper.pd_trace: (default: False) enable laser-resolved AnalogPD trace acquisition
        for the ``pulse`` method. Requires ``pd_analog`` or ``pd_spectrum``.
    :type sweeper.pd_trace: bool
    :param sweeper.pd_chop: (default: False) enable the active-high SinglePhotonCounter chop
        output. Requires ``pd_analog`` and ``pd_spectrum`` to be False and a configured ``chop``
        PG channel.
    :type sweeper.pd_chop: bool
    :param sweeper.pd_rate: (default param) analog PD sampling rate.
    :type sweeper.pd_rate: float
    :param sweeper.pd_bounds: (default: [-10.0, 10.0]) analog PD voltage bounds.
    :type sweeper.pd_bounds: tuple[float, float]
    :param sweeper.pd_data_transfer: DAQ data transfer mode for analog PD.
    :type sweeper.pd_data_transfer: str
    :param sweeper.pd_segment_granularity: (default: 16) logical post-trigger sample
        granularity for Spectrum PD segments.
    :type sweeper.pd_segment_granularity: int
    :param sweeper.pd_segment_offset: (default: 0) sample offset subtracted after Spectrum
        segment rounding.
    :type sweeper.pd_segment_offset: int
    :param sweeper.buffer_size_coeff: (default: 20) default and fallback value for
        ``pd.buffer_size_coeff``.
    :type sweeper.buffer_size_coeff: int
    :param sweeper.clock_name: (default: "clock") DAQ clock source name in target.servers.
    :type sweeper.clock_name: str

    :param sweeper.channel_remap: mapping to fix default channel names.
    :type sweeper.channel_remap: dict[str | int, str | int]
    :param sweeper.pg_freq_cw: (has preset) PG frequency for CW mode.
    :type sweeper.pg_freq_cw: float
    :param sweeper.pg_freq_pulse: (has preset) PG frequency for Pulse mode.
    :type sweeper.pg_freq_pulse: float
    :param sweeper.minimum_block_length: (has preset) minimum block length in generated blocks
    :type sweeper.minimum_block_length: int
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
    :param sweeper.gate_delay: (default param) gate delay before counting.
    :type sweeper.gate_delay: float
    :param sweeper.post_gate_delay: (default param) extra excitation after measurement window.
    :type sweeper.post_gate_delay: float
    :param sweeper.trigger_width: (default: 20e-9) detector-trigger width for trace mode.
    :type sweeper.trigger_width: float
    :param sweeper.burst_num: (default: 100) traces averaged at each frequency in trace mode.
    :type sweeper.burst_num: int
    :param sweeper.roi_head: (default: 20e-9) detector-trigger to laser delay in trace mode.
    :type sweeper.roi_head: float
    :param sweeper.roi_tail: (default: 100e-9) trace margin after the laser pulse.
    :type sweeper.roi_tail: float
    :param sweeper.sig_delay: (default: 0.0) signal-window delay after the laser.
    :type sweeper.sig_delay: float
    :param sweeper.sig_width: (default: 100e-9) signal-window width.
    :type sweeper.sig_width: float
    :param sweeper.ref_delay: (default: 0.0) reference delay after the signal window.
    :type sweeper.ref_delay: float
    :param sweeper.ref_width: (default: 100e-9) reference-window width.
    :type sweeper.ref_width: float
    :param sweeper.refmode: (default: "divide") trace reduction mode: ``subtract``, ``divide``,
        or ``ignore``.
    :type sweeper.refmode: str
    :param sweeper.chop_delay: (default: 0.0) delay from commanded laser onset to the
        SinglePhotonCounter chop window.
    :type sweeper.chop_delay: float
    :param sweeper.chop_width: (default: 100e-9) width of the SinglePhotonCounter chop window.
    :type sweeper.chop_width: float
    :param sweeper.pd.hardware_average: (default param: True) use Spectrum hardware averaging
        for trace bursts. This parameter is exposed only for Spectrum trace mode.
    :type sweeper.pd.hardware_average: bool
    :param sweeper.pd.buffer_size_coeff: (default param: 20) detector buffer size relative to
        one callback block.
    :type sweeper.pd.buffer_size_coeff: int
    :param sweeper.pd.eos_deadtime: (default param: 200e-9) required deadtime after each
        realized trace before the next detector trigger.
    :type sweeper.pd.eos_deadtime: float

    :param sweeper.am_depth: (default param) depth of AM modulation.
    :type sweeper.am_depth: float
    :param sweeper.am_log: (default param) True indicates log-scale AM depth.
    :type sweeper.am_log: bool
    :param sweeper.am_rate: (default param) rate (baseband frequency) of AM in Hz.
    :type sweeper.am_rate: float
    :param sweeper.fm_deviation: (default param) FM deviation in Hz.
    :type sweeper.fm_deviation: float
    :param sweeper.fm_rate: (default param) rate (baseband frequency) of FM in Hz.
    :type sweeper.fm_rate: float

    """

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger, conf)
        self.load_pg_conf_preset(cli)
        self.load_sg_conf_preset(cli)

        self.sg = SGInterface(cli, "sg")
        self.pg = PGInterface(cli, "pg")
        self.pd_names = self.conf.get("pd_names", ["pd0", "pd1"])
        self.pds = [PDInterface(cli, n) for n in self.pd_names]
        self._pd_spectrum = self.conf.get("pd_spectrum", False)
        self._pd_analog = self.conf.get("pd_analog", False) or self._pd_spectrum
        self._pd_trace = self.conf.get("pd_trace", False)
        if self._pd_trace and not self._pd_analog:
            raise ValueError("pd_trace requires pd_analog or pd_spectrum")
        self._pd_chop = self.conf.get("pd_chop", False)
        if self._pd_chop and self._pd_analog:
            raise ValueError("pd_chop requires a SinglePhotonCounter (pd_analog = False)")
        if self._pd_analog and not self._pd_spectrum:
            self.clock = ClockSourceInterface(cli, self.conf.get("clock_name", "clock"))
        else:
            self.clock = None
        self.add_instruments(self.sg, self.pg, self.clock, *self.pds)

        self.check_required_conf(
            ["pd_clock", "block_base", "pg_freq_cw", "pg_freq_pulse", "minimum_block_length"]
        )
        self._pd_clock = self.conf["pd_clock"]
        self._pd_data_transfer = self.conf.get("pd_data_transfer")
        self._minimum_block_length = self.conf["minimum_block_length"]
        self._block_base = self.conf["block_base"]
        self._start_delay = self.conf.get("start_delay", 0.0)
        self._sg_first = self.conf.get("sg_first", False)
        self._pg_immediate = self.conf.get("pg_immediate", False)
        self._channel_remap = self.conf.get("channel_remap")
        self._continue_mw = False
        self._samples_per_trace = None

        self.pulse_pattern = None
        self.data = ODMRData()

    def _validate_pulse_params(self, params: dict) -> tuple[bool, str, str]:
        """Validate pulse parameters using the direct sweeper configuration."""

        return self.validate_pulse_params(params)

    def load_pg_conf_preset(self, cli):
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
        loader.load_preset(self.conf, cli.class_name("pg"))

    def load_sg_conf_preset(self, cli):
        loader = PresetLoader(self.logger, PresetLoader.Mode.FORWARD)
        loader.add_preset(
            "N5182B",
            [
                ("sg_first", False),
                ("pg_immediate", False),
            ],
        )
        loader.add_preset(
            "MG3710E",
            [
                ("sg_first", True),
                ("pg_immediate", False),
            ],
        )
        loader.add_preset(
            "DS_SG",
            [
                ("pg_immediate", True),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("sg"))

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        d = self._make_param_dict(
            label, self.sg.get_bounds(), self._pd_analog, self._pd_trace, self._pd_chop
        )
        if self._pd_analog:
            d["pd"] = make_pd_param_dict(
                self.conf,
                pd_trace=label == "pulse" and self._pd_trace,
                has_hardware_average=label == "pulse" and self._pd_trace and self._pd_spectrum,
            )
        return d

    def configure_sg(self, params: dict, label: str) -> bool:
        p = params
        success = self.sg.configure_point_trig_freq_sweep(
            p["start"], p["stop"], p["num"], p["power"]
        )
        success &= configure_modulation(self.sg, label, params.get("mod", {}))
        success &= self.sg.get_opc()
        return success

    def start_apd(self, params: dict, label: str) -> bool:
        # when sg_first or pg_immediate,
        # drop the first line because it contains invalid data at the first point
        # (line will be [f_N-1, f_0, f_1, ..., f_N-2) in general, however,
        #  SG is unknown state at the first point of the first line)
        drop_first = 1 if self._sg_first or self._pg_immediate else 0

        if not configure_apds(
            self.pds,
            self._pd_clock,
            self.apd_time_window(params, label),
            params["num"] * (2 if params.get("background", False) else 1),
            self.conf.get("buffer_size_coeff", 20),
            drop_first,
        ):
            return False
        return all([pd.start() for pd in self.pds])

    def start_analog_pd(self, params: dict, label: str) -> bool:
        point_count = params["num"] * (2 if params.get("background", False) else 1)
        drop_first = 1 if self._sg_first or self._pg_immediate else 0

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
            if self._samples_per_trace is None:
                return False
        else:
            if not configure_analog_pds(
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
            ):
                return False

        success = self.clock is None or self.clock.start()
        return success and all([pd.start() for pd in self.pds])

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str = ""
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)
            self._continue_mw = params.get("continue_mw", False)
        resume = params is None or params.get("resume")

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not resume:
            self.data = ODMRData(params, label)
        else:
            # TODO: check ident if resume?
            self.data.update_params(params)
        self.data.yunit = result_unit(
            self.pds[0].get_unit(), self.data.params, self.data.label, self._pd_trace
        )

        if not self.configure_sg(self.data.params, self.data.label):
            return self.fail_with_release("Failed to configure SG.")
        if self._pg_immediate:
            trig = TriggerType.IMMEDIATE
        else:
            trig = TriggerType.HARDWARE_RISING
        if not self.configure_pg(self.data.params, self.data.label, trig):
            return self.fail_with_release("Failed to configure PG.")
        if self._pd_analog:
            if not self.start_analog_pd(self.data.params, self.data.label):
                return self.fail_with_release("Failed to start PD (Analog).")
        else:
            if not self.start_apd(self.data.params, self.data.label):
                return self.fail_with_release("Failed to start APD.")

        time.sleep(self._start_delay)

        if self._sg_first or self._pg_immediate:
            if not (self.sg.set_output(True) and self.sg.start() and self.sg.get_opc()):
                return self.fail_with_release("Failed to start SG.")
            if not (self.pg.start() and self.pg.get_opc()):
                return self.fail_with_release("Failed to start PG.")
            if not self._pg_immediate:
                # implies sg_first mode, but better to have "if not pg_immediate" here
                # because pg_immediate takes precedence over sg_first.
                # (sg_first = pg_immediate = True is also a valid config)
                self.pg.trigger()
        else:
            if not (self.pg.start() and self.pg.get_opc()):
                return self.fail_with_release("Failed to start PG.")
            if not (self.sg.set_output(True) and self.sg.start()):
                return self.fail_with_release("Failed to start SG.")

        if resume:
            self.data.resume()
            self.logger.info("Resumed sweeper.")
        else:
            self.data.start()
            self.logger.info("Started sweeper.")
        return True

    def _roll_line(self, line):
        """Fix the rolling of data due to sg_first or pg_immediate operation.

        When sg_first or pg_immediate, the data will be like
        [f_N-1, f0, f1, ..., f_N-2] instead of [f0, f1, ..., f_N-1].
        Fix the former to the latter by rolling the array.

        """

        if self._sg_first or self._pg_immediate:
            return np.roll(line, -1)
        else:
            return line

    def _normalize_line(self, line):
        return self._roll_line(line)

    def work(self):
        if not self.data.running:
            return  # or raise Error?

        blocks = [pd.pop_block() for pd in self.pds]
        if self.data.label == "pulse" and self._pd_trace:
            bg_factor = 2 if self.data.measure_background() else 1
            point_count = self.data.params["num"] * bg_factor
            try:
                traces = sum_pd_blocks(blocks, point_count, self._samples_per_trace)
                line = reduce_traces(
                    traces, self.data.params["timing"], self.data.params["pd"]["rate"]
                )
            except ValueError:
                self.logger.exception("Failed to reduce AnalogPD traces")
                return
            if self._sg_first or self._pg_immediate:
                traces = np.roll(traces, -bg_factor, axis=0)
            self.append_raw_line(traces)
        else:
            line = sum_pd_channels(blocks)
        self.append_line(line)

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.sg.stop()

        if self._continue_mw:
            self.logger.warn("Skipping to turn off MW output")
        else:
            success &= self.sg.set_output(False)

        success &= all([pd.stop() for pd in self.pds])
        if self.clock is not None:
            success &= self.clock.stop()
        success &= self.pg.stop()
        success &= self.release_instruments()

        self.data.finalize()

        if success:
            self.logger.info("Stopped sweeper.")
        else:
            self.logger.error("Error stopping sweeper.")
        return success
