#!/usr/bin/env python3

"""
Worker for Analog-PD Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import typing as T
import time
import math

import numpy as np

from mahos.msgs import param_msgs as P
from mahos.msgs.inst.pg_msgs import PulsePattern
from mahos.inst.pg_interface import Block, Blocks
from mahos.inst.pd_interface import PDInterface
from mahos.inst.daq_interface import ClockSourceInterface
import mahos.util.validation as V
from mahos_dq.meas.podmr_generator import generator_kernel as K
from mahos_dq.meas.podmr_worker import (
    CommonPulserBase,
    SGPGPulserBase,
    AWGPulserBase,
    PODMRDataOperator,
    remove_analog_channels,
)
from mahos_dq.msgs.apodmr_msgs import APODMRData, MWMode
from mahos_dq.util.segments import round_segment_samples_up


class APODMRDataOperator(PODMRDataOperator):
    """Operations (set / get / analyze) on :class:`APODMRData`."""

    def update_plot_params(self, data: APODMRData, plot_params: dict) -> bool:
        """Update plot parameters, rebuilding history when sampled analysis windows change."""

        if not data.has_params():
            return False
        old = data.params.get("plot", {})
        merged = {**old, **plot_params}
        old_indices = self.get_marker_indices(data)
        updated = super().update_plot_params(data, merged)
        new_indices = self.get_marker_indices(data)
        windows_changed = not np.array_equal(old_indices, new_indices)
        if windows_changed:
            data.clear_history()
            self.analyze_with_error(data)
        return updated or windows_changed

    def update_params(self, data: APODMRData, params: dict | None):
        """Apply resume parameters without bypassing history invalidation."""

        if params is None:
            return
        params = params.copy()
        plot = params.pop("plot", None)
        was_enabled = data.params.get("save_history", True)
        data.update_params(params)
        enabled = data.params.get("save_history", True)
        if not enabled or enabled != was_enabled:
            data.clear_history()
        if plot is not None:
            self.update_plot_params(data, plot)
        if enabled != was_enabled:
            self.get_marker_indices(data)
            self.analyze_with_error(data)

    def _update_history(self, data: APODMRData):
        """Append missing retained records, rebuilding if a gap cannot be recovered."""

        if not data.params.get("save_history", True) or not data.has_raw_data():
            data.clear_history()
            return
        retained_start = data.records - data.retained_records()
        end = data.history_start_record
        if data.signal_history is not None:
            end += len(data.signal_history)
        if data.signal_history is None or end < retained_start:
            data.clear_history()
            data.history_start_record = retained_start
            end = retained_start
        if end == data.records:
            return
        means = [
            self._analyze_record(traces, data.marker_indices)
            for traces in data.raw_data[end - retained_start :]
        ]
        signals = np.stack([sig for sig, ref in means])
        references = np.stack([ref for sig, ref in means])
        if data.signal_history is None:
            data.signal_history = signals
            data.reference_history = references
        else:
            data.signal_history = np.concatenate((data.signal_history, signals))
            data.reference_history = np.concatenate((data.reference_history, references))

    def set_trace_laser_timing(self, data: APODMRData, trace_laser_timing):
        data.trace_laser_timing = float(trace_laser_timing)

    def set_trigger_timing(self, data: APODMRData, trigger_timing):
        data.trigger_timing = np.array(trigger_timing, dtype=np.float64)

    def set_instrument_params(
        self,
        data: APODMRData,
        samples_per_trace: int,
        sample_period: float,
        pg_freq: float,
        length: int,
        offsets: list[int],
        pd_rate: float,
        mw_modes: T.Sequence[MWMode],
        extra: dict | None = None,
    ):
        if "instrument" in data.params:
            return
        data.params["instrument"] = {}
        data.params["instrument"]["samples_per_trace"] = int(samples_per_trace)
        data.params["instrument"]["trange"] = float(samples_per_trace) * sample_period
        data.params["instrument"]["tbin"] = sample_period
        data.params["instrument"]["pg_freq"] = pg_freq
        data.params["instrument"]["length"] = int(length)
        data.params["instrument"]["pd_rate"] = pd_rate
        if all([ofs == 0 for ofs in offsets]):
            data.params["instrument"]["offsets"] = []
        else:
            data.params["instrument"]["offsets"] = offsets
        data.params["instrument"]["mw_modes"] = [MWMode.parse(m).name for m in mw_modes]
        if extra is not None:
            data.params["instrument"].update(extra)

    def append_record(self, data: APODMRData, traces: np.ndarray):
        traces = np.asarray(traces, dtype=np.float64)
        data.records += 1
        if data.raw_data_sum is None:
            data.raw_data_sum = traces.copy()
        else:
            data.raw_data_sum += traces

        try:
            max_records = int(data.params.get("max_records", 0))
        except (AttributeError, TypeError, ValueError):
            max_records = 0

        if max_records == 1 or data.raw_data is None:
            data.raw_data = traces[np.newaxis].copy()
        else:
            start = max(0, data.raw_data.shape[0] + 1 - max_records) if max_records else 0
            data.raw_data = np.concatenate((data.raw_data[start:], traces[np.newaxis]), axis=0)

    def get_marker_indices(self, data: APODMRData):
        tbin = data.get_bin()
        if data.params is None or tbin is None or data.trace_laser_timing is None:
            return None

        sigdelay, sigwidth, refdelay, refwidth = [
            data.params["plot"][k] for k in ("sigdelay", "sigwidth", "refdelay", "refwidth")
        ]

        signal_head = data.trace_laser_timing + sigdelay
        signal_tail = signal_head + sigwidth
        reference_head = signal_tail + refdelay
        reference_tail = reference_head + refwidth

        signal_head = np.round(signal_head / tbin).astype(np.int64)
        signal_tail = np.round(signal_tail / tbin).astype(np.int64)
        reference_head = np.round(reference_head / tbin).astype(np.int64)
        reference_tail = np.round(reference_tail / tbin).astype(np.int64)

        data.marker_indices = np.array(
            (signal_head, signal_tail, reference_head, reference_tail), dtype=np.int64
        )
        return data.marker_indices

    def _analyze_record(self, traces: np.ndarray, marker_indices: np.ndarray):
        sig_head, sig_tail, ref_head, ref_tail = (int(v) for v in marker_indices)
        sig = np.mean(traces[:, sig_head : sig_tail + 1], axis=1)
        ref = np.mean(traces[:, ref_head : ref_tail + 1], axis=1)
        return np.asarray(sig, dtype=np.float64), np.asarray(ref, dtype=np.float64)

    def _analysis_error(self, data: APODMRData) -> str | None:
        if not data.has_raw_data_sum() or data.records < 1:
            return "aggregated raw data is unavailable"
        if data.marker_indices is None:
            return "marker indices are unavailable"

        samples_per_trace = data.get_samples_per_trace()
        if samples_per_trace is None or samples_per_trace < 1:
            return "samples_per_trace is invalid"

        sig_head, sig_tail, ref_head, ref_tail = (int(v) for v in data.marker_indices)
        if not (0 <= sig_head <= sig_tail < samples_per_trace):
            return (
                "signal window is out of range "
                f"(head={sig_head}, tail={sig_tail}, samples={samples_per_trace})"
            )
        if not (0 <= ref_head <= ref_tail < samples_per_trace):
            return (
                "reference window is out of range "
                f"(head={ref_head}, tail={ref_tail}, samples={samples_per_trace})"
            )
        return None

    def analyze_with_error(self, data: APODMRData) -> str | None:
        error = self._analysis_error(data)
        if error is not None:
            data.clear_history()
            return error

        self._update_history(data)

        N = data.num_pattern()
        traces = data.raw_data_sum / data.records
        sig_avg, ref_avg = self._analyze_record(traces, data.marker_indices)

        for i in range(4):
            data.set_data(i, None)
            data.set_data_ref(i, None)

        if data.is_partial():
            p = data.partial()
            data.set_data(p, sig_avg)
            data.set_data_ref(p, ref_avg)
        else:
            for i in range(N):
                data.set_data(i, sig_avg[i::N])
                data.set_data_ref(i, ref_avg[i::N])
        return None

    def analyze(self, data: APODMRData) -> bool:
        return self.analyze_with_error(data) is None


class APODMRBlockBuilder(object):
    """Build PODMR raw blocks into APODMR-ready blocks with per-laser triggers."""

    def __init__(
        self,
        minimum_block_length: int,
        block_base: int,
        mw_modes: tuple[MWMode],
        iq_amplitude: float,
        channel_remap: dict | None,
    ):
        self.minimum_block_length = minimum_block_length
        self.block_base = block_base
        self.mw_modes = mw_modes
        self.iq_amplitude = iq_amplitude
        self.channel_remap = channel_remap
        self.eos_deadtime_ticks = 0
        self.all_trigger_timing = []

    def build_blocks(
        self, blocks: list[Blocks[Block]], freq: float, common_pulses, params: dict, num_mw: int
    ) -> tuple[Blocks[Block], list[int], list[int], int]:
        (
            base_width,
            _laser_delay,
            laser_width,
            _mw_delay,
            trigger_width,
            init_delay,
            final_delay,
        ) = common_pulses

        if params.get("divide_block", False):
            raise ValueError("divide_block=True is TODO and not implemented in APODMR.")

        if trigger_width <= 0:
            raise ValueError("trigger_width must be positive")

        roi_head = int(round(params["roi_head"] * freq))
        roi_tail = int(round(params["roi_tail"] * freq))
        if roi_head < 0 or roi_tail < 0:
            raise ValueError("roi_head and roi_tail must be non-negative")
        burst_num = int(params.get("burst_num", 1))
        if burst_num < 1:
            raise ValueError("burst_num must be positive")
        point_init_delay = float(params.get("point_init_delay", 0.0))
        if point_init_delay < 0.0:
            raise ValueError("point_init_delay must be non-negative")
        point_init_delay_ticks = int(round(point_init_delay * freq))
        if point_init_delay > 0.0 and point_init_delay_ticks <= 0:
            raise ValueError("point_init_delay must be at least one pulse-generator tick")

        out = Blocks()
        laser_timing = []
        trigger_timing = []
        self.all_trigger_timing = []
        self.eos_deadtime_ticks = math.ceil(params.get("pd", {}).get("eos_deadtime", 0.0) * freq)
        t = 0

        init_block_width = max(init_delay + laser_width, self.minimum_block_length)
        init_block_width = K.offset_base_inc(init_block_width, base_width)
        final_block_width = max(final_delay, self.minimum_block_length)
        final_block_width = K.offset_base_inc(final_block_width, base_width)

        phases = K.init_final_phases(num_mw)

        if not point_init_delay_ticks:
            init = Block(
                "INIT",
                [
                    (("sync",) + phases, init_delay),
                    (
                        (
                            "laser",
                            "sync",
                        )
                        + phases,
                        init_block_width - init_delay,
                    ),
                ],
            )
            out.append(init)
            t += init.total_length()

        point_index = 0
        for blks in blocks:
            for i in range(0, len(blks), 2):
                if point_init_delay_ticks:
                    delay = point_init_delay_ticks + (init_delay if point_index == 0 else 0)
                    block_length = K.offset_base_inc(
                        max(delay + laser_width, self.minimum_block_length), base_width
                    )
                    pattern = [
                        (("sync",) + phases, block_length - laser_width),
                        (("laser", "sync") + phases, laser_width),
                    ]
                    point_init = Block(f"POINT_INIT{point_index}", pattern)
                    out.append(point_init)
                    t += point_init.total_length()

                op = K.inject_trigger(blks[i], roi_head, trigger_width)
                rd = blks[i + 1]
                unit = op.concatenate(rd)
                trigger_offset = op.total_length() - roi_head
                laser_offset = op.total_length()
                trigger_timing.append(t + trigger_offset)
                laser_timing.append(t + laser_offset)
                for j in range(burst_num):
                    self.all_trigger_timing.append(t + trigger_offset + j * unit.total_length())
                out.append(unit.repeat(burst_num))
                t += unit.total_length() * burst_num
                point_index += 1

        out.append(Block("FINAL", [(("sync",) + phases, final_block_width)]))

        # TODO: block shaping based on params["divide_block"], self.minimum_block_length,
        # and self.block_base. But we must care about concat and repeat above;
        # it cannot be simply copied from generator_kernel.build_blocks.

        if params.get("pulse", {}).get("invertY", False):
            out = K.invert_y_phase(out)
        out = K.encode_mw_phase(out, params, self.mw_modes, num_mw, self.iq_amplitude)

        mw_offset_ticks = int(round(params.get("mw_offset", 0.0) * freq))
        if mw_offset_ticks:
            out = K.apply_mw_offset(out, mw_offset_ticks)

        if self.channel_remap is not None:
            out = out.replace(self.channel_remap)

        trace_length_ticks = roi_head + laser_width + roi_tail
        if not self.check_sample_duration(trace_length_ticks):
            raise ValueError(
                "trace window overlaps the next trigger; reduce margins or pulse rate"
            )

        return out.simplify(), laser_timing, trigger_timing, trace_length_ticks

    def check_sample_duration(self, trace_length_ticks) -> bool:
        if not self.all_trigger_timing:
            raise ValueError("check_sample_duration is called but all_trigger_timing is not set.")

        for t0, t1 in zip(self.all_trigger_timing, self.all_trigger_timing[1:]):
            if t1 - t0 < trace_length_ticks + self.eos_deadtime_ticks:
                return False
        return True


class APODMRPulserBase(CommonPulserBase):
    def __init__(self, cli, logger, conf: dict):
        super().__init__(cli, logger, conf)

        self._pd_spectrum = self.conf.get("pd_spectrum", False)
        if self._pd_spectrum:
            self.clock = None
        else:
            self.clock = ClockSourceInterface(cli, self.conf.get("clock_name", "clock"))
        self.pd_names = self.conf.get("pd_names", ["pd0"])
        self.pds = [PDInterface(cli, n) for n in self.pd_names]
        self.add_instruments(self.clock, *self.pds)

        self.check_required_conf(["pd_trigger"])
        self._pd_trigger = self.conf["pd_trigger"]
        self._pd_data_transfer = self.conf.get("pd_data_transfer")

        self.builder = APODMRBlockBuilder(
            self.conf["minimum_block_length"],
            self.conf["block_base"],
            self.mw_modes,
            self.conf.get("iq_amplitude", 0.0),
            self.conf.get("channel_remap"),
        )

        self.data = APODMRData()
        self.op = APODMRDataOperator()

        self.trace_count = None
        self.samples_per_trace = None

        self._analysis_warned = False
        self._acquisition_failed = False

    def _pd_rate_param(self):
        rate = self.conf.get("pd_rate", 2e6)
        if isinstance(rate, (tuple, list)):
            rates = self._conf_pos_integers("pd_rate")
            if not rates:
                raise V.ValidationError("pd_rate must not be empty.")
            return P.IntChoiceParam(rates[0], rates, doc="PD sampling rate")
        elif isinstance(rate, (int, np.integer)):
            rate = self._conf_pos_int("pd_rate", int(rate))
            return P.IntParam(
                rate, 1e3, 1e9, unit="Hz", SI_prefix=True, digit=9, doc="PD sampling rate"
            )
        elif isinstance(rate, (float, np.floating)):
            rate = self._conf_pos_float("pd_rate", float(rate))
            return P.FloatParam(
                rate, 1e3, 1e9, unit="Hz", SI_prefix=True, digit=9, doc="PD sampling rate"
            )
        V.check_num(rate, "pd_rate")
        raise AssertionError("unreachable")

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        d = super().get_param_dict(label)
        if d is None:
            return None
        if "head" in d["plot"]["taumode"].options():
            taumodes = tuple(m for m in d["plot"]["taumode"].options() if m != "head")
            d["plot"]["taumode"] = P.StrChoiceParam("raw", taumodes)

        # Since divide_block is TODO and not implemented, default it to False anyway.
        if "divide_block" in d:
            d["divide_block"] = P.BoolParam(False)
        # remove unused params
        if "timebin" in d:
            del d["timebin"]
        if "interval" in d:
            del d["interval"]
        if "multi_histogram" in d:
            del d["multi_histogram"]

        d["hardware_sweep_limit"] = P.BoolParam(
            False,
            doc=(
                "stop detector acquisition at the exact sweep limit in hardware; requires "
                "sweeps divisible by sweeps_per_record; acquisition is aborted if the detector "
                "software queue discards a record"
            ),
        )

        # set defaults which won't result in marker out-of-bounds
        d["plot"]["sigdelay"].set(200e-9)
        d["plot"]["sigwidth"].set(300e-9)
        d["plot"]["refdelay"].set(1000e-9)
        d["plot"]["refwidth"].set(500e-9)

        d["roi_head"] = P.FloatParam(
            self._conf_nonneg_num("roi_head", 20e-9),
            0.0,
            10e6,
            unit="s",
            doc="margin at head of sampled trace and trigger-to-laser offset",
        )
        d["roi_tail"] = P.FloatParam(
            self._conf_nonneg_num("roi_tail", 100e-9),
            0.0,
            10e6,
            unit="s",
            doc="margin at tail of sampled trace",
        )
        d["sweeps_per_record"] = P.IntParam(
            self._conf_pos_int("sweeps_per_record", 10),
            1,
            1000000,
            doc="number of sweeps accumulated in one stored raw trace record",
        )
        d["max_records"] = P.IntParam(
            self._conf_nonneg_int("max_records", 1),
            0,
            100000000,
            doc="maximum number of raw trace records to retain (0 for unlimited)",
        )
        d["save_history"] = P.BoolParam(
            True,
            doc="retain signal and reference window means for every record",
        )
        d["burst_num"] = P.IntParam(
            self._conf_pos_int("burst_num", 1),
            1,
            1000000,
            doc="number of burst (repeated shots per sweep point)",
        )
        d["point_init_delay"] = P.FloatParam(
            self._conf_nonneg_num("point_init_delay", 0.0),
            0.0,
            1.0,
            unit="s",
            SI_prefix=True,
            doc="dark delay before the initialization laser for each sweep point",
        )
        d["pd"] = P.ParamDict()
        d["pd"]["rate"] = self._pd_rate_param()
        lb, ub = self._conf_ascending_numbers("pd_bounds", 2, (-10.0, 10.0))
        d["pd"]["buffer_size_coeff"] = P.IntParam(
            self._conf_pos_int("buffer_size_coeff", 20),
            1,
            10_000,
            doc="ratio of requested buffer size to record data size",
        )
        d["pd"]["bounds"] = [
            P.FloatParam(lb, -10.0, +10.0, doc="PD voltage lower bound"),
            P.FloatParam(ub, -10.0, +10.0, doc="PD voltage upper bound"),
        ]
        d["pd"]["drop_first"] = P.IntParam(0, 0, 100, doc="drop first N records to stabilize")
        if self._pd_spectrum:
            d["pd"]["hardware_average"] = P.BoolParam(True, doc="use hardware block averaging")
        d["pd"]["eos_deadtime"] = P.FloatParam(
            200e-9, 0.0, 1.0, unit="s", SI_prefix=True, doc="end-of-sample deadtime"
        )

        return d

    def _sweeps_per_record(self, params: dict) -> int:
        return int(params.get("sweeps_per_record", 1))

    def _validate_sweep_params(self, params: dict) -> bool:
        sweeps = int(params.get("sweeps", 0))
        sweeps_per_record = self._sweeps_per_record(params)
        if sweeps_per_record < 1:
            self.logger.error(
                f"sweeps_per_record must be at least 1: sweeps_per_record={sweeps_per_record}"
            )
            return False
        if params.get("hardware_sweep_limit", False) and sweeps > 0 and sweeps % sweeps_per_record:
            self.logger.error(
                "sweeps must be an integer multiple of sweeps_per_record: "
                f"sweeps={sweeps}, sweeps_per_record={sweeps_per_record}"
            )
            return False
        return True

    def _remaining_records(self, params: dict, records: int) -> int | None:
        sweeps = int(params.get("sweeps", 0))
        if not params.get("hardware_sweep_limit", False) or sweeps == 0:
            return None
        target_records = sweeps // self._sweeps_per_record(params)
        remaining_records = target_records - int(records)
        if remaining_records <= 0:
            self.logger.error(
                "finite sweep target has already been reached: "
                f"target_records={target_records}, records={records}"
            )
            return 0
        return remaining_records

    def _burst_num(self, params: dict) -> int:
        return int(params.get("burst_num", 1))

    def generate_blocks(self, data: APODMRData | None = None):
        if data is None:
            data = self.data
        self._set_num_pattern_and_validate_params(data)
        generator = self.generators[data.label]
        params = data.get_params()
        raw_blocks, freq, common_pulses = generator.generate_raw_blocks(data.xdata, params)
        blocks, laser_timing, trigger_timing, trace_length_ticks = self.builder.build_blocks(
            raw_blocks, freq, common_pulses, params, generator.num_mw()
        )
        return blocks, freq, laser_timing, trigger_timing, trace_length_ticks

    def _init_inst(self, params: dict) -> bool:
        if not self._init_generator_inst(params):
            return False
        if not self._init_fg(params):
            self.logger.error("Error initializing FG.")
            return False
        return True

    def _samples_per_trace(
        self, pd_rate: float, trace_length_ticks: int, freq: float
    ) -> int | None:
        samples_per_trace = max(1, int(round(trace_length_ticks / freq * pd_rate)))
        if not self._pd_spectrum:
            return samples_per_trace

        granularity = self._conf_pos_int("pd_segment_granularity", 16)
        offset = self._conf_int("pd_segment_offset", 0)
        try:
            adjusted = round_segment_samples_up(samples_per_trace, granularity=granularity)
            adjusted -= offset
        except ValueError:
            self.logger.exception("failed to round samples_per_trace to segment granularity")
            return None
        if adjusted <= 0:
            self.logger.error(
                "adjusted samples_per_trace is too short; increase laser_width or ROI window"
            )
            return None
        if adjusted != samples_per_trace:
            self.logger.info(
                "PD samples_per_trace adjusted: "
                f"{samples_per_trace} to {adjusted} (granularity = {granularity} "
                f"offset = {offset})"
            )
        samples_per_trace = adjusted
        trace_length_ticks = int(np.ceil(samples_per_trace * freq / pd_rate))
        if not self.builder.check_sample_duration(trace_length_ticks):
            self.logger.error(
                "trace window overlaps the next trigger; reduce margins or pulse rate"
            )
            return None
        return samples_per_trace

    def init_start_pds(self, remaining_records: int | None) -> bool:
        params = self.data.get_params()
        rate = params["pd"]["rate"]
        sweeps_per_record = self._sweeps_per_record(params)
        burst_num = self._burst_num(params)

        params_clock = {
            "freq": rate,
            "samples": self.samples_per_trace,
            "finite": True,
            "trigger_source": self._pd_trigger,
            "trigger_dir": True,
            "retriggerable": True,
        }
        if self.clock is None:
            clock_pd = self._pd_trigger
        else:
            if not self.clock.configure(params_clock):
                self.logger.error("failed to configure clock.")
                return False
            clock_pd = self.clock.get_internal_output()

        cb_samples = self.trace_count * self.samples_per_trace
        drop_first = params["pd"].get("drop_first", 0)
        finite = remaining_records is not None
        buffer_size = cb_samples * params["pd"].get(
            "buffer_size_coeff", self.conf.get("buffer_size_coeff", 20)
        )
        samples = cb_samples * (remaining_records + drop_first) if finite else buffer_size
        if not (
            all(
                [
                    pd.configure_triggered(
                        clock_pd,
                        cb_samples,
                        samples,
                        rate,
                        buffer_size=buffer_size,
                        finite=finite,
                        drop_first=drop_first,
                        oversample=1,
                        block_samples=self.samples_per_trace,
                        block_reduce_factor=burst_num,
                        block_reduce_op="mean",
                        reduce_factor=sweeps_per_record,
                        reduce_op="mean",
                        bounds=params["pd"].get("bounds", (-10.0, 10.0)),
                        data_transfer=self._pd_data_transfer,
                        hardware_average=params["pd"].get("hardware_average", True),
                    )
                    for pd in self.pds
                ]
            )
            and (self.clock is None or self.clock.start())
            and all([pd.start() for pd in self.pds])
        ):
            self.logger.error("Error starting PDs.")
            return False

        return True

    def _reshape_sweep(self, line: np.ndarray) -> np.ndarray:
        expected = self.trace_count * self.samples_per_trace
        if len(line) != expected:
            raise ValueError(
                f"Unexpected PD record size {len(line)} for trace_count={self.trace_count}, "
                f"samples_per_trace={self.samples_per_trace}"
            )
        traces = line.reshape(self.trace_count, self.samples_per_trace)
        return traces

    def update(self) -> bool:
        if not self.data.running:
            return False

        hardware_limited = bool(
            self.data.params.get("hardware_sweep_limit", False)
            and int(self.data.params.get("sweeps", 0)) > 0
        )
        lines = []
        for name, pd in zip(self.pd_names, self.pds):
            if hardware_limited:
                ls, overflowed = pd.pop_block_with_status()
                if overflowed:
                    self.logger.error(
                        f"PD queue overflowed during finite acquisition ({name}); "
                        "acquired records were dropped."
                    )
                    self._acquisition_failed = True
                    return False
            else:
                ls = pd.pop_block()
            if isinstance(ls, list):
                # PD has multi channel
                lines.extend(ls)
            else:
                # single channel, assume ls is np.ndarray
                lines.append(ls)

        line = np.sum(lines, axis=0)
        traces = self._reshape_sweep(line)
        self.op.append_record(self.data, traces)
        self.op.get_marker_indices(self.data)
        error = self.op.analyze_with_error(self.data)
        if error is not None:
            if not self._analysis_warned:
                self.logger.warn(f"Skipping APODMR analysis: {error}")
                self._analysis_warned = True
            return True
        self._analysis_warned = False
        return True

    def update_plot_params(self, params: dict) -> bool:
        if self.data.params is None:
            return False
        if "plotmode" in params:
            try:
                options = self._plotmode_options(self.data.label)
            except ValueError as e:
                self.logger.error(f"Cannot validate plotmode for {self.data.label}: {e}")
                return False
            if params["plotmode"] not in options:
                self.logger.error(
                    f"Unknown plotmode '{params['plotmode']}' for method '{self.data.label}'. "
                    f"available: {options}"
                )
                return False
        if self.op.update_plot_params(self.data, params):
            self.data.remove_fit_data()
        if not self.data.running:
            self.op.get_marker_indices(self.data)
            error = self.op.analyze_with_error(self.data)
            if error is not None:
                self.logger.warn(f"Cannot analyze APODMR data with current plot params: {error}")
        return True

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        self._acquisition_failed = False
        if params is not None:
            params = P.unwrap(params)
        resume = params is None or ("resume" in params and params["resume"])
        if params is None:
            quick_resume = resume and self._quick_resume
        else:
            quick_resume = resume and params.get("quick_resume", self._quick_resume)
        if not resume:
            self.data = APODMRData(params, label)
            self.op.update_axes(self.data)
        else:
            self.op.update_params(self.data, params)
        if not self._validate_sweep_params(self.data.params):
            return False
        remaining_records = self._remaining_records(self.data.params, self.data.records)
        if remaining_records == 0:
            return False
        try:
            self._set_num_pattern_and_validate_params(self.data)
        except ValueError as e:
            self.logger.error(f"Invalid params for {label}: {e}")
            return False

        self._analysis_warned = False
        sweeps_limit = self.data.params.get("sweeps", 0)
        sweeps_per_record = self._sweeps_per_record(self.data.params)
        if sweeps_limit > 0 and sweeps_limit % sweeps_per_record:
            self.logger.warn(
                "sweeps is not divisible by sweeps_per_record; measurement will stop on the "
                "next completed record boundary. Enable hardware_sweep_limit to require an "
                "exact limit."
            )
        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if quick_resume:
            self.logger.info("Quick resume enabled: skipping initial inst configurations.")
        if not quick_resume and not self._init_inst(self.data.params):
            return self.fail_with_release("Error initializing instruments.")
        # PD/clock configuration is always refreshed on each start, even with quick resume.
        if not self.init_start_pds(remaining_records):
            return self.fail_with_release("Error initializing or starting PDs.")

        if not self._start_inst():
            return False

        if resume:
            self.data.resume()
            self.logger.info("Resumed pulser.")
        else:
            self.data.start()
            self.logger.info("Started pulser.")
        return True

    def stop(self) -> bool:
        if not self.data.running:
            return False

        success = self._stop_generator_inst()
        success &= all([pd.stop() for pd in self.pds]) and all([pd.release() for pd in self.pds])
        if self.clock is not None:
            success &= self.clock.stop() and self.clock.release()
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(False)
        if self.fg is not None:
            success &= self.fg.release()

        self.data.finalize()
        if success:
            self.logger.info("Stopped pulser.")
        else:
            self.logger.error("Error stopping pulser.")
        return success

    def is_finished(self) -> bool:
        if not self.data.has_params():
            return False
        if self._acquisition_failed:
            return True
        if (
            self.data.params.get("sweeps", 0) > 0
            and self.data.sweeps() >= self.data.params["sweeps"]
        ):
            return True
        if (
            self.data.params.get("duration", 0.0) > 0.0
            and self.data.measurement_time() >= self.data.params["duration"]
        ):
            return True
        return False

    def work(self):
        return self.update()

    def data_msg(self) -> APODMRData:
        return self.data


class Pulser(SGPGPulserBase, APODMRPulserBase):
    """Worker for APODMR using SG + PG signal source.

    :param pulser.start_delay: (sec.) delay before starting PG output. (default: 0.5)
    :type pulser.start_delay: float
    :param pulser.quick_resume: default value of quick_resume.
        If True, it skips instrument configurations on resume.
    :type pulser.quick_resume: bool
    :param pulser.mw_modes: mw phase control modes for each channel.
        QPSK (0) is 4-phase control using IQ modulation at SG and a switch.
        Ext2Phase (1) is 2-phase control using external 90-deg splitter and two switches.
        ArbPhase (2) is arbitrary phase control using IQ modulation at SG
        (Analog output (AWG) is required for PG).
    :type pulser.mw_modes: tuple[str | int]
    :param pulser.iq_amplitude: (only for mw_mode ArbPhase (2)) amplitude of analog IQ signal in V.
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
    :param pulser.mw_channels: Optional SG channel identifiers for MW outputs.
        The elements should have form {"sg": "sg1", ch: 1}.
    :type pulser.mw_channels: list[dict[str, str | int]]
    :param pulser.generators: Optional user generator registry mapping method labels to
        ``[module_name, class_name]``.
        These classes are loaded at worker initialization and can add or override methods.
    :type pulser.generators: dict[str, tuple[str, str]]

    :param pulser.pd_names: (default: ["pd0"]) PD names in target.servers.
    :type pulser.pd_names: list[str]
    :param pulser.clock_name: (default: ``"clock"``) Clock source instrument name.
    :type pulser.clock_name: str
    :param pulser.pd_trigger: DAQ terminal name for PD trigger.
    :type pulser.pd_trigger: str
    :param pulser.pd_data_transfer: Optional DAQ transfer mode label.
    :type pulser.pd_data_transfer: str
    :param pulser.pd_spectrum: (default: False) set True if PD is Spectrum_AnalogIn-based.
    :type pulser.pd_spectrum: bool
    :param pulser.pd_segment_granularity: (default: 16) logical post-trigger sample
        granularity for Spectrum PD segments.
    :type pulser.pd_segment_granularity: int
    :param pulser.pd_segment_offset: (default: 0) sample offset subtracted after Spectrum
        segment rounding.
    :type pulser.pd_segment_offset: int
    :param pulser.buffer_size_coeff: Buffer size coefficient multiplied by trace length.
    :type pulser.buffer_size_coeff: int
    :param pulser.roi_head: (default: 20e-9) default margin at head of sampled trace
        and trigger-to-laser offset.
    :type pulser.roi_head: float
    :param pulser.roi_tail: (default: 100e-9) default margin at tail of sampled trace.
    :type pulser.roi_tail: float
    :param pulser.sweeps_per_record: (default: 10) default number of sweeps accumulated in one
        stored raw trace record.
    :type pulser.sweeps_per_record: int
    :param pulser.burst_num: (default: 1) default number of burst (repeated shots per sweep point).
    :type pulser.burst_num: int
    :param pulser.max_records: (default: 1) default maximum number of retained raw trace records.
        Set to 0 for unlimited retention.
    :type pulser.max_records: int
    :param pulser.point_init_delay: (default: 0.0) default dark delay before the initialization
        laser for each sweep point. A positive value enables per-point initialization blocks.
    :type pulser.point_init_delay: float
    :param pulser.pd_rate: (default: 2e6) default PD sampling rate in Hz.
    :type pulser.pd_rate: float
    :param pulser.pd_bounds: (default: ``(-10.0, 10.0)``) default PD voltage bounds.
    :type pulser.pd_bounds: tuple[float, float]

    """

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        params = P.unwrap(params)
        if not self._validate_sweep_params(params):
            return False
        d = APODMRData(params, label)
        try:
            blocks, freq, _, _, trace_length_ticks = self.generate_blocks(d)
            pd_rate = params["pd"]["rate"]
        except (ValueError, KeyError) as e:
            self.logger.error(f"Invalid params for {label}: {e}")
            return False
        if self._samples_per_trace(pd_rate, trace_length_ticks, freq) is None:
            return False
        offsets = self.pg.validate_blocks(blocks, freq)
        return offsets is not None

    def _init_pg(self, params: dict) -> bool:
        if not (self.pg.stop() and self.pg.clear()):
            self.logger.error("Error stopping PG.")
            return False

        try:
            blocks, self.freq, laser_timing, trigger_timing, trace_length_ticks = (
                self.generate_blocks()
            )
        except (ValueError, KeyError):
            self.logger.exception(f"Invalid params for {self.data.label}")
            return False

        pd_rate = params["pd"]["rate"]
        sample_period = 1.0 / pd_rate
        spt = self._samples_per_trace(pd_rate, trace_length_ticks, self.freq)
        if spt is None:
            return False
        else:
            self.samples_per_trace = spt
        self.trace_count = len(laser_timing)
        self.op.set_laser_timing(self.data, np.array(laser_timing) / self.freq)
        self.op.set_trace_laser_timing(self.data, params["roi_head"])
        self.op.set_trigger_timing(self.data, np.array(trigger_timing) / self.freq)
        self.pulse_pattern = PulsePattern(blocks, self.freq, markers=trigger_timing)

        if not (self.pg.configure_blocks(blocks, self.freq) and self.pg.get_opc()):
            self.logger.error("Error configuring PG.")
            return False

        self.length = self.pg.get_length()
        self.offsets = self.pg.get_offsets()
        self.op.set_instrument_params(
            self.data,
            self.samples_per_trace,
            sample_period,
            self.freq,
            self.length,
            self.offsets,
            pd_rate,
            self.mw_modes,
        )
        return True

    def _start_inst(self) -> bool:
        success = self._start_sg(self.data.params)
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)

        time.sleep(self._start_delay)

        if success and self.pg.start():
            return True

        # fail: stop and release everything
        self.pg.stop()
        if self._fg_enabled(self.data.params):
            self.fg.set_output(False)
        self._stop_sg()
        for pd in self.pds:
            pd.stop()
        if self.clock is not None:
            self.clock.stop()
        return self.fail_with_release("Error starting pulser.")

    def _stop_generator_inst(self) -> bool:
        return self.pg.stop() and self.pg.release() and self._stop_sg()


class AWGPulser(AWGPulserBase, APODMRPulserBase):
    """Worker for APODMR using AWG as signal source.

    :param pulser.start_delay: (sec.) delay before starting AWG output. (default: 0.5)
    :type pulser.start_delay: float
    :param pulser.quick_resume: default value of quick_resume.
        If True, it skips instrument configurations on resume.
    :type pulser.quick_resume: bool
    :param pulser.split_fraction: (default: 4) fraction factor (F) to split the free period
        for MW phase modulation. the period (T) is split into (T // F, T - T // F) and MW phase
        is switched at T // F. Thus, larger F results in "quicker start" of the phase
        modulation (depending on hardware, but its response may be a bit slow).
    :type pulser.split_fraction: int
    :param pulser.minimum_block_length: (has preset) minimum block length in generated blocks
    :type pulser.minimum_block_length: int
    :param pulser.block_base: (has preset) block base granularity of pulse generator.
    :type pulser.block_base: int
    :param pulser.divide_block: (has preset) Default value of divide_block.
    :type pulser.divide_block: bool
    :param pulser.awg_rate: default value of AWG sampling frequency
    :type pulser.awg_rate: float | int | list[int]
    :param pulser.file_transport_dir: (optional) writer-side directory for shared HDF5 waveform
        files.
    :type pulser.file_transport_dir: str
    :param pulser.remove_awg_file: (default: True) remove each HDF5 transport file after the
        synchronous AWG configure attempt.
    :type pulser.remove_awg_file: bool
    :param pulser.awg_freq: default value of AWG-generated MW frequency
    :type pulser.awg_freq: float
    :param pulser.awg_monitor_max_samples: (default: 10_000_000) maximum source prefix length
        inspected for the AWG monitor preview.
    :type pulser.awg_monitor_max_samples: int
    :param pulser.awg_monitor_max_points: (default: 500_000) maximum reduced analog points
        published per AWG channel.
    :type pulser.awg_monitor_max_points: int
    :param pulser.channel_remap: mapping to fix default channel names.
    :type pulser.channel_remap: dict[str | int, str | int]
    :param pulser.mw_channels: Mapping from MW tone source index (``freq``, ``freq1``, ...)
        to physical AWG channel. Keys must be contiguous starting at 0. Up to two distinct
        physical channels are supported.
    :type pulser.mw_channels: dict[int, int]
    :param pulser.generators: Optional user generator registry mapping method labels to
        ``[module_name, class_name]``.
        These classes are loaded at worker initialization and can add or override methods.
    :type pulser.generators: dict[str, tuple[str, str]]

    :param pulser.pd_names: (default: ["pd0"]) PD names in target.servers.
    :type pulser.pd_names: list[str]
    :param pulser.clock_name: (default: ``"clock"``) Clock source instrument name.
    :type pulser.clock_name: str
    :param pulser.pd_trigger: DAQ terminal name for PD trigger.
    :type pulser.pd_trigger: str
    :param pulser.pd_data_transfer: Optional DAQ transfer mode label.
    :type pulser.pd_data_transfer: str
    :param pulser.pd_spectrum: (default: False) set True if PD is Spectrum_AnalogIn-based.
    :type pulser.pd_spectrum: bool
    :param pulser.pd_segment_granularity: (default: 16) logical post-trigger sample
        granularity for Spectrum PD segments.
    :type pulser.pd_segment_granularity: int
    :param pulser.pd_segment_offset: (default: 0) sample offset subtracted after Spectrum
        segment rounding.
    :type pulser.pd_segment_offset: int
    :param pulser.buffer_size_coeff: Buffer size coefficient multiplied by trace length.
    :type pulser.buffer_size_coeff: int
    :param pulser.roi_head: (default: 20e-9) default margin at head of sampled trace
        and trigger-to-laser offset.
    :type pulser.roi_head: float
    :param pulser.roi_tail: (default: 100e-9) default margin at tail of sampled trace.
    :type pulser.roi_tail: float
    :param pulser.sweeps_per_record: (default: 10) default number of sweeps accumulated in one
        stored raw trace record.
    :type pulser.sweeps_per_record: int
    :param pulser.burst_num: (default: 1) default number of burst (repeated shots per sweep point).
    :type pulser.burst_num: int
    :param pulser.max_records: (default: 1) default maximum number of retained raw trace records.
        Set to 0 for unlimited retention.
    :type pulser.max_records: int
    :param pulser.point_init_delay: (default: 0.0) default dark delay before the initialization
        laser for each sweep point. A positive value enables per-point initialization blocks.
    :type pulser.point_init_delay: float
    :param pulser.pd_rate: (default: 2e6) default PD sampling rate in Hz.
    :type pulser.pd_rate: float
    :param pulser.pd_bounds: (default: ``(-10.0, 10.0)``) default PD voltage bounds.
    :type pulser.pd_bounds: tuple[float, float]

    """

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        params = P.unwrap(params)
        awg_bounds = self._get_awg_bounds()
        if awg_bounds is None:
            return False
        if not self._validate_sweep_params(params):
            return False
        if not self._validate_file_transport(params, awg_bounds):
            return False
        d = APODMRData(params, label)
        try:
            self.make_generators(self.awg.get_digital_rate(params["awg"]["rate"]))
            blocks, freq, _, _, trace_length_ticks = self.generate_blocks(d)
            pd_rate = params["pd"]["rate"]
        except (ValueError, KeyError):
            self.logger.exception(f"Invalid params for {label}")
            return False
        if self._samples_per_trace(pd_rate, trace_length_ticks, freq) is None:
            return False
        try:
            num_mw = self.generators[label].num_mw()
            tones = self.make_mw_tones(num_mw, params)
            ret = self.renderer.render(blocks, freq, tones, num_mw, params, awg_bounds)
            self.logger.info(f"AWG rendering success: {ret}")
            return True
        except (ValueError, KeyError):
            self.logger.exception(f"Invalid params for {label}")
            return False

    def _init_generator_inst(self, params: dict) -> bool:
        self.awg_waveform = None
        awg_bounds = self._get_awg_bounds()
        if awg_bounds is None:
            return False
        if not self._validate_file_transport(params, awg_bounds):
            return False
        if not self.awg.stop():
            self.logger.error("Error stopping AWG.")
            return False
        try:
            self.make_generators(self.awg.get_digital_rate(params["awg"]["rate"]))
            blocks, self.freq, laser_timing, trigger_timing, trace_length_ticks = (
                self.generate_blocks()
            )
        except (ValueError, KeyError):
            self.logger.exception(f"Invalid params for {self.data.label}")
            return False
        try:
            num_mw = self.generators[self.data.label].num_mw()
            tones = self.make_mw_tones(num_mw, params)
            self.renderer.render(blocks, self.freq, tones, num_mw, params, awg_bounds)
            self.logger.info("AWG rendering success.")
        except (ValueError, KeyError):
            self.logger.exception(f"Invalid params for {self.data.label}")
            return False

        pd_rate = params["pd"]["rate"]
        sample_period = 1.0 / pd_rate
        spt = self._samples_per_trace(pd_rate, trace_length_ticks, self.freq)
        if spt is None:
            return False
        else:
            self.samples_per_trace = spt
        self.trace_count = len(laser_timing)
        self.op.set_laser_timing(self.data, np.array(laser_timing) / self.freq)
        self.op.set_trace_laser_timing(self.data, params["roi_head"])
        self.op.set_trigger_timing(self.data, np.array(trigger_timing) / self.freq)
        pulse_blocks = remove_analog_channels(blocks, blocks.analog_channels())
        self.pulse_pattern = PulsePattern(pulse_blocks, self.freq, markers=trigger_timing)

        if not self.renderer.upload(file_transport=params["awg"].get("file_transport", False)):
            return False
        self.awg_waveform = self.renderer.waveform_msg(
            trigger_timing,
            self.freq,
            max_samples=self._awg_monitor_max_samples,
            max_points=self._awg_monitor_max_points,
        )
        inst_extra = {"awg": self.renderer.get_meta_data()}

        # set ideal PG-specific meta data
        self.length = blocks.total_length()
        self.offsets = []
        self.op.set_instrument_params(
            self.data,
            self.samples_per_trace,
            sample_period,
            self.freq,
            self.length,
            self.offsets,
            pd_rate,
            self.mw_modes,
            inst_extra,
        )
        return True

    def _start_inst(self) -> bool:
        if self._fg_enabled(self.data.params):
            success = self.fg.set_output(True)
        else:
            success = True

        time.sleep(self._start_delay)

        if success and self.awg.start():
            return True

        # fail: stop and release everything
        self.awg.stop()
        if self._fg_enabled(self.data.params):
            self.fg.set_output(False)
        for pd in self.pds:
            pd.stop()
        if self.clock is not None:
            self.clock.stop()
        return self.fail_with_release("Error starting pulser.")
