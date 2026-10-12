#!/usr/bin/env python3

"""
Worker for Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import typing as T
import time
import copy
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import isotonic_regression

from mahos.util.timer import IntervalTimer
from mahos_dq.msgs.podmr_msgs import (
    PODMRData,
    TDCStatus,
    is_sweepN,
    is_CPlike,
    is_correlation,
    MWMode,
    TimingInfo,
)
from mahos.msgs.inst.pg_msgs import AnalogChannel, PulsePattern
from mahos.msgs.inst.tdc_msgs import ChannelStatus
from mahos.msgs import param_msgs as P
from mahos.inst.sg_interface import SGInterface
from mahos.inst.pg_interface import PGInterface
from mahos.inst.tdc_interface import TDCInterface
from mahos.inst.awg_interface import AWGInterface
from mahos.inst.fg_interface import FGInterface
from mahos.util.conf import ConfAccessorMixin, PresetLoader
from mahos.util.param import ParamAccessor
import mahos.util.validation as V
from mahos.meas.common_worker import Worker
from mahos.node.log import DummyLogger

from mahos_dq.meas.podmr_generator.generator import make_generators
from mahos_dq.meas.awg_renderer import AWGRenderer
from mahos_dq.meas.awg_renderer.renderer_kernel import MWTone


def remove_analog_channels(blocks, channel_names: T.Iterable[str]):
    """Return blocks without the named AnalogChannels, preserving all other channels."""

    names = set(channel_names)

    def remove(channels):
        return tuple(
            channel
            for channel in channels
            if not isinstance(channel, AnalogChannel) or channel.name() not in names
        )

    return blocks.apply(remove)


@dataclass(frozen=True)
class LaserTimingParams:
    """Parameters for laser edge timing detection."""

    scope: tuple[float, float]
    smooth_window: int = 5
    fraction: float = 0.5
    monotonic: bool = True

    def validate(self) -> bool:
        """Return True if this config is valid."""

        return self.smooth_window >= 1 and 0.0 < self.fraction < 1.0


@dataclass(frozen=True)
class LaserTimingResult:
    """Result of laser edge timing detection."""

    success: bool
    offsets: np.ndarray | None = None


class LaserTimingDetector(object):
    """Detector for laser edge timing and offset fitting."""

    def __init__(self, logger=None):
        if logger is None:
            self.logger = DummyLogger()
        else:
            self.logger = logger

    @staticmethod
    def _odd_window_length(length: int, smooth_window: int) -> int:
        if length <= 1:
            return 1
        window = max(1, int(smooth_window))
        window = min(window, length)
        if window % 2 == 0:
            window = max(1, window - 1)
        return window

    @staticmethod
    def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return data
        kernel = np.full(window, 1.0 / window, dtype=np.float64)
        return np.convolve(data, kernel, mode="same")

    def _estimate_edge_constant_fraction(
        self, seg: np.ndarray, smooth_window: int, fraction: float
    ) -> float | None:
        """Estimate rising edge timing in seg and return edge index in float."""

        seg = np.asarray(seg, dtype=np.float64)
        if seg.ndim != 1 or len(seg) < 3:
            return None

        smooth = self._moving_average(seg, self._odd_window_length(len(seg), smooth_window))

        q = max(1, len(smooth) // 4)
        baseline = float(np.median(smooth[:q]))
        level = float(np.median(smooth[-q:]))
        amp = level - baseline
        if not np.isfinite(amp) or amp <= 0.0:
            return None

        threshold = baseline + fraction * amp
        crossings = np.flatnonzero((smooth[:-1] < threshold) & (smooth[1:] >= threshold))
        if len(crossings):
            i0 = int(crossings[0])
            i1 = i0 + 1
        else:
            # fallback to the closest sample to threshold
            i1 = int(np.argmin(np.abs(smooth - threshold)))
            if i1 <= 0:
                i0, i1 = 0, 1
            else:
                i0 = i1 - 1

        y0, y1 = smooth[i0], smooth[i1]
        if not np.isfinite(y0) or not np.isfinite(y1):
            return None
        if y1 == y0:
            edge = float(i1)
        else:
            # sub-bin linear interpolation
            frac = (threshold - y0) / (y1 - y0)
            edge = float(i0) + float(np.clip(frac, 0.0, 1.0))

        return edge

    def _fit_monotonic_offsets(self, offsets: np.ndarray) -> np.ndarray:
        """Fit monotonic drift (increasing or decreasing) by isotonic regression."""

        offsets = np.asarray(offsets, dtype=np.float64)
        inc = isotonic_regression(offsets, increasing=True).x
        dec = isotonic_regression(offsets, increasing=False).x

        err_inc = np.sum(np.square(offsets - inc))
        err_dec = np.sum(np.square(offsets - dec))
        return dec if err_dec < err_inc else inc

    def _finalize_offsets(self, offsets: np.ndarray, monotonic: bool) -> np.ndarray | None:
        valid = np.isfinite(offsets)
        if not np.any(valid):
            return None

        if not np.all(valid):
            idx = np.arange(len(offsets))
            if np.count_nonzero(valid) == 1:
                offsets = np.full_like(offsets, offsets[valid][0], dtype=np.float64)
            else:
                offsets = np.interp(idx, idx[valid], offsets[valid])

        if monotonic and len(offsets) > 1:
            offsets = self._fit_monotonic_offsets(offsets)
        return offsets

    def detect_roi(
        self,
        traces: np.ndarray,
        rois: list[tuple[int, int]],
        laser_timing: np.ndarray,
        tbin: float,
        config: LaserTimingParams,
    ) -> LaserTimingResult:
        """Detect laser timing offsets in ROI mode."""

        head, tail = config.scope
        if traces.ndim != 2 or len(traces) != len(laser_timing):
            self.logger.error(
                "FindLaserTiming (ROI): invalid trace shape. "
                f"ndim={traces.ndim}, traces={len(traces)}, laser_timing={len(laser_timing)}"
            )
            return LaserTimingResult(False)

        starts = np.round((laser_timing - head) / tbin).astype(int)
        stops = np.round((laser_timing + tail) / tbin).astype(int)

        offsets = np.full(len(laser_timing), np.nan, dtype=np.float64)
        short_segments = 0
        no_edge = 0

        for i, (timing, start, stop, (roi_start, _), trace) in enumerate(
            zip(laser_timing, starts, stops, rois, traces)
        ):
            local_start = max(0, start - roi_start)
            local_stop = min(len(trace), stop - roi_start)
            if local_stop - local_start < 3:
                short_segments += 1
                continue

            edge = self._estimate_edge_constant_fraction(
                trace[local_start:local_stop], config.smooth_window, config.fraction
            )
            if edge is None:
                no_edge += 1
                continue

            offsets[i] = (roi_start + local_start + edge) * tbin - timing

        valid = np.isfinite(offsets)
        valid_num = int(np.count_nonzero(valid))
        if valid_num == 0:
            self.logger.error(
                "FindLaserTiming (ROI): failed to detect any laser edge. "
                f"short_segments={short_segments}, no_edge={no_edge}"
            )
            return LaserTimingResult(False)
        if valid_num < len(offsets):
            self.logger.warn(
                "FindLaserTiming (ROI): partial detection, "
                f"valid={valid_num}/{len(offsets)} (short={short_segments}, no_edge={no_edge}). "
                "Missing offsets will be interpolated."
            )

        offsets = self._finalize_offsets(offsets, config.monotonic)
        if offsets is None:
            self.logger.error("FindLaserTiming (ROI): failed at final offset aggregation.")
            return LaserTimingResult(False)
        return LaserTimingResult(True, offsets=offsets)

    def detect_noroi(
        self, raw: np.ndarray, laser_timing: np.ndarray, tbin: float, config: LaserTimingParams
    ) -> LaserTimingResult:
        """Detect laser timing offsets in no-ROI mode."""

        head, tail = config.scope
        if raw.ndim != 1:
            self.logger.error(f"FindLaserTiming (no-ROI): invalid raw_data ndim={raw.ndim}.")
            return LaserTimingResult(False)

        starts = np.round((laser_timing - head) / tbin).astype(int)
        stops = np.round((laser_timing + tail) / tbin).astype(int)

        offsets = np.full(len(laser_timing), np.nan, dtype=np.float64)
        short_segments = 0
        no_edge = 0

        for i, (start, stop, timing) in enumerate(zip(starts, stops, laser_timing)):
            start = max(0, start)
            stop = min(len(raw), stop)
            if stop - start < 3:
                short_segments += 1
                continue

            edge = self._estimate_edge_constant_fraction(
                raw[start:stop], config.smooth_window, config.fraction
            )
            if edge is None:
                no_edge += 1
                continue

            offsets[i] = (start + edge) * tbin - timing

        valid = np.isfinite(offsets)
        valid_num = int(np.count_nonzero(valid))
        if valid_num == 0:
            self.logger.error(
                "FindLaserTiming (no-ROI): failed to detect any laser edge. "
                f"short_segments={short_segments}, no_edge={no_edge}"
            )
            return LaserTimingResult(False)
        if valid_num < len(offsets):
            self.logger.warn(
                "FindLaserTiming (no-ROI): partial detection, "
                f"valid={valid_num}/{len(offsets)} (short={short_segments}, no_edge={no_edge}). "
                "Missing offsets will be interpolated."
            )

        offsets = self._finalize_offsets(offsets, config.monotonic)
        if offsets is None:
            self.logger.error("FindLaserTiming (no-ROI): failed at final offset aggregation.")
            return LaserTimingResult(False)
        return LaserTimingResult(True, offsets=offsets)


class PODMRDataOperator(object):
    """Operations (set / get / analyze) on PODMRData."""

    def __init__(self, logger=None):
        if logger is None:
            self.logger = DummyLogger()
        else:
            self.logger = logger
        self.laser_timing_detector = LaserTimingDetector(self.logger)

    def set_laser_timing(self, data: PODMRData, laser_timing):
        if data.laser_timing is not None:
            return
        data.laser_timing = np.array(laser_timing)  # unit is [sec]

    def set_instrument_params(
        self,
        data: PODMRData,
        trange: float,
        tbin: float,
        pg_freq: float,
        length: int,
        offsets: list[int],
        mw_modes: T.Sequence[MWMode],
        extra: dict | None = None,
    ):
        if "instrument" in data.params:
            return
        data.params["instrument"] = {}
        data.params["instrument"]["trange"] = trange
        data.params["instrument"]["tbin"] = tbin
        data.params["instrument"]["pg_freq"] = pg_freq
        data.params["instrument"]["length"] = int(length)
        if all([ofs == 0 for ofs in offsets]):
            data.params["instrument"]["offsets"] = []
        else:
            data.params["instrument"]["offsets"] = offsets
        data.params["instrument"]["mw_modes"] = [MWMode.parse(m).name for m in mw_modes]
        if extra is not None:
            data.params["instrument"].update(extra)

    def update(self, data: PODMRData, data_new, tdc_status):
        data.raw_data = data_new
        data.tdc_status = tdc_status

    def _apply_laser_timing_result(self, data: PODMRData, result: LaserTimingResult) -> bool:
        if not result.success or result.offsets is None:
            return False

        data.laser_timing_offset = result.offsets
        mn, mx = np.min(data.laser_timing_offset) * 1e9, np.max(data.laser_timing_offset) * 1e9
        self.logger.info(
            f"Set laser timing offset: min {mn:.1f}, max {mx:.1f}, delta {mx - mn:.1f} ns"
        )
        return True

    def _find_laser_timing_roi(self, data: PODMRData, config: LaserTimingParams) -> bool:
        tbin = data.get_bin()
        if tbin is None:
            return False

        result = self.laser_timing_detector.detect_roi(
            traces=np.asarray(data.raw_data),
            rois=data.get_rois(),
            laser_timing=np.asarray(data.laser_timing, dtype=np.float64),
            tbin=tbin,
            config=config,
        )
        return self._apply_laser_timing_result(data, result)

    def _find_laser_timing_noroi(self, data: PODMRData, config: LaserTimingParams) -> bool:
        tbin = data.get_bin()
        if tbin is None:
            return False

        result = self.laser_timing_detector.detect_noroi(
            raw=np.asarray(data.raw_data, dtype=np.float64),
            laser_timing=np.asarray(data.laser_timing, dtype=np.float64),
            tbin=tbin,
            config=config,
        )
        return self._apply_laser_timing_result(data, result)

    def find_laser_timing(
        self,
        data: PODMRData,
        scope: tuple[float, float],
        smooth_window: int = 5,
        fraction: float = 0.5,
        monotonic: bool = True,
    ) -> bool:
        """Find actual laser timing from raw data and store the offset in laser_timing_offset.

        :param scope: (head, tail) the scope around laser_timing to look for the laser edge
            in real time unit (sec).

        :param smooth_window: smoothing window length in bins. Even values are rounded down
            to odd so moving-average smoothing stays centered on a bin and avoids half-bin bias.
        :param fraction: constant-fraction level for edge timing (0.0 to 1.0).
        :param monotonic: if True, fit offsets with monotonic drift constraint.

        """

        config = LaserTimingParams(scope, smooth_window, fraction, monotonic)
        if not config.validate():
            return False
        if not data.has_raw_data() or data.get_bin() is None or data.laser_timing is None:
            return False
        if data.has_roi():
            return self._find_laser_timing_roi(data, config)
        else:
            return self._find_laser_timing_noroi(data, config)

    def clear_laser_timing(self, data: PODMRData) -> bool:
        if not data.has_raw_data() or data.laser_timing_offset is None:
            return False
        data.laser_timing_offset = None
        self.logger.info("Cleared laser timing offset.")
        return True

    def get_marker_indices(self, data: PODMRData):
        """Get marker indices, that is the analysis timings in unit of time bins."""

        tbin = data.get_bin()
        if data.params is None or tbin is None:
            return None

        sigdelay, sigwidth, refdelay, refwidth = [
            data.params["plot"][k] for k in ("sigdelay", "sigwidth", "refdelay", "refwidth")
        ]

        if data.laser_timing_offset is not None:
            # apply the offsets as relative values to the first laser timing.
            # global offset is handled by user-tuned timing parameters below (e.g. sigdelay).
            ofs = data.laser_timing_offset - data.laser_timing_offset[0]
            laser_timing = data.laser_timing + ofs
        else:
            laser_timing = data.laser_timing

        signal_head = laser_timing + sigdelay
        signal_tail = signal_head + sigwidth
        reference_head = signal_tail + refdelay
        reference_tail = reference_head + refwidth

        # sec to time bin index
        signal_head = np.round(signal_head / tbin).astype(np.int64)
        signal_tail = np.round(signal_tail / tbin).astype(np.int64)
        reference_head = np.round(reference_head / tbin).astype(np.int64)
        reference_tail = np.round(reference_tail / tbin).astype(np.int64)

        data.marker_indices = np.vstack((signal_head, signal_tail, reference_head, reference_tail))
        return data.marker_indices

    def analyze(self, data: PODMRData) -> bool:
        if not data.has_raw_data() or data.marker_indices is None or data.tdc_status.sweeps < 1:
            return False

        if data.is_partial():
            if data.has_roi():
                return self._analyze_partial_roi(data)
            else:
                return self._analyze_partial_noroi(data)
        else:
            if data.has_roi():
                return self._analyze_complementary_roi(data)
            else:
                return self._analyze_complementary_noroi(data)

    def _store_partial(self, data: PODMRData, sig: np.ndarray, ref: np.ndarray):
        p = data.partial()
        if p is None or p < 0 or p >= data.num_pattern():
            raise ValueError(f"invalid partial {p} for num_pattern {data.num_pattern()}")
        data._set_pattern_data(p, sig)
        data._set_pattern_ref(p, ref)

    def _store_complementary(self, data: PODMRData, sig: np.ndarray, ref: np.ndarray):
        N = data.num_pattern()
        for i in range(N):
            data._set_pattern_data(i, sig[i::N])
            data._set_pattern_ref(i, ref[i::N])

    def _analyze_partial_roi(self, data: PODMRData) -> bool:
        sig = np.zeros(len(data.xdata))
        ref = np.zeros(len(data.xdata))
        sig_head, sig_tail, ref_head, ref_tail = data.marker_indices

        for i, d in enumerate(data.raw_data):
            s, _ = data.get_roi(i)
            try:
                sig[i] = np.mean(d[sig_head[i] - s : sig_tail[i] - s + 1])
            except IndexError as e:
                self.logger.error("analyze_sig (sig %d): %r" % (i, e))

            try:
                ref[i] = np.mean(d[ref_head[i] - s : ref_tail[i] - s + 1])
            except IndexError as e:
                self.logger.error("analyze_sig (ref %d): %r" % (i, e))

        sweeps = data.tdc_status.sweeps
        self._store_partial(data, sig / sweeps, ref / sweeps)

        return True

    def _analyze_partial_noroi(self, data: PODMRData) -> bool:
        sig = np.zeros(len(data.xdata))
        ref = np.zeros(len(data.xdata))
        sig_head, sig_tail, ref_head, ref_tail = data.marker_indices

        for i in range(len(data.xdata)):
            try:
                sig[i] = np.mean(data.raw_data[sig_head[i] : sig_tail[i] + 1])
            except IndexError as e:
                self.logger.error("analyze_sig (sig %d): %r" % (i, e))

            try:
                ref[i] = np.mean(data.raw_data[ref_head[i] : ref_tail[i] + 1])
            except IndexError as e:
                self.logger.error("analyze_sig (ref %d): %r" % (i, e))

        sweeps = data.tdc_status.sweeps
        self._store_partial(data, sig / sweeps, ref / sweeps)

        return True

    def _analyze_complementary_roi(self, data: PODMRData) -> bool:
        N = data.num_pattern()
        sig = np.zeros(len(data.xdata) * N)
        ref = np.zeros(len(data.xdata) * N)
        sig_head, sig_tail, ref_head, ref_tail = data.marker_indices

        for i, d in enumerate(data.raw_data):
            s, _ = data.get_roi(i)
            try:
                sig[i] = np.mean(d[sig_head[i] - s : sig_tail[i] - s + 1])
            except IndexError as e:
                self.logger.error("analyze_sig (sig %d): %r" % (i, e))

            try:
                ref[i] = np.mean(d[ref_head[i] - s : ref_tail[i] - s + 1])
            except IndexError as e:
                self.logger.error("analyze_sig (ref %d): %r" % (i, e))

        sweeps = data.tdc_status.sweeps
        self._store_complementary(data, sig / sweeps, ref / sweeps)

        return True

    def _analyze_complementary_noroi(self, data: PODMRData) -> bool:
        N = data.num_pattern()
        sig = np.zeros(len(data.xdata) * N)
        ref = np.zeros(len(data.xdata) * N)
        sig_head, sig_tail, ref_head, ref_tail = data.marker_indices

        for i in range(len(data.xdata) * N):
            try:
                sig[i] = np.mean(data.raw_data[sig_head[i] : sig_tail[i] + 1])
            except IndexError as e:
                self.logger.error("analyze_sig (sig %d): %r" % (i, e))

            try:
                ref[i] = np.mean(data.raw_data[ref_head[i] : ref_tail[i] + 1])
            except IndexError as e:
                self.logger.error("analyze_sig (ref %d): %r" % (i, e))

        sweeps = data.tdc_status.sweeps
        self._store_complementary(data, sig / sweeps, ref / sweeps)

        return True

    def update_plot_params(self, data, plot_params: dict[str, P.RawPDValue]) -> bool:
        """update plot_params. returns True if param is actually updated."""

        if not data.has_params():
            return False
        updated = not P.isclose(data.params["plot"], plot_params)
        if "plot" not in data.params:
            data.params["plot"] = plot_params
        else:
            data.params["plot"].update(plot_params)
        if updated:
            self.update_axes(data)
        return updated

    def update_axes(self, data):
        plot = data.params["plot"]

        data.ylabel = "Intensity ({})".format(plot["plotmode"])
        data.yunit = ""

        taumode = plot["taumode"]
        if taumode == "raw":
            if data.is_sweepN():
                data.xlabel, data.xunit = "N", "pulses"
            else:
                data.xlabel, data.xunit = "tau", "s"
        elif taumode == "total":
            data.xlabel, data.xunit = "total precession time", "s"
        elif taumode == "freq":
            data.xlabel, data.xunit = "detecting frequency", "Hz"
        elif taumode == "index":
            data.xlabel, data.xunit = "sweep index", "#"
        elif taumode == "head":
            data.xlabel, data.xunit = "signal head time", "s"

        # transform
        if plot.get("logX"):
            data.xscale = "log"
        else:
            data.xscale = "linear"

        if plot.get("logY"):
            data.yscale = "log"
        else:
            data.yscale = "linear"


class Bounds(object):
    def __init__(self):
        self._sgs = {}
        self._fg = None
        self._awg = {}

    def has_sg(self, i):
        return i in self._sgs and self._sgs[i] is not None

    def sg(self, i):
        return self._sgs[i]

    def set_sg(self, i, sg_bounds):
        self._sgs[i] = sg_bounds

    def has_fg(self):
        return self._fg is not None

    def fg(self):
        return self._fg

    def set_fg(self, fg_bounds):
        self._fg = fg_bounds

    def has_awg(self) -> bool:
        return bool(self._awg)

    def awg(self) -> dict:
        return self._awg

    def set_awg(self, awg_bounds: dict):
        self._awg = awg_bounds


class CommonPulserBase(Worker, ConfAccessorMixin, ABC):
    @abstractmethod
    def _init(self, cli): ...

    @abstractmethod
    def _add_generator_params(self, params: P.ParamDict) -> bool: ...

    @abstractmethod
    def _init_generator_inst(self, params: dict) -> bool: ...

    @abstractmethod
    def _start_inst(self) -> bool: ...

    @abstractmethod
    def _stop_generator_inst(self) -> bool: ...

    @abstractmethod
    def get_timing_info(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> TimingInfo | None: ...

    @abstractmethod
    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool: ...

    def __init__(self, cli, logger, conf: dict):
        super().__init__(cli, logger, conf)

        self.mw_modes = None
        self._init(cli)
        if self.mw_modes is None:
            raise RuntimeError("mw_modes is not set after _init().")

        if "fg" in cli:
            self.fg = FGInterface(cli, "fg")
        else:
            self.fg = None
        self.add_instruments(self.fg)

        # important PG config meta data (to be stored in data.params["instrument"])
        self.length = self.offsets = self.freq = None

        self._quick_resume = self._conf_bool("quick_resume", True)
        self._start_delay = self._conf_nonneg_num("start_delay", 0.5)

        self.bounds = Bounds()
        self.pulse_pattern = None
        self.awg_waveform = None

    def _fg_enabled(self, params: dict) -> bool:
        return "fg" in params and params["fg"] is not None and params["fg"]["mode"] != "disable"

    def _init_fg(self, params: dict) -> bool:
        if not self._fg_enabled(params):
            return True
        if self.fg is None:
            self.logger.error("FG is required but not enabled.")
            return False

        c = params["fg"]
        # TODO: let some kwargs loaded from Pulser conf.
        if c["mode"] == "cw":
            self.fg.configure_cw(c["wave"], c["freq"], c["ampl"], reset=True)
        elif c["mode"] == "gate":
            self.fg.configure_gate(c["wave"], c["freq"], c["ampl"], c["phase"], reset=True)
        else:
            self.logger.error("Unknown mode {} for fg.".format(c["mode"]))
            return False
        return True

    def _plotmode_options(self, label: str) -> tuple[str, ...]:
        """Get plotmode choices for method `label`.

        Complementary plotting is currently supported only for N=2 and N=4.

        """

        if label not in self.generators:
            raise ValueError(f"unknown method '{label}'")
        num_pattern = self.generators[label].num_pattern()

        if num_pattern == 2:
            return (
                "data01",
                "data0",
                "data1",
                "diff",
                "average",
                "normalize",
                "linear-bg0",
                "linear-bg1",
                "concatenate",
                "ref",
            )
        if num_pattern == 4:
            return (
                "data01",
                "data23",
                "data0",
                "data1",
                "data2",
                "data3",
                "diff",
                "diff01-23",
                "average",
                "ref01",
                "ref23",
                "concatenate",
            )
        raise ValueError(f"unsupported num_pattern={num_pattern} for method '{label}'")

    def _set_num_pattern(self, data: PODMRData) -> int:
        if data.label not in self.generators:
            raise ValueError(f"unknown method '{data.label}'")
        num_pattern = self.generators[data.label].num_pattern()
        data.params["num_pattern"] = num_pattern
        return num_pattern

    def _validate_partial(self, data: PODMRData):
        n = data.num_pattern()
        p = data.params.get("partial", -1)
        if p != -1 and (p < 0 or p >= n):
            raise ValueError(f"partial {p} is invalid for num_pattern={n}")

    def _validate_plotmode(self, data: PODMRData):
        plotmode = data.params.get("plot", {}).get("plotmode", "data01")
        options = self._plotmode_options(data.label)
        if plotmode not in options:
            raise ValueError(
                f"plotmode '{plotmode}' is invalid for method '{data.label}'. available: {options}"
            )

    def _set_num_pattern_and_validate_params(self, data: PODMRData):
        """Set data.params["num_pattern"] and validate params in given data.

        :raises ValueError: if there is any invalid parameter.

        """

        self._set_num_pattern(data)
        self._validate_partial(data)
        self._validate_plotmode(data)

    def _get_param_dict_pulse(self, label: str, d: dict):
        num_pattern = self.generators[label].num_pattern()

        ## common_pulses
        d["base_width"] = P.FloatParam(8e-9, 1e-9, 1e-4)
        d["laser_delay"] = P.FloatParam(0.0, 0.0, 1e-3)
        d["laser_width"] = P.FloatParam(3e-6, 1e-9, 1e-3)
        d["mw_delay"] = P.FloatParam(1e-6, 0.0, 1e-3)
        d["trigger_width"] = P.FloatParam(20e-9, 1e-9, 1e-6)
        d["init_delay"] = P.FloatParam(0.0, 0.0, 1e-6)
        d["final_delay"] = P.FloatParam(5e-6, 0.0, 1e-2)
        ### global mw offset
        d["mw_offset"] = P.FloatParam(0.0, -1e-4, 1e-4)

        ## common switches
        d["invert_sweep"] = P.BoolParam(False)
        d["enable_reduce"] = P.BoolParam(False)
        d["divide_block"] = P.BoolParam(self.conf.get("divide_block", False))
        d["partial"] = P.IntParam(-1, -1, num_pattern - 1)

        ## sweep params (tau / N)
        if self.generators[label].is_sweepN():
            d["Nstart"] = P.IntParam(1, 1, 10000)
            d["Nnum"] = P.IntParam(50, 1, 10000)
            d["Nstep"] = P.IntParam(1, 1, 10000)
        else:
            d["start"] = P.FloatParam(1e-9, 1e-9, 1e-3)
            d["num"] = P.IntParam(50, 1, 10000)
            d["step"] = P.FloatParam(1e-9, 1e-9, 1e-3)
            d["log"] = P.BoolParam(False)

        return d

    def get_param_dict_labels(self) -> list:
        return list(self.generators.keys())

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        if label not in self.generators:
            self.logger.error(f"Unknown label {label}")
            return None

        # fundamentals
        d = P.ParamDict(
            resume=P.BoolParam(False),
            quick_resume=P.BoolParam(False),
            timebin=P.FloatParam(3.2e-9, 0.1e-9, 100e-9),
            interval=P.FloatParam(1.0, 0.1, 10.0),
            sweeps=P.IntParam(0, 0, doc="limit number of sweeps"),
            duration=P.FloatParam(0.0, 0.0, unit="s", doc="limit measurement duration"),
            ident=P.UUIDParam(optional=True, enable=False),
            roi_head=P.FloatParam(
                -1e-9,
                -1e-9,
                10e6,
                unit="s",
                doc="margin at head of ROI. negative value disables ROI.",
            ),
            roi_tail=P.FloatParam(
                -1e-9,
                -1e-9,
                10e6,
                unit="s",
                doc="margin at tail of ROI. negative value disables ROI.",
            ),
            multi_histogram=P.BoolParam(False, doc="use TDC multi histogram mode to realize ROI."),
        )

        self._get_param_dict_pulse(label, d)
        if not self._add_generator_params(d):
            return None
        d["pulse"] = self.generators[label].pulse_params()

        if self.fg is not None:
            if self.bounds.has_fg():
                fg = self.bounds.fg()
            else:
                fg = self.fg.get_bounds()
                if fg is None:
                    self.logger.error("Failed to get FG bounds.")
                    return None
                self.bounds.set_fg(fg)

            f_min, f_max = fg["freq"]
            a_min, a_max = fg["ampl"]
            d["fg"] = {
                "mode": P.StrChoiceParam("disable", ("disable", "cw", "gate")),
                "wave": P.StrParam("sinusoid"),
                "freq": P.FloatParam(1e6, f_min, f_max),
                "ampl": P.FloatParam(a_min, a_min, a_max),
                "phase": P.FloatParam(0.0, 0.0, 360.0),
            }

        taumodes = ["raw", "total", "freq", "index", "head"]
        if not (is_CPlike(label) or is_correlation(label) or label in ("spinecho", "trse")):
            taumodes.remove("total")
        if not ((is_CPlike(label) and not is_sweepN(label)) or label == "spinecho"):
            taumodes.remove("freq")

        d["plot"] = {
            "plotmode": P.StrChoiceParam(
                "data01",
                self._plotmode_options(label),
            ),
            "taumode": P.StrChoiceParam("raw", taumodes),
            "logX": P.BoolParam(False),
            "logY": P.BoolParam(False),
            "sigdelay": P.FloatParam(200e-9, 0.0, 1e-3),
            "sigwidth": P.FloatParam(300e-9, 1e-9, 1e-3),
            "refdelay": P.FloatParam(2200e-9, 1e-9, 1e-3),
            "refwidth": P.FloatParam(2400e-9, 1e-9, 1e-3),
            "refmode": P.StrChoiceParam("subtract", ("subtract", "divide", "ignore")),
            "refaverage": P.BoolParam(False),
            "flipY": P.BoolParam(False),
        }
        return d

    def pulse_msg(self) -> PulsePattern | None:
        return self.pulse_pattern

    def wave_msg(self):
        return self.awg_waveform


class PODMRPulserBase(CommonPulserBase):
    def __init__(self, cli, logger, conf: dict):
        super().__init__(cli, logger, conf)

        self.tdc = TDCInterface(cli, "tdc")
        self.add_instruments(self.tdc)
        self.timer = None

        self._tdc_ch0 = self._conf_nonneg_int("tdc_primary_ch", 0)
        self._tdc_ch1 = self._conf_nonneg_int("tdc_secondary_ch", 1)
        self._tdc_ch1_enable = self._conf_bool("tdc_secondary_enable", True)
        self._resume_raw_data = None
        self._resume_tdc_status = None

        # extra meta data (used for AWG now)
        self._inst_extra = None

        self.eos_margin = self._conf_nonneg_num("eos_margin", 1e-6)

        self.data = PODMRData()
        self.op = PODMRDataOperator(self.logger)

    @abstractmethod
    def _tdc_range(self) -> float: ...

    def generate_blocks(self, data: PODMRData | None = None):
        if data is None:
            data = self.data
        self._set_num_pattern_and_validate_params(data)
        generate = self.generators[data.label].generate
        params = data.get_params()
        if not self.conf.get("divide_block", False) and params["divide_block"]:
            self.logger.warn("divide_block is recommended to be False.")
        if params.get("divide_block", False) and params.get("mw_offset", 0.0) != 0.0:
            self.logger.warn(
                "divide_block=True with non-zero mw_offset can break down Nrep optimization."
            )
        return generate(data.xdata, params)

    def find_laser_timing(self, scope, smooth_window, fraction, monotonic) -> bool:
        success = self.op.find_laser_timing(self.data, scope, smooth_window, fraction, monotonic)
        if success:
            self.op.get_marker_indices(self.data)
            self.op.analyze(self.data)
        return success

    def clear_laser_timing(self) -> bool:
        success = self.op.clear_laser_timing(self.data)
        if success:
            self.op.get_marker_indices(self.data)
            self.op.analyze(self.data)
        return success

    def _init_inst(self, params: dict) -> bool:
        if not self._init_generator_inst(params):
            return False
        if not self._init_fg(params):
            self.logger.error("Error initializing FG.")
            return False
        d = self._init_tdc(params)
        if d is None:
            self.logger.error("Error initializing TDC.")
            return False

        self.op.set_instrument_params(
            self.data,
            d["range"],
            d["bin"],
            self.freq,
            self.length,
            self.offsets,
            self.mw_modes,
            self._inst_extra,
        )

        return True

    def _init_tdc(self, params: dict) -> dict | None:
        if params.get("multi_histogram", False):
            trange = params["roi_head"] + params["laser_width"] + params["roi_tail"]
            if not self.tdc.configure_multi_histogram(
                "podmr", trange, params["timebin"], len(self.data.laser_timing)
            ):
                self.logger.error("Error configuring TDC.")
                return None
        else:
            trange = self._tdc_range() - self.eos_margin
            if not self.tdc.configure_histogram("podmr", trange, params["timebin"]):
                self.logger.error("Error configuring TDC.")
                return None
        if params.get("sweeps", 0) and not self.tdc.set_sweeps(params["sweeps"]):
            # Even if the TDC doesn't support sweeps limit, we can auto-finish by is_finished().
            # However, the sweeps value can be larger than the limit.
            self.logger.warn("Error setting sweeps limit for TDC. Limit may not be strict.")
        if params.get("duration", 0.0) and not self.tdc.set_duration(params["duration"]):
            self.logger.error("Error setting duration limit for TDC.")
            return None
        return self.tdc.get_range_bin()

    def _get_tdc_status(self) -> TDCStatus:
        """Get status from TDC."""

        st0 = self.tdc.get_status(self._tdc_ch0)
        if self._tdc_ch1_enable:
            st1 = self.tdc.get_status(self._tdc_ch1)
        else:
            # dummy status to fill st1.total with zero
            st1 = ChannelStatus(True, 0.0, 0, 0)

        if self._resume_tdc_status:
            r: TDCStatus = self._resume_tdc_status
            return TDCStatus(
                round(st0.runtime) + r[0], st0.starts + r[1], st0.total + r[2], st1.total + r[3]
            )
        else:
            return TDCStatus(round(st0.runtime), st0.starts, st0.total, st1.total)

    def _get_tdc_running(self) -> bool:
        """return True if TDC is running."""

        st0 = self.tdc.get_status(self._tdc_ch0)
        return st0.running if st0 is not None else False

    def _wait_tdc_stop(self, timeout_sec=60.0, interval_sec=0.2) -> bool:
        """Wait for TDC status become not-running (stopped)."""

        self.logger.debug("Waiting TDC stop")
        for i in range(int(round(timeout_sec / interval_sec))):
            if not self._get_tdc_running():
                return True
            time.sleep(interval_sec)

        self.logger.error(f"Timeout ({timeout_sec} sec) encountered in wait_tdc_stop!")
        return False

    def update(self) -> bool:
        if not self.data.running:
            return False

        if self.data.has_roi() and not self.data.is_multi_histogram():
            roi = self.data.get_rois()
            data0 = self.tdc.get_data_roi(self._tdc_ch0, roi)
            # because length of each ROI fragments are all same,
            # we can convert list[ndarray] to 2D ndarray.
            if data0 is not None:
                data0 = np.array(data0)

            if self._tdc_ch1_enable:
                data1 = self.tdc.get_data_roi(self._tdc_ch1, roi)
                if data1 is not None:
                    data1 = np.array(data1)
            elif data0 is not None:
                data1 = np.zeros_like(data0)
            else:
                data1 = None
        else:
            data0 = self.tdc.get_data(self._tdc_ch0)
            if self._tdc_ch1_enable:
                data1 = self.tdc.get_data(self._tdc_ch1)
            elif data0 is not None:
                data1 = np.zeros_like(data0)
            else:
                data1 = None

        if data0 is not None and data1 is not None:
            if self._resume_raw_data is not None:
                new_data = data0 + data1 + self._resume_raw_data
            else:
                new_data = data0 + data1
            self.op.update(self.data, new_data, self._get_tdc_status())
            self.op.get_marker_indices(self.data)
            self.op.analyze(self.data)

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
            # when measument is running, re-analysis is done on next data update.
            # re-analyze here when measurement isn't running (viewing finished / loaded data).
            self.op.get_marker_indices(self.data)
            self.op.analyze(self.data)
        return True

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)
        resume = params is None or ("resume" in params and params["resume"])
        if params is None:
            quick_resume = resume and self._quick_resume
        else:
            quick_resume = resume and params.get("quick_resume", self._quick_resume)
        if not resume:
            self.data = PODMRData(params, label)
            self.op.update_axes(self.data)
        else:
            _last_duration = self.data.params.get("duration", 0.0)
            self.data.update_params(params)
        try:
            self._set_num_pattern_and_validate_params(self.data)
        except ValueError as e:
            self.logger.error(f"Invalid params for {label}: {e}")
            return False

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if quick_resume:
            self.logger.info("Quick resume enabled: skipping initial inst configurations.")
        if not quick_resume and not self._init_inst(self.data.params):
            return self.fail_with_release("Error initializing instruments.")
        if resume:
            self._resume_raw_data = self.data.raw_data.copy()
            self._resume_tdc_status = copy.copy(self.data.tdc_status)
        else:
            self._resume_raw_data = None
            self._resume_tdc_status = None

        # update duration here because duration means duration of additional measurement.
        # this treatise is different from sweeps (sweeps limit is considered total).
        if (
            quick_resume
            and params is not None
            and params.get("duration", 0.0) != _last_duration
            and not self.tdc.set_duration(params["duration"])
        ):
            return self.fail_with_release("Failed to set tdc duration.")

        if not self._start_inst():
            return False

        self.timer = IntervalTimer(self.data.params["interval"])

        if resume:
            self.data.resume()
            self.logger.info("Resumed pulser.")
        else:
            self.data.start()
            self.logger.info("Started pulser.")
        return True

    def discard(self) -> bool:
        if not self.data.running:
            return False
        return self.tdc.stop() and self._wait_tdc_stop() and self.tdc.clear() and self.tdc.resume()

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = (
            self._stop_generator_inst()
            and self.tdc.stop()
            and self._wait_tdc_stop()
            and self.update()
            and self.tdc.release()
        )
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(False)
        if self.fg is not None:
            success &= self.fg.release()

        if success:
            self.timer = None
            self.data.finalize()
            self.logger.info("Stopped pulser.")
        else:
            self.logger.error("Error stopping pulser.")
        return success

    def is_finished(self) -> bool:
        if not self.data.has_params() or not self.data.has_data():
            return False
        if self.data.params.get("sweeps", 0) > 0:
            return self.data.sweeps() >= self.data.params["sweeps"]
        # TDC may stop running by itself if duration limit is set
        return not self._get_tdc_running()

    def work(self):
        if not self.data.running:
            return

        if self.timer.check():
            self.update()

    def data_msg(self) -> PODMRData:
        return self.data


class SGPGPulserBase(CommonPulserBase):
    def _init(self, cli):
        self._load_conf_preset(cli)

        self.sgs = {"sg": SGInterface(cli, "sg")}
        _default_channels = [{"sg": "sg"}]
        for i in range(1, 10):
            name = f"sg{i}"
            if name in cli.insts():
                self.sgs[name] = SGInterface(cli, name)
                _default_channels.append({"sg": name})

        self.mw_modes = tuple(
            MWMode.parse(m) for m in self.conf.get("mw_modes", (0,) * len(self.sgs))
        )
        self.mw_channels = self.conf.get("mw_channels", _default_channels)

        self.pg = PGInterface(cli, "pg")

        self.add_instruments(self.pg, *self.sgs.values())

        self.check_required_conf(
            ["block_base", "pg_freq", "reduce_start_divisor", "minimum_block_length"]
        )
        self.generators = make_generators(
            freq=self.conf["pg_freq"],
            reduce_start_divisor=self.conf["reduce_start_divisor"],
            split_fraction=self.conf.get("split_fraction", 4),
            minimum_block_length=self.conf["minimum_block_length"],
            block_base=self.conf["block_base"],
            mw_modes=self.mw_modes,
            iq_amplitude=self.conf.get("iq_amplitude", 0.0),
            channel_remap=self.conf.get("channel_remap"),
            generators=self.conf.get("generators"),
            allowed_num_pattern=(2, 4),
            print_fn=self.logger.info,
        )

    def _load_conf_preset(self, cli):
        loader = PresetLoader(self.logger, PresetLoader.Mode.FORWARD)
        loader.add_preset(
            "DTG",
            [
                ("block_base", 4),
                ("pg_freq", 2.0e9),
                ("reduce_start_divisor", 2),
                ("minimum_block_length", 1000),
                ("divide_block", True),
            ],
        )
        loader.add_preset(
            "PulseStreamer",
            [
                ("block_base", 1),
                ("pg_freq", 1.0e9),
                ("reduce_start_divisor", 10),
                ("minimum_block_length", 1),
                ("divide_block", False),
            ],
        )
        loader.add_preset(
            "SpinCore_PulseBlaster",
            [
                ("block_base", 1),
                ("pg_freq", 0.5e9),
                ("reduce_start_divisor", 5),
                ("minimum_block_length", 5),
                ("divide_block", False),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("pg"))

    def _add_generator_params(self, params: P.ParamDict) -> bool:
        for i in range(len(self.mw_channels)):
            idx = "" if not i else i
            if self.bounds.has_sg(i):
                sg = self.bounds.sg(i)
            else:
                sg = self._get_sg_bounds(i)
                if sg is None:
                    self.logger.error(f"Failed to get SG{idx} bounds.")
                    return False
                self.bounds.set_sg(i, sg)

            f_min, f_max = sg["freq"]
            p_min, p_max = sg["power"]
            sg_freq = max(min(self.conf.get(f"sg{idx}_freq", 2.8e9), f_max), f_min)
            params[f"freq{idx}"] = P.FloatParam(sg_freq, f_min, f_max)
            params[f"power{idx}"] = P.FloatParam(p_min, p_min, p_max)
            params[f"nomw{idx}"] = P.BoolParam(False)

        return True

    def get_timing_info(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> TimingInfo | None:
        pg_freq = self.conf["pg_freq"]
        period = 1.0 / pg_freq
        return TimingInfo(pg_freq=pg_freq, period=period)

    @abstractmethod
    def _init_pg(self, params: dict) -> bool: ...

    def _init_sg(self, params: dict) -> bool:
        configured = []
        for i, (channel, mode) in enumerate(zip(self.mw_channels, self.mw_modes)):
            sg: SGInterface = self.sgs[channel["sg"]]
            ch = channel.get("ch", 1)
            reset = channel["sg"] not in configured
            idx = "" if not i else i
            freq = params[f"freq{idx}"]
            power = params[f"power{idx}"]

            if mode in (MWMode.QPSK, MWMode.ArbPhase):
                if not sg.configure_cw_iq_ext(freq, power, ch=ch, reset=reset):
                    self.logger.error(f"Error initializing SG{idx}.")
                    return False
            else:  # MWMode.Ext2Phase
                if not sg.configure_cw(freq, power, ch=ch, reset=reset):
                    self.logger.error(f"Error initializing SG{idx}.")
                    return False
            configured.append(channel["sg"])
        return True

    def _start_sg(self, params: dict) -> bool:
        for i, channel in enumerate(self.mw_channels):
            sg: SGInterface = self.sgs[channel["sg"]]
            ch = channel.get("ch", 1)
            idx = "" if not i else i
            nomw = params.get(f"nomw{idx}", False)
            if not sg.set_output(not nomw, ch=ch):
                self.logger.error(f"Error starting SG{idx}.")
                return False
        return True

    def _stop_sg(self) -> bool:
        success = True
        for _, channel in enumerate(self.mw_channels):
            sg: SGInterface = self.sgs[channel["sg"]]
            ch = channel.get("ch", 1)
            success &= sg.set_output(False, ch=ch) and sg.release()
        return success

    def _get_sg_bounds(self, i: int):
        channel = self.mw_channels[i]
        sg: SGInterface = self.sgs[channel["sg"]]
        ch = channel.get("ch", 1)
        return sg.get_bounds(ch)

    def _init_generator_inst(self, params: dict) -> bool:
        if not self._init_sg(params):
            self.logger.error("Error initializing SG.")
            return False
        if not self._init_pg(params):
            self.logger.error("Error initializing PG.")
            return False
        return True


class Pulser(SGPGPulserBase, PODMRPulserBase):
    """Worker for Pulse ODMR using SG + PG signal source.

    :param pulser.start_delay: (sec.) delay before starting PG output. (default: 0.5)
        A non-zero value is recommended when multi_histogram mode is used.
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

    :param pulser.eos_margin: (default: 1e-6) End-of-sequence timing margin in seconds.
    :type pulser.eos_margin: float
    :param pulser.tdc_primary_ch: (default: 0) TDC channel id for primary (mandatory) channel.
    :type pulser.tdc_primary_ch: int
    :param pulser.tdc_secondary_ch: (default: 1) TDC channel id for secondary (optional) channel.
    :type pulser.tdc_secondary_ch: int
    :param pulser.tdc_secondary_enable: (default: True) If True, secondary channel is enabled.
    :type pulser.tdc_secondary_enable: bool

    """

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        params = P.unwrap(params)
        d = PODMRData(params, label)
        try:
            blocks, freq, _ = self.generate_blocks(d)
        except (ValueError, KeyError):
            self.logger.exception(f"Invalid params for {label}")
            return False
        offsets = self.pg.validate_blocks(blocks, freq)
        return offsets is not None

    def _init_pg(self, params: dict) -> bool:
        if not (self.pg.stop() and self.pg.clear()):
            self.logger.error("Error stopping PG.")
            return False

        try:
            blocks, self.freq, laser_timing = self.generate_blocks()
        except (ValueError, KeyError):
            self.logger.exception(f"Invalid params for {self.data.label}")
            return False
        self.op.set_laser_timing(self.data, np.array(laser_timing) / self.freq)
        self.pulse_pattern = PulsePattern(blocks, self.freq, markers=laser_timing)
        pg_params = {"blocks": blocks, "freq": self.freq}

        if not (self.pg.configure(pg_params) and self.pg.get_opc()):
            self.logger.error("Error configuring PG.")
            return False

        self.length = self.pg.get_length()
        self.offsets = self.pg.get_offsets()

        return True

    def _tdc_range(self) -> float:
        return self.length / self.freq

    def _start_inst(self) -> bool:
        success = self.tdc.stop() and self.tdc.clear() and self.tdc.start()
        success &= self._start_sg(self.data.params)
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)

        # A non-zero delay here can be important in multi_histogram mode
        # to prevent the TDC from dropping the first trigger.
        time.sleep(self._start_delay)

        if success and self.pg.start():
            return True

        # fail: stop and release everything
        self.pg.stop()
        if self._fg_enabled(self.data.params):
            self.fg.set_output(False)
        self._stop_sg()
        self.tdc.stop()
        return self.fail_with_release("Error starting pulser.")

    def _stop_generator_inst(self) -> bool:
        return self.pg.stop() and self.pg.release() and self._stop_sg()


class AWGPulserBase(CommonPulserBase):
    def _parse_mw_channels(self) -> dict[int, int]:
        raw = self.conf.get("mw_channels", {0: 0, 1: 0})
        if not isinstance(raw, dict):
            raise V.ValidationError(
                "mw_channels must map source indices to physical AWG channels."
            )
        if any(
            isinstance(source, (bool, np.bool_)) or not isinstance(source, (str, int, np.integer))
            for source in raw
        ):
            raise V.ValidationError("mw_channels keys must be integer strings or integers.")
        try:
            sources = {source: int(source) for source in raw}
        except ValueError:
            raise V.ValidationError(
                "mw_channels keys must be integer strings or integers."
            ) from None
        channels = {
            sources[source]: int(V.check_nonneg_int(physical, f"mw_channels[{source!r}]"))
            for source, physical in raw.items()
        }

        if len(channels) != len(raw) or set(channels) != set(range(len(channels))):
            raise V.ValidationError("mw_channels keys must be contiguous starting at 0.")
        if any(physical not in (0, 1) for physical in channels.values()):
            raise V.ValidationError("mw_channels values must be 0 or 1.")
        return channels

    def _init(self, cli):
        self._load_conf_preset(cli)

        self.awg = AWGInterface(cli, "awg")
        self.add_instruments(self.awg)

        self.mw_channels = self._parse_mw_channels()
        self.awg_channels = tuple(sorted(set(self.mw_channels.values())))
        if not (1 <= len(self.awg_channels) <= 2):
            raise ValueError(
                f"mw_channels must select one or two physical outputs: {self.mw_channels}"
            )

        self.renderer = AWGRenderer(
            self.awg,
            channels=self.awg_channels,
            logger=self.logger,
            file_transport_dir=self._conf_str("file_transport_dir", ""),
            remove_transport_file=self._conf_bool("remove_awg_file", True),
        )
        self.mw_modes = (MWMode.AWG,) * len(self.mw_channels)
        self._awg_monitor_max_samples = self._conf_pos_int("awg_monitor_max_samples", 10_000_000)
        self._awg_monitor_max_points = self._conf_pos_int("awg_monitor_max_points", 500_000)
        if self._awg_monitor_max_points < 2:
            raise ValueError("awg_monitor_max_points must be at least 2.")

        awg_rate = self.conf.get("awg_rate", 10e9)
        if isinstance(awg_rate, (tuple, list)):
            rates = self._conf_pos_integers("awg_rate")
            if not rates:
                raise V.ValidationError("awg_rate must not be empty.")
            awg_rate = rates[0]
        awg_rate = V.check_pos_num(awg_rate, "awg_rate")
        self._pg_freq = 0.0
        self.make_generators(self.awg.get_digital_rate(awg_rate))

    def get_timing_info(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> TimingInfo | None:
        try:
            params = P.unwrap(params)
            awg_rate = ParamAccessor(params).child("awg").pos_num("rate")
            pg_freq = self.awg.get_digital_rate(awg_rate)
            if pg_freq is not None:
                try:
                    V.check_pos_num(pg_freq, "Digital rate from AWG")
                    period = 1.0 / pg_freq
                    return TimingInfo(pg_freq=pg_freq, period=period)
                except V.ValidationError as e:
                    self.logger.error(str(e))
            else:
                self.logger.error("Failed to get digital rate from AWG.")
        except (KeyError, ValueError):
            self.logger.exception("Invalid params to get timing info:")
        return None

    def make_generators(self, pg_freq: float | None):
        if pg_freq == self._pg_freq:
            return self.generators

        pg_freq = V.check_pos_num(pg_freq, "pg_freq")

        self.generators = make_generators(
            freq=pg_freq,
            reduce_start_divisor=10,  # unused
            split_fraction=self.conf.get("split_fraction", 4),
            minimum_block_length=self.conf["minimum_block_length"],
            block_base=self.conf["block_base"],
            mw_modes=self.mw_modes,
            iq_amplitude=1.0,  # unused
            channel_remap=self.conf.get("channel_remap"),
            generators=self.conf.get("generators"),
            allowed_num_pattern=(2, 4),
            print_fn=self.logger.info,
        )
        self._pg_freq = pg_freq
        return self.generators

    def _load_conf_preset(self, cli):
        loader = PresetLoader(self.logger, PresetLoader.Mode.FORWARD)
        loader.add_preset(
            "Spectrum_AWG",
            [
                ("block_base", 1),
                ("reduce_start_divisor", 10),
                ("minimum_block_length", 1),
                ("divide_block", False),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("awg"))

    def _get_awg_bounds(self) -> dict | None:
        if self.bounds.has_awg():
            return self.bounds.awg()

        bounds = self.awg.get_bounds()
        if bounds is None:
            self.logger.error("Failed to get AWG bounds.")
            return None

        self.bounds.set_awg(bounds)
        return bounds

    def _awg_rate_param(self, awg_bounds: dict):
        mn, mx = awg_bounds["sample_rate"]
        rate = self.conf.get("awg_rate", 10e9)
        if isinstance(rate, (tuple, list)):
            rates = self._conf_pos_integers("awg_rate")
            if not rates:
                raise V.ValidationError("awg_rate must not be empty.")
            return P.IntChoiceParam(rates[0], rates, doc="AWG sampling rate")
        elif isinstance(rate, (int, np.integer)):
            rate = self._conf_pos_int("awg_rate", int(rate))
            r = min(max(rate, mn), mx)
            return P.IntParam(
                r, mn, mx, unit="Hz", SI_prefix=True, digit=9, doc="AWG sampling rate"
            )
        elif isinstance(rate, (float, np.floating)):
            rate = self._conf_pos_float("awg_rate", float(rate))
            r = min(max(rate, mn), mx)
            return P.FloatParam(
                r, mn, mx, unit="Hz", SI_prefix=True, digit=9, doc="AWG sampling rate"
            )

        V.check_num(rate, "awg_rate")
        raise AssertionError("unreachable")

    def _add_generator_params(self, params: P.ParamDict) -> bool:
        del params["enable_reduce"]
        awg_bounds = self._get_awg_bounds()
        if awg_bounds is None:
            return False

        for i in range(len(self.mw_channels)):
            idx = "" if not i else i

            # Nyquist frequency for max. sampling rate.
            f_max = awg_bounds["sample_rate"][1] / 2.0
            p_max = awg_bounds["power_dBm"]
            awg_freq = max(min(self.conf.get(f"awg_freq{idx}", 2.8e9), f_max), 0.0)
            params[f"freq{idx}"] = P.FloatParam(awg_freq, 0.0, f_max)
            params[f"power{idx}"] = P.FloatParam(min(0, p_max), None, p_max)
            params[f"nomw{idx}"] = P.BoolParam(bool(i))

        params["awg"] = P.ParamDict(
            rate=self._awg_rate_param(awg_bounds),
            local_phase=P.BoolParam(
                False,
                doc="Reset the carrier phase at the beginning of each MW pulse",
            ),
            file_transport=P.BoolParam(False, doc="Send AWG waveform via the file transport"),
        )
        return True

    def _validate_file_transport(self, params: dict, awg_bounds: dict) -> bool:
        if not params["awg"].get("file_transport", False):
            return True
        if not self.renderer.has_file_transport():
            self.logger.error("awg.file_transport requires pulser.file_transport_dir.")
            return False
        if not awg_bounds.get("file_transport", False):
            self.logger.error("AWG instrument does not have file_transport_dir configured.")
            return False
        return True

    def make_mw_tones(self, num_logical_mw: int, params: dict) -> list[MWTone]:
        """Create active MW tones for the selected pulse-pattern generator."""

        tones = []

        for source_i in range(len(self.mw_channels)):
            source_suffix = "" if source_i == 0 else str(source_i)
            if params.get(f"nomw{source_suffix}", False):
                continue

            # A single-channel generator can drive multiple physical tones
            # with the same logical envelope.
            logical_i = 0 if num_logical_mw == 1 else source_i
            if logical_i >= num_logical_mw:
                continue

            logical_suffix = "" if logical_i == 0 else str(logical_i)
            tones.append(
                MWTone(
                    channel=f"mw{logical_suffix}",
                    phase_channel=f"mw{logical_suffix}_phase",
                    freq=params[f"freq{source_suffix}"],
                    power=params[f"power{source_suffix}"],
                    awg_channel=self.mw_channels[source_i],
                )
            )

        return tones

    def _stop_generator_inst(self) -> bool:
        return self.awg.stop() and self.awg.release()


class AWGPulser(AWGPulserBase, PODMRPulserBase):
    """Worker for Pulse ODMR using an AWG as signal source.

    :param pulser.start_delay: (sec.) delay before starting AWG output. (default: 0.5)
        A non-zero value is recommended when multi_histogram mode is used.
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

    :param pulser.eos_margin: (default: 1e-6) End-of-sequence timing margin in seconds.
    :type pulser.eos_margin: float
    :param pulser.tdc_primary_ch: (default: 0) TDC channel id for primary (mandatory) channel.
    :type pulser.tdc_primary_ch: int
    :param pulser.tdc_secondary_ch: (default: 1) TDC channel id for secondary (optional) channel.
    :type pulser.tdc_secondary_ch: int
    :param pulser.tdc_secondary_enable: (default: True) If True, secondary channel is enabled.
    :type pulser.tdc_secondary_enable: bool

    """

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        params = P.unwrap(params)
        awg_bounds = self._get_awg_bounds()
        if awg_bounds is None:
            return False
        if not self._validate_file_transport(params, awg_bounds):
            return False
        d = PODMRData(params, label)
        try:
            self.make_generators(self.awg.get_digital_rate(params["awg"]["rate"]))
            blocks, freq, _ = self.generate_blocks(d)
        except (ValueError, KeyError):
            self.logger.exception(f"Invalid params for {label}")
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
            blocks, self.freq, laser_timing = self.generate_blocks()
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

        # set ideal PG-specific meta data
        self.length = blocks.total_length()
        self.offsets = []

        self.op.set_laser_timing(self.data, np.array(laser_timing) / self.freq)
        pulse_blocks = remove_analog_channels(blocks, blocks.analog_channels())
        self.pulse_pattern = PulsePattern(pulse_blocks, self.freq, markers=laser_timing)

        if not self.renderer.upload(file_transport=params["awg"].get("file_transport", False)):
            return False
        self.awg_waveform = self.renderer.waveform_msg(
            laser_timing,
            self.freq,
            max_samples=self._awg_monitor_max_samples,
            max_points=self._awg_monitor_max_points,
        )
        self._inst_extra = {"awg": self.renderer.get_meta_data()}
        return True

    def _tdc_range(self) -> float:
        return self._inst_extra["awg"]["actual_duration"]

    def _start_inst(self) -> bool:
        success = self.tdc.stop() and self.tdc.clear() and self.tdc.start()
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)

        # A non-zero delay here can be important in multi_histogram mode
        # to prevent the TDC from dropping the first trigger.
        time.sleep(self._start_delay)

        if success and self.awg.start():
            return True

        # fail: stop and release everything
        self.awg.stop()
        if self._fg_enabled(self.data.params):
            self.fg.set_output(False)
        self.tdc.stop()
        return self.fail_with_release("Error starting pulser.")
