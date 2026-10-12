#!/usr/bin/env python3

"""
Shared PD-related logic for ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mahos.msgs import param_msgs as P
from mahos_dq.util.segments import round_segment_samples_up, round_segment_samples_down


@dataclass(frozen=True)
class TraceSamples:
    """Logical and realized sample counts for one detector trace."""

    logical: int
    realized: int


def make_pd_rate_param(conf: dict, *, pd_trace: bool = False):
    """Make the AnalogPD rate parameter from sweeper configuration."""

    rate = conf.get("pd_rate", 250e6 if pd_trace else 400e3)
    if isinstance(rate, (tuple, list)) and isinstance(rate[0], int):
        return P.IntChoiceParam(rate[0], rate, doc="PD sampling rate")
    elif isinstance(rate, int):
        return P.IntParam(
            rate, 1e3, 1e9, unit="Hz", SI_prefix=True, digit=9, doc="PD sampling rate"
        )
    elif isinstance(rate, float):
        return P.FloatParam(
            rate, 1e3, 1e9, unit="Hz", SI_prefix=True, digit=9, doc="PD sampling rate"
        )
    raise TypeError("conf['pd_rate'] has invalid type.")


def make_pd_buffer_size_coeff_param(conf: dict):
    return P.IntParam(
        conf.get("buffer_size_coeff", 20),
        1,
        10_000,
        doc="ratio of requested buffer size to callback data size",
    )


def make_pd_param_dict(
    conf: dict, *, pd_trace: bool = False, has_hardware_average: bool = False
) -> P.ParamDict[str, P.PDValue]:
    """Make the AnalogPD parameter dictionary used by ODMR sweepers."""

    d = P.ParamDict()
    d["rate"] = make_pd_rate_param(conf, pd_trace=pd_trace)
    d["buffer_size_coeff"] = make_pd_buffer_size_coeff_param(conf)
    lb, ub = conf.get("pd_bounds", (-10.0, 10.0))
    lower_doc = "lower bound of expected voltage"
    upper_doc = "upper bound of expected voltage"
    d["bounds"] = [
        P.FloatParam(lb, -10.0, 10.0, unit="V", doc=lower_doc),
        P.FloatParam(ub, -10.0, 10.0, unit="V", doc=upper_doc),
    ]
    if has_hardware_average:
        d["hardware_average"] = P.BoolParam(True, doc="use hardware block averaging")
    if pd_trace:
        d["eos_deadtime"] = P.FloatParam(
            200e-9, 0.0, 1.0, unit="s", SI_prefix=True, doc="end-of-sample deadtime"
        )
    return d


def make_trace_timing_param_dict(conf: dict) -> P.ParamDict[str, P.PDValue]:
    """Make the pulse timing dictionary for AnalogPD trace acquisition."""

    return P.ParamDict(
        laser_delay=P.FloatParam(
            100e-9, 0.0, 1e-3, unit="s", SI_prefix=True, doc="delay before laser"
        ),
        laser_width=P.FloatParam(
            300e-9, 0.0, 1e-3, unit="s", SI_prefix=True, doc="width of laser"
        ),
        mw_delay=P.FloatParam(
            1e-6, 0.0, 1e-3, unit="s", SI_prefix=True, doc="delay before microwave"
        ),
        mw_width=P.FloatParam(1e-6, 0.0, 1e-3, unit="s", SI_prefix=True, doc="width of microwave"),
        trigger_width=P.FloatParam(
            conf.get("trigger_width", 20e-9),
            0.0,
            10e-6,
            unit="s",
            SI_prefix=True,
            doc="width of detector trigger",
        ),
        mw_offset=P.FloatParam(0.0, -1e-4, 1e-4, unit="s", SI_prefix=True, doc="global mw offset"),
        burst_num=P.IntParam(
            conf.get("burst_num", 100), 1, 100_000, doc="number of traces at each frequency"
        ),
        roi_head=P.FloatParam(
            conf.get("roi_head", 20e-9),
            0.0,
            1e-3,
            unit="s",
            SI_prefix=True,
            doc="trigger-to-laser delay",
        ),
        roi_tail=P.FloatParam(
            conf.get("roi_tail", 100e-9),
            0.0,
            1e-3,
            unit="s",
            SI_prefix=True,
            doc="trace margin after the laser",
        ),
        sig_delay=P.FloatParam(
            conf.get("sig_delay", 0.0),
            0.0,
            1e-3,
            unit="s",
            SI_prefix=True,
            doc="signal-window delay after the laser",
        ),
        sig_width=P.FloatParam(
            conf.get("sig_width", 100e-9),
            0.0,
            1e-3,
            unit="s",
            SI_prefix=True,
            doc="signal-window width",
        ),
        ref_delay=P.FloatParam(
            conf.get("ref_delay", 0.0),
            0.0,
            1e-3,
            unit="s",
            SI_prefix=True,
            doc="reference-window delay after the signal window",
        ),
        ref_width=P.FloatParam(
            conf.get("ref_width", 100e-9),
            0.0,
            1e-3,
            unit="s",
            SI_prefix=True,
            doc="reference-window width",
        ),
        refmode=P.StrChoiceParam(
            conf.get("refmode", "divide"),
            ("subtract", "divide", "ignore"),
            doc="signal/reference reduction mode",
        ),
    )


def trace_samples(params: dict, conf: dict, pd_spectrum: bool) -> TraceSamples:
    """Calculate logical and realized samples in one trace."""

    timing = params["timing"]
    rate = params["pd"]["rate"]
    duration = timing["roi_head"] + timing["laser_width"] + timing["roi_tail"]
    logical = max(1, int(round(duration * rate)))
    realized = logical
    if pd_spectrum:
        granularity = conf.get("pd_segment_granularity", 16)
        offset = conf.get("pd_segment_offset", 0)
        realized = round_segment_samples_up(logical, granularity=granularity) - offset
        if realized < 1:
            raise ValueError(
                f"realized samples_per_trace must be positive: {realized} "
                f"(logical={logical}, offset={offset})"
            )
    return TraceSamples(logical, realized)


def marker_indices(timing: dict, rate: float) -> np.ndarray:
    """Calculate signal and reference marker indices with APODMR rounding semantics."""

    signal_head = timing["roi_head"] + timing["sig_delay"]
    signal_tail = signal_head + timing["sig_width"]
    reference_head = signal_tail + timing["ref_delay"]
    reference_tail = reference_head + timing["ref_width"]
    return np.round(
        np.asarray((signal_head, signal_tail, reference_head, reference_tail)) * rate
    ).astype(np.int64)


def validate_trace_params(params: dict, conf: dict, pd_spectrum: bool) -> None:
    """Validate trace timing, analysis windows, and detector-trigger separation."""

    timing = params["timing"]
    burst_num = timing["burst_num"]
    if isinstance(burst_num, bool) or not isinstance(burst_num, (int, np.integer)):
        raise ValueError("burst_num must be an integer")
    if burst_num <= 0:
        raise ValueError("burst_num must be positive")

    if timing["trigger_width"] <= 0.0:
        raise ValueError("trigger_width must be positive")

    nonnegative_timing = (
        "laser_delay",
        "laser_width",
        "mw_delay",
        "mw_width",
        "roi_head",
        "roi_tail",
        "sig_delay",
        "sig_width",
        "ref_delay",
        "ref_width",
    )
    for key in nonnegative_timing:
        if timing[key] < 0.0:
            raise ValueError(f"{key} must be non-negative")
    for key in ("delay", "background_delay", "final_delay"):
        if params.get(key, 0.0) < 0.0:
            raise ValueError(f"{key} must be non-negative")
    eos_deadtime = params["pd"].get("eos_deadtime", 0.0)
    if eos_deadtime < 0.0:
        raise ValueError("pd.eos_deadtime must be non-negative")
    if params["pd"].get("buffer_size_coeff", conf.get("buffer_size_coeff", 20)) < 1:
        raise ValueError("pd.buffer_size_coeff must be positive")

    trigger_width = timing["trigger_width"]
    roi_head = timing["roi_head"]
    if trigger_width > roi_head:
        raise ValueError("trigger_width <= roi_head must be satisfied")
    pre_laser = timing["mw_delay"] + timing["mw_width"] + timing["laser_delay"]
    if roi_head > pre_laser:
        raise ValueError("roi_head must fit inside the complete pre-laser sequence")

    samples = trace_samples(params, conf, pd_spectrum)
    markers = marker_indices(timing, params["pd"]["rate"])
    sig_head, sig_tail, ref_head, ref_tail = (int(v) for v in markers)
    if not 0 <= sig_head <= sig_tail < samples.realized:
        raise ValueError(
            "signal window is out of range "
            f"(head = {sig_head}, tail = {sig_tail}, samples = {samples})"
        )
    if not 0 <= ref_head <= ref_tail < samples.realized:
        raise ValueError(
            "reference window is out of range "
            f"(head = {ref_head}, tail = {ref_tail}, samples = {samples})"
        )

    pg_freq = conf["pg_freq_pulse"]
    interval_ticks = sum(
        round(timing[key] * pg_freq)
        for key in ("mw_delay", "mw_width", "laser_delay", "laser_width")
    )
    trace_ticks = math.ceil(samples.realized / params["pd"]["rate"] * pg_freq)
    eos_deadtime_ticks = math.ceil(eos_deadtime * pg_freq)
    if interval_ticks < trace_ticks + eos_deadtime_ticks:
        raise ValueError(
            "trace window including eos_deadtime overlaps the next detector trigger "
            f"(samples = {samples}, trace = {trace_ticks}, "
            f"eos = {eos_deadtime_ticks}, interval = {interval_ticks})"
        )

    if timing["refmode"] not in ("subtract", "divide", "ignore"):
        raise ValueError(f"unknown refmode: {timing['refmode']}")


def configure_apds(
    pds,
    pd_clock: str,
    time_window: float,
    point_count: int,
    buffer_size_coeff: int,
    drop_first: int,
) -> bool:
    """Configure APDs (daq.SinglePhotonCounters) for ODMR."""

    # max. expected sampling rate. double expected freq due to gate mode.
    # this max rate is achieved if freq switching time was zero (it's non-zero in reality).
    rate = 2.0 / time_window
    buffer_size = point_count * buffer_size_coeff
    params_pd = {
        "clock": pd_clock,
        "cb_samples": point_count,
        "samples": buffer_size,
        "buffer_size": buffer_size,
        "rate": rate,
        "finite": False,
        "every": False,
        "drop_first": drop_first,
        "gate": True,
        "time_window": time_window,
    }

    return all([pd.configure(params_pd) for pd in pds])


def configure_analog_pds(
    clock,
    pds,
    pd_trigger: str,
    params: dict,
    conf: dict,
    pd_spectrum: bool,
    point_count: int,
    drop_first: int,
    data_transfer: str | None,
    logger,
) -> bool:
    """Configure a retriggerable clock and AnalogPDs for integrated acquisition."""

    rate = params["pd"]["rate"]
    oversamp = round(params["timing"]["time_window"] * rate)

    if pd_spectrum:
        granularity = conf.get("pd_segment_granularity", 16)
        offset = conf.get("pd_segment_offset", 0)
        try:
            adjusted = round_segment_samples_down(oversamp, granularity=granularity)
            adjusted -= offset
        except ValueError:
            logger.exception("failed to round oversample to segment granularity")
            return False
        if adjusted != oversamp:
            logger.info(
                "PD oversample adjusted down: "
                f"{oversamp} to {adjusted} (granularity = {granularity} offset = {offset})"
            )
        oversamp = adjusted

    logger.info(f"Analog PD oversample: {oversamp}")

    params_clock = {
        "freq": rate,
        "samples": oversamp,
        "finite": True,
        "trigger_source": pd_trigger,
        "trigger_dir": True,
        "retriggerable": True,
    }
    if clock is None:
        clock_pd = pd_trigger
    else:
        if not clock.configure(params_clock):
            logger.error("failed to configure clock")
            return False
        clock_pd = clock.get_internal_output()

    buffer_size = point_count * params["pd"].get(
        "buffer_size_coeff", conf.get("buffer_size_coeff", 20)
    )
    params_pd = {
        "trigger_source": clock_pd,
        "clock": clock_pd,
        "cb_samples": point_count,
        "samples": buffer_size,
        "buffer_size": buffer_size,
        "rate": rate,
        "bounds": params["pd"].get("bounds", (-10.0, 10.0)),
        "finite": False,
        "every": False,
        "drop_first": drop_first,
        "oversample": oversamp,
        "clock_mode": True,
        "clock_dir": True,
        "trigger_dir": True,
        "data_transfer": data_transfer,
    }
    success = all(pd.configure(params_pd, "triggered") for pd in pds)
    if not success:
        return False
    return True


def configure_trace_pds(
    clock,
    pds,
    pd_trigger: str,
    params: dict,
    conf: dict,
    pd_spectrum: bool,
    point_count: int,
    drop_first: int,
    data_transfer: str | None,
    logger,
) -> int | None:
    """Configure a retriggerable clock and PDs for averaged trace acquisition."""

    try:
        validate_trace_params(params, conf, pd_spectrum)
        samples_per_trace = trace_samples(params, conf, pd_spectrum).realized
    except (KeyError, TypeError, ValueError):
        logger.exception("invalid AnalogPD trace parameters")
        return None

    rate = params["pd"]["rate"]
    params_clock = {
        "freq": rate,
        "samples": samples_per_trace,
        "finite": True,
        "trigger_source": pd_trigger,
        "trigger_dir": True,
        "retriggerable": True,
    }
    if clock is None:
        clock_pd = pd_trigger
    else:
        if not clock.configure(params_clock):
            logger.error("failed to configure clock")
            return None
        clock_pd = clock.get_internal_output()

    cb_samples = point_count * samples_per_trace
    buffer_size = cb_samples * params["pd"].get(
        "buffer_size_coeff", conf.get("buffer_size_coeff", 20)
    )
    params_pd = {
        "trigger_source": clock_pd,
        "clock": clock_pd,
        "cb_samples": cb_samples,
        "samples": buffer_size,
        "buffer_size": buffer_size,
        "rate": rate,
        "bounds": params["pd"].get("bounds", (-10.0, 10.0)),
        "finite": False,
        "every": False,
        "drop_first": drop_first,
        "oversample": 1,
        "block_samples": samples_per_trace,
        "block_reduce_factor": timing_burst_num(params),
        "block_reduce_op": "mean",
        "clock_mode": True,
        "clock_dir": True,
        "trigger_dir": True,
        "data_transfer": data_transfer,
        "hardware_average": params["pd"].get("hardware_average", True),
    }
    success = all(pd.configure(params_pd, "triggered") for pd in pds)
    if not success:
        return None
    return samples_per_trace


def timing_burst_num(params: dict) -> int:
    """Return the trace burst count as a builtin integer."""

    return int(params["timing"]["burst_num"])


def result_unit(pd_unit: str, params: dict, label: str, pd_trace: bool) -> str:
    """Return the unit of the reduced ODMR result."""

    if label == "pulse" and pd_trace and params["timing"]["refmode"] == "divide":
        return ""
    return pd_unit


def reshape_trace_block(data, point_count: int, samples_per_trace: int) -> np.ndarray:
    """Reshape one reduced detector/channel block into point-major traces."""

    array = np.asarray(data)
    expected = point_count * samples_per_trace
    if array.size != expected:
        raise ValueError(
            f"unexpected PD block size {array.size}; expected {expected} "
            f"for point_count={point_count}, samples_per_trace={samples_per_trace}"
        )
    return array.reshape(point_count, samples_per_trace)


def reduce_traces(traces: np.ndarray, timing: dict, rate: float) -> np.ndarray:
    """Reduce point-major averaged traces to scalar ODMR values."""

    sig_head, sig_tail, ref_head, ref_tail = (int(v) for v in marker_indices(timing, rate))
    samples_per_trace = traces.shape[1]
    if not 0 <= sig_head <= sig_tail < samples_per_trace:
        raise ValueError("signal window is out of range")
    if not 0 <= ref_head <= ref_tail < samples_per_trace:
        raise ValueError("reference window is out of range")

    signal = np.mean(traces[:, sig_head : sig_tail + 1], axis=1)
    mode = timing["refmode"]
    if mode == "ignore":
        return signal
    reference = np.mean(traces[:, ref_head : ref_tail + 1], axis=1)
    if mode == "subtract":
        return signal - reference
    elif mode == "divide":
        return signal / reference
    raise ValueError(f"unknown refmode: {mode}")


def reduce_pd_blocks(
    blocks: list, params: dict, point_count: int, samples_per_trace: int
) -> np.ndarray:
    """Reduce and sum blocks from all detector channels."""

    traces = sum_pd_blocks(blocks, point_count, samples_per_trace)
    return reduce_traces(traces, params["timing"], params["pd"]["rate"])


def _collect_pd_channels(blocks: list) -> list:
    """Collect data from multiple PDs, flattening multi-channel blocks."""

    channels = []
    for block in blocks:
        if isinstance(block, list):
            # PD has multiple channels
            channels.extend(block)
        else:
            # single channel
            channels.append(block)
    return channels


def sum_pd_channels(blocks: list) -> np.ndarray:
    """Sum ordinary non-``pd_trace`` data from all detector channels as-is."""

    return np.sum(_collect_pd_channels(blocks), axis=0)


def sum_pd_blocks(blocks: list, point_count: int, samples_per_trace: int) -> np.ndarray:
    """Reshape and sum ``pd_trace`` blocks from all detector channels as point-major traces."""

    channels = _collect_pd_channels(blocks)
    if not channels:
        raise ValueError("no PD data")
    return np.sum(
        [reshape_trace_block(channel, point_count, samples_per_trace) for channel in channels],
        axis=0,
    )
