#!/usr/bin/env python3

"""
Measurement-layer AWG rendering utilities.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

PODMR/APODMR generators keep producing logical :class:`Blocks[Block]
<mahos.msgs.inst.pg_msgs.Blocks>`; this module converts them into the data
forms of :class:`AWGInterface <mahos.inst.awg_interface.AWGInterface>`:

- :func:`render_flat`: flat per-channel arrays for ``configure_waveforms``.

Block-channel interpretation
----------------------------

- ``trigger``, ``laser`` (any name in ``RenderParams.digital_channels``)
  become digital outputs.
- Per MW tone k (:class:`MWTone`): digital channel ``mw``/``mw1`` gates the
  tone on/off; ``AnalogChannel("mw_phase"/"mw1_phase", value)`` sets its
  carrier phase (degrees by default), and ``MWTone.awg_channel`` selects its
  physical analog output.
- ``sync`` (any name in ``RenderParams.drop_channels``) is discarded.
- Unknown channels are an error (catches e.g. QPSK-encoded ``mw_i``/``mw_q``
  patterns: AWG rendering expects ``mw_modes = "ArbPhase"`` *without*
  phase encoding).

Conventions
-----------

- Carrier synthesis uses the global-time LO model (phase-coherent) by default: each
  physical analog output is the sum of its assigned tones,
  ``mw_c(t) = sum_k on_k(t) * a_k * sin(2 pi f_k t + phi_k(t))``, with ``t``
  the global sample time. With ``RenderParams.local_phase``, each tone instead
  uses time relative to the first rendered sample of every contiguous high interval
  of its MW gate. Tone amplitudes come from ``power`` in dBm via the
  configured load impedance: ``V_k = dBm_to_mVpeak(power_k)``. The card
  amplitude of each output should be set to :func:`required_amplitude_mV`
  (the sum of tone peaks assigned to that output); pass the values actually
  set as ``RenderParams.amplitude_mV`` so the normalized fractions
  ``a_k = V_k / amplitude_mV[channel]`` are exact.
- Blocks are timed in integer samples at ``pg_freq``; AWG boundaries use
  *cumulative* rounding: boundary ``i`` lands on sample
  ``round(T_i * rate / pg_freq)`` (exact Fraction arithmetic, no drift).
  Each edge is off by at most half an AWG sample; results report the maximum error.
  raises ValueError if any intervals that collapsed to zero length.

"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from mahos.msgs.inst.pg_msgs import Block, Blocks
from mahos.util.unit import dBm_to_Vpeak
import mahos.util.validation as V


def dBm_to_mVpeak(power_dBm: float, impedance: float = 50.0) -> float:
    try:
        peak = 1e3 * dBm_to_Vpeak(power_dBm, impedance)
    except OverflowError as e:
        raise ValueError(f"power is too large to render: {power_dBm}") from e
    V.check_num(peak, "peak voltage")
    return peak


@dataclass
class MWTone:
    """One direct-MW carrier synthesized on a physical analog output.

    :ivar channel: digital block channel gating this tone (e.g. ``"mw"``).
    :ivar phase_channel: AnalogChannel name carrying this tone's phase
        (e.g. ``"mw_phase"``).
    :ivar freq: carrier frequency in Hz.
    :ivar power: tone power in dBm (into the configured load impedance).
    :ivar awg_channel: physical AWG output channel carrying this tone.

    """

    channel: str
    phase_channel: str
    freq: float
    power: float
    awg_channel: int = 0

    def __post_init__(self):
        V.check_nonempty_str(self.channel, "tone channel")
        V.check_nonempty_str(self.phase_channel, "tone phase_channel")
        V.check_pos_num(self.freq, "tone frequency")
        V.check_num(self.power, "tone power")
        V.check_nonneg_int(self.awg_channel, "awg_channel")


@dataclass
class RenderParams:
    """Parameters of block-to-AWG rendering.

    :ivar tones: MW tones assigned to physical outputs by ``MWTone.awg_channel``.
    :ivar analog_channels: one or two physical outputs to render. Channels remain
        present as all-zero arrays even when no active tone is assigned to them.
    :ivar num_logical_mw: Number of logical MW channels.
    :ivar load_impedance: Ohm, for dBm -> voltage conversion.
    :ivar phase_degree: if True (False), phase AnalogChannel values are in degrees (radians).
    :ivar local_phase: reset each tone's carrier phase on the first rendered sample of every
        contiguous high interval of its logical MW gate.
    :ivar digital_channels: block channels routed to digital outputs.
    :ivar drop_channels: block channels to discard silently (e.g. ``sync``).
    :ivar amplitude_mV: mapping physical output channel to the full-scale card
        amplitude against which its samples are normalized. None: use
        :func:`required_amplitude_mV` (sum of tone peaks per output).

    """

    tones: list[MWTone] = field(default_factory=list)
    analog_channels: tuple[int, ...] = (0,)
    num_logical_mw: int = 1
    load_impedance: float = 50.0
    phase_degree: bool = True
    local_phase: bool = False
    digital_channels: tuple[str, ...] = ("trigger", "laser")
    drop_channels: tuple[str, ...] = ("sync",)
    amplitude_mV: dict[int, float] | None = None


@dataclass
class FlatResult:
    """Result of :func:`render_flat` (arguments of ``configure_waveforms``).

    ``actual_samples`` is the padded (upload) length; ``rendered_samples``
    the rendered sample length before granularity padding.

    """

    analog: dict[int, np.ndarray]
    digital: dict[str, np.ndarray]
    rate: float
    amplitude_mV: dict[int, float]
    actual_samples: int
    rendered_samples: int
    max_rounding_error_sec: float

    def digital_rle(self) -> dict[str, list[tuple[bool, int]]]:
        """Digital tracks as compact RLE lists (value, n_samples) for cheap transport."""

        return {name: rle_encode(track) for name, track in self.digital.items()}


@dataclass
class _ToneState:
    """Keeps Tone's active state and current sampling point origin for local-phase rendering."""

    active: bool = False
    origin: int = 0


def rle_encode(track: np.ndarray) -> list[tuple[bool, int]]:
    """Encode a bool array as [(value, n_samples), ...]."""

    track = np.asarray(track, dtype=bool)
    if track.size == 0:
        return []
    edges = np.flatnonzero(np.diff(track)) + 1
    bounds = np.concatenate(([0], edges, [track.size]))
    return [(bool(track[b]), int(e - b)) for b, e in zip(bounds[:-1], bounds[1:])]


def _analog_channels(params: RenderParams) -> tuple[int, ...]:
    """Validate and return physical analog channels in render order."""

    channels = tuple(params.analog_channels)
    if not (1 <= len(channels) <= 2):
        raise ValueError(f"analog_channels must contain one or two channels: {channels}")
    for i, channel in enumerate(channels):
        V.check_nonneg_int(channel, f"analog_channels[{i}]")
    if len(set(channels)) != len(channels):
        raise ValueError(f"analog_channels must be unique: {channels}")
    return channels


def required_amplitude_mV(params: RenderParams) -> dict[int, float]:
    """Per-output card amplitude needed for exact per-tone powers.

    Each value is the sum of peak voltages of tones assigned to that output
    (worst-case constructive sum).

    """

    V.check_pos_num(params.load_impedance, "load_impedance")
    amplitudes = {channel: 0.0 for channel in _analog_channels(params)}
    for tone in params.tones:
        if tone.awg_channel not in amplitudes:
            raise ValueError(
                f"tone output CH{tone.awg_channel} is not in configured analog channels "
                f"{tuple(amplitudes)}"
            )
        amplitudes[tone.awg_channel] += dBm_to_mVpeak(tone.power, params.load_impedance)
    return amplitudes


def _as_blocks(blocks: Blocks[Block] | T.Sequence[Block]) -> list[Block]:
    lst = list(blocks)
    if not lst:
        raise ValueError("blocks must be non-empty.")
    for b in lst:
        V.check_pos_int(b.Nrep, f"block {b.name!r} Nrep")
        if not b.pattern:
            raise ValueError(f"block {b.name!r} pattern must be non-empty")
        for _, duration in b.pattern:
            if duration < 0:
                raise ValueError(
                    f"block {b.name!r} contains a negative interval duration: {duration}"
                )
        if b.raw_length() <= 0:
            raise ValueError(f"block {b.name!r} duration must be positive")
        if b.trigger:
            raise ValueError(
                f"block {b.name!r} has trigger=True; trigger-wait blocks are not supported "
                "by the AWG backend (the whole pattern is replayed per AWG trigger instead)."
            )
    return lst


def _validate_channels(blocks: list[Block], params: RenderParams):
    analog_channels = _analog_channels(params)
    V.check_bool(params.local_phase, "local_phase")
    V.check_pos_int(params.num_logical_mw, "num_logical_mw")

    logical_mw = {}
    for i in range(params.num_logical_mw):
        suffix = "" if i == 0 else str(i)
        logical_mw[f"mw{suffix}"] = f"mw{suffix}_phase"

    invalid_tones = [
        tone for tone in params.tones if logical_mw.get(tone.channel) != tone.phase_channel
    ]
    if invalid_tones:
        raise ValueError(
            f"tones reference invalid logical MW channels: {invalid_tones}; "
            f"expected channel pairs: {logical_mw}"
        )
    invalid_outputs = [tone for tone in params.tones if tone.awg_channel not in analog_channels]
    if invalid_outputs:
        raise ValueError(
            f"tones reference unconfigured physical outputs: {invalid_outputs}; "
            f"configured analog channels: {analog_channels}"
        )

    known = set(params.digital_channels) | set(params.drop_channels)
    for channel, phase_channel in logical_mw.items():
        known.add(channel)
        known.add(phase_channel)

    used = set()
    for block in blocks:
        used.update(block.channels())

    unknown = sorted(str(channel) for channel in used - known)
    if unknown:
        raise ValueError(
            f"blocks contain unknown channel(s) {unknown}; "
            f"known: {sorted(str(channel) for channel in known)}. "
            "(AWG rendering expects AWG-style channels such as mw / mw_phase.)"
        )


def _fractions(params: RenderParams) -> tuple[dict[int, float], list[float]]:
    """Return per-output amplitudes and per-tone full-scale fractions."""

    channels = _analog_channels(params)
    required = required_amplitude_mV(params)
    if params.amplitude_mV is None:
        amplitudes = required
    else:
        if set(params.amplitude_mV) != set(channels):
            raise ValueError(
                f"amplitude_mV keys {sorted(params.amplitude_mV)} do not match "
                f"analog_channels {channels}"
            )
        amplitudes = {
            channel: float(
                V.check_nonneg_num(params.amplitude_mV[channel], f"CH{channel} amplitude_mV")
            )
            for channel in channels
        }

    for channel in channels:
        amplitude = amplitudes[channel]
        peak_sum = required[channel]
        if peak_sum > 0.0 and amplitude <= 0.0:
            raise ValueError(f"CH{channel} amplitude_mV is 0 but tones are active")
        if amplitude > 0.0 and peak_sum > amplitude * (1.0 + 1e-9):
            raise ValueError(
                f"CH{channel} tone peak sum {peak_sum:.3f} mV exceeds "
                f"amplitude_mV {amplitude:.3f} mV (waveform would clip)"
            )

    fractions = []
    for tone in params.tones:
        peak = dBm_to_mVpeak(tone.power, params.load_impedance)
        amplitude = amplitudes[tone.awg_channel]
        fractions.append(peak / amplitude if amplitude > 0.0 else 0.0)
    return amplitudes, fractions


def _sample_ratio(pg_freq: float, rate: float) -> Fraction:
    V.check_pos_num(pg_freq, "pg_freq")
    V.check_pos_num(rate, "rate")
    pg, rt = int(round(pg_freq)), int(round(rate))
    return Fraction(rt, pg)


def _tone_phase(block: Block, tone: MWTone, channels, degree: bool) -> float:
    phi = block.analog_value(tone.phase_channel, channels)
    V.check_num(phi, f"block {block.name!r} phase on {tone.phase_channel!r}")
    return math.radians(phi) if degree else float(phi)


def _render_pattern(
    pattern,
    start_sample: int,
    n: int,
    rate: float,
    ratio: Fraction,
    params: RenderParams,
    fractions: list[float],
    block: Block,
    tone_states: list[_ToneState],
    t0_pg: int = 0,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray], float]:
    """Render one pulse pattern into analog outputs, digital tracks, and max error.

    ``start_sample`` is the pattern's global start sample, used for
    phase-coherent LO synthesis. ``t0_pg`` is the pattern's global start in pg
    samples, used as the boundary-rounding origin: every boundary is rounded
    on the *global* time axis (``round((t0_pg + t_local) * ratio) -
    round(t0_pg * ratio)``), so multi-block flat patterns have no per-block
    rounding offsets. Pass ``t0_pg=0`` for local (segment-relative) rounding,
    e.g. for reusable sequence segments.

    """

    analog = {channel: np.zeros(n) for channel in _analog_channels(params)}
    tracks = {ch: np.zeros(n, dtype=bool) for ch in params.digital_channels}
    max_err = 0.0

    s_origin = round(t0_pg * ratio)
    t_pg = 0
    s_prev = 0
    for channels, duration in pattern:
        t_pg += int(duration)
        exact = (t0_pg + t_pg) * ratio
        s_next = round(exact) - s_origin
        max_err = max(max_err, abs(float(exact - round(exact))))

        if s_next == s_prev:
            if duration == 0:
                continue
            raise ValueError(
                "positive interval collapsed on AWG rendering. "
                f"duration = {duration} PG ticks, "
                f"PG boundary = {t0_pg + t_pg}, "
                f"AWG boundary = {s_prev}, ratio = {ratio}"
            )
        if s_next < s_prev:
            raise RuntimeError(f"AWG boundary moved backwards: {s_prev} -> {s_next}")

        sl = slice(s_prev, s_next)
        for ch in params.digital_channels:
            if ch in channels:
                tracks[ch][sl] = True
        sample_indices = start_sample + np.arange(s_prev, s_next, dtype=np.float64)
        for tone, state, frac in zip(params.tones, tone_states, fractions):
            on = tone.channel in channels
            if on and not state.active:
                state.origin = start_sample + s_prev
            state.active = on
            if frac > 0.0 and on:
                phi = _tone_phase(block, tone, channels, params.phase_degree)
                if params.local_phase:
                    t = (sample_indices - state.origin) / rate
                else:
                    t = sample_indices / rate
                analog[tone.awg_channel][sl] += frac * np.sin(2.0 * math.pi * tone.freq * t + phi)
        s_prev = s_next

    if s_prev != n:  # defensive; n is computed from the same rounding
        raise RuntimeError(f"internal: rendered length {s_prev} != expected {n}")
    for channel, output in analog.items():
        if not np.isfinite(output).all():
            raise ValueError(f"CH{channel} analog waveform contains a non-finite sample")
        if (np.abs(output) > 1.0).any():
            raise ValueError(f"CH{channel} analog waveform over-range")
    return analog, tracks, max_err


def flat_samples_estimate(
    blocks: Blocks[Block] | T.Sequence[Block], pg_freq: float, rate: float
) -> int:
    """Logical sample count of flat rendering (before granularity padding)."""

    blocks = _as_blocks(blocks)
    ratio = _sample_ratio(pg_freq, rate)
    total = sum(b.total_length() for b in blocks)
    return int(round(total * ratio))


def render_flat(
    blocks: Blocks[Block] | T.Sequence[Block],
    pg_freq: float,
    rate: float,
    params: RenderParams,
    granularity: tuple[int, int],
) -> FlatResult:
    """Render blocks into the flat waveform form of ``configure_waveforms``.

    :param blocks: logical pattern (``Block.Nrep`` is expanded).
    :param pg_freq: block time base in Hz (samples of `blocks` are at this rate).
    :param rate: AWG sample rate in Hz (use the instrument's actual rate).
    :param granularity: (min_samples, step) of the card memory; output is
        zero-padded to comply (see ``get_bounds()['granularity']``).
    """

    blocks = _as_blocks(blocks)
    _validate_channels(blocks, params)
    amplitudes, fractions = _fractions(params)
    ratio = _sample_ratio(pg_freq, rate)
    mn, step = map(int, V.check_pos_integers(granularity, 2, "granularity"))

    analog_parts: dict[int, list[np.ndarray]] = {
        channel: [] for channel in _analog_channels(params)
    }
    track_parts: dict[str, list[np.ndarray]] = {ch: [] for ch in params.digital_channels}
    max_err = 0.0
    tone_states = [_ToneState() for _ in params.tones]

    t_pg = 0  # global cumulative pg samples
    s0 = 0  # global cumulative AWG samples
    for b in blocks:
        # all boundaries (block-internal ones included) follow global cumulative rounding
        t0 = t_pg
        t_pg += b.total_length()
        s1 = round(t_pg * ratio)
        a, tracks, err = _render_pattern(
            b.total_pattern(),
            s0,
            s1 - s0,
            rate,
            ratio,
            params,
            fractions,
            b,
            tone_states,
            t0_pg=t0,
        )
        max_err = max(max_err, err)
        for channel, output in a.items():
            analog_parts[channel].append(output)
        for ch in params.digital_channels:
            track_parts[ch].append(tracks[ch])
        s0 = s1

    analog = {channel: np.concatenate(parts) for channel, parts in analog_parts.items()}
    rendered_samples = s0

    target = max(rendered_samples, mn)
    if target % step:
        target += step - (target % step)
    pad = target - rendered_samples
    if pad:
        analog = {
            channel: np.concatenate([output, np.zeros(pad)]) for channel, output in analog.items()
        }

    digital = {}
    for ch in params.digital_channels:
        track = np.concatenate(track_parts[ch]) if track_parts[ch] else np.zeros(0, dtype=bool)
        if pad:
            track = np.concatenate([track, np.zeros(pad, dtype=bool)])
        if track.any():
            digital[ch] = track

    return FlatResult(
        analog=analog,
        digital=digital,
        rate=float(rate),
        amplitude_mV=amplitudes,
        actual_samples=target,
        rendered_samples=rendered_samples,
        max_rounding_error_sec=max_err / rate,
    )
