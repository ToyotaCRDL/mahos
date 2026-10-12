#!/usr/bin/env python3

"""
Spectrum Instrumentation AWG instruments.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

This currently exposes two classes for Spectrum Instrumentation's AWG.
- :class:`Spectrum_AWG_Core`: a plain, reusable wrapper around the official
  ``spcm`` Python package. It has no MAHOS-server dependency and can
  be used from standalone scripts.
- :class:`Spectrum_AWG`: the MAHOS :class:`Instrument <mahos.inst.instrument.Instrument>`
  wrapping the core, exposing standard APIs for ``InstrumentServer``.

"""

from __future__ import annotations

import logging
import time
import typing as T
from dataclasses import dataclass, field

import numpy as np

from mahos.inst.instrument import Instrument
from mahos.inst.awg_file import load_waveforms
from mahos.msgs import param_msgs as P
from mahos.msgs.inst.awg_msgs import TriggerType
from mahos.util.unit import dBm_to_Vpeak, Vpeak_to_dBm
from mahos.util.conf import ConfAccessorMixin
from mahos.util.param import ParamAccessor
from mahos.util.file_transport import SharedFileTransport
import mahos.util.validation as V


@dataclass
class Waveform:
    """Analog samples plus named digital marker tracks at a common sample rate.

    :ivar sample_rate: sample rate in Hz.
    :ivar analog: float64 array, full-scale fraction in [-1, +1].
    :ivar markers: mapping marker name -> boolean array (same length as analog).
    :ivar segments: list of (name, start_index, stop_index) annotations,
        stop exclusive. Useful to align scope / digitizer windows.

    """

    sample_rate: float
    # supposed to contain normalized floating-point samples before conversion to int16.
    # IMPORTANT: this is only the software-normalized waveform.
    # It still has to be converted into the integer DAC format before being sent to the card
    analog: np.ndarray
    # stores digital marker tracks by name.
    markers: dict[str, np.ndarray] = field(default_factory=dict)
    # stores named regions of the waveform.
    # Each segment is a tuple: (name, start_index, stop_index) where stop_index is EXCLUSIVE.
    segments: list[tuple[str, int, int]] = field(default_factory=list)

    def __post_init__(self):
        V.check_pos_num(self.sample_rate, "sample_rate")

        self.analog = np.asarray(
            self.analog, dtype=np.float64
        )  # Takes whatever was passed as analog and converts it into a float64 array

        if self.analog.ndim != 1:
            raise ValueError(f"analog must be 1D, got shape {self.analog.shape}")
        if not np.all(np.isfinite(self.analog)):
            raise ValueError("analog contains NaN or infinite values")
        if self.analog.size == 0:
            raise ValueError("analog waveform must contain at least one sample")
        if np.max(np.abs(self.analog), initial=0.0) > 1.0:
            raise ValueError("analog exceeds full-scale range [-1, +1]")

        for name in list(self.markers):
            m = np.asarray(
                self.markers[name], dtype=bool
            )  # converts the marker track into a bool array
            if m.shape != self.analog.shape:
                raise ValueError(
                    f"marker '{name}' length {m.shape} != analog length {self.analog.shape}"
                )
            self.markers[name] = m

        n = self.analog.size
        for segment_name, start, stop in self.segments:
            if not (0 <= start <= stop <= n):
                raise ValueError(
                    f"invalid segment {segment_name!r}: ({start}, {stop}) for waveform length {n}"
                )

    def n_samples(self) -> int:
        """returns the number of samples in the analog waveform."""

        return len(self.analog)

    def duration(self) -> float:
        return self.n_samples() / self.sample_rate

    def max_abs(self) -> float:
        """returns the largest absolute value of the analog waveform."""

        return float(np.max(np.abs(self.analog))) if self.n_samples() else 0.0

    def marker_names(self) -> list[str]:
        return list(self.markers.keys())

    def validate(self, limit: float = 1.0):
        """Checks/ Raise ValueError if any analog sample exceeds `limit` (full-scale fraction)."""

        m = self.max_abs()
        if m > limit + 1e-12:
            raise ValueError(
                f"analog samples exceed full-scale limit: max |x| = {m:.6f} > {limit}"
            )

    def segment_times(self) -> list[tuple[str, float, float]]:
        """Segments as (name, start_time, stop_time) in seconds."""

        return [(n, i0 / self.sample_rate, i1 / self.sample_rate) for n, i0, i1 in self.segments]


@dataclass(frozen=True)
class MarkerRoute:
    """Physical routing of a named synchronous digital output."""

    line: int
    source: int


def quantize_analog(analog: np.ndarray) -> np.ndarray:
    """Quantize full-scale fraction [-1, +1] floats to int16 two's complement."""

    a = np.asarray(analog, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Analog waveform contains infinite value")
    if (np.abs(a) > 1.0).any():
        raise ValueError("Analog waveform contains over-range value (abs(analog) > 1)")
    full_scale = np.iinfo(np.int16).max
    return np.round(a * full_scale).astype(np.int16)


def pack_waveform(
    wf: Waveform, marker_order: T.Sequence[str] | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    """Pack analog + markers into int16 card samples (M5i synchronous digital out).

    Mechanism (verified against ``spcm.SynchronousDigitalIOs.process``): with
    ``n`` markers, the int16 analog word is logically right-shifted by ``n``
    on its unsigned view -- this converts 16 bit two's complement to (16-n) bit
    two's complement in the low bits -- and marker ``j`` (index in
    ``marker_order``) is OR-ed into sample bit ``15 - j``.

    The driver must configure the X line carrying marker ``j`` with::

        SPCM_XMODE_DIGOUT
        | SPCM_XMODE_DIGOUTSRC_CHx
        | (SPCM_XMODE_DIGOUTSRC_BIT15 << j)

    Here, ``CHx`` is the physical analog channel containing the packed marker
    bit.

    :param marker_order: marker names, highest bit first. Defaults to the
        waveform's marker insertion order. Waveform tracks not listed are an
        error; listed-but-absent tracks are treated as all-low.
    :returns: (int16 samples, mapping marker name -> sample bit index).

    """

    if marker_order is None:
        marker_order = wf.marker_names()
    marker_order = list(marker_order)
    unknown = [m for m in wf.markers if m not in marker_order]
    if unknown:
        raise ValueError(
            f"waveform has marker tracks not in marker_order: {unknown}"
        )  # Every marker track present in the waveform must be listed in marker_order.
    n = len(marker_order)
    if n > 4:
        raise ValueError(f"at most 4 markers supported (X0..X3), got {n}")

    data = quantize_analog(wf.analog)
    if n == 0:
        return data, {}

    packed = data.view(np.uint16) >> np.uint16(n)
    # before:
    # [analog analog analog analog ... analog]
    # after shift by n:
    # [0 ... n-2 zeros ... 0 analog analog ... analog]

    bit_map: dict[str, int] = {}
    for j, name in enumerate(marker_order):
        bit = 15 - j
        bit_map[name] = bit
        track = wf.markers.get(name)
        if track is not None:
            packed = packed | (track.astype(np.uint16) << np.uint16(bit))
    return packed.astype(np.uint16).view(np.int16), bit_map


def unpack_waveform(data: np.ndarray, n_markers: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Inverse of :func:`pack_waveform` (for tests and debugging).

    :returns: (analog full-scale fraction, markers in ``marker_order`` order).
        The low ``n_markers`` bits of the analog value are lost to
        quantization; the amplitude scale is preserved.

    """

    u = np.asarray(data).view(np.uint16)
    markers = [((u >> np.uint16(15 - j)) & np.uint16(1)).astype(bool) for j in range(n_markers)]
    restored = (u << np.uint16(n_markers)).astype(np.uint16).view(np.int16)
    full_scale = np.iinfo(np.int16).max
    analog = restored.astype(np.float64) / full_scale
    return analog, markers


class Spectrum_AWG_Core(object):
    """Standalone wrapper for a Spectrum M5i.63xx AWG card.

    :param resource: card resource such as ``/dev/spcm0``. If None, the card
        is searched by `serial_number` or, failing that, the first AO card.
    :param serial_number: card serial number (e.g. 24423).
    :param amp_limit_mV: hard software cap for the output amplitude in mV.
    :param verbose: pass verbose flag to spcm.Card.
    :param logger: logger object (std logging style). Defaults to module logger.

    Hardware notes (M5i.6360-x16, verified against manual & spcm sources)
    ---------------------------------------------------------------------

    - 1 channel, 16 bit, base sample rate 10 GS/s; valid rates are the base rate
      divided by powers of two (..., 625, 312.5 MS/s, ...). The driver reads back
      the actual rate after setting; always build waveforms with the actual rate.
    - Output range (SPC_AMP) is programmable up to +-500 mV into 50 Ohm; this
      driver enforces ``amp_limit_mV`` (project cap: 100 mV) in software.
    - Sequence mode is not supported yet because SPC_PCIFEATURES of this card
      decodes to Multiple-Replay + Digital-Outputs; the sequence-mode feature bit is absent.
    - Replay modes: "singlerestart" (default; armed, one replay per trigger),
      "single" (one trigger, replay ``loops`` times).
    - Markers are embedded in the upper sample bits (see :func:`pack_waveform`)
      and routed to X0..X3 with
      ``SPCM_XMODE_DIGOUT``. X lines update at 1/4 of the sample clock (1/8 above
      5 GS/s).
    - Demo cards configured in Spectrum Control Center work transparently
      (resource string like ``/dev/spcm_demo0``); for fully hardware-less
      development use conf ``mock: true`` (:class:`Dummy_AWG_Core`).

    """

    def __init__(
        self,
        resource: str | None = None,  # Optional card resource path, e.g. /dev/spcm0.
        serial_number: int | None = None,  # Optional card serial number.
        amp_limit_mV: int = 100,  # Software amplitude safety cap in millivolts.
        granularity: int = 64,
        verbose: bool = False,  # Whether to use verbose hardware output.
        logger=None,  # Optional logger.
    ):
        self.resource = resource
        self.serial_number = serial_number
        self.amp_limit_mV = int(amp_limit_mV)
        self.granularity = granularity
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)

        self._spcm = None
        self._card = None
        self._channels = None
        self._transfer = None
        self.running = False
        self._uploaded_samples = 0

    def _import_spcm(self):
        if self._spcm is None:
            import spcm

            self._spcm = spcm
        return self._spcm

    def open(self):
        """Open the card (idempotent)."""

        spcm = self._import_spcm()
        if self._card is not None:
            return self

        kwargs = {"verbose": self.verbose}
        if self.serial_number is not None and not self.resource:
            self._card = spcm.Card(serial_number=self.serial_number, **kwargs)
        elif self.resource:
            if self.serial_number is not None:
                kwargs["serial_number"] = self.serial_number
            self._card = spcm.Card(self.resource, **kwargs)
        else:
            self._card = spcm.Card(card_type=spcm.SPCM_TYPE_AO, **kwargs)
        self._card.open()

        fnc = self._card.get_i(spcm.SPC_FNCTYPE)
        if fnc != spcm.SPCM_TYPE_AO:
            self.logger.warning(f"card function type is {fnc}, expected AO ({spcm.SPCM_TYPE_AO})")

        # disable driver timeout; we never block on WAITREADY
        # this looks like the default, but do this explicitly, just in case
        self._card.timeout(0)

        self.logger.info(f"Opened Spectrum AWG: {self.card_info()}")
        return self

    def close(self):
        self.stop()
        if self._card is not None:
            try:
                self._card.close()
            except Exception:
                self.logger.exception("Exception while closing Spectrum AWG card.")
            self._card = None
        self._channels = None
        self._transfer = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _require_card(self):
        if self._card is None:
            raise RuntimeError("card is not opened. call open() first.")
        return self._card

    def card_info(self) -> dict:
        """Read basic card information / capability registers."""

        card = self._require_card()
        spcm = self._import_spcm()

        def geti(reg):
            try:
                return card.get_i(reg)
            except Exception:
                return None

        features = geti(spcm.SPC_PCIFEATURES) or 0
        info = {
            "card_type": hex(geti(spcm.SPC_PCITYP) or 0),
            "serial_number": geti(spcm.SPC_PCISERIALNO),
            "memory_bytes": geti(spcm.SPC_PCIMEMSIZE),
            "bytes_per_sample": geti(spcm.SPC_MIINST_BYTESPERSAMPLE),
            "max_sample_rate": geti(spcm.SPC_MIINST_MAXADCLOCK),
            "min_sample_rate": geti(spcm.SPC_MIINST_MINADCLOCK),
            "num_channels": int(card.num_channels()),
            "num_xio_lines": geti(spcm.SPC_NUM_XIO_LINES),
            "features": features,
            "has_sequence_mode": bool(features & spcm.SPCM_FEAT_SEQUENCE),
            "avail_card_modes": geti(spcm.SPC_AVAILCARDMODES),
        }
        bps = info["bytes_per_sample"] or 2
        if info["memory_bytes"]:
            info["memory_samples"] = info["memory_bytes"] // bps
        return info

    def memory_granularity(self) -> tuple[int, int]:
        """(min_size, step) of SPC_MEMSIZE in samples; falls back to (64, 64)."""

        card = self._require_card()
        spcm = self._import_spcm()
        try:
            mn = int(card.get_i(spcm.SPC_AVAILMEMSIZE_MIN))
            step = int(card.get_i(spcm.SPC_AVAILMEMSIZE_STEP))
            if mn > 0 and step > 0:
                return mn, step
        except Exception:
            pass
        return self.granularity, self.granularity

    def status(self) -> dict:
        """Decoded SPC_M2STATUS card status."""

        card = self._require_card()
        spcm = self._import_spcm()
        st = card.get_i(spcm.SPC_M2STATUS)
        return {
            "raw": st,
            "pretrigger": bool(st & spcm.M2STAT_CARD_PRETRIGGER),
            "triggered": bool(st & spcm.M2STAT_CARD_TRIGGER),
            "ready": bool(st & spcm.M2STAT_CARD_READY),
            "running": self.running,
            "uploaded_samples": self._uploaded_samples,
        }

    def configure_clock(
        self,
        sample_rate: float,
        mode: str = "internal",
        ref_clock_freq: float = 10e6,
        clock_output: bool = False,
        termination_50: bool = True,
    ) -> int:
        """Set up the clock engine and sample rate. Returns the *actual* rate in Hz.

        :param mode: "internal" (internal PLL) or "ext_ref" (external
            reference clock of `ref_clock_freq`, e.g. a lab 10 MHz reference).
        :param termination_50: clock input termination (ext_ref only).

        """

        sample_rate = V.check_pos_num(sample_rate, "sample_rate")
        mode = V.check_str(mode, "clock mode").lower()
        card = self._require_card()
        spcm = self._import_spcm()
        clock = spcm.Clock(card)
        if mode in ("internal", "int", "intpll"):
            clock.mode(spcm.SPC_CM_INTPLL)
        elif mode in ("ext_ref", "extref", "external_reference", "reference"):
            ref_clock_freq = V.check_pos_num(ref_clock_freq, "ref_clock_freq")
            clock.mode(spcm.SPC_CM_EXTREFCLOCK)
            clock.reference_clock(int(round(ref_clock_freq)))
            try:
                clock.termination(1 if termination_50 else 0)
            except Exception:
                self.logger.exception("could not set clock termination (continuing).")
        else:
            raise ValueError(f"unsupported clock mode: {mode}")
        clock.clock_output(bool(clock_output))

        requested = int(round(sample_rate))
        actual = int(clock.sample_rate(requested))
        if actual != requested:
            self.logger.warning(
                f"sample rate: requested {requested:_d} Hz, card set {actual:_d} Hz "
                "(usually supports base rate / 2^n)."
            )
        else:
            self.logger.debug(f"sample rate: {actual:_d} Hz")
        return actual

    def _validate_amplitude(self, amplitude_mV: int | float) -> int:
        V.check_num(amplitude_mV, "amplitude_mV")
        amplitude_mV = int(round(amplitude_mV))
        if not (1 <= amplitude_mV <= self.amp_limit_mV):
            raise ValueError(
                f"requested amplitude {amplitude_mV} mV is outside 1 .. {self.amp_limit_mV} mV"
            )
        return amplitude_mV

    def _configure_output_channel(
        self, ch, amplitude_mV: int, filter_: int | None, stop_level: str
    ) -> int:
        amplitude_mV = self._validate_amplitude(amplitude_mV)
        ch.enable(True)
        ch.amp(amplitude_mV)  # plain int -> mV, no output-load conversion
        if filter_ is not None:
            ch.filter(int(filter_))

        spcm = self._import_spcm()
        levels = {
            "zero": ("SPCM_STOPLVL_ZERO", "SPCM_STOPLVL_TOZERO"),
            "low": ("SPCM_STOPLVL_LOW", "SPCM_STOPLVL_TOMIN"),
            "high": ("SPCM_STOPLVL_HIGH", "SPCM_STOPLVL_TOMAX"),
            "holdlast": ("SPCM_STOPLVL_HOLDLAST",),
        }
        key = stop_level.lower()
        if key not in levels:
            raise ValueError(f"unsupported stop_level: {stop_level}")
        for name in levels[key]:
            if hasattr(spcm, name):
                ch.stop_level(getattr(spcm, name))
                break
        else:
            self.logger.warning(f"no stop-level constant found for '{stop_level}'; left default.")

        actual = ch.amp()
        actual = int(getattr(actual, "magnitude", actual))
        if actual != amplitude_mV:
            self.logger.warning(
                f"CH{ch.index} amplitude: requested {amplitude_mV} mV, card set {actual} mV."
            )
        return actual

    def configure_channels(
        self,
        channels: T.Iterable[int],
        amplitudes_mV: dict[int, int],
        filter_: int | None = None,
        stop_level: str = "zero",
    ) -> dict[int, int]:
        """Enable and configure one or two physical analog output channels."""

        card = self._require_card()
        spcm = self._import_spcm()
        active = sorted(
            set(int(V.check_nonneg_int(ch, f"channels[{i}]")) for i, ch in enumerate(channels))
        )
        available = range(int(card.num_channels()))
        if not active or len(active) > 2 or any(ch not in available for ch in active):
            raise ValueError(
                f"invalid active channels {active}; available channels are {list(available)}"
            )

        mask = 0
        for channel in active:
            mask |= getattr(spcm, f"CHANNEL{channel}")
        self._channels = spcm.Channels(card, card_enable=mask)

        actual = {}
        for ch in self._channels.channels:
            actual[ch.index] = self._configure_output_channel(
                ch, amplitudes_mV[ch.index], filter_, stop_level
            )
        return actual

    def set_amplitude(self, channel: int, amplitude_mV: int | float) -> int:
        """Set amplitude if `channel` is active and return the requested or actual value."""

        channel = int(V.check_nonneg_int(channel, "channel"))
        amplitude_mV = self._validate_amplitude(amplitude_mV)
        if self._channels is None:
            return amplitude_mV
        for ch in self._channels.channels:
            if ch.index == channel:
                ch.amp(amplitude_mV)
                actual = ch.amp()
                return int(getattr(actual, "magnitude", actual))
        return amplitude_mV

    def configure_trigger(
        self,
        source: str = "ext0",
        level: float = 1.0,
        edge: bool = True,
        termination_50: bool = False,
    ) -> bool:
        """Set up the trigger engine.

        :param source: "ext0" (external trigger input), "software"
            (immediate trigger on start) or "none" (only force_trigger()).
        :param level: ext0 trigger level in volts.
        :param edge: True for positive, False for negative edge (ext0).

        """

        V.check_bool(edge, "edge")
        V.check_str(source, "trigger source")
        card = self._require_card()
        spcm = self._import_spcm()
        trigger = spcm.Trigger(card)
        source = source.lower()
        if source in ("ext0", "external", "external0"):
            V.check_num(level, "trigger level")
            trigger.or_mask(spcm.SPC_TMASK_EXT0)
            trigger.and_mask(spcm.SPC_TMASK_NONE)
            trigger.ext0_mode(spcm.SPC_TM_POS if edge else spcm.SPC_TM_NEG)
            trigger.termination(1 if termination_50 else 0)
            trigger.ext0_coupling(spcm.COUPLING_DC)
            trigger.ext0_level0(float(level) * spcm.units.V)
        elif source in ("software", "soft"):
            trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
            trigger.and_mask(spcm.SPC_TMASK_NONE)
        elif source in ("none", ""):
            trigger.or_mask(spcm.SPC_TMASK_NONE)
            trigger.and_mask(spcm.SPC_TMASK_NONE)
        else:
            raise ValueError(f"unsupported trigger source: {source}")
        self.logger.debug(f"trigger: source={source}, level={level} V, edge={edge}")
        return True

    def configure_markers(self, line_to_source_bit: dict[int, tuple[int, int]]):
        """Route sample bits to X lines (synchronous digital output).

        :param line_to_source_bit: mapping X line index (0..3) to
            ``(physical analog source channel, sample bit)``.

        """

        card = self._require_card()
        spcm = self._import_spcm()
        for x_index, (source, bit) in line_to_source_bit.items():
            V.check_nonneg_int(x_index, "marker X line")
            V.check_nonneg_int(source, f"marker X{x_index} source channel")
            V.check_nonneg_int(bit, f"marker X{x_index} sample bit")
            if not (0 <= int(source) < int(card.num_channels())):
                raise ValueError(f"invalid marker source channel: {source}")
            if not (0 <= int(bit) <= 15):
                raise ValueError(f"invalid sample bit: {bit}")
            mode = (
                spcm.SPCM_XMODE_DIGOUT
                | (spcm.SPCM_XMODE_DIGOUTSRC_CH0 << int(source))
                | (spcm.SPCM_XMODE_DIGOUTSRC_BIT15 << (15 - int(bit)))
            )
            card.set_i(spcm.SPCM_X0_MODE + int(x_index), mode)
            self.logger.debug(f"X{x_index}: DIGOUT of CH{source} bit {bit}")

    REPLAY_MODES = ("singlerestart", "single")

    def _card_mode(self, replay_mode: str) -> int:
        spcm = self._import_spcm()
        modes = {
            "singlerestart": spcm.SPC_REP_STD_SINGLERESTART,
            "single": spcm.SPC_REP_STD_SINGLE,
        }
        key = V.check_str(replay_mode, "replay_mode").lower()
        if key not in modes:
            raise ValueError(f"unsupported replay mode: {replay_mode} (use {self.REPLAY_MODES})")
        return modes[key]

    def upload_samples(
        self, samples: dict[int, np.ndarray], replay_mode: str = "singlerestart", loops: int = 0
    ) -> int:
        """Upload raw int16 samples for one or two enabled physical channels.

        :param samples: mapping physical channel index to a 1-D int16 array.
            Markers, if any, are already packed into the upper bits.
        :param loops: SPC_LOOPS. 0 = infinite. Meaning depends on mode:
            singlerestart: number of accepted triggers;
            single: number of replays after the one trigger.
        :returns: number of uploaded samples.

        """

        card = self._require_card()
        spcm = self._import_spcm()
        if not samples or len(samples) > 2:
            raise ValueError("samples must contain one or two physical channels")
        arrays = {}
        lengths = set()
        for channel, data in samples.items():
            channel = int(V.check_nonneg_int(channel, "sample channel"))
            data = np.ascontiguousarray(data)
            if data.dtype != np.int16 or data.ndim != 1:
                raise ValueError(
                    f"CH{channel} samples must be a 1-D int16 array, "
                    f"got {data.dtype}, ndim={data.ndim}"
                )
            arrays[channel] = data
            lengths.add(len(data))
        if len(lengths) != 1:
            raise ValueError(f"channel sample lengths differ: {sorted(lengths)}")
        n = lengths.pop()

        if self._channels is None:
            raise RuntimeError("configure_channels() must be called before upload_samples()")
        row_by_channel = {ch.index: ch.data_index for ch in self._channels.channels}
        if set(arrays) != set(row_by_channel):
            raise ValueError(
                f"sample channels {sorted(arrays)} do not match enabled channels "
                f"{sorted(row_by_channel)}"
            )
        mn, step = self.memory_granularity()
        if n < mn or n % step:
            raise ValueError(
                f"waveform length {n} violates card granularity (min {mn}, step {step})"
            )
        info = self.card_info()
        mem = info.get("memory_samples")  # checks whether the waveform fits in onboard AWG memory.
        total = n * len(arrays)
        if mem and total > mem:
            raise ValueError(
                f"waveform uses {total:_d} sample words across {len(arrays)} channels, "
                f"exceeding card memory {mem:_d}"
            )

        self.stop()
        card.card_mode(self._card_mode(replay_mode))
        card.loops(int(V.check_nonneg_int(loops, "loops")))

        self._transfer = spcm.DataTransfer(card)
        self._transfer.memory_size(n)
        self._transfer.allocate_buffer(n)
        buf = self._transfer.buffer
        if buf.ndim == 1:
            if len(arrays) != 1:
                raise RuntimeError(f"unexpected one-dimensional buffer for {len(arrays)} channels")
            buf[:] = next(iter(arrays.values()))
        elif buf.ndim == 2:
            if buf.shape != (len(arrays), n):
                raise RuntimeError(
                    f"unexpected transfer buffer shape {buf.shape}; expected {(len(arrays), n)}"
                )
            for channel, data in arrays.items():
                buf[row_by_channel[channel], :] = data
        else:
            raise RuntimeError(f"unexpected transfer buffer shape: {buf.shape}")
        self._transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)
        self._uploaded_samples = n
        self.logger.debug(f"uploaded {n:_d} samples, mode = {replay_mode}, loops = {loops}.")
        return n

    def upload(
        self,
        waveforms: dict[int, Waveform],
        marker_routes: dict[str, MarkerRoute] | None = None,
        replay_mode: str = "singlerestart",
        loops: int = 0,
    ) -> dict:
        """Pack and upload waveforms keyed by physical analog channel.

        :param marker_routes: mapping marker name to its physical XIO route.
        :returns: upload information including marker bits, lines, and sources.

        """

        marker_routes = dict(marker_routes or {})
        packed = {}
        bit_map = {}
        line_to_source_bit = {}
        for channel, waveform in waveforms.items():
            marker_order = [
                name for name, route in marker_routes.items() if route.source == channel
            ]
            data, channel_bits = pack_waveform(waveform, marker_order=marker_order)
            packed[channel] = data
            for name, bit in channel_bits.items():
                route = marker_routes[name]
                bit_map[name] = bit
                line_to_source_bit[route.line] = (route.source, bit)

        n = self.upload_samples(packed, replay_mode=replay_mode, loops=loops)
        if line_to_source_bit:
            self.configure_markers(line_to_source_bit)
        active_routes = {name: marker_routes[name] for name in bit_map}
        return {
            "n_samples": n,
            "marker_bits": bit_map,
            "marker_lines": {name: route.line for name, route in active_routes.items()},
            "marker_sources": {name: route.source for name, route in active_routes.items()},
        }

    def start(self, wait_ready: bool = False, immediate: bool = False):
        """Start the card and enable the trigger engine (arm).

        With replay mode "singlerestart" the card then replays the uploaded
        pattern once per trigger edge (or :meth:`force_trigger`).

        """

        card = self._require_card()
        spcm = self._import_spcm()
        if self._uploaded_samples <= 0:
            raise RuntimeError("no waveform uploaded; call upload() before start().")
        flags = spcm.M2CMD_CARD_ENABLETRIGGER
        if wait_ready:
            flags |= spcm.M2CMD_CARD_WAITREADY
        if immediate:
            flags |= spcm.M2CMD_CARD_FORCETRIGGER
        card.start(flags)
        self.running = True
        self.logger.debug("started (armed).")

    def stop(self):
        if self._card is None or not self.running:
            self.running = False
            return
        spcm = self._import_spcm()
        try:
            self._card.stop(spcm.M2CMD_DATA_STOPDMA)
        except Exception:
            self.logger.exception("Exception while stopping Spectrum AWG.")
        self.running = False
        self.logger.debug("stopped.")

    def force_trigger(self):
        """Issue one software-forced trigger event (works with any trigger mask)."""

        card = self._require_card()
        spcm = self._import_spcm()
        card.set_i(spcm.SPC_M2CMD, spcm.M2CMD_CARD_FORCETRIGGER)

    def reset(self):
        """Stop and reset the card state."""

        self.stop()
        card = self._require_card()
        # de-refer these objects for safety before reset.
        self._channels = None
        self._transfer = None
        card.reset()
        # reset bookkeeping attributes
        self._uploaded_samples = 0

        # disable driver timeout; we never block on WAITREADY
        # this looks like the default, but do this explicitly, just in case
        card.timeout(0)


class Dummy_AWG_Core(object):
    """Hardware-less stand-in for :class:`Spectrum_AWG_Core` (conf ``mock: true``).

    Stores configuration and uploaded data so measurement logic can be
    developed and tested on machines without the card / driver.

    """

    def __init__(
        self,
        amp_limit_mV: int = 100,
        granularity: int = 64,
        num_channels: int = 2,
        logger=None,
        **_,
    ):
        self.amp_limit_mV = int(amp_limit_mV)
        self.granularity = granularity
        self.num_channels = int(num_channels)
        self.logger = logger or logging.getLogger(__name__)

        self.running = False
        self._uploaded_samples = 0
        self.last_upload: dict = {}
        self.config: dict = {}

    def open(self):
        self.logger.info("Dummy_AWG_Core: open().")
        return self

    def close(self):
        self.running = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def card_info(self) -> dict:
        return {
            "card_type": "0xa6360 (mock)",
            "serial_number": 0,
            "memory_samples": 2 * 1024**3,
            "bytes_per_sample": 2,
            "max_sample_rate": int(10e9),
            "num_channels": self.num_channels,
            "num_xio_lines": 4,
            "has_sequence_mode": False,
        }

    def memory_granularity(self):
        return self.granularity, self.granularity

    def status(self) -> dict:
        return {
            "raw": 0,
            "pretrigger": False,
            "triggered": False,
            "ready": not self.running,
            "running": self.running,
            "uploaded_samples": self._uploaded_samples,
        }

    def configure_clock(
        self,
        sample_rate: float,
        mode: str = "internal",
        ref_clock_freq: float = 10e6,
        clock_output: bool = False,
        termination_50: bool = True,
    ) -> int:
        # emulate M5i.63xx clock: base 10 GS/s divided by powers of two
        base = 10e9
        req = float(V.check_pos_num(sample_rate, "sample_rate"))

        mode = V.check_str(mode, "clock mode").lower()
        if mode in ("internal", "int", "intpll"):
            pass
        elif mode in ("ext_ref", "extref", "external_reference", "reference"):
            ref_clock_freq = float(V.check_pos_num(ref_clock_freq, "ref_clock_freq"))
        else:
            raise ValueError(f"unsupported clock mode: {mode}")

        div = max(1, 2 ** round(np.log2(base / req))) if req < base else 1
        actual = int(base / div)
        self.config["clock"] = {
            "sample_rate": actual,
            "mode": mode,
            "ref_clock_freq": ref_clock_freq,
            "clock_output": bool(clock_output),
            "termination_50": bool(termination_50),
        }
        if actual != int(round(req)):
            self.logger.warning(
                f"Dummy_AWG_Core: sample rate {int(req):_d} -> {actual:_d} Hz (base / 2^n)."
            )
        return actual

    def _validate_amplitude(self, amplitude_mV: int | float) -> int:
        V.check_num(amplitude_mV, "amplitude_mV")
        amplitude_mV = int(round(amplitude_mV))
        if not (1 <= amplitude_mV <= self.amp_limit_mV):
            raise ValueError(
                f"requested amplitude {amplitude_mV} mV is outside 1 .. {self.amp_limit_mV} mV"
            )
        return amplitude_mV

    def configure_channels(
        self, channels, amplitudes_mV: dict[int, int], filter_=None, stop_level="zero"
    ) -> dict[int, int]:
        active = sorted(
            set(int(V.check_nonneg_int(ch, f"channels[{i}]")) for i, ch in enumerate(channels))
        )
        invalid = any(not (0 <= channel < self.num_channels) for channel in active)
        if not active or len(active) > 2 or invalid:
            raise ValueError(
                f"invalid active channels {active}; available channels are "
                f"{list(range(self.num_channels))}"
            )
        stop_level = stop_level.lower()
        if stop_level not in ("zero", "low", "high", "holdlast"):
            raise ValueError(f"unsupported stop_level: {stop_level}")
        if filter_ is not None:
            filter_ = int(filter_)

        actual = {}
        for channel in active:
            actual[channel] = self._validate_amplitude(amplitudes_mV[channel])
        self.config["channels"] = {
            "active": active,
            "amplitudes_mV": actual,
            "filter": filter_,
            "stop_level": stop_level,
        }
        return actual

    def set_amplitude(self, channel: int, amplitude_mV: int | float) -> int:
        channel = int(V.check_nonneg_int(channel, "channel"))
        if not (0 <= channel < self.num_channels):
            raise ValueError(f"invalid analog channel: {channel}")
        amplitude_mV = self._validate_amplitude(amplitude_mV)
        channels = self.config.get("channels")
        if channels is not None and channel in channels["active"]:
            channels["amplitudes_mV"][channel] = amplitude_mV
        return amplitude_mV

    def configure_trigger(
        self,
        source: str = "ext0",
        level: float = 1.0,
        edge: bool = True,
        termination_50: bool = False,
    ) -> bool:
        V.check_bool(edge, "edge")
        source = V.check_str(source, "trigger source").lower()
        if source in ("ext0", "external", "external0"):
            level = float(V.check_num(level, "trigger level"))
        elif source not in ("software", "soft", "none", ""):
            raise ValueError(f"unsupported trigger source: {source}")
        self.config["trigger"] = {
            "source": source,
            "level": level,
            "edge": edge,
            "termination_50": bool(termination_50),
        }
        return True

    def configure_markers(self, line_to_source_bit: dict[int, tuple[int, int]]):
        routes = {}
        num_lines = int(self.card_info()["num_xio_lines"])
        for line, (source, bit) in line_to_source_bit.items():
            line = int(V.check_nonneg_int(line, "marker X line"))
            source = int(V.check_nonneg_int(source, f"marker X{line} source channel"))
            bit = int(V.check_nonneg_int(bit, f"marker X{line} sample bit"))
            if not (0 <= line < num_lines):
                raise ValueError(f"invalid marker X line: {line}")
            if not (0 <= source < self.num_channels):
                raise ValueError(f"invalid marker source channel: {source}")
            if not (0 <= bit <= 15):
                raise ValueError(f"invalid sample bit: {bit}")
            routes[line] = (source, bit)
        self.config["markers"] = routes

    REPLAY_MODES = Spectrum_AWG_Core.REPLAY_MODES

    def _validate_replay_mode(self, replay_mode: str) -> str:
        key = V.check_str(replay_mode, "replay_mode").lower()
        if key not in self.REPLAY_MODES:
            raise ValueError(f"unsupported replay mode: {replay_mode} (use {self.REPLAY_MODES})")
        return key

    def upload_samples(self, samples, replay_mode="singlerestart", loops=0) -> int:
        if not samples or len(samples) > 2:
            raise ValueError("samples must contain one or two physical channels")
        samples = {
            int(V.check_nonneg_int(channel, "sample channel")): np.ascontiguousarray(data)
            for channel, data in samples.items()
        }
        if any(data.dtype != np.int16 or data.ndim != 1 for data in samples.values()):
            raise ValueError("each channel must contain a 1-D int16 sample array")
        active = self.config.get("channels", {}).get("active")
        if active is None:
            raise RuntimeError("configure_channels() must be called before upload_samples()")
        if set(samples) != set(active):
            raise ValueError(
                f"sample channels {sorted(samples)} do not match enabled channels {active}"
            )
        lengths = {len(data) for data in samples.values()}
        if len(lengths) != 1:
            raise ValueError(f"channel sample lengths differ: {sorted(lengths)}")
        n = lengths.pop()
        mn, step = self.memory_granularity()
        if n < mn or n % step:
            raise ValueError(
                f"waveform length {n} violates card granularity (min {mn}, step {step})"
            )
        memory_samples = self.card_info()["memory_samples"]
        total = n * len(samples)
        if total > memory_samples:
            raise ValueError(
                f"waveform uses {total:_d} sample words across {len(samples)} channels, "
                f"exceeding card memory {memory_samples:_d}"
            )
        replay_mode = self._validate_replay_mode(replay_mode)
        loops = int(V.check_nonneg_int(loops, "loops"))

        self.stop()
        self._uploaded_samples = n
        self.last_upload = {
            "samples": {channel: data.copy() for channel, data in samples.items()},
            "replay_mode": replay_mode,
            "loops": loops,
        }
        return self._uploaded_samples

    def upload(self, waveforms, marker_routes=None, replay_mode="singlerestart", loops=0) -> dict:
        marker_routes = dict(marker_routes or {})
        packed = {}
        bit_map = {}
        line_to_source_bit = {}
        for channel, waveform in waveforms.items():
            marker_order = [
                name for name, route in marker_routes.items() if route.source == channel
            ]
            data, channel_bits = pack_waveform(waveform, marker_order=marker_order)
            packed[channel] = data
            for name, bit in channel_bits.items():
                route = marker_routes[name]
                bit_map[name] = bit
                line_to_source_bit[route.line] = (route.source, bit)
        n = self.upload_samples(packed, replay_mode, loops)
        self.last_upload["waveforms"] = waveforms
        if line_to_source_bit:
            self.configure_markers(line_to_source_bit)
        active_routes = {name: marker_routes[name] for name in bit_map}
        return {
            "n_samples": n,
            "marker_bits": bit_map,
            "marker_lines": {name: route.line for name, route in active_routes.items()},
            "marker_sources": {name: route.source for name, route in active_routes.items()},
        }

    def start(self, wait_ready: bool = False, immediate: bool = False):
        if self._uploaded_samples <= 0:
            raise RuntimeError("no waveform uploaded; call upload() before start().")
        self.running = True

    def stop(self):
        self.running = False

    def force_trigger(self):
        self.logger.debug("Dummy_AWG_Core: force_trigger().")

    def reset(self):
        self.stop()
        self._uploaded_samples = 0
        self.last_upload = {}
        self.config = {}


class Spectrum_AWG(Instrument, ConfAccessorMixin):
    """MAHOS Instrument for the Spectrum M5i.63xx AWG.

    :param resource: (optional) card resource such as ``/dev/spcm0``.
    :type resource: str
    :param serial_number: (optional) card serial number, e.g. 24423.
        If neither resource nor serial_number is given, the first AO card is used.
    :type serial_number: int
    :param mock: (default: False) use :class:`Dummy_AWG_Core` (no hardware / driver).
    :type mock: bool
    :param mock_num_channels: (default: 2) number of physical channels reported by
        :class:`Dummy_AWG_Core`.
    :type mock_num_channels: int
    :param verbose: (default: False) enable verbose output from the ``spcm`` card object.
    :type verbose: bool
    :param amplitude_mV: (default: 100) output amplitude, zero-to-peak into 50 Ohm.
        An integer applies to every analog channel; a mapping sets values by physical
        channel index.
    :type amplitude_mV: int | dict[int, int]
    :param amp_limit_mV: (default: 100) hard software cap for amplitude_mV.
    :type amp_limit_mV: int
    :param memory_granularity: (default: 64) waveform memory granularity.
    :type memory_granularity: int
    :param clock_mode: (default: "internal") "ext_ref" (external 10 MHz
        reference) or "internal".
    :type clock_mode: str
    :param ref_clock_freq: (default: 10e6) external reference frequency in Hz.
    :type ref_clock_freq: float
    :param clock_output: (default: False) enable clock output connector.
    :type clock_output: bool
    :param trigger_level: (default: 1.0) ext0 trigger level in volts.
    :type trigger_level: float
    :param trigger_termination_50: (default: False) ext0 input termination.
    :type trigger_termination_50: bool
    :param markers: mapping marker name to a route with ``line`` (XIO line index)
        and ``source`` (physical analog channel carrying the embedded bit). The default is
        ``{"trigger": {"line": 0, "source": 0}, "laser": {"line": 1, "source": 0}}``.
        A integer value is interpreted as an XIO line sourced from channel 0.
        Every configured source channel must be active when uploading a waveform.
    :type markers: dict[str, dict[str, int] | int]
    :param digital_min_duration: (default: 4e-9) minimum duration in seconds of a
        constant synchronous digital-output level, reported through ``get("bounds")``.
    :type digital_min_duration: float
    :param load_impedance: (default: 50.0) load impedance in Ohm used for
        dBm <-> voltage conversion (set("power", dBm), get("power")).
    :type load_impedance: float
    :param stop_level: (default: "zero") analog level while not replaying:
        "zero", "low", "high", "holdlast".
    :type stop_level: str
    :param filter: (optional) SPC_FILTER0 value, see manual.
    :type filter: int
    :param file_transport_dir: (optional) reader-side directory for HDF5 waveform transport.
        This may differ from the writer-side path if both refer to the same shared storage.
    :type file_transport_dir: str

    Standard API
    ------------

    - ``configure(params, label)`` with labels:

      - ``"waveforms"`` (see :class:`AWGInterface <mahos.inst.awg_interface.AWGInterface>`):
        params ``analog`` (dict of one or two physical channel indices -> normalized float
        samples), ``digital`` (dict name -> bit array or RLE), ``rate``, ``trigger_type``,
        ``n_runs``, ``trigger_level`` (optional, defaults to conf["trigger_level"]).
      - ``"waveforms_file"``: like ``"waveforms"``, but ``file_name`` names an HDF5 payload
        in ``file_transport_dir`` and the waveform arrays are not sent through ZeroMQ.
      - ``"cw"``: Not supported yet.
      - ``"sequence"``: Not supported yet.

    - ``start()``: arm (each trigger replays the pattern once in singlerestart).
    - ``stop()``: stop replay.
    - ``set("trigger")``: force one trigger.
      ``set("amplitude", (channel, mv))``. ``set("power", (channel, dBm))``:
      amplitude from dBm via load_impedance.
    - ``get("opc")``, ``get("status")``, ``get("info")``, ``get("sample_rate")``,
      ``get("waveform_info")``, ``get("finished")``, ``get("length")``,
      ``get("offsets")``, ``get("power")``, ``get("bounds")``.

    """

    def __init__(self, name: str, conf: dict | None = None, prefix: str | None = None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self._mock = self._conf_bool("mock", False)
        core_kwargs = dict(
            resource=self.conf.get("resource"),
            serial_number=self.conf.get("serial_number"),
            amp_limit_mV=self._conf_pos_int("amp_limit_mV", 100),
            granularity=self._conf_pos_int("memory_granularity", 64),
            verbose=self._conf_bool("verbose", False),
            logger=self.logger,
        )
        if self._mock:
            core_kwargs["num_channels"] = self._conf_pos_int("mock_num_channels", 2)
            self._core = Dummy_AWG_Core(**core_kwargs)
        else:
            self._core = Spectrum_AWG_Core(**core_kwargs)

        raw_markers = self.conf.get(
            "markers",
            {
                "trigger": {"line": 0, "source": 0},
                "laser": {"line": 1, "source": 0},
            },
        )
        self.markers = self._parse_marker_routes(raw_markers)

        self.load_impedance = self._conf_pos_num("load_impedance", 50.0)
        self._stop_level = self._conf_str("stop_level", "zero")
        self._filter = None if self.conf.get("filter") is None else int(self._conf_int("filter"))
        file_transport_dir = self._conf_str("file_transport_dir", "")
        self._file_transport = (
            SharedFileTransport(file_transport_dir) if file_transport_dir else None
        )
        self.digital_min_duration = self._conf_nonneg_num("digital_min_duration", 4e-9)
        self._hardware_trigger = {
            "level": self._conf_num("trigger_level", 1.0),
            "termination_50": self._conf_bool("trigger_termination_50", False),
        }
        self._clock_conf = {
            "mode": self._conf_str("clock_mode", "internal"),
            "ref_clock_freq": self._conf_pos_num("ref_clock_freq", 10e6),
            "clock_output": self._conf_bool("clock_output", False),
        }

        # Open the card and discover capabilities.
        self._core.open()
        info = self._core.card_info()
        self.analog_channels = tuple(range(int(info["num_channels"])))
        if any(route.source not in self.analog_channels for route in self.markers.values()):
            raise ValueError(
                f"marker source must be in available analog channels {self.analog_channels}: "
                f"{self.markers}"
            )

        # bookkeeping of the last configure()
        self._trigger_type: TriggerType = TriggerType.IMMEDIATE
        self._length: int = 0
        self._sample_rate: int = 0
        self._offsets: list[int] = []
        self._wave_info: dict = {}

        # Reset the card state and set amplitude only (that's managed independently of configure).
        self._reset()
        amplitudes = self._parse_amplitudes(
            self.conf.get("amplitude_mV", 100), self.analog_channels
        )
        self.amplitude_mV = {
            channel: self._core.set_amplitude(channel, amplitude)
            for channel, amplitude in amplitudes.items()
        }

        self.logger.info(
            f"initialized ({'mock' if self._mock else 'spcm'}): "
            f"amplitudes {self.amplitude_mV} mV, markers {self.markers}."
        )

    @staticmethod
    def _parse_amplitudes(raw, analog_channels: tuple[int, ...]) -> dict[int, int]:
        """Normalize strictly typed per-channel amplitude configuration."""

        if not isinstance(raw, dict):
            amplitude = V.check_int(raw, "amplitude_mV")
            return {channel: amplitude for channel in analog_channels}

        normalized = {}
        for key, value in raw.items():
            if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
                channel = int(V.check_int(key, "amplitude_mV channel"))
            elif isinstance(key, str) and key in {str(ch) for ch in analog_channels}:
                channel = int(key)
            else:
                raise ValueError(
                    f"invalid amplitude_mV channel {key!r}; available: {list(analog_channels)}"
                )
            if channel not in analog_channels:
                raise ValueError(
                    f"invalid amplitude_mV channel {channel}; available: {list(analog_channels)}"
                )
            if channel in normalized:
                raise ValueError(f"duplicate amplitude_mV configuration for channel {channel}")
            normalized[channel] = V.check_int(value, f"amplitude_mV[{key!r}]")

        return {channel: normalized.get(channel, 100) for channel in analog_channels}

    @staticmethod
    def _parse_marker_routes(raw_markers: dict) -> dict[str, MarkerRoute]:
        """Normalize and validate named marker route configuration."""

        if not isinstance(raw_markers, dict) or len(raw_markers) > 4:
            raise ValueError("markers must be a mapping with at most four entries")
        markers = {}
        for name, value in raw_markers.items():
            if isinstance(value, dict):
                if set(value) != {"line", "source"}:
                    raise ValueError(
                        f"marker {name!r} route must contain exactly 'line' and 'source'"
                    )
                line, source = value["line"], value["source"]
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                line, source = value
            else:
                line, source = value, 0
            V.check_int(line, f"marker {name!r} line")
            V.check_int(source, f"marker {name!r} source")
            route = MarkerRoute(line, source)
            markers[str(name)] = route

        lines = [route.line for route in markers.values()]
        if len(set(lines)) != len(lines) or any(not (0 <= line <= 3) for line in lines):
            raise ValueError(f"marker X lines must be unique and within 0..3: {markers}")
        if any(not (0 <= route.source <= 1) for route in markers.values()):
            raise ValueError(f"marker source channels must be 0 or 1: {markers}")
        return markers

    @staticmethod
    def _norm_trigger_type(tt) -> TriggerType:
        if isinstance(tt, TriggerType):
            return tt
        if isinstance(tt, str):
            return TriggerType[tt.upper()]
        if isinstance(tt, int) and not isinstance(tt, bool):
            return TriggerType(tt)
        raise ValueError(f"invalid trigger_type: {tt!r}")

    def _granularity(self) -> tuple[int, int]:
        return self._core.memory_granularity()

    def _set_rate(self, rate: int) -> bool:
        sample_rate = self._core.configure_clock(rate, **self._clock_conf)
        if rate != sample_rate:
            return self.fail_with(
                f"sample rate {rate:_d} Hz is not realizable (card: {sample_rate:_d} Hz);"
                "valid rates are base rate / 2^n (see get('bounds'))."
            )
        return True

    def _configure_trigger(
        self, trigger_type: TriggerType, n_runs: int | None, level: float
    ) -> tuple[str, int]:
        """Configure trigger from AWGInterface API semantics.

        :returns: (replay_mode, loops) to use for the upload.

        """

        self._trigger_type = trigger_type
        loops = 0 if n_runs is None else int(n_runs)

        if self._trigger_type == TriggerType.IMMEDIATE:
            # output starts right at start(); n_runs=None repeats infinitely until stop().
            self._core.configure_trigger(source="software")
            return "single", loops
        if self._trigger_type == TriggerType.SOFTWARE:
            # start() arms; each set("trigger") replays the pattern once.
            self._core.configure_trigger(source="none")
            return "singlerestart", loops
        if self._trigger_type in (TriggerType.HARDWARE_RISING, TriggerType.HARDWARE_FALLING):
            # start() arms; each ext0 edge replays the pattern once.
            self._core.configure_trigger(
                source="ext0",
                level=level,
                edge=True if self._trigger_type == TriggerType.HARDWARE_RISING else False,
                termination_50=self._hardware_trigger["termination_50"],
            )
            return "singlerestart", loops
        raise ValueError(f"unsupported trigger_type: {self._trigger_type}")

    @staticmethod
    def _digital_level(name: str, value) -> bool:
        """Validate and normalize one digital level."""

        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        V.check_num(value, f"digital channel {name!r} level")
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"digital channel {name!r}: level must be bool, 0, or 1; got {value!r}")

    @staticmethod
    def _expand_digital(name: str, value, n_expected: int | None) -> np.ndarray:
        """Expand a logical digital channel to a bool array.

        Accepts a bool/0-1 array or a compact RLE list of (value, n_samples) pairs.

        """

        is_rle = (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and isinstance(value[0], (list, tuple))
        )
        if is_rle:
            levels = []
            counts = []
            for run in value:
                if not isinstance(run, (list, tuple)) or len(run) != 2:
                    raise ValueError(
                        f"digital channel {name!r}: each RLE entry must be "
                        f"(value, n_samples); got {run!r}"
                    )
                level, count = run
                levels.append(Spectrum_AWG._digital_level(name, level))
                count = int(V.check_nonneg_int(count, f"digital channel {name!r} RLE n_samples"))
                counts.append(count)

            total = sum(counts)
            if n_expected is not None and total != n_expected:
                raise ValueError(
                    f"digital channel {name!r} has {total} RLE samples, expected {n_expected}"
                )
            return np.repeat(np.asarray(levels, dtype=bool), counts)

        raw = np.asarray(value)
        if raw.ndim != 1:
            raise ValueError(f"digital channel {name!r} must be 1-D, got shape {raw.shape}")
        if n_expected is not None and raw.size != n_expected:
            raise ValueError(
                f"digital channel {name!r} has {raw.size} samples, expected {n_expected}"
            )
        if raw.dtype.kind == "b":
            return np.ascontiguousarray(raw)
        if raw.dtype.kind in "iuf":
            if raw.dtype.kind == "f" and not np.all(np.isfinite(raw)):
                raise ValueError(f"digital channel {name!r} contains a non-finite value")
            if not np.all((raw == 0) | (raw == 1)):
                raise ValueError(f"digital channel {name!r} contains a value other than 0 or 1")
            return np.ascontiguousarray(raw, dtype=bool)
        raise ValueError(
            f"digital channel {name!r} must contain bool or numeric 0/1 values; "
            f"got dtype {raw.dtype}"
        )

    def _check_keys(self, params: dict, allowed: T.Iterable[str], label: str) -> bool:
        unknown = [k for k in params if k not in set(allowed)]
        if unknown:
            return self.fail_with(f"unknown param(s) for label '{label}': {unknown}")
        return True

    def _check_upload_size(self, n: int, num_channels: int) -> bool:
        mem = self._core.card_info().get("memory_samples")
        total = n * num_channels
        if mem and total > mem:
            return self.fail_with(
                f"waveform uses {total:_d} sample words across {num_channels} channels, "
                f"exceeding card memory {mem:_d}."
            )
        return True

    def _upload(
        self,
        waveforms: dict[int, Waveform],
        rate: int,
        replay_mode: str,
        loops: int,
        label: str,
        offsets: list[int] | None = None,
        extra: dict | None = None,
    ) -> bool:
        representative = next(iter(waveforms.values()))
        if not self._check_upload_size(representative.n_samples(), len(waveforms)):
            return False
        info = self._core.upload(
            waveforms, marker_routes=self.markers, replay_mode=replay_mode, loops=loops
        )
        n = int(info["n_samples"])
        self._length = n
        self._sample_rate = rate
        self._offsets = [0] if offsets is None else list(offsets)
        self._wave_info = {
            "label": label,
            "n_samples": n,
            "duration": n / rate,
            "sample_rate": rate,
            "replay_mode": replay_mode,
            "loops": loops,
            "analog_channels": sorted(waveforms),
            "marker_bits": info["marker_bits"],
            "marker_lines": info["marker_lines"],
            "marker_sources": info["marker_sources"],
            "segments": list(representative.segments),
        }
        if extra:
            self._wave_info.update(extra)
        self.logger.info(
            f"configured '{label}': {n:_d} samples ({n / rate * 1e6:.2f} us), "
            f"replay = {replay_mode}, loops = {loops}."
        )
        return True

    # Public API: AWGInterface.configure_waveforms

    def _configure_waveforms(self, params: dict) -> bool:
        self.logger.debug("entering configure_waveforms()")

        if not self._check_keys(
            params,
            ("analog", "digital", "rate", "trigger_type", "n_runs", "trigger_level"),
            "waveforms",
        ):
            return False
        analog = params.get("analog")
        if not isinstance(analog, dict) or not (1 <= len(analog) <= 2):
            return self.fail_with("analog must map one or two physical channel indices to samples")
        if any(isinstance(channel, bool) or not isinstance(channel, int) for channel in analog):
            return self.fail_with(f"analog channel keys must be integers, got {list(analog)}")
        channels = sorted(analog)
        if any(channel not in self.analog_channels for channel in channels):
            return self.fail_with(
                f"analog channels {channels} are not within available channels "
                f"{list(self.analog_channels)}"
            )

        arrays = {}
        lengths = set()
        for channel in channels:
            a = np.asarray(analog[channel], dtype=np.float64)
            if a.ndim != 1 or a.size == 0:
                return self.fail_with(f"CH{channel} analog samples must be a non-empty 1-D array")
            arrays[channel] = a
            lengths.add(int(a.size))
        if len(lengths) != 1:
            return self.fail_with(f"analog channel sample lengths differ: {sorted(lengths)}")
        n = lengths.pop()
        mn, step = self._granularity()
        if n < mn or n % step:
            return self.fail_with(
                f"sample count {n} violates memory granularity (min {mn}, step {step})"
            )

        digital = params.get("digital")
        if digital is None:
            digital = {}
        elif not isinstance(digital, dict):
            return self.fail_with(f"digital must be a mapping, got {type(digital).__name__}")
        unknown = [k for k in digital if k not in self.markers]
        if unknown:
            return self.fail_with(
                f"unknown digital channel(s) {unknown}; configured markers: {list(self.markers)}"
            )
        tracks = {name: self._expand_digital(name, val, n) for name, val in digital.items()}

        inactive_sources = {
            name: route.source
            for name, route in self.markers.items()
            if route.source not in channels
        }
        if inactive_sources:
            return self.fail_with(
                f"configured marker source channels are not active: {inactive_sources}; "
                f"active channels are {channels}"
            )

        p = ParamAccessor(params)
        rate = int(round(p.pos_num("rate")))
        trigger_type = self._norm_trigger_type(p.get("trigger_type"))
        n_runs = p.get("n_runs")
        if n_runs is not None:
            n_runs = int(p.pos_int("n_runs"))
        trigger_level = float(p.num("trigger_level", self._hardware_trigger["level"]))

        waveforms = {
            channel: Waveform(
                sample_rate=rate,
                analog=arrays[channel],
                markers={
                    name: track
                    for name, track in tracks.items()
                    if self.markers[name].source == channel
                },
            )
            for channel in channels
        }
        # pre-check the waveform size to fail fast before modifying the card state.
        if not self._check_upload_size(waveforms[channels[0]].n_samples(), len(waveforms)):
            return False

        # reset before starting actual card state modification.
        self._reset()

        actual_amplitudes = self._core.configure_channels(
            channels,
            self.amplitude_mV,
            filter_=self._filter,
            stop_level=self._stop_level,
        )
        self.amplitude_mV.update(actual_amplitudes)

        if not self._set_rate(rate):
            return False

        replay, loops = self._configure_trigger(trigger_type, n_runs, trigger_level)

        self.logger.debug("starting to upload waveforms")
        return self._upload(waveforms, rate, replay, loops, "waveforms", offsets=[0])

    def _configure_sequence(self, params: dict) -> bool:
        return self.fail_with("Sequence mode is not supported yet.")

    def _configure_waveforms_file(self, params: dict) -> bool:
        if not self._check_keys(
            params,
            ("file_name", "rate", "trigger_type", "n_runs", "trigger_level"),
            "waveforms_file",
        ):
            return False
        if self._file_transport is None:
            return self.fail_with("file_transport is not configured")
        file_name = params.get("file_name")
        try:
            path = self._file_transport.resolve(file_name)
        except ValueError:
            return self.fail_with(f"file_name must be a basename: {file_name!r}")
        analog, digital = load_waveforms(path)
        self.logger.debug(f"loaded waveforms from {path}")
        waveform_params = {key: value for key, value in params.items() if key != "file_name"}
        waveform_params.update(analog=analog, digital=digital)
        return self._configure_waveforms(waveform_params)

    def _configure_cw(self, params: dict) -> bool:
        return self.fail_with("CW mode (DDS) is not supported yet.")

    def _reset(self):
        """Reset core (card) state and bookkeeping attributes."""

        try:
            self._core.reset()
        finally:
            self._length = 0
            self._sample_rate = 0
            self._offsets = []
            self._wave_info = {}

    # Standard API

    def close_resources(self):
        if getattr(self, "_core", None) is not None:
            self._core.close()

    def configure(self, params: dict, label: str = "") -> bool:
        if self.is_closed():
            return False
        params = P.unwrap(params) if params is not None else {}
        try:
            if label == "waveforms":
                return self._configure_waveforms(params)
            elif label == "waveforms_file":
                return self._configure_waveforms_file(params)
            elif label == "sequence":
                return self._configure_sequence(params)
            elif label == "cw":
                return self._configure_cw(params)
            else:
                return self.fail_with(f"unknown configure() label: {label!r}")
        except Exception:
            self.logger.exception(f"error in configure(label={label!r}).")
            return False

    def start(self, label: str = "") -> bool:
        if self.is_closed():
            return False
        try:
            self._core.start(immediate=self._trigger_type == TriggerType.IMMEDIATE)
            return True
        except Exception:
            self.logger.exception("error in start().")
            return False

    def stop(self, label: str = "") -> bool:
        if self.is_closed():
            return False
        try:
            self._core.stop()
            return True
        except Exception:
            self.logger.exception("error in stop().")
            return False

    def reset(self, label: str = "") -> bool:
        if self.is_closed():
            return False
        try:
            self._reset()
            return True
        except Exception:
            self.logger.exception("error in reset().")
            return False

    def shutdown(self) -> bool:
        """Safe stop for power-off.

        stop replay, reset the card state, clear the pattern, and close the card.

        """

        try:
            if not self._closed:
                self._reset()
        except Exception:
            self.logger.exception("error while stopping during shutdown().")
        self.close()
        self.logger.info("shutdown complete.")
        return True

    def set(self, key: str, value=None, label: str = "") -> bool:
        if self.is_closed():
            return False
        key = key.lower()
        if key == "trigger":
            self._core.force_trigger()
            return True
        elif key == "amplitude":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                return self.fail_with("amplitude value must be (channel, amplitude_mV)")
            channel, raw_mv = int(value[0]), value[1]
            if channel not in self.analog_channels:
                return self.fail_with(
                    f"invalid analog channel {channel}; available: {list(self.analog_channels)}"
                )
            mv = int(round(float(raw_mv)))
            if not (1 <= mv <= self._core.amp_limit_mV):
                return self.fail_with(
                    f"amplitude {mv} mV out of bounds (1 .. {self._core.amp_limit_mV} mV)."
                )
            self.amplitude_mV[channel] = self._core.set_amplitude(channel, mv)
            return True
        elif key == "power":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                return self.fail_with("power value must be (channel, power_dBm)")
            channel, power = int(value[0]), float(value[1])
            mv = 1e3 * dBm_to_Vpeak(power, self.load_impedance)
            if mv > self._core.amp_limit_mV + 1e-6:
                max_p = Vpeak_to_dBm(self._core.amp_limit_mV / 1e3, self.load_impedance)
                return self.fail_with(
                    f"power {power:.2f} dBm -> {mv:.1f} mV exceeds the amplitude "
                    f"limit {self._core.amp_limit_mV} mV ({max_p:.2f} dBm)."
                )
            return self.set("amplitude", (channel, mv))
        else:
            return self.fail_with(f"unknown set() key: {key!r}")

    def _get_bounds(self) -> dict:
        info = self._core.card_info()
        mn, step = self._granularity()
        max_rate = info.get("max_sample_rate") or int(10e9)
        min_rate = info.get("min_sample_rate") or 0
        return {
            "analog_channels": self.analog_channels,
            "sample_rate": (float(min_rate), float(max_rate)),
            "amplitude_mV": (1, int(self._core.amp_limit_mV)),
            "power_dBm": Vpeak_to_dBm(self._core.amp_limit_mV / 1e3, self.load_impedance),
            "load_impedance": self.load_impedance,
            "memory_samples": info.get("memory_samples"),
            "granularity": (mn, step),
            "num_xio_lines": info.get("num_xio_lines"),
            "digital_lines": {
                name: {"line": route.line, "source": route.source}
                for name, route in self.markers.items()
            },
            "trigger_types": [t.name for t in TriggerType],
            "has_sequence_mode": bool(info.get("has_sequence_mode", False)),
            "file_transport": self._file_transport is not None,
            "digital_min_duration": self.digital_min_duration,
        }

    def _get_digital_rate(self, sample_rate: float) -> float:
        divisor = 4.0 if sample_rate <= 5.0e9 else 8.0
        return sample_rate / divisor

    def get(self, key: str, args=None, label: str = ""):
        if self.is_closed():
            return None
        key = key.lower()
        if key == "opc":
            # all driver operations are synchronous; optional args is a settle delay in sec.
            if args:
                time.sleep(float(args))
            return True
        elif key == "status":
            return self._core.status()
        elif key == "info":
            return self._core.card_info()
        elif key == "sample_rate":
            return float(self._sample_rate)
        elif key == "waveform_info":
            return dict(self._wave_info)
        elif key == "finished":
            return bool(self._core.status().get("ready", False))
        elif key == "length":
            return int(self._length)
        elif key == "offsets":
            return list(self._offsets)
        elif key == "power":
            channel = 0 if args is None else int(args)
            if channel not in self.amplitude_mV:
                self.logger.error(
                    f"invalid analog channel {channel}; available: {list(self.analog_channels)}"
                )
                return None
            return Vpeak_to_dBm(self.amplitude_mV[channel] / 1e3, self.load_impedance)
        elif key == "bounds":
            return self._get_bounds()
        elif key == "digital_rate":
            return self._get_digital_rate(float(args))
        else:
            self.logger.error(f"unknown get() key: {key!r}")
            return None
