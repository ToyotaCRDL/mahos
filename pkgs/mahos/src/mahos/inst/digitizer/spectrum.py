#!/usr/bin/env python3

"""
Spectrum Instrumentation Digitizer instruments.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import threading
import time
import enum
import math

import numpy as np

from mahos.inst.instrument import Instrument
from mahos.util.queue import RollingQueue


class Spectrum_AnalogIn(Instrument):
    """Spectrum Instrumentation digitizer (AnalogIn card).

    In ``triggered`` mode, each hardware trigger records a fixed-length segment with
    multiple-recording FIFO mode. ``finite=True`` stops after a positive logical ``samples``
    count divisible by ``cb_samples``. In ``stream`` mode, data are streamed continuously with
    single FIFO mode. Triggered logical segments contain post-trigger samples only. The physical
    segment is enlarged as needed for mandatory pre-trigger samples and byte alignment, and the
    pre-trigger region is removed before data reduction and publication. Software triggering is
    rejected in ``triggered`` mode; ``stream`` mode uses a one-time software trigger internally.

    :param lines: channel indices to enable, for example ``[0]`` or ``[0, 1]``.
    :type lines: list[int | str]
    :param channel_mask: explicit channel bit mask. Takes precedence over ``lines``.
    :type channel_mask: int
    :param resource: card resource such as ``/dev/spcm0`` or ``TCPIP::192.168.1.10::inst0::INSTR``.
    :type resource: str
    :param serial_number: card serial number.
    :type serial_number: int
    :param queue_size: (default: 10000) software buffer size.
    :type queue_size: int
    :param silent: (default: False) suppress queue overflow warnings.
    :type silent: bool
    :param unit: (default: V) reported data unit.
    :type unit: str
    :param gain: (default: 1.0) conversion gain in V per output unit.
    :type gain: float
    :param timeout_sec: (default: 5.0) card timeout in seconds.
    :type timeout_sec: float
    :param notify_alignment_bytes: (default: 4096) required byte alignment for FIFO
        notify sizes. Set to ``1`` to disable alignment.
    :type notify_alignment_bytes: int
    :param segment_alignment_bytes: (default: ``notify_alignment_bytes``) required byte
        alignment for physical segments in triggered mode. Set to ``1`` to disable alignment.
    :type segment_alignment_bytes: int
    :param trigger_sample_granularity: (default: 16) sample-count granularity required for
        triggered logical segments and pre-trigger regions.
    :type trigger_sample_granularity: int
    :param min_segment_samples: (default: 32) minimum physical segment length in samples.
    :type min_segment_samples: int
    :param min_pre_trigger_samples: (default: 16) minimum pre-trigger region in samples.
    :type min_pre_trigger_samples: int
    :param min_post_trigger_samples: (default: 16) minimum logical post-trigger segment length
        in samples.
    :type min_post_trigger_samples: int
    :param strict: (default: True) If True, check realized sampling rate strictly -
        the configuration fails whenever requested and realized rates don't match.
    :type strict: bool

    """

    class Mode(enum.Enum):
        UNCONFIGURED = 0
        STREAM = 1
        TRIGGERED = 2

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        if "channel_mask" not in self.conf:
            self.check_required_conf(("lines",))
        self.lines = self.conf.get("lines", [0])
        self.queue_size = self.conf.get("queue_size", 10000)
        self.queue = RollingQueue(self.queue_size)
        self._silent = self.conf.get("silent", False)
        self.unit = self.conf.get("unit", "V")
        self.gain = float(self.conf.get("gain", 1.0))
        self._strict = self.conf.get("strict", True)
        self._trigger_sample_granularity = max(
            1, int(self.conf.get("trigger_sample_granularity", 16))
        )
        self._min_segment_samples = max(1, int(self.conf.get("min_segment_samples", 32)))
        self._min_pre_trigger_samples = max(1, int(self.conf.get("min_pre_trigger_samples", 16)))
        self._min_post_trigger_samples = max(1, int(self.conf.get("min_post_trigger_samples", 16)))

        self._spcm = None
        self._card = None
        self._channels = None
        self._transfer = None
        self._reader = None
        self._stop_ev = threading.Event()
        self.running = False

        self._mode = self.Mode.UNCONFIGURED
        self._trigger_source = None
        self._stamp = False
        self._line_num = 0
        self._oversample = 1
        self._block_samples = 1
        self._block_reduce_factor = 1
        self._block_reduce_op = "mean"
        self._reduce_factor = 1
        self._reduce_op = "mean"
        self._hardware_average = False
        self._averages = 1
        self._drop_records_left = 0
        self._logical_segment_samples = 0
        self._segment_samples = 0
        self._pre_trigger_samples = 0
        self._post_trigger_samples = 0
        self._record_samples = 0
        self._stream_samples = 0
        self._sampling_rate = 0
        self._stream_epoch_ns: int | None = None
        self._stream_emitted_samples = 0
        self._notify_samples = 0
        self._buffer_samples = 0
        self._pending: list[list[np.ndarray]] = []

        if not self.conf.get("mock", False):
            self._import_spcm()
            self._open_card()

    def _import_spcm(self):
        if self._spcm is None:
            import spcm

            self._spcm = spcm
        return self._spcm

    def _open_card(self):
        spcm = self._import_spcm()
        if self._card is not None:
            return self._card

        kwargs = {"verbose": self.conf.get("verbose", False)}
        if self.conf.get("serial_number") is not None:
            kwargs["serial_number"] = self.conf["serial_number"]
            self._card = spcm.Card(**kwargs)
        elif self.conf.get("resource"):
            self._card = spcm.Card(self.conf["resource"], **kwargs)
        else:
            self._card = spcm.Card(card_type=spcm.SPCM_TYPE_AI, **kwargs)
        self._card.open()
        return self._card

    def _channel_index(self, line) -> int:
        if isinstance(line, str):
            s = line.strip().lower()
            for prefix in ("channel", "ch"):
                if s.startswith(prefix):
                    s = s[len(prefix) :]
                    break
            return int(s)
        return int(line)

    def _channel_mask(self) -> int:
        spcm = self._import_spcm()
        if "channel_mask" in self.conf:
            return int(self.conf["channel_mask"])
        mask = 0
        for line in self.lines:
            idx = self._channel_index(line)
            mask |= int(getattr(spcm, f"CHANNEL{idx}", 1 << idx))
        return mask

    def _get_amp_offset(self, path: int, lower_V: float, upper_V: float) -> tuple[int, int]:
        amps = {0: [200, 500, 1000, 2000, 5000, 10000], 1: [500, 1000, 2500, 5000]}[path]
        lower_V, upper_V = float(lower_V), float(upper_V)
        if not (math.isfinite(lower_V) and math.isfinite(upper_V)) or lower_V >= upper_V:
            raise ValueError(f"requested bounds are invalid: ({lower_V}, {upper_V})")

        center_V = (lower_V + upper_V) / 2.0
        half_span_V = (upper_V - lower_V) / 2.0
        required_amp_V = max(half_span_V, abs(center_V))
        for amp_mV in amps:
            amp_V = amp_mV / 1e3
            if amp_V < required_amp_V and not math.isclose(amp_V, required_amp_V):
                continue

            offset = int(round(-center_V / amp_V * 100))
            actual_center_V = -amp_V * offset / 100.0
            if actual_center_V - amp_V <= lower_V and upper_V <= actual_center_V + amp_V:
                return amp_mV, offset

        raise ValueError(f"requested bounds cannot be represented: ({lower_V}, {upper_V})")

    def _setup_channels(self, params) -> bool:
        spcm = self._import_spcm()
        card = self._open_card()
        self._channels = spcm.Channels(card, card_enable=self._channel_mask())
        self._line_num = len(self._channels)

        # Don't change the order of settings.
        # 1: input path
        path = int(params.get("input_path", self.conf.get("input_path", 0)))
        if path not in (0, 1):
            return self.fail_with(f"Invalid input path {path} (must be 0 or 1)")
        self._channels.path(path)

        # 2: termination
        termination = int(params.get("termination", self.conf.get("termination", 1_000_000)))
        if path == 1 and termination != 50:
            return self.fail_with("For path 1, termination is always 50.")
        if path == 0:
            # don't set termination for path 1 that's always 50 Ohm (attempt will raise error).
            if termination == 1_000_000:
                self._channels.termination(0)
            elif termination == 50:
                self._channels.termination(1)
            else:
                return self.fail_with(f"invalid termination: {termination} (must be 50 or 1 M)")

        # 3: coupling
        DC_coupling = bool(params.get("DC_coupling", self.conf.get("DC_coupling", True)))
        if DC_coupling:
            self._channels.coupling(spcm.COUPLING_DC)
        else:
            self._channels.coupling(spcm.COUPLING_AC)

        self.logger.debug(
            "Channels: path = {}, termination = {}, coupling = {}".format(
                path, "50" if termination == 50 else "HighZ", "DC" if DC_coupling else "AC"
            )
        )

        # 4: bounds (amp and offset)
        bounds = self._get_bounds(params)
        if len(bounds) != len(self._channels):
            return self.fail_with(
                f"len(bounds) is invalid: {len(bounds)} != {len(self._channels)}."
            )
        for ch, bnds in zip(self._channels, bounds):
            if len(bnds) == 2 and isinstance(bnds[0], (float, int, np.integer, np.floating)):
                lb, ub = bnds
                amp, offset = self._get_amp_offset(path, lb, ub)
                amp_set = ch.amp(amp * spcm.units.mV, return_unit=spcm.units.mV)
                ofs_set = ch.offset(offset * spcm.units.percent, return_unit=spcm.units.V)
                self.logger.debug(f"{ch} range: Amp = {amp_set}, Offset = {ofs_set}")
            else:
                return self.fail_with(f"invalid bounds: {bnds}")

        return True

    def _setup_clock(self, params):
        spcm = self._import_spcm()
        clock = spcm.Clock(self._open_card())
        mode = str(
            params.get("clock_mode_spcm", self.conf.get("clock_mode_spcm", "intpll"))
        ).lower()
        if mode == "intpll":
            clock.mode(spcm.SPC_CM_INTPLL)
        else:
            return self.fail_with(f"unsupported clock mode: {mode}")
        rate_req = int(round(params["rate"])) * spcm.units.Hz
        rate_ret = clock.sample_rate(rate_req, return_unit=spcm.units.Hz)
        msg = f"Sampling rate: requested = {rate_req.magnitude:_d} Hz, "
        msg += f"actual = {rate_ret.magnitude:_d} Hz"
        if self._strict and rate_req != rate_ret:
            return self.fail_with(msg)
        self._sampling_rate = int(rate_ret.magnitude)
        self.logger.debug(msg)
        return True

    def _setup_trigger(self, params):
        spcm = self._import_spcm()
        trigger = spcm.Trigger(self._open_card())
        source = str(params.get("trigger_source", self.conf.get("trigger_source", "ext0"))).lower()
        direction = params.get("trigger_dir", True)
        mode = spcm.SPC_TM_POS if direction else spcm.SPC_TM_NEG

        if source in ("ext0", "external0"):
            trigger.or_mask(spcm.SPC_TMASK_EXT0)
            trigger.ext0_mode(mode)
            if "trigger_termination" in params or "trigger_termination" in self.conf:
                termination = int(
                    params.get("trigger_termination", self.conf.get("trigger_termination"))
                )
            else:
                termination = int(
                    params.get("termination", self.conf.get("termination", 1_000_000))
                )
            if termination == 1_000_000:
                trigger.termination(0)
                self.logger.debug("trigger ext0 termination: HighZ (1 M)")
            elif termination == 50:
                trigger.termination(1)
                self.logger.debug("trigger ext0 termination: 50 Ohm")
            else:
                return self.fail_with(f"invalid termination: {termination} (must be 50 or 1 M)")
            trigger.ext0_coupling(spcm.COUPLING_DC)
            trigger.ext0_level0(
                float(params.get("trigger_level", self.conf.get("trigger_level", 0.25)))
                * spcm.units.V
            )
            trigger.and_mask(spcm.SPC_TMASK_NONE)
            self._trigger_source = "ext0"
        elif source in ("software", "soft"):
            trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
            trigger.and_mask(spcm.SPC_TMASK_NONE)
            self._trigger_source = "software"
        elif source in ("none", ""):
            trigger.or_mask(spcm.SPC_TMASK_NONE)
            trigger.and_mask(spcm.SPC_TMASK_NONE)
            self._trigger_source = "none"
        else:
            return self.fail_with(f"unsupported trigger_source: {source}")
        return True

    def _start_transfer(self, *commands):
        self._transfer.start_buffer_transfer(*commands)

    def _notify_alignment_bytes(self) -> int:
        return max(1, int(self.conf.get("notify_alignment_bytes", 4096)))

    def _segment_alignment_bytes(self) -> int:
        alignment = self.conf.get("segment_alignment_bytes")
        if alignment is None:
            return self._notify_alignment_bytes()
        return max(1, int(alignment))

    def _samples_to_bytes(self, samples: int) -> int:
        return int(self._transfer.samples_to_bytes(int(samples)))

    def _ceil_to_multiple(self, value: int, step: int) -> int:
        value = max(1, int(value))
        step = max(1, int(step))
        return ((value + step - 1) // step) * step

    def _align_notify_samples(self, min_samples: int, step_samples: int = 1) -> int:
        alignment = self._notify_alignment_bytes()
        samples = self._ceil_to_multiple(min_samples, step_samples)
        if alignment <= 1:
            return samples

        max_trials = int(self.conf.get("notify_alignment_trials", alignment))
        for _ in range(max_trials):
            if self._samples_to_bytes(samples) % alignment == 0:
                return samples
            samples += max(1, int(step_samples))
        raise ValueError(
            f"could not align notify_samples to {alignment} bytes "
            f"from {min_samples} samples with step {step_samples}"
        )

    def _align_segment_samples(self, logical_samples: int) -> int:
        step = self._trigger_sample_granularity
        min_samples = max(
            self._min_segment_samples,
            int(logical_samples) + self._min_pre_trigger_samples,
        )
        alignment = self._segment_alignment_bytes()
        samples = self._ceil_to_multiple(min_samples, step)
        if alignment <= 1:
            return samples

        max_trials = max(1, int(self.conf.get("segment_alignment_trials", alignment)))
        for _ in range(max_trials):
            if self._samples_to_bytes(samples) % alignment == 0:
                return samples
            samples += step
        raise ValueError(
            f"could not align physical segment to {alignment} bytes "
            f"from {min_samples} samples with step {step}"
        )

    def _align_buffer_samples(self, min_samples: int, notify_samples: int) -> int:
        return self._ceil_to_multiple(max(min_samples, notify_samples), notify_samples)

    def _finite_notify_samples(
        self, min_samples: int, total_samples: int, step_samples: int
    ) -> int:
        """Find an aligned segment-boundary notify size that divides a finite transfer."""

        min_samples = self._ceil_to_multiple(min_samples, step_samples)
        alignment = self._notify_alignment_bytes()
        divisors = set()
        for divisor in range(1, math.isqrt(total_samples) + 1):
            if total_samples % divisor == 0:
                divisors.add(divisor)
                divisors.add(total_samples // divisor)
        for samples in sorted(divisors):
            if samples < min_samples:
                continue
            if samples % step_samples == 0 and self._samples_to_bytes(samples) % alignment == 0:
                return samples
        raise ValueError(
            "finite transfer has no valid notify size that divides total_samples: "
            f"total_samples={total_samples}, min_samples={min_samples}, "
            f"segment_samples={step_samples}, alignment_bytes={alignment}"
        )

    def _log_fifo_sizes(
        self,
        mode: str,
        requested_notify: int,
        notify_samples: int,
        buffer_samples: int,
        segment_samples: int | None = None,
    ):
        msg = "{:s} FIFO notify_samples: requested = {:_d} ({:_d} B), ".format(
            mode,
            requested_notify,
            self._samples_to_bytes(requested_notify),
        )
        msg += "aligned = {:_d} ({:_d} B); ".format(
            notify_samples,
            self._samples_to_bytes(notify_samples),
        )
        msg += "buffer_samples = {:_d} ({:_d} B), buffer/notify = {:.1f}".format(
            buffer_samples,
            self._samples_to_bytes(buffer_samples),
            buffer_samples / notify_samples,
        )
        if segment_samples is not None:
            msg += "; segment_samples = {:_d}, ".format(segment_samples)
            msg += "notify/segment: requested = {:.1f}, aligned = {:.1f}".format(
                requested_notify / segment_samples,
                notify_samples / segment_samples,
            )
        self.logger.debug(msg)

    def _append_data(self, data: np.ndarray | list[np.ndarray], stamp_ns: int | None = None):
        data = self._convert_gain(data)
        if self._stamp:
            data = (data, time.time_ns() if stamp_ns is None else stamp_ns)
        if not self.queue.append(data) and not self._silent:
            self.logger.warn("queue is overflowing. The oldest data is discarded.")

    def _convert_gain(self, data):
        # early return for efficiency in default NOP (gain = 1) case
        if self.gain == 1.0:
            return data

        if isinstance(data, list):
            return [d / self.gain for d in data]
        return data / self.gain

    def _as_float_array(self, data) -> np.ndarray:
        if hasattr(data, "magnitude"):
            data = data.magnitude
        return np.asarray(data, dtype=np.float64)

    def _convert_segment(self, segment) -> list[np.ndarray]:
        spcm = self._import_spcm()
        if len(segment) != self._segment_samples:
            raise ValueError(
                "physical segment length does not match configured segment size: "
                f"{len(segment)} != {self._segment_samples}"
            )
        start = self._pre_trigger_samples
        segment = segment[start : start + self._logical_segment_samples]
        out = []
        for i, ch in enumerate(self._channels):
            raw = segment[:, i] if segment.ndim > 1 else segment
            data = ch.convert_data(raw, spcm.units.V, averages=self._averages)
            data = self._as_float_array(data)
            out.append(data)
        return out

    def _convert_fifo_block(self, block) -> list[np.ndarray]:
        spcm = self._import_spcm()
        out = []
        for i, ch in enumerate(self._channels):
            raw = block[i, :] if block.ndim > 1 else block
            data = ch.convert_data(raw, spcm.units.V, averages=self._averages)
            data = self._as_float_array(data)
            out.append(data)
        return out

    def _reduce_stream(self, data: np.ndarray) -> np.ndarray:
        if self._oversample > 1:
            if len(data) % self._oversample:
                raise ValueError("stream block length must be divisible by oversample")
            data = data.reshape(-1, self._oversample).mean(axis=1)
        return data

    def _stream_elapsed_ns(self, samples: int) -> int:
        numerator = int(samples) * 1_000_000_000
        return (numerator + self._sampling_rate // 2) // self._sampling_rate

    def _note_stream_notification(self, samples: int):
        if not self._stamp or self._stream_epoch_ns is not None:
            return
        self._stream_epoch_ns = time.time_ns() - self._stream_elapsed_ns(samples)

    def _next_stream_stamp_ns(self) -> int:
        if self._stream_epoch_ns is None:
            raise RuntimeError("stream timestamp epoch is not initialized")
        self._stream_emitted_samples += self._stream_samples
        return self._stream_epoch_ns + self._stream_elapsed_ns(self._stream_emitted_samples)

    def _reduce_record(self, data: np.ndarray) -> np.ndarray:
        if self._oversample > 1:
            if len(data) % self._oversample:
                raise ValueError("record length must be divisible by oversample")
            data = data.reshape(-1, self._oversample).mean(axis=1)
        if self._block_reduce_factor > 1 and not self._hardware_average:
            data = data.reshape(-1, self._block_reduce_factor, self._block_samples)
            if self._block_reduce_op == "sum":
                data = data.sum(axis=1)
            else:
                data = data.mean(axis=1)
            data = data.reshape(-1)
        if self._reduce_factor > 1:
            data = data.reshape(self._reduce_factor, len(data) // self._reduce_factor)
            if self._reduce_op == "sum":
                data = data.sum(axis=0)
            else:
                data = data.mean(axis=0)
        return data

    def _append_record_segments(self, segments: list[np.ndarray]):
        for ch_pending, segment in zip(self._pending, segments):
            ch_pending.append(segment)

        while all(
            sum(len(a) for a in ch_pending) >= self._record_samples for ch_pending in self._pending
        ):
            record = []
            for ch_pending in self._pending:
                data = np.concatenate(ch_pending)
                current = data[: self._record_samples]
                rest = data[self._record_samples :]
                ch_pending.clear()
                if len(rest):
                    ch_pending.append(rest)
                record.append(self._reduce_record(current))
            if self._drop_records_left > 0:
                self._drop_records_left -= 1
                continue
            self._append_data(record[0] if len(record) == 1 else record)

    def _append_stream_samples(self, samples: list[np.ndarray]):
        for ch_pending, data in zip(self._pending, samples):
            ch_pending.append(data)

        while all(
            sum(len(a) for a in ch_pending) >= self._stream_samples for ch_pending in self._pending
        ):
            block = []
            for ch_pending in self._pending:
                data = np.concatenate(ch_pending)
                current = data[: self._stream_samples]
                rest = data[self._stream_samples :]
                ch_pending.clear()
                if len(rest):
                    ch_pending.append(rest)
                block.append(self._reduce_stream(current))
            stamp_ns = self._next_stream_stamp_ns() if self._stamp else None
            self._append_data(block[0] if len(block) == 1 else block, stamp_ns=stamp_ns)

    def _reader_loop(self):
        spcm = self._import_spcm()
        try:
            if self._mode == self.Mode.TRIGGERED:
                for block in self._transfer:
                    if self._stop_ev.is_set():
                        break
                    for segment in block:
                        self._append_record_segments(self._convert_segment(segment))
            else:
                for block in self._transfer:
                    if self._stop_ev.is_set():
                        break
                    self._note_stream_notification(self._notify_samples)
                    converted = self._convert_fifo_block(block)
                    self._append_stream_samples(converted)
        except spcm.SpcmException as e:
            if e.error.value == 0:
                # "no error", no need to log exception.
                return
            self.logger.exception("SpcmException caught in FIFO reader thread:")
        except Exception:
            self.logger.exception("Exception caught in FIFO reader thread:")

    def _get_bounds(self, params):
        bounds = params.get("bounds", self.conf.get("bounds", (-10.0, 10.0)))
        if len(bounds) == 2 and isinstance(bounds[0], (float, int, np.integer, np.floating)):
            return [tuple(bounds)] * self._line_num
        return bounds

    def _validate_reduce_params(self, params) -> bool:
        self._oversample = int(params.get("oversample", 1))
        self._block_samples = int(params.get("block_samples", 1))
        self._block_reduce_factor = int(params.get("block_reduce_factor", 1))
        self._block_reduce_op = str(params.get("block_reduce_op", "mean")).lower()
        self._reduce_factor = int(params.get("reduce_factor", 1))
        self._reduce_op = str(params.get("reduce_op", "mean")).lower()

        if self._oversample < 1:
            return self.fail_with("oversample must be positive.")
        if self._block_samples < 1:
            return self.fail_with("block_samples must be positive.")
        if self._block_reduce_factor < 1:
            return self.fail_with("block_reduce_factor must be positive.")
        if self._block_reduce_op not in ("sum", "mean"):
            return self.fail_with(
                f"block_reduce_op must be 'sum' or 'mean': {self._block_reduce_op}"
            )
        if self._reduce_factor < 1:
            return self.fail_with("reduce_factor must be positive.")
        if self._reduce_op not in ("sum", "mean"):
            return self.fail_with(f"reduce_op must be 'sum' or 'mean': {self._reduce_op}")
        return True

    def _reset_card(self):
        """Stop and reset the card state."""

        self.stop()
        self._mode = self.Mode.UNCONFIGURED
        card = self._open_card()
        # de-refer these objects for safety before reset.
        self._transfer = None
        self._channels = None
        card.reset()

        return card

    def _setup_triggered_card(self, params, card) -> bool:
        spcm = self._import_spcm()

        self._stamp = params.get("stamp", False)
        self._logical_segment_samples = self._oversample * self._block_samples
        self._segment_samples = 0
        self._pre_trigger_samples = 0
        self._post_trigger_samples = 0
        if (
            self._logical_segment_samples < self._min_post_trigger_samples
            or self._logical_segment_samples % self._trigger_sample_granularity
        ):
            return self.fail_with(
                "derived logical segment samples must be an integer multiple of "
                f"{self._trigger_sample_granularity} and at least "
                f"{self._min_post_trigger_samples}: {self._logical_segment_samples}"
            )
        self._drop_records_left = abs(int(params.get("drop_first", 0)))
        self._hardware_average = (
            bool(params.get("hardware_average", True)) and self._block_reduce_factor > 1
        )
        if self._hardware_average and self._block_reduce_op != "mean":
            return self.fail_with("hardware averaging supports block_reduce_op='mean' only.")
        if int(params["cb_samples"]) % self._block_samples:
            return self.fail_with("cb_samples must be integer multiple of block_samples.")
        self._averages = self._block_reduce_factor if self._hardware_average else 1
        self._record_samples = int(params["cb_samples"]) * self._oversample * self._reduce_factor
        if not self._hardware_average:
            self._record_samples *= self._block_reduce_factor
        if self._record_samples % self._logical_segment_samples:
            return self.fail_with(
                "record samples must be an integer multiple of logical segment samples."
            )

        timeout_sec = float(params.get("timeout_sec", self.conf.get("timeout_sec", 5.0)))
        if self._hardware_average:
            card.card_mode(spcm.SPC_REC_FIFO_AVERAGE)
            self.logger.debug("Card mode: SPC_REC_FIFO_AVERAGE")
        else:
            card.card_mode(spcm.SPC_REC_FIFO_MULTI)
            self.logger.debug("Card mode: SPC_REC_FIFO_MULTI")
        card.timeout(timeout_sec * spcm.units.s)

        if not self._setup_trigger(params):
            return False

        if not self._setup_channels(params):
            return False

        if not self._setup_clock(params):
            return False

        if self._hardware_average:
            self._transfer = spcm.BlockAverage(card)
            allocated = self._transfer.averages(self._block_reduce_factor)
            self.logger.debug(f"allocated averages: {allocated}")
            if allocated != self._block_reduce_factor:
                return self.fail_with("failed to allocate requested averages")
        else:
            self._transfer = spcm.Multi(card)

        return True

    def _setup_triggered_buffer(
        self, params, finite: bool, logical_samples: int, cb_samples: int, card
    ) -> bool:
        try:
            self._segment_samples = self._align_segment_samples(self._logical_segment_samples)
        except ValueError:
            self.logger.exception()
            return False

        segments_per_record = self._record_samples // self._logical_segment_samples
        physical_record_samples = segments_per_record * self._segment_samples
        record_count = logical_samples // cb_samples if finite else 0
        card.loops(record_count * segments_per_record)
        requested_notify_samples = physical_record_samples

        if params.get("buffer_segments"):
            requested_buffer_samples = int(params["buffer_segments"]) * self._segment_samples
        else:
            if params.get("buffer_size", 0):
                buffer_samples = params["buffer_size"]
            else:
                buffer_samples = params["samples"]
            buffer_samples *= self._oversample * self._reduce_factor
            if not self._hardware_average:
                buffer_samples *= self._block_reduce_factor
            buffer_records = max(
                1,
                (buffer_samples + self._record_samples - 1) // self._record_samples,
            )
            requested_buffer_samples = buffer_records * physical_record_samples

        try:
            if finite:
                self._notify_samples = self._finite_notify_samples(
                    requested_notify_samples,
                    record_count * physical_record_samples,
                    self._segment_samples,
                )
            else:
                self._notify_samples = self._align_notify_samples(
                    requested_notify_samples, step_samples=self._segment_samples
                )
        except ValueError:
            self.logger.exception()
            return False

        self._buffer_samples = self._align_buffer_samples(
            requested_buffer_samples, self._notify_samples
        )
        buffer_segments = self._buffer_samples // self._segment_samples
        self._log_fifo_sizes(
            "triggered",
            requested_notify_samples,
            self._notify_samples,
            self._buffer_samples,
            self._segment_samples,
        )
        self._transfer.allocate_buffer(
            segment_samples=self._segment_samples, num_segments=buffer_segments
        )
        post_trigger = self._transfer.post_trigger(self._logical_segment_samples)
        self._post_trigger_samples = int(
            post_trigger.magnitude if hasattr(post_trigger, "magnitude") else post_trigger
        )
        if self._post_trigger_samples != self._logical_segment_samples:
            return self.fail_with(
                "realized post-trigger samples do not match the requested logical segment: "
                f"{self._post_trigger_samples} != {self._logical_segment_samples}"
            )
        self._pre_trigger_samples = self._segment_samples - self._post_trigger_samples
        if (
            self._pre_trigger_samples < self._min_pre_trigger_samples
            or self._pre_trigger_samples % self._trigger_sample_granularity
        ):
            return self.fail_with(
                "realized pre-trigger samples must be an integer multiple of "
                f"{self._trigger_sample_granularity} and at least "
                f"{self._min_pre_trigger_samples}: {self._pre_trigger_samples}"
            )
        self._transfer.notify_samples(self._notify_samples)
        if finite:
            self._transfer.to_transfer_samples(record_count * physical_record_samples)

        self.logger.debug(
            "triggered segment samples: "
            f"logical (post-trigger) = {self._logical_segment_samples:_d}, "
            f"physical = {self._segment_samples:_d}, pre-trigger = {self._pre_trigger_samples:_d}"
        )

        return True

    def configure_triggered(self, params: dict) -> bool:
        required = ("trigger_source", "cb_samples", "samples", "rate")
        if not self.check_required_params(params, required):
            return False
        if str(params["trigger_source"]).lower() in ("software", "soft"):
            return self.fail_with("software trigger is not supported in triggered mode.")
        if not self._validate_reduce_params(params):
            return False

        finite = bool(params.get("finite", False))
        logical_samples = int(params["samples"])
        cb_samples = int(params["cb_samples"])
        if cb_samples <= 0:
            return self.fail_with("cb_samples must be positive.")
        if logical_samples <= 0:
            return self.fail_with("samples must be positive.")
        if finite and logical_samples % cb_samples:
            return self.fail_with(
                "samples must be an integer multiple of cb_samples in finite triggered mode: "
                f"samples = {logical_samples}, cb_samples = {cb_samples}"
            )

        card = self._reset_card()

        if not self._setup_triggered_card(params, card):
            return False
        if not self._setup_triggered_buffer(params, finite, logical_samples, cb_samples, card):
            return False

        self.queue = RollingQueue(self.queue_size)
        self._pending = [[] for _ in range(self._line_num)]
        self._mode = self.Mode.TRIGGERED
        self.logger.debug("Configured triggered FIFO mode.")
        return True

    def configure_stream(self, params: dict) -> bool:
        if not self.check_required_params(params, ("cb_samples", "buffer_size", "rate")):
            return False
        if not self._validate_reduce_params(params):
            return False
        cb_samples = int(params["cb_samples"])
        if cb_samples <= 0:
            return self.fail_with("cb_samples must be positive.")

        spcm = self._import_spcm()
        card = self._reset_card()

        self._stamp = params.get("stamp", False)
        self._hardware_average = False
        self._averages = 1

        card.card_mode(spcm.SPC_REC_FIFO_SINGLE)
        self.logger.debug("Card mode: SPC_REC_FIFO_SINGLE")
        timeout_sec = float(params.get("timeout_sec", self.conf.get("timeout_sec", 5.0)))
        card.timeout(timeout_sec * spcm.units.s)
        if not self._setup_trigger({"trigger_source": "software"}):
            return False
        if not self._setup_channels(params):
            return False
        if not self._setup_clock(params):
            return False

        self._transfer = spcm.DataTransfer(card)
        self._stream_samples = cb_samples * self._oversample
        try:
            self._notify_samples = self._align_notify_samples(self._stream_samples)
        except ValueError:
            self.logger.exception()
            return False
        requested_buffer_samples = max(
            self._stream_samples, int(params["buffer_size"]) * self._oversample
        )
        self._buffer_samples = self._align_buffer_samples(
            requested_buffer_samples, self._notify_samples
        )
        self._log_fifo_sizes(
            "stream", self._stream_samples, self._notify_samples, self._buffer_samples
        )
        self._transfer.allocate_buffer(self._buffer_samples)
        self._transfer.notify_samples(self._notify_samples)
        pre_trigger = int(params.get("pre_trigger", self.conf.get("pre_trigger", 16)))
        self._transfer.pre_trigger(pre_trigger)

        self.queue = RollingQueue(self.queue_size)
        self._pending = [[] for _ in range(self._line_num)]
        self._stream_epoch_ns = None
        self._stream_emitted_samples = 0
        self._mode = self.Mode.STREAM
        self.logger.debug("Configured stream FIFO mode.")
        return True

    def get_fifo_status(self) -> dict:
        if self._mode == self.Mode.TRIGGERED:
            return {
                "segment_samples": self._segment_samples,
                "logical_segment_samples": self._logical_segment_samples,
                "pre_trigger_samples": self._pre_trigger_samples,
                "post_trigger_samples": self._post_trigger_samples,
                "record_samples": self._record_samples,
                "notify_samples": self._notify_samples,
                "buffer_samples": self._buffer_samples,
            }
        elif self._mode == self.Mode.STREAM:
            return {
                "stream_samples": self._stream_samples,
                "notify_samples": self._notify_samples,
                "buffer_samples": self._buffer_samples,
            }
        else:
            self.logger.error("fifo status is requested but not configured.")
            return {}

    # Standard API

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "triggered":
            return self.configure_triggered(params)
        elif label == "stream":
            return self.configure_stream(params)
        else:
            return self.fail_with(f"unknown configure label: {label}")

    def start(self, label: str = "") -> bool:
        if self.running:
            self.logger.warn("start() is called while running.")
            return True
        if self._mode == self.Mode.UNCONFIGURED or self._transfer is None:
            return self.fail_with("Must call configure() before start().")

        spcm = self._import_spcm()
        self._stop_ev.clear()
        self._start_transfer(spcm.M2CMD_DATA_STARTDMA)
        start_commands = [spcm.M2CMD_CARD_ENABLETRIGGER]
        if self._trigger_source == "software":
            start_commands.append(spcm.M2CMD_CARD_FORCETRIGGER)
        self._card.start(*start_commands)
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self.running = True
        self.logger.debug("Started Spectrum_AnalogIn.")
        return True

    def stop(self, label: str = "") -> bool:
        if not self.running:
            return True
        self._stop_ev.set()
        try:
            spcm = self._import_spcm()
            self._card.stop(spcm.M2CMD_DATA_STOPDMA)
        except Exception:
            self.logger.exception("Exception while stopping the card.")
        if self._reader is not None:
            self._reader.join(timeout=float(self.conf.get("stop_join_sec", 2.0)))
            if self._reader.is_alive():
                self.logger.error(
                    "Reader thread did not stop before timeout."
                    " Closing card handle to force it to abort."
                )
                # STOPDMA didn't unblock the thread's WAITDMA call (driver-level hang).
                # Closing the handle forces the driver to error out any pending call on it,
                # and the reader thread exits via its own exception handler. A fresh handle
                # is opened on the next configure_*() call so the zombie thread can never
                # touch the new session's card/transfer/queue.
                try:
                    self._card.close()
                except Exception:
                    self.logger.exception("Exception while force-closing the card.")
                self._card = None
                self._reader.join(timeout=float(self.conf.get("stop_join_sec", 2.0)))
                if self._reader.is_alive():
                    self.logger.error("Reader thread still alive after force-close.")
        self._reader = None
        self.running = False
        self._mode = self.Mode.UNCONFIGURED
        self.logger.debug("Stopped Spectrum_AnalogIn.")
        return True

    def reset(self, label: str = "") -> bool:
        if self.is_closed():
            return False
        try:
            self._reset_card()
            return True
        except Exception:
            self.logger.exception("error in reset().")
            return False

    def close_resources(self):
        self.stop()
        if self._card is not None:
            try:
                self._card.close()
            except Exception:
                self.logger.exception("Exception while closing the card.")
            self._card = None

    def pop_opt(self):
        return self.queue.pop_opt()

    def pop_all_opt(self):
        return self.queue.pop_all_opt()

    def pop_block(self):
        return self.queue.pop_block()

    def pop_block_with_status(self):
        return self.queue.pop_block_with_status()

    def pop_all_block(self):
        return self.queue.pop_all_block()

    def set(self, key: str, value=None, label: str = "") -> bool:
        key = key.lower()
        if key == "gain":
            self.gain = float(value)
            return True
        elif key == "unit":
            self.unit = str(value)
            return True
        return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None, label: str = ""):
        if key == "data":
            return self.pop_block() if args else self.pop_opt()
        elif key == "data_with_status":
            return self.pop_block_with_status()
        elif key == "all_data":
            return self.pop_all_block() if args else self.pop_all_opt()
        elif key == "fifo_status":
            return self.get_fifo_status()
        elif key == "unit":
            return self.unit
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
