#!/usr/bin/env python3

"""
Typed Interface for Arbitrary Waveform Generator.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

Two public data forms are provided so that workers are not forced to materialize huge arrays:

- Flat waveform form (:meth:`AWGInterface.configure_waveforms`):
  per-channel sample arrays.
- File-backed flat waveform form (:meth:`AWGInterface.configure_waveforms_file`):
  the same payload in a shared HDF5 transport file.
- Sequence form (:meth:`AWGInterface.configure_sequence`): reusable
  :class:`AWGSegment` sample segments referenced by :class:`AWGStep` entries
  carrying loop counts (e.g. ``Block.Nrep`` maps to ``AWGStep.loops``).

Conventions:

- ``analog`` keys are physical output channel indices and values are NumPy float
  arrays in *normalized* full-scale fraction [-1.0, +1.0]; the physical voltage
  is amplitude (mV or dBm) set on the instrument.
- ``digital`` values are logical bit arrays (bool / 0-1) or compact RLE lists
  of ``(value, n_samples)`` pairs, keyed by logical channel names such as
  ``"laser"`` and ``"trigger"``; the instrument config maps names to digital lines.
- All channels of one waveform / segment must represent the same sample count
  after expansion.

"""

from __future__ import annotations

import typing as T

import numpy as np

from mahos.inst.interface import InstrumentInterface
from mahos.msgs.inst.awg_msgs import TriggerType, AWGSegment, AWGStep


class AWGInterface(InstrumentInterface):
    """Interface for Arbitrary Waveform Generator."""

    def configure_waveforms(
        self,
        analog: dict[int, np.ndarray],
        digital: dict[str, T.Any],
        rate: float,
        trigger_type: TriggerType = TriggerType.IMMEDIATE,
        n_runs: int | None = None,
    ) -> bool:
        """Configure the AWG with flat per-channel waveforms.

        Standardized run semantics (mirroring PGInterface.configure_blocks):

        - ``TriggerType.IMMEDIATE``: output starts immediately on start();
          ``n_runs=None`` repeats infinitely until stop().
        - ``TriggerType.SOFTWARE``: start() arms the AWG; each trigger()
          replays the pattern once.
        - ``TriggerType.HARDWARE_RISING`` / ``HARDWARE_FALLING``: start() arms
          the AWG; each hardware trigger edge replays the pattern once.

        :param analog: mapping physical analog channel index -> float samples in
            normalized full-scale fraction [-1, +1].
        :param digital: mapping logical digital channel name -> bool samples or
            RLE list of (value, n_samples) pairs.
        :param rate: sample rate in Hz. Must be realizable by the instrument
            (the samples are pre-rendered, so silent rate coercion is an error;
            see get_bounds()).
        :param trigger_type: the trigger type.
        :param n_runs: repetition number. if None, runs infinitely.

        """

        return self.configure(
            {
                "analog": analog,
                "digital": digital,
                "rate": rate,
                "trigger_type": trigger_type,
                "n_runs": n_runs,
            },
            label="waveforms",
        )

    def configure_waveforms_file(
        self,
        file_name: str,
        rate: float,
        trigger_type: TriggerType = TriggerType.IMMEDIATE,
        n_runs: int | None = None,
    ) -> bool:
        """Configure the AWG from a waveform file in its static transport directory.

        ``file_name`` must be a basename. The measurement and instrument hosts may map the
        shared directory to different paths; only this basename is transported by ZeroMQ.

        """

        return self.configure(
            {
                "file_name": file_name,
                "rate": rate,
                "trigger_type": trigger_type,
                "n_runs": n_runs,
            },
            label="waveforms_file",
        )

    def configure_sequence(
        self,
        segments: list[AWGSegment],
        steps: list[AWGStep],
        rate: float,
        trigger_type: TriggerType = TriggerType.IMMEDIATE,
        n_runs: int | None = None,
    ) -> bool:
        """Configure the AWG with a sequence of reusable segments.

        TODO (reserved): Note that there is no real consumer/provider of this interface yet.

        Instruments with hardware sequence mode map this to segment memory and
        a step program; instruments without it may flatten the expanded
        sequence into a single waveform (memory permitting).

        :param segments: reusable sample segments (unique names).
        :param steps: sequence program; each step references a segment by name
            and carries a loop count and transition behavior.
        :param rate: sample rate in Hz (see configure_waveforms()).
        :param trigger_type: the trigger type.
        :param n_runs: repetition number of the whole sequence. if None, runs infinitely.

        """

        return self.configure(
            {
                "segments": segments,
                "steps": steps,
                "rate": rate,
                "trigger_type": trigger_type,
                "n_runs": n_runs,
            },
            label="sequence",
        )

    def trigger(self) -> bool:
        """Issue software trigger (one replay when armed)."""

        return self.set("trigger")

    def get_digital_rate(self, sample_rate: float) -> float | None:
        """Get synchronous digital update rate for sample_rate."""

        return self.get("digital_rate", sample_rate)

    def get_finished(self) -> bool:
        """Get if the AWG has finished the configured runs and is ready."""

        return self.get("finished")

    def get_opc(self, delay=None) -> bool:
        """Get OPC (operation complete) status."""

        return self.get("opc", delay)

    def get_length(self) -> int:
        """Get total logical sample length of last configure().

        Zero means no active configuration.

        """

        return self.get("length")

    def get_sample_rate(self) -> float:
        """Get sampling rate of last configure().

        Zero means no active configuration.

        """

        return self.get("sample_rate")

    def get_offsets(self) -> list[int]:
        """Get sample offsets of the sequence steps (or [0] for flat waveforms)."""

        return self.get("offsets")

    def get_bounds(self) -> dict:
        """Get instrument bounds / capabilities.

        Reported keys include: ``analog_channels`` (physical channel indices),
        ``sample_rate`` (min, max), ``amplitude_mV`` (min, max),
        ``power_dBm`` (max at the amplitude limit),
        ``memory_samples``, ``granularity`` (min size, step),
        ``num_xio_lines``, ``digital_lines`` (name -> route with X line and
        physical source channel),
        ``trigger_types``, ``has_sequence_mode`` and ``file_transport``.

        """

        return self.get("bounds")

    def set_amplitude(self, channel: int, amplitude_mV: float) -> bool:
        """Set the full-scale amplitude of a physical analog output channel."""

        return self.set("amplitude", (channel, amplitude_mV))
