#!/usr/bin/env python3

"""
Message Types for Arbitrary Waveform Generator Instruments.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from dataclasses import dataclass, field
import typing as T
import enum
import uuid

import numpy as np

from mahos.msgs.common_msgs import Message


class TriggerType(enum.Enum):
    IMMEDIATE = 0  # no trigger, free-run
    SOFTWARE = 1  # software trigger
    HARDWARE_RISING = 2  # hardware trigger, rising edge
    HARDWARE_FALLING = 3  # hardware trigger, falling edge


class AWGWaveform(Message):
    """Bounded preview of a rendered AWG waveform for visualization.

    Analog traces contain normalized full-scale samples and their original AWG sample indices.
    They may be min/max reduced and cover only a prefix of the uploaded waveform. Digital tracks
    use run-length encoding in the original sample grid.

    :ivar analog: physical channel mapped to ``(sample_indices, normalized_samples)`` arrays.
    :ivar digital: logical channel mapped to ``(value, n_samples)`` run-length pairs.
    :ivar rate: AWG sample rate in Hz.
    :ivar amplitude_mV: physical channel mapped to its zero-to-peak output amplitude in mV.
    :ivar rendered_samples: waveform length before hardware-granularity padding.
    :ivar actual_samples: uploaded waveform length including padding.
    :ivar preview_samples: number of source samples covered by this prefix preview.
    :ivar markers: optional region markers in original AWG sample coordinates.
    :ivar reduction: analog reduction method, ``"none"`` or ``"minmax"``.
    :ivar reduction_factor: approximate source samples represented by each analog point.
    :ivar ident: UUID identifying this preview publication.

    """

    def __init__(
        self,
        analog: dict[int, tuple[np.ndarray, np.ndarray]],
        digital: dict[str, list[tuple[bool, int]]],
        rate: float,
        amplitude_mV: dict[int, int | float],
        rendered_samples: int,
        actual_samples: int,
        preview_samples: int,
        markers: list[int] | None = None,
        reduction: str = "none",
        reduction_factor: float = 1.0,
        ident: uuid.UUID | None = None,
    ):

        self.analog = analog
        self.digital = digital
        self.rate = rate
        self.amplitude_mV = amplitude_mV
        self.rendered_samples = rendered_samples
        self.actual_samples = actual_samples
        self.preview_samples = preview_samples
        self.markers = markers
        self.reduction = reduction
        self.reduction_factor = reduction_factor
        self.ident = ident or uuid.uuid4()


@dataclass
class AWGSegment:
    """One reusable analog / digital sample segment of an AWG sequence.

    TODO (reserved): Note that there is no real consumer/provider of this class yet.

    :ivar name: unique segment name, referenced by :class:`AWGStep`.
    :ivar analog: mapping physical analog channel index -> float samples in
        normalized full-scale fraction [-1, +1].
    :ivar digital: mapping logical digital channel name (e.g. ``"laser"``) ->
        bool samples, or RLE list of ``(value, n_samples)`` pairs.

    """

    name: str
    analog: dict[int, np.ndarray] = field(default_factory=dict)
    digital: dict[str, T.Any] = field(default_factory=dict)


@dataclass
class AWGStep:
    """One step of an AWG sequence program.

    TODO (reserved): Note that there is no real consumer/provider of this class yet.

    :ivar segment: name of the :class:`AWGSegment` to replay.
    :ivar loops: number of consecutive replays of the segment (>= 1).
    :ivar transition: behavior after the loops finish:

        - ``"end_loop"`` (default): continue with the next step.
        - ``"end_sequence"``: final step; pattern ends here.
        - ``"end_loop_on_trigger"``: repeat the segment until a trigger event,
          then continue. Requires hardware sequence mode; instruments without
          it (e.g. the M5i.6360-x16, whose feature register reports no
          sequence mode) reject this transition.

    """

    segment: str
    loops: int = 1
    transition: str = "end_loop"
