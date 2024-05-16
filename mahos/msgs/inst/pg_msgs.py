#!/usr/bin/env python3

"""
Message Types for Pulse Generator Instruments.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import typing as T
from collections import UserList
import enum
import copy

import numpy as np
from numpy.typing import NDArray

from ..common_msgs import Message


class TriggerType(enum.Enum):
    IMMEDIATE = 0  # no trigger, free-run
    SOFTWARE = 1  # software trigger
    HARDWARE_RISING = 2  # hardware trigger, rising edge
    HARDWARE_FALLING = 3  # hardware trigger, falling edge


Channels = T.NewType("Channels", T.Union[T.Tuple[str, ...], T.Tuple[int, ...]])


class Pulse(T.NamedTuple):
    channels: Channels
    duration: int


Pattern = T.NewType("Pattern", T.List[Pulse])

AcceptedChannels = T.NewType(
    "AcceptedChannels", T.Union[None, str, int, T.Tuple[str, ...], T.Tuple[int, ...]]
)
AcceptedPulse = T.NewType("AcceptedPulse", T.Tuple[AcceptedChannels, int])
AcceptedPattern = T.NewType("AcceptedPattern", T.List[AcceptedPulse])


class Block(Message):
    def __init__(
        self,
        name: str,
        pattern: AcceptedPattern,
        Nrep: int = 1,
        trigger: bool = False,
    ):
        """The block representation of pulse pattern.

        :param pattern: pattern is list of 2-tuple:
            (channels (tuple/list of str/int or str or int or None),
            period (int))

        channel names is None or empty tuple, output is all-zero during the period.

        """

        self.name = name
        self.pattern = self.regularize_pattern(pattern)
        self.Nrep = Nrep
        self.trigger = trigger

    def regularize_channels(self, ch: AcceptedChannels) -> Channels:
        if ch is None:
            return ()
        if isinstance(ch, (str, int)):
            return (ch,)
        elif isinstance(ch, (tuple, list)):
            return tuple(ch)
        else:
            raise TypeError("channel {} has unrecognizable type {}".format(ch, type(ch)))

    def regularize_pulse(self, pulse: AcceptedPulse) -> Pulse:
        # cast duration to builtin int because numpy types (like np.int64) may be incorpolated
        return Pulse(self.regularize_channels(pulse[0]), int(pulse[1]))

    def regularize_pattern(self, pattern: AcceptedPattern) -> Pattern:
        return [self.regularize_pulse(p) for p in pattern]

    # Mutating (list-like) operations

    def update_duration(self, index: int, duration: int):
        """update the duration of Pulse at `index` to new value `duration`."""

        pulse = self.pattern[index]
        self.pattern[index] = Pulse(pulse.channels, duration)

    def insert(self, index: int, pulse: AcceptedPulse):
        """Insert a `pulse` to `index` in pattern."""

        self.pattern.insert(index, self.regularize_pulse(pulse))

    def append(self, pulse: AcceptedPulse):
        """Append a `pulse` to the tail of pattern."""

        self.pattern.append(self.regularize_pulse(pulse))

    def extend(self, ptn: Block | AcceptedPattern):
        """Extend the pattern with given pattern or pattern in a Block."""

        if isinstance(ptn, Block):
            ptn = ptn.total_pattern()
        self.pattern.extend(self.regularize_pattern(ptn))

    # Non-mutating (copying) operations

    def suffix(self, suffix: str) -> Block:
        """Return new Block with a name suffix. Can be used to avoid name-duplication."""

        ret = self.copy()
        ret.name += suffix
        return ret

    def repeat(self, num: int) -> Block:
        """Return new Block with `num`-times repeat."""

        ret = self.copy()
        ret.Nrep *= num
        return ret

    def collapse(self) -> Block:
        """Return new Block with collapsed pattern.

        The pattern is physically repeated and Nrep is set 1.

        """

        return Block(self.name, self.total_pattern(), Nrep=1, trigger=self.trigger)

    def concatenate(self, other: Block) -> Block:
        """Generate a concatenated Block of this and `other`."""

        ret = self.collapse()
        ret.extend(other)
        return ret

    def raw_length(self) -> int:
        """Raw block length without considering Nrep."""

        return sum(elem[1] for elem in self.pattern)

    def total_length(self) -> int:
        """Total block length considering Nrep."""

        return self.Nrep * self.raw_length()

    def raw_channel_length(self, channel: str | int, high: bool) -> int:
        """Count raw block length when given channel is high or low."""

        if high:
            return sum(elem[1] for elem in self.pattern if channel in elem[0])
        else:
            return sum(elem[1] for elem in self.pattern if channel not in elem[0])

    def total_channel_length(self, channel: str | int, high: bool) -> int:
        """Count total block length when given channel is high or low."""

        return self.Nrep * self.raw_channel_length(channel, high)

    def total_pattern(self) -> Pattern:
        """Total (repeated) pattern considering Nrep."""

        return self.Nrep * self.pattern

    def total_pattern_num(self) -> int:
        """Total (repeated) pattern number (instruction size) considering Nrep."""

        return self.Nrep * self.raw_pattern_num()

    def raw_pattern_num(self) -> int:
        """Raw pattern number (instruction size) without considering Nrep."""

        return len(self.pattern)

    def channels(self) -> set[str | int]:
        """Get set of channels included in this Block."""

        s = set()
        for ch, duration in self.pattern:
            s.update(ch)
        return s

    def decode(self, channel: str | int, max_len: int | None = None) -> list[bool]:
        """Decode the pulse pattern to bool list."""

        ptn = []
        for ch, duration in self.total_pattern():
            elem = [True] if channel in ch else [False]
            ptn.extend(elem * duration)
            if max_len is not None and len(ptn) > max_len:
                break
        if max_len is None:
            return ptn
        else:
            return ptn[:max_len]

    def plottable_time(self, max_len: int | None = None) -> NDArray:
        """Generate timeline (x-axis data) for plottable(), that's common for all channels."""

        t = 0
        x = []
        for ch, duration in self.total_pattern():
            x.extend((t, duration + t))
            t += duration
            if max_len is not None and len(x) > max_len:
                break
        if max_len is None:
            return np.array(x)
        else:
            return np.array(x[:max_len])

    def plottable(self, channel: str | int, max_len: int | None = None) -> list[int]:
        """Decode the pulse pattern to plottable array."""

        ptn = []
        for ch, duration in self.total_pattern():
            elem = 1 if channel in ch else 0
            ptn.extend((elem, elem))
            if max_len is not None and len(ptn) > max_len:
                break
        if max_len is None:
            return ptn
        else:
            return ptn[:max_len]

    def decode_all(self, max_len: int | None = None) -> tuple[list[str | int], list[list[bool]]]:
        """Decode the patterns for all included channels."""

        channels = self.channels()
        try:
            channels = list(sorted(channels))
        except TypeError:
            channels = list(channels)
        patterns = [self.decode(ch, max_len=max_len) for ch in channels]
        return channels, patterns

    def plottable_all(self, max_len: int | None = None) -> tuple[list[str | int], list[list[int]]]:
        """Decode the patterns for all included channels to plottable array."""

        channels = self.channels()
        try:
            channels = list(sorted(channels))
        except TypeError:
            channels = list(channels)
        patterns = [self.plottable(ch, max_len=max_len) for ch in channels]
        return channels, patterns

    def union(self, other: Block) -> Block:
        """Return new united Block of this and `other`.

        union() operation is used to merge two Blocks generated in independent ways.
        The channels described in this and `other` must be exclusive, i.e.,
        if channel named "a" is defined in this Block, "a" cannot be appeared in the `other`.
        Two Blocks must have same total_length() too.

        """

        ch0 = self.channels()
        ch1 = other.channels()
        if ch0.intersection(ch1):
            raise ValueError(f"Channels have intersection: {ch0}, {ch1}")
        if self.total_length() != other.total_length():
            raise ValueError("Length mismatch")

        # copy the blocks
        p0 = self.simplify().total_pattern()
        p1 = other.simplify().total_pattern()
        t = self.total_length()
        ptn = []
        while t > 0:
            h0, h1 = p0[0], p1[0]
            if h0.duration == h1.duration:
                ptn.append((h0.channels + h1.channels, h0.duration))
                p0.pop(0)
                p1.pop(0)
                t -= h0.duration
            elif h0.duration > h1.duration:
                ptn.append((h0.channels + h1.channels, h1.duration))
                p0[0] = Pulse(h0.channels, h0.duration - h1.duration)
                p1.pop(0)
                t -= h1.duration
            else:  # h0.duration < h1.duration
                ptn.append((h0.channels + h1.channels, h0.duration))
                p1[0] = Pulse(h1.channels, h1.duration - h0.duration)
                p0.pop(0)
                t -= h0.duration
        assert not p0
        assert not p1
        return Block(self.name, ptn, Nrep=1, trigger=self.trigger)

    def __len__(self) -> int:
        """Total block length considering Nrep."""

        return self.total_length()

    def __eq__(self, other: Block) -> bool:
        if len(self.pattern) != len(other.pattern):
            return False
        for (ch, duration), (ch_, duration_) in zip(self.pattern, other.pattern):
            if set(ch) != set(ch_) or duration != duration_:
                return False
        return self.name == other.name and self.Nrep == other.Nrep

    def __str__(self):
        if self.Nrep == 1 and not self.trigger:
            return "Block({:s}, {:d})".format(self.name, len(self))
        elif self.Nrep == 1 and self.trigger:
            return "Block({:s}, {:d}, trigger={})".format(self.name, len(self), self.trigger)
        elif self.Nrep != 1 and not self.trigger:
            return "Block({:s}, {:d}, Nrep={:d})".format(self.name, len(self), self.Nrep)
        else:
            return "Block({:s}, {:d}, Nrep={:d}, trigger={})".format(
                self.name, len(self), self.Nrep, self.trigger
            )

    def copy(self) -> Block:
        return copy.deepcopy(self)

    def replace(
        self,
        replace_dict: dict[str | int, tuple[str, ...] | tuple[int, ...]],
    ) -> Block:
        """Return copy of this Block with replaced channels.

        :param replace_dict: (before, after) mapping for replace operation.
                             `after` can be empty tuple to remove `before`.

        """

        new_pattern = []
        for ch, period in self.pattern:
            new_ch = list(ch)
            for before, after in replace_dict.items():
                if before in ch:
                    new_ch.remove(before)
                    new_ch.extend(after)
            new_pattern.append((tuple(new_ch), period))

        return Block(self.name, new_pattern, Nrep=self.Nrep, trigger=self.trigger)

    def remove(
        self,
        remove_channels: str | int | tuple[str, ...] | tuple[int, ...],
    ) -> Block:
        """Return copy of this Block with removed channels.

        :param remove: single channel or tuple of channels to remove.

        """

        if isinstance(remove_channels, (str, int)):
            remove_channels = (remove_channels,)

        new_pattern = [
            (tuple((c for c in chs if c not in remove_channels)), t) for chs, t in self.pattern
        ]
        return Block(self.name, new_pattern, Nrep=self.Nrep, trigger=self.trigger)

    def apply(self, func: T.Callable[Channels, Channels]) -> Block:
        """Return copy of this Block applying given function over channels."""

        new_pattern = []
        for ch, period in self.pattern:
            new_pattern.append((func(ch), period))

        return Block(self.name, new_pattern, Nrep=self.Nrep, trigger=self.trigger)

    def simplify(self) -> Block:
        """return simplified copy of this Block.

        simplification is done by:
        - removing zero-duration pulses
        - merging contiguous pulses with same channels

        """

        new_pattern = []
        for ch, period in self.pattern:
            if period == 0:
                continue

            if len(new_pattern) == 0:
                new_pattern.append((ch, period))
                continue

            _ch, _period = new_pattern[-1]

            if set(_ch) == set(ch):
                new_pattern[-1] = (_ch, _period + period)
            else:
                new_pattern.append((ch, period))

        return Block(self.name, new_pattern, Nrep=self.Nrep, trigger=self.trigger)

    def scale(self, s: int) -> Block:
        """return scaled copy of this Block.

        :param s: the scale. must be 1 or greater.

        """

        s = int(s)
        if s < 1:
            raise ValueError("s must be 1 or greater.")

        pat = [(ch, s * duration) for ch, duration in self.pattern]
        return Block(self.name, pat, Nrep=self.Nrep, trigger=self.trigger)

    def pattern_to_strs(self) -> list[str]:
        def pulse_to_str(pulse: Pulse):
            channels, period = pulse
            n_str = ",".join([str(ch) for ch in channels])
            if not n_str:
                n_str = "NOP"
            return f"{n_str:s}:{period:d}"

        return [pulse_to_str(pat) for pat in self.pattern]


class Blocks(UserList):
    """list for Block with convenient functions."""

    def collapse(self) -> Block:
        """Collapse the contained Blocks to single Block.

        Resultant Block is almost equivalent to the original Blocks[Block], but
        the trigger attributes of non-first Blocks are silently discarded.

        """

        blk = self.data[0].collapse()
        for b in self.data[1:]:
            blk.extend(b.total_pattern())
        return blk

    def repeat(self, num: int) -> Blocks[Block]:
        """Return new Blocks[Block] with `num`-times repeat.

        Each element of new Blocks is a copy of the original element.
        (Possibly) unique names are assigned to avoid name collison
        (that is not allowed for some instruments).

        """

        ret = Blocks()
        for i in range(num):
            for block in self.data:
                b = block.copy()
                if i:
                    b.name += f"_{i}"
                ret.append(b)
        return ret

    def channels(self) -> list[str | int]:
        """Get list of included channels.

        If types of all channels are identical (all str or all int),
        the result is sorted.
        Otherwise the order is unpredictable (because mixed strs and ints cannot be sorted).

        """

        channels = set()
        for block in self.data:
            channels.update(block.channels())
        try:
            return list(sorted(channels))
        except TypeError:
            return list(channels)

    def decode(self, channel: str | int, max_len: int | None = None) -> NDArray:
        """Decode the pattern of given channel.

        If channel is not included, all-zero array will be returned.

        """

        ptn = []
        for block in self.data:
            ptn.extend(block.decode(channel, max_len=max_len))
            if max_len is not None and len(ptn) > max_len:
                break
        if max_len is None:
            return np.array(ptn, dtype=np.uint8)
        else:
            return np.array(ptn[:max_len], dtype=np.uint8)

    def plottable_time(self, max_len: int | None = None) -> NDArray:
        """Generate timeline (x-axis data) for plottable(), that's common for all channels."""

        ptn = np.empty(0, dtype=np.uint64)
        for block in self.data:
            t = np.array(block.plottable_time(max_len=max_len), dtype=np.uint64)
            if len(ptn):
                t += ptn[-1]
            ptn = np.append(ptn, t)
            if max_len is not None and len(ptn) > max_len:
                break
        if max_len is None:
            return ptn
        else:
            return ptn[:max_len]

    def plottable(self, channel: str | int, max_len: int | None = None) -> NDArray:
        """Decode the pattern of given channel to plottable array."""

        ptn = []
        for block in self.data:
            ptn.extend(block.plottable(channel, max_len=max_len))
            if max_len is not None and len(ptn) > max_len:
                break
        if max_len is None:
            return np.array(ptn, dtype=np.uint8)
        else:
            return np.array(ptn[:max_len], dtype=np.uint8)

    def decode_all(self, max_len: int | None = None) -> tuple[list[str | int], list[NDArray]]:
        """Decode the patterns for all included channels."""

        channels = self.channels()
        patterns = [self.decode(ch, max_len=max_len) for ch in channels]
        return channels, patterns

    def plottable_all(self, max_len: int | None = None) -> tuple[list[str | int], list[NDArray]]:
        """Decode the patterns for all included channels to plottable array."""

        channels = self.channels()
        patterns = [self.plottable(ch, max_len=max_len) for ch in channels]
        return channels, patterns

    def equivalent(self, other: Blocks[Block]) -> bool:
        """Check if this and other blocks are equivalent.

        Compare the channels and patterns after decode.

        """

        channels, patterns = self.decode_all()
        other_channels, other_patterns = other.decode_all()

        if len(patterns) != len(other_patterns):
            return False

        return channels == other_channels and all(
            [len(p0) == len(p1) and (p0 == p1).all() for p0, p1 in zip(patterns, other_patterns)]
        )

    def replace(
        self,
        replace_dict: dict[str | int, tuple[str, ...] | tuple[int, ...]],
    ) -> Blocks[Block]:
        """Return copy of the Blocks with replaced channels."""

        return Blocks([b.replace(replace_dict) for b in self.data])

    def remove(
        self,
        remove_channels: str | int | tuple[str, ...] | tuple[int, ...],
    ) -> Blocks[Block]:
        """Return copy of the Blocks with removed channels."""

        return Blocks([b.remove(remove_channels) for b in self.data])

    def apply(self, func: T.Callable[Channels, Channels]) -> Blocks[Block]:
        """Return copy of this Blocks applying given function over channels."""

        return Blocks([b.apply() for b in self.data])

    def simplify(self) -> Blocks[Block]:
        """return simplified copy of this Blocks.

        simplification is done by:
        - removing zero-times repeated Blocks
        - removing zero-duration pulses in Blocks
        - merging contiguous pulses with same channels in Blocks

        """

        return Blocks([b.simplify() for b in self.data if b.Nrep])

    def total_length(self) -> int:
        """Total block length considering Nrep."""

        return sum([b.total_length() for b in self.data])

    def total_channel_length(self, channel: str | int, high: bool) -> int:
        """Count total block length when given channel is high or low."""

        return sum([b.total_channel_length(channel, high) for b in self.data])

    def total_pattern_num(self) -> int:
        """Total pattern number (instruction size) considering Nrep."""

        return sum([b.total_pattern_num() for b in self.data])

    def scale(self, s: int) -> Blocks[Block]:
        """Scale all the durations in the Blocks."""

        return Blocks([b.scale(s) for b in self.data])


class BlockSeq(Message):
    """Nestable and named sequence of Blocks.

    Methods compatible with Blocks are implemented.
    Blocks[Block] and BlockSeq are almost equivalent if BlockSeq contains Block only.
    Blocks[Block] may be a bit handier if you don't need the nested repeat
    because it supports list-like operations.

    """

    def __init__(
        self,
        name: str,
        data: list[Block | BlockSeq],
        Nrep: int = 1,
        trigger: bool = False,
    ):
        self.name = name
        self.data = data
        self.Nrep = Nrep
        self.trigger = trigger

    def _new(self, data) -> BlockSeq:
        return BlockSeq(self.name, data, self.Nrep, self.trigger)

    def collapse(self) -> Block:
        """Collapse the contained Blocks | BlockSeq to single Block.

        Resultant Block is almost equivalent to the original BlockSeq, but
        the trigger attributes of non-first Blocks are silently discarded.

        """

        blk = self.data[0].collapse()
        for b in self.data[1:]:
            blk.extend(b.collapse())
        return blk.repeat(self.Nrep).collapse()

    def repeat(self, num: int) -> BlockSeq:
        """Return new BlockSeq with `num`-times repeat."""

        return BlockSeq(self.name, self.data, self.Nrep * num, self.trigger)

    def total_sequence(self):
        return self.data * self.Nrep

    def unique_blocks(self) -> Blocks[Block]:
        """Extract unique (in terms of their names) Blocks inside."""

        ret = Blocks()
        names = []
        for bs in self.data:
            if isinstance(bs, Block):
                if bs.name in names:
                    continue
                ret.append(bs)
                names.append(bs.name)
            else:  # BlockSeq
                blks = bs.unique_blocks()
                ret.extend([b for b in blks if b.name not in names])
                names.extend([b.name for b in blks])

        return ret

    def channels(self) -> list[str | int]:
        """Get list of included channels.

        If types of all channels are identical (all str or all int),
        the result is sorted.
        Otherwise the order is unpredictable (because mixed strs and ints cannot be sorted).

        """

        channels = set()
        for blk_or_seq in self.data:
            channels.update(blk_or_seq.channels())
        try:
            return list(sorted(channels))
        except TypeError:
            return list(channels)

    def _decode(self, channel: str | int, max_len: int | None = None) -> list[bool]:
        ptn = []
        for blk_or_seq in self.total_sequence():
            if isinstance(blk_or_seq, Block):
                ptn.extend(blk_or_seq.decode(channel, max_len=max_len))
            else:
                ptn.extend(blk_or_seq._decode(channel, max_len=max_len))
            if max_len is not None and len(ptn) > max_len:
                break
        if max_len is None:
            return ptn
        else:
            return ptn[:max_len]

    def decode(self, channel: str | int, max_len: int | None = None) -> NDArray:
        """Decode the pattern of given channel.

        If channel is not included, all-zero array will be returned.

        """

        return np.array(self._decode(channel, max_len=max_len), dtype=np.uint8)

    def plottable_time(self, max_len: int | None = None) -> NDArray:
        """Generate timeline (x-axis data) for plottable(), that's common for all channels."""

        ptn = np.empty(0, dtype=np.uint64)
        for blk_or_seq in self.total_sequence():
            t = np.array(blk_or_seq.plottable_time(max_len=max_len), dtype=np.uint64)
            if len(ptn):
                t += ptn[-1]
            ptn = np.append(ptn, t)
            if max_len is not None and len(ptn) > max_len:
                break
        if max_len is None:
            return ptn
        else:
            return ptn[:max_len]

    def _plottable(self, channel: str | int, max_len: int | None = None) -> list[int]:
        """Decode the pattern of given channel to plottable array."""

        ptn = []
        for blk_or_seq in self.total_sequence():
            if isinstance(blk_or_seq, Block):
                ptn.extend(blk_or_seq.plottable(channel, max_len=max_len))
            else:
                ptn.extend(blk_or_seq._plottable(channel, max_len=max_len))
            if max_len is not None and len(ptn) > max_len:
                break
        if max_len is None:
            return ptn
        else:
            return ptn[:max_len]

    def plottable(self, channel: str | int, max_len: int | None = None) -> NDArray:
        return np.array(self._plottable(channel, max_len=max_len), dtype=np.uint8)

    def decode_all(self, max_len: int | None = None) -> tuple[list[str | int], list[NDArray]]:
        """Decode the patterns for all included channels."""

        channels = self.channels()
        patterns = [self.decode(ch, max_len=max_len) for ch in channels]
        return channels, patterns

    def plottable_all(self, max_len: int | None = None) -> tuple[list[str | int], list[NDArray]]:
        """Decode the patterns for all included channels to plottable array."""

        channels = self.channels()
        patterns = [self.plottable(ch, max_len=max_len) for ch in channels]
        return channels, patterns

    def equivalent(self, other: BlockSeq | Blocks[Block] | Block) -> bool:
        """Check if this and other sequences are equivalent.

        Compare the channels and patterns after decode.

        """

        channels, patterns = self.decode_all()
        other_channels, other_patterns = other.decode_all()

        if len(patterns) != len(other_patterns):
            return False

        return channels == other_channels and all(
            [len(p0) == len(p1) and (p0 == p1).all() for p0, p1 in zip(patterns, other_patterns)]
        )

    def replace(
        self,
        replace_dict: dict[str | int, tuple[str, ...] | tuple[int, ...]],
    ) -> BlockSeq:
        """Return copy of this BlockSeq with replaced channels."""

        return self._new([b.replace(replace_dict) for b in self.data])

    def remove(
        self,
        remove_channels: str | int | tuple[str, ...] | tuple[int, ...],
    ) -> BlockSeq:
        """Return copy of this BlockSeq with removed channels."""

        return self._new([b.remove(remove_channels) for b in self.data])

    def apply(self, func: T.Callable[Channels, Channels]) -> BlockSeq:
        """Return copy of this BlockSeq applying given function over channels."""

        return self._new([b.apply() for b in self.data])

    def simplify(self) -> BlockSeq:
        """return simplified copy of this BlockSeq.

        simplification is done by:
        - removing zero-times repeated Blocks
        - removing zero-duration pulses in Blocks
        - merging contiguous pulses with same channels in Blocks

        """

        return self._new([b.simplify() for b in self.data if b.Nrep])

    def total_length(self) -> int:
        """Total block length considering Nrep."""

        return sum([b.total_length() for b in self.data]) * self.Nrep

    def total_channel_length(self, channel: str | int, high: bool) -> int:
        """Count total block length when given channel is high or low."""

        return sum([b.total_channel_length(channel, high) for b in self.data]) * self.Nrep

    def total_pattern_num(self) -> int:
        """Total pattern number (instruction size) considering Nrep."""

        return sum([b.total_pattern_num() for b in self.data]) * self.Nrep

    def scale(self, s: int) -> BlockSeq:
        """Scale all the durations in this BlockSeq."""

        return self._new([b.scale(s) for b in self.data])

    def nest_depth(self) -> int:
        return max([b.nest_depth() if isinstance(b, BlockSeq) else 0 for b in self.data]) + 1
