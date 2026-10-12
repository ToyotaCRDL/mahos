#!/usr/bin/env python3

"""
Core functions for pattern generators for Pulse ODMR

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import re
from bisect import bisect_right

import numpy as np
from itertools import chain

from mahos.msgs.inst.pg_msgs import Block, Blocks, BlockSeq, AnalogChannel, Channels
from mahos_dq.msgs.podmr_msgs import MWMode

mw_x = AnalogChannel("mw_phase", 0)
mw_y = AnalogChannel("mw_phase", 90)
mw_x_inv = AnalogChannel("mw_phase", 180)
mw_y_inv = AnalogChannel("mw_phase", 270)


def is_mw_related_channel(ch: str | int | AnalogChannel) -> bool:
    if isinstance(ch, str):
        return ch.startswith("mw")
    if isinstance(ch, AnalogChannel):
        return ch.name().startswith("mw")
    return False


def _split_mw_channels(channels: Channels) -> tuple[Channels, Channels]:
    non_mw = []
    mw = []
    for ch in channels:
        if is_mw_related_channel(ch):
            mw.append(ch)
        else:
            non_mw.append(ch)
    return tuple(non_mw), tuple(mw)


def _expand_pattern(blocks: Blocks[Block]):
    names = []
    triggers = []
    lengths = []
    pattern = []
    for block in blocks:
        names.append(block.name)
        triggers.append(block.trigger)
        lengths.append(block.total_length())
        pattern.extend(block.total_pattern())
    return names, triggers, lengths, pattern


def _normalize_mw_modes(mw_modes: tuple[MWMode | str | int]) -> tuple[MWMode]:
    return tuple(MWMode.parse(m) for m in mw_modes)


def apply_mw_offset(blocks: Blocks[Block], offset_ticks: int) -> Blocks[Block]:
    """Apply circular delay to mw-related channels in whole pulse sequence."""

    names, triggers, lengths, pattern = _expand_pattern(blocks)
    if not pattern:
        return blocks

    total_length = blocks.total_length()
    if total_length <= 0:
        return blocks

    offset_ticks %= total_length
    if offset_ticks == 0:
        return blocks

    starts = []
    non_mw_states = []
    mw_states = []
    t = 0
    boundaries = {0}
    for channels, duration in pattern:
        starts.append(t)
        non_mw, mw = _split_mw_channels(channels)
        non_mw_states.append(non_mw)
        mw_states.append(mw)
        t += duration
        boundaries.add(t)

    assert t == total_length

    for s in starts:
        boundaries.add((s + offset_ticks) % total_length)

    # total_length is equivalent to 0 in circular timeline.
    boundaries.discard(total_length)
    boundaries = sorted(boundaries)
    if boundaries[0] != 0:
        boundaries.insert(0, 0)

    # Rebuild RLE by sampling non-MW at t and MW at (t - offset) for each boundary interval.
    shifted = []
    for i, t0 in enumerate(boundaries):
        t1 = boundaries[i + 1] if i + 1 < len(boundaries) else total_length
        assert t0 < t1

        idx_non = bisect_right(starts, t0) - 1
        idx_mw = bisect_right(starts, (t0 - offset_ticks) % total_length) - 1
        channels = non_mw_states[idx_non] + mw_states[idx_mw]
        duration = t1 - t0

        if shifted and shifted[-1][0] == channels:
            shifted[-1] = (channels, shifted[-1][1] + duration)
        else:
            shifted.append((channels, duration))

    # Restore original block boundaries so all downstream code keeps block-wise behavior.
    new_blocks = []
    pat_i = 0
    head_ch, head_dur = shifted[0]
    for name, trigger, length in zip(names, triggers, lengths):
        consumed = 0
        block_pattern = []
        while consumed < length:
            take = min(head_dur, length - consumed)
            block_pattern.append((head_ch, take))
            consumed += take
            head_dur -= take
            if head_dur == 0:
                pat_i += 1
                if pat_i < len(shifted):
                    head_ch, head_dur = shifted[pat_i]
        new_blocks.append(Block(name, block_pattern, trigger=trigger))

    return Blocks(new_blocks).simplify()


def round_pulsesN(
    freq: float,
    xdata: list[float],
    common_pulses: list[float],
    pulses: list[list[float]],
    reduce_start_divisor: int,
    print_fn,
) -> tuple[float, list[int], list[int], list[list[int]]]:
    """Round the float (real time) pulse parameters according to freq to get integer parameters.

    pulses[i] must have same lengths.

    """

    def _reduce(values, freq, divisor):
        if any([v % divisor != 0 for v in values]):
            return values, freq
        else:
            print_fn(f"reducing parameters {values} by {divisor}")
            return _reduce([v // divisor for v in values], freq / divisor, 10)

    flat_pulses = list(chain.from_iterable(pulses))
    N_pulses = len(flat_pulses)
    n_pulses = len(pulses[0])
    for p in pulses:
        if len(p) != n_pulses:
            raise ValueError("Each element of pulses should have same lengths.")

    values = list(xdata) + common_pulses + flat_pulses
    values = [round(v * freq) for v in values]

    if reduce_start_divisor:
        print_fn("starting to reduce parameters")
        values, freq = _reduce(values, freq, reduce_start_divisor)
        print_fn(f"reduced to: {values} freq: {freq:.2e}")

    _xdata = np.round(values[: -len(common_pulses) - N_pulses]).astype(int)

    if N_pulses > 0:
        _common_pulses = values[-len(common_pulses) - N_pulses : -N_pulses]
        _pulses = []
        for i in range(len(pulses)):
            end = len(values) - n_pulses * i
            start = end - n_pulses
            _pulses.append(values[start:end])
        _pulses.reverse()
    else:
        _common_pulses = values[-len(common_pulses) :]
        _pulses = [[] for _ in pulses]

    return freq, _xdata, _common_pulses, _pulses


def round_pulses(
    freq: float,
    xdata: list[float],
    common_pulses: list[float],
    pulses0: list[float],
    pulses1: list[float],
    reduce_start_divisor: int,
    print_fn,
) -> tuple[float, list[int], list[int], list[int], list[int]]:
    """Round the float (real time) pulse parameters according to freq to get integer parameters.

    pulses0 and pulses1 must have same lengths.

    """

    freq, _xdata, _common_pulses, _pulses = round_pulsesN(
        freq, xdata, common_pulses, [pulses0, pulses1], reduce_start_divisor, print_fn
    )
    return freq, _xdata, _common_pulses, _pulses[0], _pulses[1]


def split_int(n, A):
    return n // A, n - n // A


def offset_base_inc(duration: int, base: int) -> int:
    """Offset a `duration` to make it an integer multiple of `base`.

    Let duration = base * N + M,
    when M == 0: return duration = base * N
    when M != 0: return base * (N + 1)

    """

    M = duration % base
    if M:
        return duration + base - M
    return duration


def generate_blocksN(
    i: int,
    v: int,
    common_pulses: list[int],
    gen_single_ptn,
    argsN: list,
    read_phases: list[AnalogChannel | tuple[AnalogChannel]] | None = None,
    laser_phase: AnalogChannel | tuple[AnalogChannel] = mw_x,
    partial: int = -1,
    fix_base_width: int | None = None,
) -> Blocks[Block]:
    (
        base_width,
        laser_delay,
        laser_width,
        mw_delay,
        trigger_width,
        init_delay,
        final_delay,
    ) = common_pulses

    if fix_base_width is not None:
        base_width = fix_base_width

    laser_width = offset_base_inc(laser_width, base_width)

    # blocks
    ptnN = [gen_single_ptn(v, *args) for args in argsN]
    N = len(ptnN)
    if N < 2:
        raise ValueError("generate_blocksN() requires at least two patterns.")
    if read_phases is None:
        if N == 2:
            read_phases = [mw_x, mw_x_inv]
        else:
            read_phases = [mw_x] * N
    if len(read_phases) != N:
        raise ValueError(f"read_phases length {len(read_phases)} must match pattern count {N}")

    if isinstance(laser_phase, AnalogChannel):
        ptn_former = [((laser_phase,), mw_delay)]
        ptn_laser = [((laser_phase, "laser", "sync"), laser_width)]
    else:
        ptn_former = [(tuple(laser_phase), mw_delay)]
        ptn_laser = [(tuple(laser_phase) + ("laser", "sync"), laser_width)]

    blocks = []
    for j, (ptn, read_phase) in enumerate(zip(ptnN, read_phases)):
        if isinstance(read_phase, AnalogChannel):
            ptn_latter = [((read_phase,), laser_delay)]
        else:
            ptn_latter = [(tuple(read_phase), laser_delay)]

        # base width offset (operation)
        total = sum([s[1] for s in ptn_former + ptn + ptn_latter])
        if total % base_width == 0:
            ptn_operate = ptn_former + ptn + ptn_latter
        else:
            ofs = base_width - total % base_width
            ptn_offset = [(ptn_laser[0][0], ofs)]
            ptn_operate = ptn_offset + ptn_former + ptn + ptn_latter

        blocks.extend([Block(f"P{i}-{j}", ptn_operate), Block(f"read{i}-{j}", ptn_laser)])

    if 0 <= partial < N:
        return Blocks(blocks[2 * partial : 2 * partial + 2])
    return Blocks(blocks)


def generate_blocks(
    i: int,
    v: int,
    common_pulses: list[int],
    gen_single_ptn,
    args0,
    args1,
    read_phase0: AnalogChannel | tuple[AnalogChannel] = mw_x,
    read_phase1: AnalogChannel | tuple[AnalogChannel] = mw_x_inv,
    laser_phase: AnalogChannel | tuple[AnalogChannel] = mw_x,
    partial: int = -1,
    fix_base_width: int | None = None,
) -> Blocks[Block]:
    return generate_blocksN(
        i,
        v,
        common_pulses,
        gen_single_ptn,
        [args0, args1],
        [read_phase0, read_phase1],
        laser_phase=laser_phase,
        partial=partial,
        fix_base_width=fix_base_width,
    )


def init_final_phases(num_mw):
    phases = []
    for ch in range(num_mw):
        if ch == 0:
            ch = ""
        phases.append(AnalogChannel(f"mw{ch}_phase", 0))
    return tuple(phases)


def inject_trigger(op_block: Block, roi_head: int, trigger_width: int) -> Block:
    if roi_head < trigger_width:
        raise ValueError("roi_head must be equal to or longer than trigger_width")
    total_length = op_block.total_length()
    if roi_head > total_length:
        raise ValueError("roi_head must fit in the operation block before each laser pulse")

    trigger_start = total_length - roi_head
    trigger_tail = roi_head - trigger_width
    pattern = []
    if trigger_start > 0:
        pattern.append((None, trigger_start))
    pattern.append((("trigger",), trigger_width))
    if trigger_tail > 0:
        pattern.append((None, trigger_tail))

    return op_block.union(Block(op_block.name + "_trig", pattern))


def build_blocks(
    blocks: list[Blocks[Block]],
    freq: float,
    common_pulses,
    params: dict,
    merge=True,
    invertY=False,
    minimum_block_length: int = 1000,
    block_base: int = 4,
    mw_modes: tuple[MWMode | str | int] = (MWMode.QPSK,),
    num_mw: int = 1,
    iq_amplitude: float = 0.0,
    mw_offset: int = 0,
    channel_remap: dict | None = None,
) -> tuple[Blocks[Block], list[int]]:
    """Build up the blocks by adding init and final blocks.

    final is kind of dummy. however, should be much larger than
    TDC's EOS deadtime (100 ns for MCS6A).

    """

    (
        base_width,
        laser_delay,
        laser_width,
        mw_delay,
        trigger_width,
        init_delay,
        final_delay,
    ) = common_pulses

    divide = params.get("divide_block", False)
    multi_histogram = params.get("multi_histogram", False)

    init_block_width = max(trigger_width + init_delay + laser_width, minimum_block_length)
    init_block_width = offset_base_inc(init_block_width, base_width)
    final_block_width = max(final_delay, minimum_block_length)
    final_block_width = offset_base_inc(final_block_width, base_width)

    phases = init_final_phases(num_mw)

    if multi_histogram:
        roi_head = int(round(params["roi_head"] * freq))
        roi_tail = int(round(params["roi_tail"] * freq))
        if roi_head < 0 or roi_tail < 0:
            raise ValueError("roi_head and roi_tail must be non-negative")
        _trig_ch = ("sync",)
    else:
        _trig_ch = ("trigger", "sync")

    ptn_init = [
        (_trig_ch + phases, trigger_width),
        (("sync",) + phases, init_delay),
        (
            (
                "laser",
                "sync",
            )
            + phases,
            init_block_width - trigger_width - init_delay,
        ),
    ]
    ptn_final = [(("sync",) + phases, final_block_width)]

    if multi_histogram:
        for blks in blocks:
            for i in range(0, len(blks), 2):
                blks[i] = inject_trigger(blks[i], roi_head, trigger_width)
                if blks[i].total_length() < roi_head + roi_tail:
                    raise ValueError("ROI overlaps; reduce ROI margins or laser rate")

    # flatten list[Blocks[Block]] to Blocks[Block]
    blocks = Blocks(chain.from_iterable(blocks))

    # insert init/final
    blocks.insert(0, Block("INIT", ptn_init))
    blocks.append(Block("FINAL", ptn_final))
    laser_timing = extract_laser_timing(blocks)

    # block shaping
    if divide:
        blocks = divide_long_operation(blocks, minimum_block_length, block_base)
        blocks = divide_long_laser(blocks, minimum_block_length)
        blocks = divide_long_final(blocks, minimum_block_length)
    if merge:
        blocks = merge_short_blocks(blocks, minimum_block_length)

    # phase encoding
    if invertY:
        blocks = invert_y_phase(blocks)
    blocks = encode_mw_phase(blocks, params, mw_modes, num_mw, iq_amplitude)

    if mw_offset:
        blocks = apply_mw_offset(blocks, mw_offset)

    if channel_remap is not None:
        blocks = blocks.replace(channel_remap)

    return blocks.simplify(), laser_timing


def print_blocks(blocks: Blocks[Block], print_fn=print):
    for b in blocks:
        print_fn(str(b))
        print_fn("|".join(b.pattern_to_strs()))


def encode_mw_phase(
    blocks: Blocks[Block] | BlockSeq,
    params: dict,
    mw_modes: tuple[MWMode | str | int],
    num_mw: int,
    iq_amplitude: float,
) -> Blocks[Block] | BlockSeq:
    """encode mw phase from x/y(_inv) to i/q."""

    mw_modes = _normalize_mw_modes(mw_modes)

    if MWMode.AWG in mw_modes:
        if not all(mode == MWMode.AWG for mode in mw_modes):
            raise ValueError("Mixing AWG and non-AWG modes is not supported.")
        return blocks

    if num_mw == 1:
        return _encode_mw_phase_single(blocks, params, mw_modes, iq_amplitude)
    else:
        return _encode_mw_phase_multi(blocks, params, mw_modes, iq_amplitude)


def _encode_ext2phase(
    blocks: Blocks[Block] | BlockSeq, ch_from=0, ch_to=0, remove_pulse: bool = True
) -> Blocks[Block] | BlockSeq:
    f = "" if ch_from == 0 else str(ch_from)
    if isinstance(ch_to, int):
        ch_to = (ch_to,)
    mw_x = AnalogChannel(f"mw{f}_phase", 0)
    mw_y = AnalogChannel(f"mw{f}_phase", 90)

    def encode(ch):
        if ch is None:
            return ch
        new_ch = list(ch)
        if f"mw{f}" in new_ch:
            if mw_x in new_ch:
                if remove_pulse:
                    new_ch.remove(f"mw{f}")
                new_ch.remove(mw_x)
                for c in ch_to:
                    t = "" if c == 0 else str(c)
                    new_ch.append(f"mw{t}_i")
            elif mw_y in new_ch:
                if remove_pulse:
                    new_ch.remove(f"mw{f}")
                new_ch.remove(mw_y)
                for c in ch_to:
                    t = "" if c == 0 else str(c)
                    new_ch.append(f"mw{t}_q")
            else:
                raise ValueError(f"Unknown MW phase: {new_ch}")
        # remove phase-only parts
        elif mw_x in new_ch:
            new_ch.remove(mw_x)
        elif mw_y in new_ch:
            new_ch.remove(mw_y)

        return tuple(new_ch)

    return blocks.apply(encode)


def _encode_arb_phase(
    blocks: Blocks[Block] | BlockSeq, amplitude: float, ch_from=0, ch_to=0, phase0=45.0
) -> Blocks[Block] | BlockSeq:
    f = "" if ch_from == 0 else str(ch_from)
    if isinstance(ch_to, int):
        ch_to = (ch_to,)

    def encode(ch):
        if ch is None:
            return ch
        new_ch = list(ch)
        for c in new_ch:
            if isinstance(c, AnalogChannel) and c.name() == f"mw{f}_phase":
                new_ch.remove(c)
                I = amplitude * np.cos(np.radians(c.value() + phase0))
                Q = amplitude * np.sin(np.radians(c.value() + phase0))
                for c in ch_to:
                    t = "" if c == 0 else str(c)
                    new_ch.append(AnalogChannel(f"mw{t}_i", I))
                    new_ch.append(AnalogChannel(f"mw{t}_q", Q))
                break
        return tuple(new_ch)

    return blocks.apply(encode)


def _encode_mw_phase_single(
    blocks: Blocks[Block] | BlockSeq, params, mw_modes: tuple[MWMode], iq_amplitude: float
) -> Blocks[Block] | BlockSeq:
    """Encode mw phase from x/y(_inv) to i/q for single channel sequence."""

    for i in range(2, len(mw_modes)):
        if not params[f"nomw{i}"]:
            msg = f"mw{i} is enabled but cannot encode MW channel #2 or greater."
            raise NotImplementedError(msg)

    # if two mw channels are available, output same timing signal for mw0 and/or mw1
    # according to nomw parameters.
    nomw = params.get("nomw", False)
    nomw1 = params.get("nomw1", True)
    if nomw and nomw1:
        d = {
            "mw": (),
            mw_x: (),
            mw_y: (),
            mw_x_inv: (),
            mw_y_inv: (),
        }
        return blocks.replace(d)
    elif nomw1:  # mw0 only
        if mw_modes[0] == MWMode.QPSK:
            d = {
                mw_x: ("mw_i", "mw_q"),
                mw_y: ("mw_q",),
                mw_x_inv: (),
                mw_y_inv: ("mw_i",),
            }
            return blocks.replace(d)
        elif mw_modes[0] == MWMode.Ext2Phase:
            return _encode_ext2phase(blocks, ch_from=0, ch_to=0)
        elif mw_modes[0] == MWMode.ArbPhase:
            return _encode_arb_phase(blocks, iq_amplitude, ch_from=0, ch_to=0)
        else:
            raise ValueError(f"Unknown mw_mode: {mw_modes[0]}")
    elif nomw:  # mw1 only
        if mw_modes[1] == MWMode.QPSK:
            d = {
                mw_x: ("mw1_i", "mw1_q"),
                mw_y: ("mw1_q",),
                mw_x_inv: (),
                mw_y_inv: ("mw1_i",),
            }
            return blocks.replace(d)
        elif mw_modes[1] == MWMode.Ext2Phase:
            return _encode_ext2phase(blocks, ch_from=0, ch_to=1)
        elif mw_modes[1] == MWMode.ArbPhase:
            return _encode_arb_phase(blocks, iq_amplitude, ch_from=0, ch_to=1)
        else:
            raise ValueError(f"Unknown mw_mode: {mw_modes[0]}")
    else:  # both mw0 and mw1
        if mw_modes == (MWMode.QPSK, MWMode.QPSK):
            d = {
                mw_x: ("mw_i", "mw_q", "mw1_i", "mw1_q"),
                mw_y: ("mw_q", "mw1_q"),
                mw_x_inv: (),
                mw_y_inv: ("mw_i", "mw1_i"),
            }
            return blocks.replace(d)
        elif mw_modes == (MWMode.Ext2Phase, MWMode.Ext2Phase):
            return _encode_ext2phase(blocks, ch_from=0, ch_to=(0, 1))
        elif mw_modes == (MWMode.ArbPhase, MWMode.ArbPhase):
            return _encode_arb_phase(blocks, iq_amplitude, ch_from=0, ch_to=(0, 1))
        elif mw_modes == (MWMode.QPSK, MWMode.Ext2Phase):
            d = {
                mw_x: (mw_x, "mw_i", "mw_q"),
                mw_y: (mw_y, "mw_q"),
            }
            return _encode_ext2phase(blocks.replace(d), ch_from=0, ch_to=1, remove_pulse=False)
        elif mw_modes == (MWMode.Ext2Phase, MWMode.QPSK):
            d = {
                mw_x: (mw_x, "mw1_i", "mw1_q"),
                mw_y: (mw_y, "mw1_q"),
            }
            return _encode_ext2phase(blocks.replace(d), ch_from=0, ch_to=0, remove_pulse=False)
        else:
            # TODO: the other combination of modes.
            raise ValueError(f"Unsupported mw_modes: {mw_modes} to multiplex")


def _encode_mw_phase_multi(
    blocks: Blocks[Block] | BlockSeq, params, mw_modes: tuple[MWMode], iq_amplitude: float
) -> Blocks[Block] | BlockSeq:
    """encode mw phase from x/y(_inv) to i/q for multi mw channel sequence."""

    for ch, mode in enumerate(mw_modes):
        c = "" if ch == 0 else str(ch)
        if mode == MWMode.QPSK:
            mw_x = AnalogChannel(f"mw{c}_phase", 0)
            mw_y = AnalogChannel(f"mw{c}_phase", 90)
            mw_x_inv = AnalogChannel(f"mw{c}_phase", 180)
            mw_y_inv = AnalogChannel(f"mw{c}_phase", 270)
            d = {
                mw_x: (f"mw{c}_i", f"mw{c}_q"),
                mw_y: (f"mw{c}_q",),
                mw_x_inv: (),
                mw_y_inv: (f"mw{c}_i",),
            }
            blocks = blocks.replace(d)
        elif mode == MWMode.Ext2Phase:
            blocks = _encode_ext2phase(blocks, ch_from=ch, ch_to=ch)
        elif mode == MWMode.ArbPhase:
            blocks = _encode_arb_phase(blocks, iq_amplitude, ch_from=ch, ch_to=ch)
        else:
            raise ValueError(f"Unknown mw_mode: {mode}")

    return blocks


def invert_y_phase(blocks: Blocks[Block] | BlockSeq) -> Blocks[Block] | BlockSeq:
    """swap mw_y and mw_y_inv."""

    def invert(ch):
        if ch is None:
            return ch

        new_ch = list(ch)

        if mw_y in new_ch:
            new_ch.remove(mw_y)
            new_ch.append(mw_y_inv)
        elif mw_y_inv in new_ch:
            new_ch.remove(mw_y_inv)
            new_ch.append(mw_y)

        return tuple(new_ch)

    return blocks.apply(invert)


_op_pattern = re.compile(r"P(\d+)-(\d+)")
_read_pattern = re.compile(r"read(\d+)-(\d+)")


def divide_long_operation(blocks: Blocks[Block], length: int, base: int) -> Blocks[Block]:
    """Divide long blocks for operation (mw) into multiple blocks.

    This is meaningful only for hardware where Nrep > 1 reduces memory consumption (e.g., DTG).

    :param length: length of the repeated block.
    :param base: base granularity requirement for blocks.
                 This function tries to keep each block length being integer multiple of base.
                 length % base should be 0.
                 This adjustment is ignored by setting base = 1.

    """

    new_blocks = Blocks()

    for b in blocks:
        _b = b.copy()

        match = _op_pattern.match(b.name)
        if match is None:
            new_blocks.append(_b)
            continue

        base_name = "P{:s}-{:s}".format(*match.group(1, 2))

        def make_name(i):
            if i == 0:
                return base_name
            else:
                return f"{base_name}-{i:d}"

        pattern = []
        i = 0
        for ch, period in _b.pattern:
            if period < length * 4:
                pattern.append((ch, period))
                continue

            # divide long period into three:
            # Remaining (surplus) - Repeated (Nrep >= 2) - Remaining (length)
            # the third one assures that we won't make too short block
            # even if we have short pattern after this.

            Nrep = period // length - 2
            surplus = period % length + length

            # try to meet base requirement
            total = surplus + sum([p for _, p in pattern])
            m = total % base

            pattern.append((ch, surplus - m))

            new_blocks.append(Block(make_name(i), pattern))
            i += 1

            new_blocks.append(Block(make_name(i), [(ch, length)], Nrep=Nrep))
            i += 1

            pattern = [(ch, length + m)]

        if pattern:
            new_blocks.append(Block(make_name(i), pattern))
        # if i:
        #     print("divided operation block: {:s}".format(base_name))

    return new_blocks


def divide_long_laser(blocks: Blocks[Block], length: int) -> Blocks[Block]:
    """Divide long blocks for laser (read) into two blocks: remaining + repeated block.

    This is meaningful only for hardware where Nrep > 1 reduces memory consumption (e.g., DTG).

    :param length: length of the repeated block.

    """

    new_blocks = Blocks()

    for b in blocks:
        _b = b.copy()

        match = _read_pattern.match(b.name)
        if match is None:
            new_blocks.append(_b)
            continue

        ch, period = _b.pattern[0]

        # avoiding to make too short laser block.
        Nrep = period // length - 1
        surplus = period % length + length

        if Nrep <= 1:
            new_blocks.append(_b)
            continue

        _b.pattern = [(ch, surplus)]
        new_blocks.append(_b)

        rep_block = Block("read{:s}-{:s}-1".format(*match.group(1, 2)), [(ch, length)], Nrep=Nrep)
        new_blocks.append(rep_block)
        # print("divided laser block: read{:s}-{:s}".format(*match.group(1, 2)))

    return new_blocks


def divide_long_final(blocks: Blocks[Block], length: int) -> Blocks[Block]:
    """Divide long final block into two blocks: remaining + repeated block.

    This is meaningful only for hardware where Nrep > 1 reduces memory consumption (e.g., DTG).

    :param length: length of the repeated block.

    """

    _blocks = blocks.copy()
    final = _blocks.pop()

    ch, period = final.pattern[0]

    # avoiding to make too short laser block.
    Nrep = period // length - 1
    surplus = period % length + length

    if Nrep <= 1:
        _blocks.append(final)
    else:
        final.pattern = [(ch, surplus)]
        _blocks.append(final)

        rep_block = Block("FINAL-1", [(ch, length)], Nrep=Nrep)
        _blocks.append(rep_block)
        # print("divided FINAL block")

    return _blocks


def merge_short_blocks(blocks: Blocks[Block], length: int) -> Blocks[Block]:
    """merge too short blocks (period is shorter than `length`).

    This is meaningful only for hardware where too short block is not accepted (e.g., DTG).

    """

    new_blocks = Blocks()
    for b in blocks:
        new_b = b.copy()

        if len(new_blocks) == 0:
            new_blocks.append(new_b)
            continue

        last_b = new_blocks[-1]
        last_b_length = last_b.raw_length()
        is_read = _read_pattern.match(b.name) is not None
        if (last_b_length < length or is_read) and (last_b.Nrep == 1 and b.Nrep == 1):
            # print("merging {:s} and {:s}".format(last_b.name, b.name))
            last_b.pattern += new_b.pattern
            last_b.name += "_" + new_b.name
            continue

        new_blocks.append(new_b)

    return new_blocks


def extract_laser_timing(blocks: Blocks[Block]) -> list[int]:
    laser_timing = []
    t = 0
    for b in blocks:
        t += b.total_length()
        m = _op_pattern.match(b.name)
        if m is not None:
            laser_timing.append(t)
    return laser_timing
