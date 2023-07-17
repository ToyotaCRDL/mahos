#!/usr/bin/env python3

"""
Core functions for pattern generators for Pulse ODMR

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import typing as T
import re

import numpy as np
from itertools import chain

from ...msgs.inst_pg_msgs import Block, Blocks


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

    pulses0 and pulses1 should have same length.

    """

    def _reduce(values, freq, divisor):
        if any([v % divisor != 0 for v in values]):
            return values, freq
        else:
            print_fn(f"reducing parameters {values} by {divisor}")
            return _reduce([v // divisor for v in values], freq / divisor, 10)

    values = list(xdata) + common_pulses + pulses0 + pulses1
    values = [round(v * freq) for v in values]

    if reduce_start_divisor:
        print_fn("starting to reduce parameters")
        values, freq = _reduce(values, freq, reduce_start_divisor)
        print_fn(f"reduced to: {values} freq: {freq:.2e}")

    _xdata = np.round(values[: -len(common_pulses) - len(pulses0) - len(pulses1)]).astype(int)

    if len(pulses0) > 0:
        _common_pulses = values[
            -len(common_pulses) - len(pulses0) - len(pulses1) : -len(pulses0) - len(pulses1)
        ]
        _pulses0 = values[-len(pulses0) - len(pulses1) : -len(pulses1)]
        _pulses1 = values[-len(pulses1) :]
    else:
        _common_pulses = values[-len(common_pulses) :]
        _pulses0, _pulses1 = [], []

    return freq, _xdata, _common_pulses, _pulses0, _pulses1


def split_int(n, A):
    return n // A, n - n // A


def generate_blocks(
    i: int,
    v: int,
    common_pulses: list[int],
    gen_single_ptn,
    args0,
    args1,
    read_phase0: str = "mw_x",
    read_phase1: str = "mw_x_inv",
    partial: int = -1,
    nomw: bool = False,
    ignore_basewidth: bool = False,
):
    (
        base_width,
        laser_delay,
        laser_width,
        mw_delay,
        trigger_width,
        init_delay,
        final_delay,
    ) = common_pulses

    if ignore_basewidth:
        base_width = 1

    # base length offset (laser)
    laser_width = laser_width + base_width - laser_width % base_width

    # blocks
    ptn0 = gen_single_ptn(v, *args0)
    ptn1 = gen_single_ptn(v, *args1)

    ptn_former = [(("mw_x",), mw_delay)]
    ptn_latter0 = [((read_phase0,), laser_delay)]
    ptn_latter1 = [((read_phase1,), laser_delay)]
    ptn_laser = [(("mw_x", "laser", "sync"), laser_width)]

    if nomw:
        ptn0 = [((read_phase0,), sum([s[1] for s in ptn0]))]
        ptn1 = [((read_phase1,), sum([s[1] for s in ptn1]))]

    # base length offset (operation)
    total0 = sum([s[1] for s in ptn_former + ptn0 + ptn_latter0])
    if total0 % base_width == 0:
        ptn_operate0 = ptn_former + ptn0 + ptn_latter0
    else:
        ofs0 = base_width - total0 % base_width
        ptn_offset0 = [(ptn_laser[0][0], ofs0)]
        ptn_operate0 = ptn_offset0 + ptn_former + ptn0 + ptn_latter0

    total1 = sum([s[1] for s in ptn_former + ptn1 + ptn_latter1])
    if total1 % base_width == 0:
        ptn_operate1 = ptn_former + ptn1 + ptn_latter1
    else:
        ofs1 = base_width - total1 % base_width
        ptn_offset1 = [(ptn_laser[0][0], ofs1)]
        ptn_operate1 = ptn_offset1 + ptn_former + ptn1 + ptn_latter1

    blocks0 = [
        Block(f"P{i}-0", ptn_operate0),
        Block(f"read{i}-0", ptn_laser),
    ]
    blocks1 = [Block(f"P{i}-1", ptn_operate1), Block(f"read{i}-1", ptn_laser)]
    if partial == 0:
        return Blocks(blocks0)
    elif partial == 1:
        return Blocks(blocks1)
    else:  # complementary
        return Blocks(blocks0 + blocks1)


def build_blocks(
    blocks: Blocks[Block],
    common_pulses,
    divide=False,
    merge=True,
    invertY=False,
    minimum_block_length: int = 1000,
    block_base: int = 4,
) -> T.Tuple[Blocks[Block], T.List[int]]:
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

    # base length offset
    init_block_width = max(trigger_width + init_delay + laser_width, minimum_block_length)
    init_block_width = init_block_width + base_width - init_block_width % base_width
    final_block_width = max(final_delay, minimum_block_length)
    final_block_width = final_block_width + base_width - final_block_width % base_width

    ptn_init = [
        (("trigger", "sync", "mw_x"), trigger_width),
        (("sync", "mw_x"), init_delay),
        (("laser", "sync", "mw_x"), init_block_width - trigger_width - init_delay),
    ]
    ptn_final = [(("sync", "mw_x"), final_block_width)]

    # flatten
    blocks = list(chain.from_iterable(blocks))

    # insert init/final
    blocks.insert(0, Block("INIT", ptn_init))
    blocks.append(Block("FINAL", ptn_final))
    laser_timing = extract_laser_timing(blocks)

    # shaping blocks
    if divide:
        blocks = divide_long_operation(blocks, minimum_block_length, block_base)
        blocks = divide_long_laser(blocks, minimum_block_length)
        blocks = divide_long_final(blocks, minimum_block_length)
    if merge:
        blocks = merge_short_blocks(blocks, minimum_block_length)

    blocks = blocks.simplify()
    blocks = encode_mw_phase(blocks)
    if invertY:
        blocks = invert_y_phase(blocks)

    return blocks, laser_timing


def print_blocks(blocks: Blocks[Block], print_fn=print):
    for b in blocks:
        print_fn(str(b))
        print_fn("|".join(b.pattern_to_strs()))


def encode_mw_phase(blocks: Blocks[Block]) -> Blocks[Block]:
    """encode mw phase from x/y(_inv) to i/q."""

    iq_phase_dict = {
        "mw_x": ("mw_i", "mw_q"),
        "mw_y": ("mw_q",),
        "mw_x_inv": (),
        "mw_y_inv": ("mw_i",),
    }

    return blocks.replace(iq_phase_dict)


def invert_y_phase(blocks: Blocks[Block]) -> Blocks[Block]:
    """y ==> y_inv, y_inv ==> y (invert i <==> q)."""

    def invert(ch):
        if ch is None:
            return ch

        new_ch = list(ch)

        _i = "mw_i" in new_ch
        _q = "mw_q" in new_ch

        if _i and not _q:
            new_ch.remove("mw_i")
            new_ch.append("mw_q")
        if _q and not _i:
            new_ch.remove("mw_q")
            new_ch.append("mw_i")

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


def extract_laser_timing(blocks: Blocks[Block]) -> T.List[int]:
    laser_timing = []
    t = 0
    for b in blocks:
        t += b.total_length()
        m = _op_pattern.match(b.name)
        if m is not None:
            laser_timing.append(t)
    return laser_timing
