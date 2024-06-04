#!/usr/bin/env python3

"""
SpinCore Pulse Blaster part of Pulse Generator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import typing as T
import sys
import os
import time
import threading

import ctypes as C

from ..instrument import Instrument
from ...msgs.inst.pg_msgs import TriggerType, Block, Blocks, BlockSeq


class PulseBlasterStatus(T.NamedTuple):
    stopped: bool
    reset: bool
    running: bool
    waiting: bool


class SpinCore_PulseBlasterESR_PRO(Instrument):
    """SpinCore PulseBlasterESR-PRO Pulse Generator."""

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self.CHANNELS = {str(i): i for i in range(24)}
        if "channels" in conf:
            self.CHANNELS.update(conf.get("channels", {}))

        if sys.platform == "win32":
            dll_dir = self.conf.get("dll_dir", r"C:\SpinCore\SpinAPI\lib")
            fn = self.conf.get("dll_name", "spinapi64.dll")
            path = os.path.join(dll_dir, fn)
            self.dll = C.windll.LoadLibrary(path)
        else:  # assume linux
            dll_dir = self.conf.get("dll_dir", os.path.expanduser("~/.local/lib"))
            fn = self.conf.get("dll_name", "libspinapi.so")
            path = os.path.join(dll_dir, fn)
            self.dll = C.cdll.LoadLibrary(path)

        # last total block length and offsets.
        self.length = 0
        self.offsets = None
        self._trigger_type = None

        self.logger.info("Opened SpinAPI ({}) version: {}".format(path, self._get_pb_version()))

        num_boards = self.dll.pb_count_boards()
        self.logger.debug(f"boards -> {num_boards}")

        if num_boards <= 0:
            msg = "No PulseBlaster board available."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if num_boards > 1:
            if "board_index" not in self.conf:
                msg = "Multiple PulseBlaster found. Must specify conf['board_index']."
                self.logger.error(msg)
                raise RuntimeError(msg)
            i = self.conf["board_index"]
            self.check_error(self.dll.pb_select_board(i))
        else:
            i = 0

        self._freq = self.conf.get("freq", 500.0e6)
        self._min_duration_ns = 5 * round(1e9 / self._freq)
        clock_MHz = self._freq * 1e-6

        if self.check_error(self.dll.pb_init()):
            self.dll.pb_core_clock(
                C.c_double(clock_MHz)
            )  # Cannot check error as this one is void.
            self.logger.info(f"Initialized PulseBlasterESR-PRO ({i}) at {clock_MHz:.1f} MHz")
        else:
            msg = f"Failed to initialize PulseBlasterESR-PRO ({i})"
            self.logger.error(msg)
            raise RuntimeError(msg)

        self._max_instructions = self.conf.get("max_instructions", 4096)
        self._verbose = self.conf.get("verbose", False)
        self._sanity_check = self.conf.get("sanity_check", False)
        self._running = False
        self._bs_list = []
        self._bs_list_i = 0
        self._bs_list_mode = False
        self._bs_list_stop_ev = self._bs_list_thread = None

    def close_resources(self):
        if hasattr(self, "dll"):
            self.dll.pb_close()

    def _get_pb_version(self) -> str:
        self.dll.pb_get_version.restype = C.c_char_p
        return self.dll.pb_get_version().decode()

    def _inst(self, output: int, inst: int, inst_data: int, duration_ns: float) -> int:
        return self.dll.pb_inst_pbonly(output, inst, inst_data, C.c_double(duration_ns))

    def _log_inst(self, msg: str):
        if not self._verbose:
            return
        self.logger.debug(msg)

    def _CONTINUE(self, output: int, duration_ns: float) -> int:
        """CONTINUE instruction. No branching / jump is performed."""

        ret = self._inst(output, 0, 0, duration_ns)
        self._log_inst(f"{ret}: CONTINUE Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _STOP(self, output: int, duration_ns: float) -> int:
        """STOP instruction."""

        ret = self._inst(output, 1, 0, duration_ns)
        self._log_inst(f"{ret}: STOP Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _LOOP(self, output: int, loop_num: int, duration_ns: float) -> int:
        """LOOP instruction."""

        ret = self._inst(output, 2, loop_num, duration_ns)
        self._log_inst(f"{ret}: LOOP ({loop_num}) Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _END_LOOP(self, output: int, loop_addr: int, duration_ns: float) -> int:
        """END_LOOP instruction. loop_addr is instruction addr (ret value) of loop instruction."""

        ret = self._inst(output, 3, loop_addr, duration_ns)
        self._log_inst(f"{ret}: END_LOOP ({loop_addr}) Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _BRANCH(self, output: int, index: int, duration_ns: float) -> int:
        """BRANCH instruction."""

        ret = self._inst(output, 6, index, duration_ns)
        self._log_inst(f"{ret}: BRANCH ({index}) Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _WAIT(self, output: int, duration_ns: float) -> int:
        """WAIT instruction. Wait for trigger."""

        ret = self._inst(output, 8, 0, duration_ns)
        self._log_inst(f"{ret}: WAIT Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def check_error(self, ret: int) -> bool:
        """Check ret value of int function and log message if error. return True if OK."""

        if ret >= 0:
            return True

        size = 512
        buf = C.create_string_buffer(512)
        self.dll.pb_get_last_error(buf, size)
        msg = buf.value.decode()
        if msg:
            self.logger.error(f"Error ({ret}): {msg}")
        else:
            self.logger.error(f"There is an error ({ret}) but no message found.")
        return False

    def check_inst(self, ret: int) -> bool:
        """Check ret value of instruction function. return True if OK.

        :raises ValueError: when max number of instructions exceeded.

        """

        if ret > self._max_instructions - 1:
            raise ValueError(f"Max number of instructions ({self._max_instructions}) exceeded.")
        return self.check_error(ret)

    def channels_to_ints(self, channels) -> list[int]:
        def parse(c):
            if isinstance(c, (bytes, str)):
                return self.CHANNELS[c]
            elif isinstance(c, int):
                return c
            else:
                raise TypeError("Invalid type for channel. Use bytes, str or int.")

        if channels is None:
            return []
        elif isinstance(channels, (bytes, str, int)):
            return [parse(channels)]
        else:  # container (list or tuple) of (bytes, str, int)
            return [parse(c) for c in channels]

    def channels_to_output(self, channels) -> int:
        data = 0
        for bit in self.channels_to_ints(channels):
            data |= 1 << bit
        return data

    def _included_channels_blocks(self, blocks: Blocks[Block]) -> list[int]:
        """create sorted list of channels in blocks converted to ints."""

        # take set() here because channels_to_ints() may contain non-unique integers
        # (CHANNELS is not always injective), though it's quite rare usecase.
        return sorted(list(set(self.channels_to_ints(blocks.channels()))))

    def validate_blocks(self, blocks: Blocks[Block], freq: float) -> list[int] | None:
        # TODO
        offsets = [0] * len(blocks)
        return offsets

    def validate_blockseq(self, blockseq: BlockSeq[Block], freq: float) -> list[int] | None:
        # TODO
        offsets = [0] * len(blockseq.unique_blocks())
        return offsets

    def _extract_trigger_blocks(self, blocks: Blocks[Block]) -> bool | None:
        trigger = False
        for i, block in enumerate(blocks):
            if block.trigger:
                if i:
                    self.logger.error("Can set trigger for first block only.")
                    return None
                trigger = True
        return trigger

    def _extract_trigger_blockseq(self, blockseq: BlockSeq) -> bool | None:
        def any_trigger(bs):
            if isinstance(bs, BlockSeq):
                return bs.trigger or any([any_trigger(b) for b in bs.data])
            else:  # Block
                return bs.trigger

        # check other trigger is all False
        if any([any_trigger(b) for b in blockseq.data]):
            self.logger.error("Can set trigger for outermost BlockSeq only.")

        return blockseq.trigger

    def _configure_blocks_no_loop_block(self, block: Block, is_last: bool, head_addr: int) -> bool:
        Nj = block.total_pattern_num()
        for j, (channels, duration) in enumerate(block.total_pattern()):
            if duration < 5:
                # TODO: short pulse feature can be used to generate pulse shorter
                # than this limit. But we don't implement this as it can introduce little
                # change of total length and break the measurement.
                return self.fail_with("Duration cannot be shorter than 5 periods.")

            output = self.channels_to_output(channels)
            dur = duration * round(1e9 / self._freq)
            if is_last and j == Nj - 1:
                # the last pattern, branch to the head.
                ret = self._BRANCH(output, head_addr, dur)
            else:
                ret = self._CONTINUE(output, dur)
            if not self.check_inst(ret):
                return False
        return True

    def _configure_blocks_loop_block(self, block: Block, is_last: bool, head_addr: int) -> bool:
        loop_addr = 0
        Nj = block.raw_pattern_num()
        for j, (channels, duration) in enumerate(block.pattern):
            if duration < 5:
                return self.fail_with("Duration cannot be shorter than 5 periods.")
            output = self.channels_to_output(channels)
            dur = duration * round(1e9 / self._freq)
            if j == 0:
                loop_addr = ret = self._LOOP(output, block.Nrep, dur)
            elif is_last and j == Nj - 1:
                # Can reach here only when trigger, where BRANCH can be safely added after END_LOOP
                if not self.check_inst(self._END_LOOP(output, loop_addr, dur)):
                    return False
                ret = self._BRANCH(0, head_addr, self._min_duration_ns)
            elif j == Nj - 1:
                ret = self._END_LOOP(output, loop_addr, dur)
            else:
                ret = self._CONTINUE(output, dur)
            if not self.check_inst(ret):
                return False
        return True

    def _make_trigger_block(self) -> int:
        # Insert short CONTINUE as we cannot set WAIT at the very beginning.
        if not self.check_inst(self._CONTINUE(0, self._min_duration_ns)):
            return -1
        head_addr = self._WAIT(0, self._min_duration_ns)
        if not self.check_inst(head_addr):
            return -1
        # Insert short CONTINUE again. Without this and non-zero data the head of blocks,
        # first start() shows unexpected behaviour. (output can be high after first start())
        if not self.check_inst(self._CONTINUE(0, self._min_duration_ns)):
            return -1
        return head_addr

    def _configure_blocks(self, blocks: Blocks[Block], trigger: bool) -> bool:
        """Configure infinite loop using Blocks. If trigger is True, wait trigger at the head.

        Note that trigger at the other position is ignored.

        """

        if trigger:
            head_addr = self._make_trigger_block()
            if head_addr < 0:
                return False
        else:
            head_addr = 0

        Ni = len(blocks)
        for i, block in enumerate(blocks):
            is_last = i == Ni - 1
            if block.Nrep == 1 or (not trigger and is_last):
                # without trigger, we cannot use LOOP for the last block
                # because BRANCH at the tail will disturb the pattern.
                if not self._configure_blocks_no_loop_block(block, is_last, head_addr):
                    return False
            else:
                if not self._configure_blocks_loop_block(block, is_last, head_addr):
                    return False
        return True

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_type: TriggerType | None = None,
    ) -> bool:
        """Make sequence from blocks."""

        if freq != self._freq:
            return self.fail_with("freq must be {:.1f} MHz".format(self._freq * 1e-6))

        trigger = self._extract_trigger_blocks(blocks)
        if trigger is None:
            return False

        self._bs_list_mode = False
        self.offsets = [0] * len(blocks)
        self.length = blocks.total_length()

        # self.check_error(self.dll.pb_reset())
        self.check_error(self.dll.pb_stop())

        # 0 is PULSE_PROGRAM
        self.check_error(self.dll.pb_start_programming(0))
        try:
            success = self._configure_blocks(blocks, trigger)
        except ValueError:
            self.logger.exception("Exception while programming")
            success = False
        self.check_error(self.dll.pb_stop_programming())

        if not success:
            return self.fail_with("Failed to configure with blocks.")

        msg = f"Configured with blocks. length: {self.length}"
        self.logger.info(msg)
        return True

    def _add_block_no_loop(self, block: Block, head_addr: int) -> bool:
        Nj = block.total_pattern_num()
        for j, (channels, duration) in enumerate(block.total_pattern()):
            if duration < 5:
                return self.fail_with("Duration cannot be shorter than 5 periods.")

            output = self.channels_to_output(channels)
            dur = duration * round(1e9 / self._freq)
            if head_addr >= 0 and j == Nj - 1:
                # the last pattern, branch to the head.
                ret = self._BRANCH(output, head_addr, dur)
            else:
                ret = self._CONTINUE(output, dur)
            if not self.check_inst(ret):
                return False
        return True

    def _add_block_loop(self, block: Block, head_addr: int) -> bool:
        loop_addr = 0
        Nj = block.raw_pattern_num()
        for j, (channels, duration) in enumerate(block.pattern):
            if duration < 5:
                return self.fail_with("Duration cannot be shorter than 5 periods.")
            output = self.channels_to_output(channels)
            dur = duration * round(1e9 / self._freq)
            if j == 0:
                loop_addr = ret = self._LOOP(output, block.Nrep, dur)
            elif head_addr >= 0 and j == Nj - 1:
                # Can reach here only when trigger, where BRANCH can be safely added after END_LOOP
                if not self.check_inst(self._END_LOOP(output, loop_addr, dur)):
                    return False
                ret = self._BRANCH(0, head_addr, self._min_duration_ns)
            elif j == Nj - 1:
                ret = self._END_LOOP(output, loop_addr, dur)
            else:
                ret = self._CONTINUE(output, dur)
            if not self.check_inst(ret):
                return False
        return True

    def _add_block(self, block: Block, head_addr: int) -> bool:
        if block.Nrep == 1:
            return self._add_block_no_loop(block, head_addr)
        else:
            return self._add_block_loop(block, head_addr)

    def _add_block_loop_start(self, block: Block, Nrep: int) -> tuple[bool, int]:
        assert block.Nrep == 1

        loop_addr = 0
        for j, (channels, duration) in enumerate(block.total_pattern()):
            if duration < 5:
                self.logger.error("Duration cannot be shorter than 5 periods.")
                return False, 0

            output = self.channels_to_output(channels)
            dur = duration * round(1e9 / self._freq)
            if j == 0:
                loop_addr = ret = self._LOOP(output, Nrep, dur)
            else:
                ret = self._CONTINUE(output, dur)
            if not self.check_inst(ret):
                return False, 0
        return True, loop_addr

    def _add_block_loop_end(self, block: Block, loop_addr: int, head_addr: int) -> bool:
        assert block.Nrep == 1

        Nj = block.total_pattern_num()
        for j, (channels, duration) in enumerate(block.total_pattern()):
            if duration < 5:
                return self.fail_with("Duration cannot be shorter than 5 periods.")

            output = self.channels_to_output(channels)
            dur = duration * round(1e9 / self._freq)
            if head_addr >= 0 and j == Nj - 1:
                # Can reach here only when trigger, where BRANCH can be safely added after END_LOOP
                if not self.check_inst(self._END_LOOP(output, loop_addr, dur)):
                    return False
                ret = self._BRANCH(0, head_addr, self._min_duration_ns)
            elif j == Nj - 1:
                ret = self._END_LOOP(output, loop_addr, dur)
            else:
                ret = self._CONTINUE(output, dur)
            if not self.check_inst(ret):
                return False
        return True

    def _add_blockseq_loop_start(self, blockseq: BlockSeq, Nrep: int) -> tuple[bool, int]:
        assert blockseq.Nrep == 1

        success = True
        fst = blockseq.data[0]
        if fst.Nrep > 1:
            self.logger.error("First element cannot be loop (Nrep > 1).")
            return False, 0
        if isinstance(fst, BlockSeq):
            s, addr = self._add_blockseq_loop_start(fst, Nrep)
            success &= s
        else:
            s, addr = self._add_block_loop_start(fst, Nrep)
            success &= s

        for b in blockseq.data[1:]:
            if isinstance(b, BlockSeq):
                success &= self._add_blockseq(b, -1)
            else:
                success &= self._add_block(b, -1)

        return success, addr

    def _add_blockseq_loop_end(self, blockseq: BlockSeq, addr: int, head_addr: bool) -> bool:
        assert blockseq.Nrep == 1

        success = True

        for b in blockseq.data[:-1]:
            if isinstance(b, BlockSeq):
                success &= self._add_blockseq(b, -1)
            else:
                success &= self._add_block(b, -1)

        last = blockseq.data[-1]
        if last.Nrep > 1:
            return self.fail_with("Last element cannot be loop (Nrep > 1).")
        if isinstance(last, BlockSeq):
            success &= self._add_blockseq_loop_end(last, addr, head_addr)
        else:
            success &= self._add_block_loop_end(last, addr, head_addr)
        return success

    def _add_blockseq(self, blockseq: BlockSeq, head_addr: int) -> bool:
        """When head_addr >= 0, it contains the last block."""

        success = True
        fst = blockseq.data[0]
        if blockseq.Nrep > 1:
            if fst.Nrep > 1:
                return self.fail_with("Bad nested loop. First element cannot be loop (Nrep > 1).")
            if len(blockseq.data) == 1:
                return self.fail_with("Length of looped BlockSeq must be more than 1.")
            if isinstance(fst, BlockSeq):
                s, addr = self._add_blockseq_loop_start(fst, blockseq.Nrep)
                success &= s
            else:
                s, addr = self._add_block_loop_start(fst, blockseq.Nrep)
                success &= s
        else:
            if head_addr >= 0 and len(blockseq.data) == 1:
                # first and last blockseq / block
                if isinstance(fst, BlockSeq):
                    success &= self._add_blockseq(fst, head_addr)
                else:
                    success &= self._add_block(fst, head_addr)
            else:
                if isinstance(fst, BlockSeq):
                    success &= self._add_blockseq(fst, -1)
                else:
                    success &= self._add_block(fst, -1)

        if len(blockseq.data) == 1:
            return success

        # for more than 3 elements
        for b in blockseq.data[1:-1]:
            if isinstance(b, BlockSeq):
                success &= self._add_blockseq(b, -1)
            else:
                success &= self._add_block(b, -1)

        last = blockseq.data[-1]
        if blockseq.Nrep > 1:
            if last.Nrep > 1:
                return self.fail_with("Bad nested loop. Last element cannot be loop (Nrep > 1).")
            if isinstance(last, BlockSeq):
                success &= self._add_blockseq_loop_end(last, addr, head_addr)
            else:
                success &= self._add_block_loop_end(last, addr, head_addr)
        else:
            if isinstance(last, BlockSeq):
                success &= self._add_blockseq(last, head_addr)
            else:
                success &= self._add_block(last, head_addr)

        return success

    def _configure_blockseq(self, blockseq: BlockSeq, trigger: bool) -> bool:
        """Configure infinite loop using BlockSeq. If trigger is True, wait trigger at the head.

        Note that trigger at the other position is ignored.

        """

        if trigger:
            head_addr = self._make_trigger_block()
            if head_addr < 0:
                return False
        else:
            head_addr = 0

        return self._add_blockseq(blockseq, head_addr)

    def _configure_blockseq_oneshot(self, blockseq: BlockSeq) -> bool:
        """Configure infinite loop using BlockSeq."""

        # disable infinite loop (BRANCH to head)
        head_addr = -1
        success = (
            self._add_blockseq(blockseq, head_addr)
            # CONTINUE is inserted here to turning off the output properly.
            and self.check_inst(self._CONTINUE(0, 10 * self._min_duration_ns))
            and self.check_inst(self._STOP(0, self._min_duration_ns))
        )
        return success

    def _fix_bs_unwrap(self, bs: BlockSeq | Block) -> BlockSeq | Block:
        """unwrap unnecessary BlockSeq."""

        if isinstance(bs, Block):
            return bs

        bs: BlockSeq
        if len(bs.data) == 1:
            self.logger.debug(f"Unwrap unnecessary BlockSeq {bs.name}")
            # Block may be returned here.
            return self._fix_bs_unwrap(bs.data[0].repeat(bs.Nrep))

        return BlockSeq(bs.name, [self._fix_bs_unwrap(b) for b in bs.data], bs.Nrep, bs.trigger)

    def _fix_bs_nest(
        self, bs: BlockSeq | Block, head: bool = False, tail: bool = False
    ) -> BlockSeq | Block:
        """unroll nested loop."""

        if isinstance(bs, Block):
            return bs

        fst: BlockSeq | Block = bs.data[0]
        last: BlockSeq | Block = bs.data[-1]
        if (head or bs.Nrep > 1) and fst.Nrep > 1:
            fst.Nrep -= 1
            b = fst.copy()
            b.name += "_head"
            b.Nrep = 1
            # head=True to propagate >3-times nested loop
            bs.data.insert(0, self._fix_bs_nest(b, head=True))
            self.logger.debug(f"Fix (nest head) {fst.name} -> {b.name}, {fst.name}")
        if (tail or bs.Nrep > 1) and last.Nrep > 1:
            last.Nrep -= 1
            b = last.copy()
            b.name += "_tail"
            b.Nrep = 1
            # tail=True to propagate >3-times nested loop
            bs.data.append(self._fix_bs_nest(b, tail=True))
            self.logger.debug(f"Fix (nest tail) {last.name} -> {last.name}, {b.name}")

        return BlockSeq(bs.name, [self._fix_bs_nest(b) for b in bs.data], bs.Nrep, bs.trigger)

    def _fix_bs_tail_loop(self, bs: BlockSeq | Block) -> BlockSeq | Block:
        """unroll tail loop."""

        if bs.Nrep > 1:
            bs.Nrep = bs.Nrep - 1
            b = bs.copy()
            b.name = b.name + "_tail"
            b.Nrep = 1
            self.logger.debug(f"Fix (tail loop) {bs.name} -> {bs.name}, {b.name}")
            if isinstance(bs, BlockSeq):
                return BlockSeq(bs.name + "_wrap", [bs] + [self._fix_bs_tail_loop(b)])
            else:  # isinstance(bs, Block)
                return BlockSeq(bs.name + "_wrap", [bs, b])
        if isinstance(bs, BlockSeq):
            return BlockSeq(bs.name, bs.data[:-1] + [self._fix_bs_tail_loop(bs.data[-1])])
        return bs

    def _fix_blockseq(self, blockseq: BlockSeq, trigger: bool) -> BlockSeq | None:
        """Fix BlockSeq so that we can configure.

        Requirements and fixes below

        - Length of looped BlockSeq must be more than 1:
            Unwrap unnecessary BlockSeq.
        - Nested loop cannot start / end with looped Block or BlockSeq:
            Unroll inner loop.
        - When not trigger, final Block cannot be loop:
            Unroll loop at tail.

        """

        original = blockseq.copy()
        bseq = self._fix_bs_unwrap(blockseq)
        bseq = self._fix_bs_nest(bseq)
        if isinstance(bseq, Block):
            # outermost BlockSeq has been unwrapped. we can safely set Nrep = 1.
            bseq = BlockSeq("unwrapped", bseq)

        # check final block loop here.
        if not trigger:
            bseq = self._fix_bs_tail_loop(bseq)

        if self._sanity_check:
            self.logger.info("Starting sanity check")
            success = bseq.equivalent(original)
            self.logger.info("Finish sanity check")
            if not success:
                self.logger.error("Failed sanity check of fix_blockseq.")
                return None

        return bseq

    def fix_blockseq(self, blockseq: BlockSeq) -> BlockSeq | None:
        trigger = self._extract_trigger_blockseq(blockseq)
        if trigger is None:
            return None
        return self._fix_blockseq(blockseq, trigger)

    def configure_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_type: TriggerType | None = None,
    ) -> bool:
        """Make sequence from blockseq."""

        if freq != self._freq:
            return self.fail_with("freq must be {:.1f} MHz".format(self._freq * 1e-6))

        trigger = self._extract_trigger_blockseq(blockseq)
        if trigger is None:
            return False

        # TODO: consider nest_depth() and LOOP depth
        # if blockseq.nest_depth() > N:
        #     return self.fail_with("maximum BlockSeq nest depth is N for PulseBlaster.")

        blockseq = self._fix_blockseq(blockseq, trigger)
        if blockseq is None:
            return False

        self._bs_list_mode = False
        self.offsets = [0] * len(blockseq.unique_blocks())
        self.length = blockseq.total_length()

        # self.check_error(self.dll.pb_reset())
        self.check_error(self.dll.pb_stop())

        # 0 is PULSE_PROGRAM
        self.check_error(self.dll.pb_start_programming(0))
        try:
            success = self._configure_blockseq(blockseq, trigger)
        except ValueError:
            self.logger.exception("Exception while programming")
            success = False
        self.check_error(self.dll.pb_stop_programming())

        if not success:
            return self.fail_with("Failed to configure with blockseq.")

        msg = f"Configured with blockseq. length: {self.length}"
        self.logger.info(msg)
        return True

    def configure_blockseq_list(
        self,
        blockseq_list: list[BlockSeq],
        freq: float,
        trigger_type: TriggerType | None = None,
    ) -> bool:
        """Configure using list of blockseq."""

        if freq != self._freq:
            return self.fail_with("freq must be {:.1f} MHz".format(self._freq * 1e-6))

        triggers = [self._extract_trigger_blockseq(bs) for bs in blockseq_list]
        if any(triggers):
            self.logger.error("Cannot set trigger in configure_blockseq_list.")
            return False

        # TODO: consider nest_depth() and LOOP depth
        # if blockseq.nest_depth() > N:
        #     return self.fail_with("maximum BlockSeq nest depth is N for PulseBlaster.")

        # set trigger = True to disable tail loop unrolling
        blockseq_list = [self._fix_blockseq(bs, True) for bs in blockseq_list]
        if any([bs is None for bs in blockseq_list]):
            return False

        if self._running:
            self.stop()

        self._bs_list = blockseq_list
        self._bs_list_i = 0
        self._bs_list_mode = True

        self.offsets = [0] * len(self._bs_list)
        self.length = sum([bs.total_length() for bs in self._bs_list])

        msg = f"Configured with blockseq_list. length: {self.length}"
        self.logger.info(msg)
        return True

    def _bs_list_program(self) -> bool:
        success = True
        idx = self._bs_list_i % len(self._bs_list)
        self.logger.info(f"Start programming blockseq_list[{idx}]")
        success &= self.check_error(self.dll.pb_start_programming(0))
        try:
            success &= self._configure_blockseq_oneshot(self._bs_list[idx])
        except ValueError:
            self.logger.exception("Exception while programming")
            success = False
        success &= self.check_error(self.dll.pb_stop_programming())
        if not success:
            return self.fail_with(f"Failed to configure with blockseq_list[{idx}].")
        self.logger.info(f"Done programming blockseq_list[{idx}]")
        return success

    def _bs_list_loop(self, ev: threading.Event):
        while True:
            if not self._bs_list_program():
                self.logger.error("Error programming. Quitting blockseq_list loop.")
                return
            if not self.check_error(self.dll.pb_start()):
                self.logger.error("Error starting. Quitting blockseq_list loop.")
                return
            while not self.get_status().stopped:
                time.sleep(10e-6)
            if not self.check_error(self.dll.pb_stop()):
                self.logger.error("Error stopping. Quitting blockseq_list loop.")
                return
            self._bs_list_i += 1
            if ev.is_set():
                self.logger.info("Quitting blockseq_list loop.")
                return

    def get_status(self) -> PulseBlasterStatus:
        s = self.dll.pb_read_status()
        # self.logger.debug(f"status bits: 0b{s:04b}")
        return PulseBlasterStatus(
            stopped=bool(s & 0b1),
            reset=bool(s & 0b10),
            running=bool(s & 0b100),
            waiting=bool(s & 0b1000),
        )

    def trigger(self) -> bool:
        """issue a software trigger."""

        return self.check_error(self.dll.pb_start())

    # Standard API

    # def reset(self, label: str = "") -> bool:
    #     return True

    def start(self, label: str = "") -> bool:
        success = self.check_error(self.dll.pb_start())
        self.logger.info("Starting pulse output.")
        if self._bs_list_mode:
            self._bs_list_stop_ev = threading.Event()
            self._bs_list_thread = threading.Thread(
                target=self._bs_list_loop, args=(self._bs_list_stop_ev,)
            )
            self._bs_list_thread.start()
        self._running = True
        return success

    def stop(self, label: str = "") -> bool:
        success = self.check_error(self.dll.pb_stop())
        if self._bs_list_mode:
            self._bs_list_stop_ev.set()
            self._bs_list_thread.join()
            self._bs_list_i = 0
        self.logger.info("Stopped pulse output.")
        self._running = False
        return success

    def configure(self, params: dict, label: str = "") -> bool:
        if params.get("n_runs") not in (None, 1):
            return self.fail_with("PulseBlaster only supports n_runs None or 1.")
        if "blocks" in params and "freq" in params:
            return self.configure_blocks(
                params["blocks"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
            )
        elif "blockseq" in params and "freq" in params:
            return self.configure_blockseq(
                params["blockseq"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
            )
        elif "blockseq_list" in params and "freq" in params:
            return self.configure_blockseq_list(
                params["blockseq_list"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
            )
        else:
            return self.fail_with("These params must be given: 'blocks' | 'blockseq' and 'freq'")

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "trigger":
            return self.trigger()
        elif key == "clear":  # for API compatibility
            return True
        else:
            return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None, label: str = ""):
        if key == "status":
            return self.get_status()
        elif key == "length":
            return self.length  # length of last configure_blocks
        elif key == "offsets":  # for API compatibility
            if args is None:
                return self.offsets
            elif "blocks" in args:
                return [0] * len(args["blocks"])
            elif "blockseq" in args:
                return [0]
        elif key == "opc":  # for API compatibility
            return True
        elif key == "validate":  # for API compatibility
            if "blocks" in args and "freq" in args:
                return self.validate_blocks(args["blocks"], args["freq"])
            elif "blockseq" in args and "freq" in args:
                return self.validate_blockseq(args["blockseq"], args["freq"])
            else:
                self.logger.error(f"Invalid args for get(validate): {args}")
                return None
        elif key == "fix":
            # for debug
            return self.fix_blockseq(args)
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
