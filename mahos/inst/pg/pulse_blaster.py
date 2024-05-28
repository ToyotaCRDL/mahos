#!/usr/bin/env python3

"""
SpinCore Pulse Blaster part of Pulse Generator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import sys
import os

import ctypes as C

from ..instrument import Instrument
from ...msgs.inst.pg_msgs import TriggerType, Block, Blocks, BlockSeq


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

    def close_resources(self):
        if hasattr(self, "dll"):
            self.dll.pb_close()

    def _get_pb_version(self) -> str:
        self.dll.pb_get_version.restype = C.c_char_p
        return self.dll.pb_get_version().decode()

    def _inst(self, output: int, inst: int, inst_data: int, duration_ns: float) -> int:
        return self.dll.pb_inst_pbonly(output, inst, inst_data, C.c_double(duration_ns))

    def _CONTINUE(self, output: int, duration_ns: float) -> int:
        """CONTINUE instruction. No branching / jump is performed."""

        ret = self._inst(output, 0, 0, duration_ns)
        self.logger.debug(f"{ret}: CONTINUE Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _LOOP(self, output: int, loop_num: int, duration_ns: float) -> int:
        """LOOP instruction."""

        ret = self._inst(output, 2, loop_num, duration_ns)
        self.logger.debug(f"{ret}: LOOP ({loop_num}) Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _END_LOOP(self, output: int, loop_addr: int, duration_ns: float) -> int:
        """END_LOOP instruction. loop_addr is instruction addr (ret value) of loop instruction."""

        ret = self._inst(output, 3, loop_addr, duration_ns)
        self.logger.debug(f"{ret}: END_LOOP ({loop_addr}) Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _BRANCH(self, output: int, index: int, duration_ns: float) -> int:
        """BRANCH instruction."""

        ret = self._inst(output, 6, index, duration_ns)
        self.logger.debug(f"{ret}: BRANCH ({index}) Out: 0x{output:X} Dur: {duration_ns}")
        return ret

    def _WAIT(self, output: int, duration_ns: float) -> int:
        """WAIT instruction. Wait for trigger."""

        ret = self._inst(output, 8, 0, duration_ns)
        self.logger.debug(f"{ret}: WAIT Out: 0x{output:X} Dur: {duration_ns}")
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

    def validate_blocks(self, blocks: Blocks[Block]) -> list[int] | None:
        self.logger.warn("Not implemented!")
        return True

    def validate_blockseq(self, blockseq: BlockSeq[Block]) -> list[int] | None:
        self.logger.warn("Not implemented!")
        return True

    def _extract_trigger_blocks(self, blocks: Blocks[Block]) -> bool | None:
        trigger = False
        for i, block in enumerate(blocks):
            if block.trigger:
                if i:
                    self.logger.error("Can set trigger for first block only.")
                    return None
                trigger = True
        return trigger

    def _configure_blocks_infinite(self, blocks: Blocks[Block], trigger: bool) -> bool:
        """Configure infinite loop using blocks. If trigger is True, wait trigger at the head.

        Note that trigger at the other position is ignored.

        """

        if trigger:
            # Insert short CONTINUE as we cannot set WAIT at the very beginning.
            if not self.check_error(self._CONTINUE(0, self._min_duration_ns)):
                return False
            head_addr = ret = self._WAIT(0, self._min_duration_ns)
            if not self.check_error(ret):
                return False
            # Insert short CONTINUE again. Without this and non-zero data the head of blocks, 
            # first start() shows unexpected behaviour. (output can be high after first start())
            if not self.check_error(self._CONTINUE(0, self._min_duration_ns)):
                return False
        else:
            head_addr = 0

        N_i = len(blocks)
        for i, block in enumerate(blocks):
            patterns = block.total_pattern()
            N_j = len(patterns)
            for j, (channels, duration) in enumerate(patterns):
                if duration < 5:
                    # TODO: short pulse feature can be used to generate pulse shorter
                    # than this limit. But we don't implement this as it can introduce little
                    # change of total length and break the measurement.
                    return self.fail_with("Duration cannot be shorter than 5 periods.")
                output = self.channels_to_output(channels)
                dur = duration * round(1e9 / self._freq)
                if i == N_i - 1 and j == N_j - 1:
                    # the last pattern, branch to the head.
                    ret = self._BRANCH(output, head_addr, dur)
                else:
                    ret = self._CONTINUE(output, dur)
                if not self.check_error(ret):
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

        self.offsets = [0] * len(blocks)
        self.length = blocks.total_length()

        # self.check_error(self.dll.pb_reset())
        self.check_error(self.dll.pb_stop())

        # 0 is PULSE_PROGRAM
        self.check_error(self.dll.pb_start_programming(0))
        success = self._configure_blocks_infinite(blocks, trigger)
        self.check_error(self.dll.pb_stop_programming())

        if not success:
            return self.fail_with("Failed to configure with blocks.")

        msg = f"Configured with blocks. length: {self.length}"
        self.logger.info(msg)
        return True

    def configure_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_type: TriggerType | None = None,
    ) -> bool:
        """Make sequence from blockseq."""

        if freq != self._freq:
            return self.fail_with("freq must be {:.1f} MHz".format(self._freq * 1e-6))

        # TODO
        self.logger.error("Not implemented!")
        return False

    def trigger(self) -> bool:
        """issue a software trigger."""

        return self.check_error(self.dll.pb_start())

    # Standard API

    # def reset(self, label: str = "") -> bool:
    #     return True

    def start(self, label: str = "") -> bool:
        success = self.check_error(self.dll.pb_start())
        self.logger.info("Starting pulse output.")
        return success

    def stop(self, label: str = "") -> bool:
        success = self.check_error(self.dll.pb_stop())
        self.logger.info("Stopped pulse output.")
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
        if key == "length":
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
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
