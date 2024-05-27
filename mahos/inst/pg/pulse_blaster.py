#!/usr/bin/env python3

"""
SpinCore Pulse Blaster part of Pulse Generator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os
import enum

import ctypes as C

from ..instrument import Instrument
from ...msgs.inst.pg_msgs import TriggerType, Block, Blocks, BlockSeq


class OP(enum.Enum):
    """PulseBlaster instruction codes (op-codes)."""

    CONTINUE = 0
    STOP = 1
    LOOP = 2
    END_LOOP = 3
    JSR = 4  # jump to subroutine
    RTS = 5  # return from subroutine
    BRANCH = 6
    LONG_DELAY = 7
    WAIT = 8
    RTI = 9


class SpinCore_PulseBlasterESR_PRO(Instrument):
    """SpinCore PulseBlasterESR-PRO Pulse Generator."""

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self.CHANNELS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        if "channels" in conf:
            self.CHANNELS.update(conf.get("channels", {}))

        dll_dir = self.conf.get("dll_dir", r"C:\SpinCore\SpinAPI\lib")
        fn = self.conf.get("dll_name", "spinapi64.dll")
        path = os.path.join(dll_dir, fn)
        self.dll = C.windll.LoadLibrary(path)

        # last total block length and offsets.
        self.length = 0
        self.offsets = None
        self._trigger_type = None

        self.logger.info("Opened SpinAPI at {} version: {}".format(path, self._get_pb_version()))

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

        clock_MHz = self.conf.get("clock_MHz", 500.0)

        if self.check_error(self.dll.pb_init()) and self.check_error(
            self.dll.pb_core_clock(clock_MHz)
        ):
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
        return self.dll.pb_get_version.restype().decode()

    def _inst(self, output: int, inst: int, inst_data: int, length_ns: float) -> int:
        return self.dll.pb_inst_pbonly(output, inst, inst_data, length_ns)

    def _inst_continue(self, output: int, length_ns: float) -> int:
        """CONTINUE instruction. No branching / jump is performed."""

        return self.dll.pb_inst_pbonly(output, OP.CONTINUE.value, 0, length_ns)

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
        """NOTE: if channels is a container, channels in analog_channels are excluded."""

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
            return [parse(c) for c in channels if c not in self.analog_channels]

    def _included_channels_blocks(self, blocks: Blocks[Block]) -> list[int]:
        """create sorted list of channels in blocks converted to ints."""

        # take set() here because channels_to_ints() may contain non-unique integers
        # (CHANNELS is not always injective), though it's quite rare usecase.
        return sorted(list(set(self.channels_to_ints(blocks.channels()))))

    def _extract_trigger_blocks(self, blocks: Blocks[Block]) -> bool | None:
        trigger = False
        for i, block in enumerate(blocks):
            if block.trigger:
                if i:
                    self.logger.error("Can set trigger for first block only.")
                    return None
                trigger = True
        return trigger

    def validate_blocks(self, blocks: Blocks[Block], freq: float) -> list[int] | None:
        return True

    def validate_blockseq(self, blockseq: BlockSeq[Block], freq: float) -> list[int] | None:
        return True

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Make sequence from blocks."""

        self.offsets = [0] * len(blocks)
        self.length = blocks.total_length()

        # 0 is PULSE_PROGRAM
        self.check_error(self.dll.pb_start_programming(0))

        self.check_error(self.dll.pb_stop_programming())

        msg = f"Configured with blocks. length: {self.length}"
        self.logger.info(msg)
        return True

    def configure_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Make sequence from blockseq."""

        # TODO
        return self.configure_blocks(Blocks([blockseq.collapse()]), freq, trigger_type, n_runs)

    def trigger(self) -> bool:
        """issue a software trigger."""

        return self.check_error(self.dll.pb_start())

    # Standard API

    # def reset(self, label: str = "") -> bool:
    #     return True

    def start(self, label: str = "") -> bool:
        if self.sequence is None or self.n_runs is None:
            return self.fail_with("No sequence defined. configure() first.")
        self.ps.stream(self.sequence, self.n_runs)
        self.logger.info("Start streaming.")
        return True

    def stop(self, label: str = "") -> bool:
        success = self.check_error(self.dll.pb_stop())
        self.logger.info("Stopp the output.")
        return success

    def configure(self, params: dict, label: str = "") -> bool:
        if "blocks" in params and "freq" in params:
            return self.configure_blocks(
                params["blocks"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
                n_runs=params.get("n_runs"),
            )
        elif "blockseq" in params and "freq" in params:
            return self.configure_blockseq(
                params["blockseq"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
                n_runs=params.get("n_runs"),
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
