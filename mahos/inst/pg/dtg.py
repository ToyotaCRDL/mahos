#!/usr/bin/env python3

"""
Tektronix DTG (Data Timing Generator) part of Pulse Generator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from os import path
import time

from ..visa_instrument import VisaInstrument
from ...msgs.inst.pg_msgs import Block, Blocks, BlockSeq

from ..pg_dtg_core import dtg_io
from ..pg_dtg_core.dtg_core import DTGCoreMixin


class DTG5000(VisaInstrument, DTGCoreMixin):
    """Base Class for DTG5000 series.

    :param local_dir: Data exchange directory in local computer.
    :type local_dir: str
    :param remote_dir: Data exchange directory in remote DTG.
    :type remote_dir: str
    :param channels: mapping from channel names to indices.
    :type channels: dict[str | bytes, int]

    :param start_delay_sec: (default: 1.0) Delay between output relay on and sequencer start.
        This is to wait the output relay to stabilize.
    :type start_delay_sec: float
    :param start_query_delay_sec: (default: 0.1) Delay for OPC? command after start command.
    :type start_query_delay_sec: float
    :param start_loop_delay_sec: (default: 1.0) Delay for each loop of the start loop.
    :type start_loop_delay_sec: float
    :param start_loop_num: (default: 20) Max. number of start command repeats in the start loop.
    :type start_loop_num: int
    :param strict: (default: True) If True, check generated data strictly.
        Since offset will not be allowed in strict mode, the pulse pattern data should be
        prepared considering block granularity.
    :type strict: bool

    """

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 20000.0
        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(("local_dir", "remote_dir"))
        self.LOCAL_DIR = conf["local_dir"]
        self.REMOTE_DIR = conf["remote_dir"]
        self.SCAFFOLD = conf.get("scaffold_filename", "scaffold.dtg")
        self.SETUP = conf.get("setup_filename", "setup.dtg")

        self.logger.info(
            "Default Scaffold: {} Setup: {}".format(
                path.join(self.LOCAL_DIR, self.SCAFFOLD), path.join(self.LOCAL_DIR, self.SETUP)
            )
        )

        self.CHANNELS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        if "channels" in conf:
            self.CHANNELS.update(conf["channels"])
        self._start_delay_sec = conf.get("start_delay_sec", 1.0)
        self._start_query_delay_sec = conf.get("start_query_sec", 0.1)
        self._start_loop_delay_sec = conf.get("start_loop_delay_sec", 1.0)
        self._start_loop_num = conf.get("start_loop_num", 20)
        self._strict = conf.get("strict", True)

        # last total block length and offsets.
        self.length = 0
        self.offsets = None

    def min_block_len(self, freq):
        if not hasattr(self, "MIN_BLOCK_LENGTH"):
            raise NotImplementedError("MIN_BLOCK_LENGTH is not defined")

        for i, v in enumerate(
            (400e6, 200e6, 100e6, 50e6, 25e6, 20e6, 10e6, 5e6, 2.5, 2e6, 1e6, 0.5e6)
        ):
            if freq > v:
                return self.MIN_BLOCK_LENGTH[i]

        return self.MIN_BLOCK_LENGTH[-1]

    def block_granularity(self, freq):
        if freq > 400e6:
            return 4
        elif freq > 200e6:
            return 2
        else:
            return 1

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_positive: bool = True,
        scaffold_name: str | None = None,
        endless: bool = True,
    ) -> bool:
        """Generate tree using default sequence and write it."""

        tree = self._configure_tree_blocks(
            blocks, freq, trigger_positive, scaffold_name=scaffold_name, endless=endless
        )
        if tree is None:
            return False
        return self._write_load_setup(tree)

    def configure_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_positive: bool = True,
        scaffold_name: str | None = None,
        endless: bool = True,
    ) -> bool:
        """Generate tree using explicit sequence and write it."""

        if blockseq.nest_depth() > 2:
            return self.fail_with("maximum BlockSeq nest depth is 2 for DTG.")

        tree = self._configure_tree_blockseq(
            blockseq, freq, trigger_positive, scaffold_name=scaffold_name, endless=endless
        )
        if tree is None:
            return False
        return self._write_load_setup(tree)

    def _write_load_setup(self, tree) -> bool:
        local_path = path.join(self.LOCAL_DIR, self.SETUP)
        remote_path = "\\".join((self.REMOTE_DIR, self.SETUP))  # remote (dtg) is always windows.
        self.logger.info(f"Writing DTG setup file to {local_path}.")
        with open(local_path, "wb") as f:
            dtg_io.dump_tree(tree, f)
        self.logger.info(f"DTG is loading setup file at {remote_path}.")
        self.cls()  # to clear error queue
        self.inst.write(f'MMEM:LOAD "{remote_path}"')
        return self.check_error()

    def check_error(self) -> bool:
        """Query error once and return True if there's no error.

        log error and return False if there's some error.

        """

        msg = self.query_error()
        try:
            i = msg.find(",")
            code, m = int(msg[:i]), msg[i + 1 :]
        except ValueError:
            m = "Ill-formed error message: " + msg
            self.logger.error(m)
            return False

        if code:
            self.logger.error(f"Error from inst: {code} {m}")
            return False
        else:
            return True

    def start_once(self):
        self.logger.info(
            "Setting output all-on and starting sequencer. Delay {:.1f} sec.".format(
                self._start_delay_sec
            )
        )

        self.inst.write("OUTP:STAT:ALL ON")
        time.sleep(self._start_delay_sec)
        self.inst.write("TBAS:RUN ON")

    def get_sequencer_status(self) -> str:
        return self.inst.query("TBAS:RST?")

    def start_loop(self) -> bool:
        """repeat start command until it actually starts."""

        for i in range(self._start_loop_num):
            self.start_once()
            self.query_opc(delay=self._start_query_delay_sec)
            status = self.get_sequencer_status()

            self.logger.debug(f"{i}: TBAS:RST? = {status}")
            if status in ("RUN", "WAIT"):
                return True

            time.sleep(self._start_loop_delay_sec)

        self.logger.error("Failed to start.")
        return False

    # Standard API

    def start(self, label: str = "") -> bool:
        """Set all outputs on and start sequencer."""

        return self.start_loop()

    def stop(self, label: str = "") -> bool:
        self.inst.write("TBAS:RUN OFF")
        self.inst.write("OUTP:STAT:ALL OFF")
        self.logger.info("Stopped sequencer and set output all-off.")
        return True

    def configure(self, params: dict, label: str = "") -> bool:
        if params.get("n_runs") not in (None, 1):
            return self.fail_with("DTG only supports n_runs None or 1.")
        if "blocks" in params and "freq" in params:
            return self.configure_blocks(
                params["blocks"],
                params["freq"],
                self._trigger_positive(params.get("trigger_type")),
            )
        elif "blockseq" in params and "freq" in params:
            return self.configure_blockseq(
                params["blockseq"],
                params["freq"],
                self._trigger_positive(params.get("trigger_type")),
            )
        else:
            return self.fail_with("These params must be given: 'blocks' | 'blockseq' and 'freq'")

    def set(self, key: str, value=None) -> bool:
        if key == "trigger":
            return self.trg()
        elif key == "clear":
            return self.cls()
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key == "length":
            return self.length  # length of last configure_blocks
        elif key == "offsets":
            if args is None:
                return self.offsets  # offsets of last configure_blocks
            elif "blocks" in args and "freq" in args:
                return self.validate_blocks(args["blocks"], args["freq"])
            elif "blockseq" in args and "freq" in args:
                return self.validate_blockseq(args["blockseq"], args["freq"])
            else:
                self.logger.error(f"Invalid args for get(offsets): {args}")
                return None
        elif key == "opc":
            return self.query_opc(delay=args)
        elif key == "validate":
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


class DTG5078(DTG5000):
    MIN_BLOCK_LENGTH = (240, 120, 60, 30, 15, 12, 6, 3, 2, 2, 1, 1, 1)

    def max_block_len(self):
        return 8000000

    def block_granularity(self, freq):
        return 1

    def max_freq(self):
        return 750e6


class DTG5274(DTG5000):
    def block_granularity(self, freq):
        """Default implementation of DTG5000 follows manual,
        but causes block granularity error in some cases.
        Taking the safest way temporarily.
        """

        return 4

    def max_block_len(self):
        return 32000000

    def min_block_len(self, freq):
        """Default implementation of DTG5000 follows manual,
        but causes block length too short error in some cases.
        Taking the safest way temporarily.
        """

        return 960

    def max_freq(self):
        return 2.7e9


class DTG5334(DTG5000):
    MIN_BLOCK_LENGTH = (960, 480, 240, 120, 60, 48, 24, 12, 6, 5, 3, 2, 1)

    def max_block_len(self):
        return 64000000

    def max_freq(self):
        return 3.35e9
