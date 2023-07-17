#!/usr/bin/env python3

"""
Data Timing Generator module.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import typing as T
from os import path
import time

import numpy as np

from ..instrument import Instrument
from ..visa_instrument import VisaInstrument
from ...msgs.inst_pg_msgs import TriggerType, Block, Blocks

from . import dtg_io


def gen_block(length, blockid, name):
    """generate a tree for a DTG block.

    block is based on given length, id, and name.

    'name' may be Python string, or byte string packed into ndarray (end with NULL).
    'length' and 'id' may be Python integers, or integers packed into ndarray.

    calculation of tree size is abondoned (done later by dtg_io.recalculate_space()).

    """

    if isinstance(blockid, int):
        blockid = np.array([blockid], dtype=np.int16)

    if isinstance(name, str):
        name = np.array(bytes(name, encoding="utf-8") + b"\x00", dtype="|S")

    if isinstance(length, int):
        length = np.array([length], dtype=np.int32)

    return [
        "DTG_BLOCK_RECID",
        30,
        [
            ["DTG_ID_RECID", 2, blockid],
            ["DTG_NAME_RECID", name.nbytes, name],
            ["DTG_SIZE_RECID", 4, length],
        ],
    ]


def gen_pattern(blockid, pulses):
    """generate a python tree for a pattern command.

    pattern is to fill block 'blockid' with the binary sequence 'pulses'.

    calculation of tree size is abondoned (done later by dtg_io.recalculate_space()).

    """

    if isinstance(blockid, int):
        blockid = np.array([blockid], dtype=np.int16)

    return [
        "DTG_PATTERN_RECID",
        1,
        [
            ["DTG_GROUPID_RECID", 2, np.array([0], dtype=np.int16)],
            ["DTG_BLOCKID_RECID", 2, blockid],
            ["DTG_PATTERNDATA_RECID", 1, pulses],
        ],
    ]


def gen_subsequence(subid, subname, steps):
    """generate a python tree for a subsequence entry.

    subsequence entry is based on given id, name, list of subsequences.

    'subid' may be Python integer, or integers packed into ndarray.
    'subname' may be Python strings, or byte string packed into ndarray (end with NULL).
    'steps' may be Python list of tuples consisted of block name and repeatcount.

    calculation of tree size is abondoned (done later by dtg_io.recalculate_space()).

    """

    if isinstance(subid, int):
        subid = np.array([subid], dtype=np.int16)

    if isinstance(subname, str):
        subname = np.array(bytes(subname, encoding="utf-8") + b"\x00", dtype="|S")

    subseq = [
        "DTG_SUBSEQUENCE_RECID",
        1,
        [["DTG_ID_RECID", 2, subid], ["DTG_NAME_RECID", subname.nbytes, subname]],
    ]

    for blockname, Nrep in steps:
        if isinstance(Nrep, int):
            Nrep = np.array([Nrep], dtype=np.int32)

        if isinstance(blockname, str):
            blockname = np.array(bytes(blockname, encoding="utf-8") + b"\x00", dtype="|S")

        subseq[2].append(
            [
                "DTG_SUBSEQUENCESTEP_RECID",
                1,
                [
                    ["DTG_SUBNAME_RECID", blockname.nbytes, blockname],
                    ["DTG_REPEATCOUNT_RECID", 4, Nrep],
                ],
            ]
        )

    return subseq


def gen_sequence(label, subname, Nrep, goto, jumpto, trigger):
    """generate a python tree for a sequence entry.

    sequence entry is based on given label, name of the block/subsequence,
    number of repetitions 'Nrep' and goto label 'goto'.

    label, subname, goto may be Python strings, or byte string packed into ndarray (end with NULL).
    'Nrep' and 'trigger' may be Python integer, or integers packed into ndarray.

    'Event Jump To' is currently disabled.

    calculation of tree size is abondoned (done later by dtg_io.recalculate_space()).

    """

    if isinstance(label, str):
        label = np.array(bytes(label, encoding="utf-8") + b"\x00", dtype="|S")

    if isinstance(subname, str):
        subname = np.array(bytes(subname, encoding="utf-8") + b"\x00", dtype="|S")

    if isinstance(goto, str):
        goto = np.array(bytes(goto, encoding="utf-8") + b"\x00", dtype="|S")

    if isinstance(jumpto, str):
        jumpto = np.array(bytes(jumpto, encoding="utf-8") + b"\x00", dtype="|S")

    if isinstance(Nrep, int):
        Nrep = np.array([Nrep], dtype=np.int32)

    if isinstance(trigger, int):
        trigger = np.array([trigger], dtype=np.int16)

    return [
        "DTG_MAINSEQUENCE_RECID",
        51,
        [
            ["DTG_LABEL_RECID", label.nbytes, label],
            ["DTG_WAITTRIGGER_RECID", 2, trigger],
            ["DTG_SUBNAME_RECID", subname.nbytes, subname],
            ["DTG_REPEATCOUNT_RECID", 4, Nrep],
            ["DTG_JUMPTO_RECID", jumpto.nbytes, jumpto],
            ["DTG_GOTO_RECID", goto.nbytes, goto],
        ],
    ]


class DTG5000(VisaInstrument):
    """Base Class for DTG5000 series."""

    MAX_BLOCK_NUM = 8000

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 20000.0
        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(("root_local", "root_remote"))
        self.ROOT_LOCAL = conf["root_local"]
        self.ROOT_REMOTE = conf["root_remote"]
        self.SCAFFOLD = conf.get("scaffold_filename", "scaffold.dtg")
        self.SETUP = conf.get("setup_filename", "setup.dtg")

        self.CHANNELS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        if "channels" in conf:
            self.CHANNELS.update(conf["channels"])
        self._start_delay_sec = conf.get("start_delay_sec", 0.1)

        # last total block length and offsets.
        self.length = 0
        self.offsets = None

    def load_scaffold(self, path):
        """load and return the DTG config scaffold."""

        with open(path, "rb") as f:
            data = f.read()

        return dtg_io.read_tree(data)

    def channels_to_int(self, channels):
        def parse(c):
            if isinstance(c, (bytes, str)):
                return self.CHANNELS[c]
            elif isinstance(c, int):
                return c
            else:
                raise TypeError("Invalid type for channel. Use bytes, str or int.")

        if channels is None:
            return 0

        if isinstance(channels, (bytes, str, int)):
            return 1 << parse(channels)

        bits = 0
        for channel in channels:
            bits |= 1 << parse(channel)

        return bits

    def get_total_block_length_advanced(
        self, blocks: Blocks[Block], subsequences, sequences
    ) -> T.Optional[int]:
        t = 0
        for i, seq in enumerate(sequences):
            blks = [b for b in blocks if b.name == seq["name"]]
            if len(blks) > 0:
                t += blks[0].raw_length() * seq["Nrep"]
            else:
                subt = 0
                subseq = [s for s in subsequences if s["name"] == seq["name"]]
                if len(subseq) == 0:
                    self.logger.error(
                        f"Sequence ({i:d}, {seq['label']}) refers an undef. block/subsequence:"
                        + f" {seq['name']}"
                    )
                    return None

                for j, step in enumerate(subseq[0]["steps"]):
                    blks = [b for b in blocks if b.name == step["name"]]
                    if len(blks) == 0:
                        self.logger.error(
                            "SubSequence ({:d}, {:s}) refers an undefined block ({:s}).".format(
                                j, subseq[0]["name"], step["name"]
                            )
                        )
                        return None
                    subt += blks[0].raw_length() * step["Nrep"]
                t += subt * seq["Nrep"]

        return t

    def generate_tree(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_positive: bool = True,
        scaffold_path=None,
        endless=True,
    ):
        """Generate tree using default sequence."""

        blk_trees = []
        seq_trees = []
        ptn_trees = []

        for i, block in enumerate(blocks):
            blk_trees.append(gen_block(block.raw_length(), i + 1, block.name))

            seq_trees.append(gen_sequence("", block.name, block.Nrep, "", "", int(block.trigger)))

            self.logger.debug(f"{block.name}(x{block.Nrep})|" + "|".join(block.pattern_to_strs()))
            pulses = np.concatenate(
                [
                    self.channels_to_int(channels) * np.ones(length, dtype=np.uint8)
                    for channels, length in block.pattern
                ]
            )

            ptn_trees.append(gen_pattern(i + 1, pulses))

        # label the first subsequence as 'START' and point the goto of the last subsequence there
        if endless:
            dtg_io.replace_leaf(
                seq_trees[0][2], "DTG_LABEL_RECID", 5, np.array(b"START\x00", dtype="|S")
            )
            dtg_io.replace_leaf(
                seq_trees[-1][2], "DTG_GOTO_RECID", 5, np.array(b"START\x00", dtype="|S")
            )

        tree = self.load_scaffold(scaffold_path)
        top = tree[0][-1]

        # set clock frequency
        dtg_io.replace_leaf(top, "DTG_TB_CLOCK_RECID", 8, np.array([freq], dtype=np.float64))

        # set trigger slope
        if trigger_positive:
            dtg_io.replace_leaf(top, "DTG_TRIGGER_SLOPE_RECID", 2, np.array([0], dtype=np.int16))
        else:
            dtg_io.replace_leaf(top, "DTG_TRIGGER_SLOPE_RECID", 2, np.array([1], dtype=np.int16))

        # remove parts of default setup (view, ptn, mainseq, block)
        for i in range(4):
            top.pop()

        top.extend(blk_trees + seq_trees + ptn_trees)

        # seems to be unnecessary.
        # top.append(view)

        tree = dtg_io.recalculate_space(tree)[1]

        return tree

    def _adjust_blocks(self, blocks: Blocks[Block], freq: float) -> T.Optional[T.List[int]]:
        gran = self.block_granularity(freq)
        min_len = self.min_block_len(freq)

        plus = True
        offsets = []
        for block in blocks:
            # adjust total length to integer multiple of gran (to avoid block granularity error).
            length = block.raw_length()
            m = length % gran
            if m != 0:
                if plus:
                    ofs = gran - m
                    plus = False
                else:
                    ofs = -m
                    plus = True

                block.pattern[0] = (block.pattern[0][0], block.pattern[0][1] + ofs)
                self.logger.warn(
                    "Adjusted block {}. First step length is offset by {}.".format(block.name, ofs)
                )
                offsets.append(ofs)
            else:
                offsets.append(0)

            if length > self.MAX_BLOCK_LENGTH or length < min_len:
                self.logger.error(
                    "Block length ({}) out of bounds ({}, {}).".format(
                        length, min_len, self.MAX_BLOCK_LENGTH
                    )
                )
                return None

        return offsets

    def validate_blocks(self, blocks: Blocks[Block], freq: float) -> T.Optional[T.List[int]]:
        if len(blocks) > self.MAX_BLOCK_NUM:
            self.logger.error(
                "Number of blocks ({}) exceeds max value ({}).".format(
                    len(blocks), self.MAX_BLOCK_NUM
                )
            )
            return None

        if len(blocks) > len(set([b.name for b in blocks])):
            self.logger.error("Block names cannot be duplicated.")
            return None

        return self._adjust_blocks(blocks, freq)

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

    def _trigger_positive(self, trigger_type: T.Optional[TriggerType]) -> bool:
        """Translate trigger_type to trigger_positive (bool) for API compatibility."""

        return trigger_type != TriggerType.HARDWARE_FALLING

    def _configure_tree(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_positive: bool = True,
        scaffold_name=None,
        endless=True,
    ) -> bool:
        self.offsets = self.validate_blocks(blocks, freq)
        if self.offsets is None:
            return False
        self.length = blocks.total_length()

        if scaffold_name is None:
            scaffold_name = self.SCAFFOLD
        scaffold_path = path.join(self.ROOT_LOCAL, scaffold_name)

        tree = self.generate_tree(
            blocks,
            freq,
            trigger_positive=trigger_positive,
            scaffold_path=scaffold_path,
            endless=endless,
        )

        self.logger.info(f"DTG tree prepared. Total length: {self.length}")

        return tree

    def configure_blocks(
        self, blocks, freq, trigger_positive: bool = True, scaffold_name=None, endless=True
    ) -> bool:
        """Generate tree using default sequence and write it."""

        tree = self._configure_tree(
            blocks, freq, trigger_positive, scaffold_name=scaffold_name, endless=endless
        )
        return self._write_load_setup(tree)

    def _write_load_setup(self, tree) -> bool:
        local_path = path.join(self.ROOT_LOCAL, self.SETUP)
        remote_path = "\\".join((self.ROOT_REMOTE, self.SETUP))  # remote (dtg) is always windows.
        self.logger.info(f"Writing DTG setup file to {local_path}.")
        with open(local_path, "wb") as f:
            dtg_io.dump_tree(tree, f)
        self.logger.info(f"DTG is loading setup file at {remote_path}.")
        self.cls()  # to clear error queue
        self.inst.write(f'MMEM:LOAD "{remote_path}"')
        return self.check_error()

    # NOTE: this is not tested.
    def generate_tree_advanced(
        self,
        blocks: Blocks[Block],
        subsequences,
        sequences,
        freq,
        trigger_positive: bool = True,
        scaffold_path=None,
    ):
        """Generate tree using explicit sequence.

        sequence is {'label': sequence label, 'name': blk/sub name, 'Nrep': repeat count,
                     'goto', 'jumpto', 'trigger'}
        subsequence is {'name': subname,
                        'steps': [{'name': blockname, 'Nrep': Nrep},
                                  {'name': blockname, 'Nrep': Nrep}, ...]
                       }

        """

        blk_trees = []
        seq_trees = []
        subseq_trees = []
        ptn_trees = []

        for i, block in enumerate(blocks):
            blk_trees.append(gen_block(block.raw_length(), i + 1, block.name))
            pulses = np.concatenate(
                [
                    self.channels_to_int(channels) * np.ones(length, dtype=np.uint8)
                    for channels, length in block.pattern
                ]
            )
            ptn_trees.append(gen_pattern(i + 1, pulses))

        for i, subsequence in enumerate(subsequences):
            steps = [(s["name"], s["Nrep"]) for s in subsequence["steps"]]
            subseq_trees.append(gen_subsequence(len(blocks) + i + 1, subsequence["name"], steps))

        for i, sequence in enumerate(sequences):
            if "label" in sequence:
                label = sequence["label"]
            else:
                label = ""

            if "goto" in sequence:
                goto = sequence["goto"]
            else:
                goto = ""

            if "jumpto" in sequence:
                jumpto = sequence["jumpto"]
            else:
                jumpto = ""

            if "trigger" in sequence:
                trig = sequence["trigger"]
            else:
                trig = 0

            seq_trees.append(
                gen_sequence(label, sequence["name"], sequence["Nrep"], goto, jumpto, trig)
            )

        tree = self.load_scaffold(scaffold_path)
        top = tree[0][-1]

        # set clock frequency
        dtg_io.replace_leaf(top, "DTG_TB_CLOCK_RECID", 8, np.array([freq], dtype=np.float64))

        # set trigger slope
        if trigger_positive:
            dtg_io.replace_leaf(top, "DTG_TRIGGER_SLOPE_RECID", 2, np.array([0], dtype=np.int16))
        else:
            dtg_io.replace_leaf(top, "DTG_TRIGGER_SLOPE_RECID", 2, np.array([1], dtype=np.int16))

        # remove parts of default setup (view, ptn, mainseq, block)
        for i in range(4):
            top.pop()

        top.extend(blk_trees + subseq_trees + seq_trees + ptn_trees)

        tree = dtg_io.recalculate_space(tree)[1]

        return tree

    # NOTE: this is not tested.
    def validate_blocks_advanced(
        self, blocks: Blocks[Block], subsequences, sequences, freq: float
    ) -> T.Optional[T.List[int]]:
        def fail(msg):
            self.logger.error(msg)
            return None

        if len(blocks) > self.MAX_BLOCK_NUM:
            return fail(
                "Number of blocks ({}) exceeds max value ({}).".format(
                    len(blocks), self.MAX_BLOCK_NUM
                )
            )
        if len(sequences) > self.MAX_BLOCK_NUM:
            return fail(
                "Number of sequences ({}) exceeds max value ({}).".format(
                    len(sequences), self.MAX_BLOCK_NUM
                )
            )
        if len(subsequences) > self.MAX_BLOCK_NUM:
            return fail(
                "Number of subsequences ({}) exceeds max value ({}).".format(
                    len(subsequences), self.MAX_BLOCK_NUM
                )
            )

        # name check
        names = [b.name for b in blocks] + [sub["name"] for sub in subsequences]
        if len(blocks) + len(subsequences) > len(set(names)):
            return fail("Block/Subsequence names cannot be duplicated.")

        # label & repetition check
        labels = [s["label"] for s in sequences if "label" in s]
        if len(labels) > len(set(labels)):
            return fail("Sequence labels cannot be duplicated.")
        for i, seq in enumerate(sequences):
            if "goto" in seq:
                if seq["goto"] not in labels:
                    return fail(
                        "Seq ({}): cannot go to an undefined seq ({})., sequence: {!r}".format(
                            i, seq["goto"], seq
                        )
                    )
                if seq["Nrep"] == 0:
                    return fail(
                        f"Seq ({i}): Goto ({seq['goto']}) cannot be used with infinite loop."
                        + f" sequence: {seq!r}"
                    )
            if "jumpto" in seq:
                if seq["jumpto"] not in labels:
                    return fail(
                        "Seq ({}): cannot jump to an undefined seq ({})., sequence: {!r}".format(
                            i, seq["jumpto"], seq
                        )
                    )

        return self._adjust_blocks(blocks, freq)

    # NOTE: this is not tested.
    def write_blocks_advanced(
        self,
        blocks: Blocks[Block],
        subsequences,
        sequences,
        freq: float,
        trigger_positive: bool = True,
        scaffold_name=None,
    ) -> bool:
        """Generate tree using explicit sequence and write it."""

        self.offsets = self.validate_blocks_advanced(blocks, subsequences, sequences, freq)
        if self.offsets is None:
            return False
        self.length = self.get_total_block_length_advanced(blocks, subsequences, sequences)

        if scaffold_name is None:
            scaffold_name = self.SCAFFOLD
        scaffold_path = path.join(self.ROOT_LOCAL, scaffold_name)

        tree = self.generate_tree_advanced(
            blocks,
            subsequences,
            sequences,
            freq,
            trigger_positive=trigger_positive,
            scaffold_path=scaffold_path,
        )

        self.logger.info(f"DTG tree prepared. Total length: {self.length}")
        return self._write_load_setup(tree)

    def query_start(self, delay=0.1, maxloop=20) -> bool:
        """query start until actually run.

        if loop count reaches maxloop, return False.

        """

        start_loop_counter = 0
        while self.inst.query("TBAS:RST?") != "RUN":
            self.logger.debug(f"query start DTG #{start_loop_counter}")
            # TODO: really? call start() iteratively?
            self.start()
            self.query_opc(delay=delay)
            start_loop_counter += 1
            if start_loop_counter > maxloop:
                return False
        return True

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

    # Standard API

    def start(self) -> bool:
        """Set all outputs on and start sequencer.

        Delay can be inserted for waiting output relay to stabilize.

        """

        self.logger.info(
            "Setting output all-on and starting sequencer. Delay {:.1f} sec.".format(
                self._start_delay_sec
            )
        )
        self.inst.write("OUTP:STAT:ALL ON")
        time.sleep(self._start_delay_sec)
        self.inst.write("TBAS:RUN ON")
        return True

    def stop(self) -> bool:
        self.inst.write("TBAS:RUN OFF")
        self.inst.write("OUTP:STAT:ALL OFF")
        self.logger.info("Stopped sequencer and set output all-off.")
        return True

    def configure(self, params: dict) -> bool:
        if "blocks" in params and "freq" in params:
            if params.get("n_runs") is not None:
                return self.fail_with("DTG does not support finite n_runs.")
            return self.configure_blocks(
                params["blocks"],
                params["freq"],
                self._trigger_positive(params.get("trigger_type")),
            )
        else:
            return self.fail_with("These params must be given: 'blocks' and 'freq'")

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
            else:
                return self.validate_blocks(args["blocks"], args["freq"])
        elif key == "opc":
            return self.query_opc(delay=args)
        elif key == "validate":
            return self.validate_blocks(args["blocks"], args["freq"])
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class DTG5078(DTG5000):
    MAX_FREQ = 750e6
    MAX_BLOCK_LENGTH = 8000000
    MIN_BLOCK_LENGTH = (240, 120, 60, 30, 15, 12, 6, 3, 2, 2, 1, 1, 1)

    def block_granularity(self, freq):
        return 1


class DTG5274(DTG5000):
    MAX_FREQ = 2.7e9
    MAX_BLOCK_LENGTH = 32000000
    MIN_BLOCK_LENGTH = (960, 480, 240, 120, 60, 48, 24, 12, 6, 5, 3, 2, 1)

    def block_granularity(self, freq):
        """Default implementation of DTG5000 follows manual,
        but causes block granularity error in some cases.
        Taking the safest way temporarily.
        """

        return 4

    def min_block_len(self, freq):
        """Default implementation of DTG5000 follows manual,
        but causes block length too short error in some cases.
        Taking the safest way temporarily.
        """

        return 960


class DTG5334(DTG5000):
    MAX_FREQ = 3.35e9
    MAX_BLOCK_LENGTH = 64000000
    MIN_BLOCK_LENGTH = (960, 480, 240, 120, 60, 48, 24, 12, 6, 5, 3, 2, 1)


class DTG5274_mock(DTG5274):
    """dummy class for debug.

    This class doesn't open or hold instrument resource.

    """

    class dummyInst(object):
        resource_name = "dummy_visa_resource"

        def close(self):
            pass

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix)

        self.inst = self.dummyInst()
        self.check_required_conf(("root_local",))
        self.ROOT_LOCAL = self.conf["root_local"]
        self.SCAFFOLD = self.conf.get("scaffold_filename", "scaffold.dtg")

        self.CHANNELS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        if "channels" in self.conf:
            self.CHANNELS.update(self.conf["channels"])
        self._start_delay_sec = self.conf.get("start_delay_sec", 0.1)

        # last total block length and offsets.
        self.length = 0
        self.offsets = None

        self.logger.info("opened {} on {}".format(name, self.inst.resource_name))

    def configure_blocks(
        self, blocks, freq, trigger_positive: bool = True, scaffold_name=None, endless=True
    ) -> bool:
        """Generate tree using default sequence and write it."""

        self._configure_tree(
            blocks, freq, trigger_positive, scaffold_name=scaffold_name, endless=endless
        )
        return True

    def query_opc(self, delay=None) -> bool:
        return True

    # Standard API

    def start(self) -> bool:
        self.logger.info("Start dummy DTG.")
        return True

    def stop(self) -> bool:
        self.logger.info("Stop dummy DTG.")
        return True

    def set(self, key: str, value=None) -> bool:
        if key == "trigger":
            return True
        elif key == "clear":
            return True
        else:
            return self.fail_with(f"unknown set() key: {key}")
