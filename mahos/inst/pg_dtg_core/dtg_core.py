#!/usr/bin/env python3

from __future__ import annotations
import typing as T
from dataclasses import dataclass
from os import path

import numpy as np

from . import dtg_io
from ...msgs.inst_pg_msgs import TriggerType, Block, Blocks, BlockSeq


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
    """generate a tree for a pattern command.

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
    """generate a tree for a subsequence entry.

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
    """generate a tree for a sequence entry.

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


@dataclass
class Sequence:
    name: str
    Nrep: int = 1
    label: str = ""
    goto: str = ""
    jumpto: str = ""
    trigger: bool = False


class Step(T.NamedTuple):
    name: str
    Nrep: int


@dataclass
class SubSequence:
    name: str
    steps: list[Step]


class DTGCoreMixin(object):
    MAX_BLOCK_NUM = 8000
    MAX_SEQ_NUM = 8000
    MAX_SUBSEQ_NUM = 50

    def block_granularity(self) -> int:
        raise NotImplementedError("block_granularity() not implemented")

    def max_block_len(self) -> int:
        raise NotImplementedError("max_block_len() not implemented")

    def min_block_len(self, freq: float) -> int:
        raise NotImplementedError("min_block_len() not implemented")

    def max_freq(self) -> float:
        raise NotImplementedError("max_freq() not implemented")

    def _trigger_positive(self, trigger_type: TriggerType | None) -> bool:
        """Translate trigger_type to trigger_positive (bool) for API compatibility."""

        return trigger_type != TriggerType.HARDWARE_FALLING

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

    def _find_name(self, li: list[Block | SubSequence], name: str) -> Block | SubSequence | None:
        for elem in li:
            if elem.name == name:
                return elem
        return None

    def _get_total_block_length_seq(
        self, blocks: Blocks[Block], subsequences: list[SubSequence], sequences: list[Sequence]
    ) -> int | None:
        t = 0
        for i, seq in enumerate(sequences):
            blk = self._find_name(blocks, seq.name)
            if blk is not None:
                t += blk.raw_length() * seq.Nrep
            else:
                subt = 0
                subseq = self._find_name(subsequences, seq.name)
                if subseq is None:
                    self.logger.error(
                        f"Sequence ({i:d}, {seq.label}) refers to undef block/subseq: {seq.name}"
                    )
                    return None
                for j, step in enumerate(subseq.steps):
                    blk = self._find_name(blocks, step.name)
                    if blk is None:
                        self.logger.error(
                            "SubSequence ({:d}, {:s}) refers to undef block ({:s}).".format(
                                j, subseq.name, step.name
                            )
                        )
                        return None
                    subt += blk.raw_length() * step.Nrep
                t += subt * seq.Nrep
        return t

    def _block_to_str(self, block: Block, max_len=10):
        s = f"{block.name}(x{block.Nrep})|"
        patterns = block.pattern_to_strs()
        if len(patterns) <= max_len:
            return s + "|".join(patterns)
        return s + "|".join(patterns[: max_len - 1]) + "|...|" + patterns[-1]

    def _adjust_blocks(self, blocks: Blocks[Block], freq: float) -> list[int] | None:
        gran = self.block_granularity(freq)
        max_len = self.max_block_len()
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

            if length > max_len or length < min_len:
                self.logger.error(
                    "Block length ({}) out of bounds ({}, {}).".format(length, min_len, max_len)
                )
                return None

        return offsets

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

            self.logger.debug(self._block_to_str(block))
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

    def validate_blocks(self, blocks: Blocks[Block], freq: float) -> list[int] | None:
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

    def validate_blockseq(self, blockseq: BlockSeq[Block], freq: float) -> list[int] | None:
        blocks, subsequences, sequences = self._build_seq(blockseq, True)
        return self.validate_blocks_seq(blocks, subsequences, sequences, freq)

    def generate_tree_seq(
        self,
        blocks: Blocks[Block],
        subsequences: list[SubSequence],
        sequences: list[Sequence],
        freq: float,
        trigger_positive: bool = True,
        scaffold_path=None,
    ):
        """Generate tree using explicit sequence."""

        blk_trees = []
        seq_trees = []
        subseq_trees = []
        ptn_trees = []

        for i, block in enumerate(blocks):
            blk_trees.append(gen_block(block.raw_length(), i + 1, block.name))

            self.logger.debug(self._block_to_str(block))
            pulses = np.concatenate(
                [
                    self.channels_to_int(channels) * np.ones(length, dtype=np.uint8)
                    for channels, length in block.pattern
                ]
            )
            ptn_trees.append(gen_pattern(i + 1, pulses))

        for i, subseq in enumerate(subsequences):
            subseq_trees.append(gen_subsequence(len(blocks) + i + 1, subseq.name, subseq.steps))

        for i, seq in enumerate(sequences):
            seq_trees.append(
                gen_sequence(seq.label, seq.name, seq.Nrep, seq.goto, seq.jumpto, int(seq.trigger))
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

    def validate_blocks_seq(
        self,
        blocks: Blocks[Block],
        subsequences: list[SubSequence],
        sequences: list[Sequence],
        freq: float,
    ) -> list[int] | None:
        def fail(msg):
            self.logger.error(msg)
            return None

        if len(blocks) > self.MAX_BLOCK_NUM:
            return fail(
                "Number of blocks ({}) exceeds max value ({}).".format(
                    len(blocks), self.MAX_BLOCK_NUM
                )
            )
        if len(sequences) > self.MAX_SEQ_NUM:
            return fail(
                "Number of sequences ({}) exceeds max value ({}).".format(
                    len(sequences), self.MAX_SEQ_NUM
                )
            )
        if len(subsequences) > self.MAX_SUBSEQ_NUM:
            return fail(
                "Number of subsequences ({}) exceeds max value ({}).".format(
                    len(subsequences), self.MAX_SUBSEQ_NUM
                )
            )
        self.logger.info(
            "blocks: {}/{} seqs: {}/{} subseqs: {}/{}".format(
                len(blocks),
                self.MAX_BLOCK_NUM,
                len(sequences),
                self.MAX_SEQ_NUM,
                len(subsequences),
                self.MAX_SUBSEQ_NUM,
            )
        )

        # name check
        names = [b.name for b in blocks] + [sub.name for sub in subsequences]
        if len(blocks) + len(subsequences) > len(set(names)):
            return fail("Block/Subsequence names cannot be duplicated.")

        # label & repetition check
        labels = [s.label for s in sequences if s.label]
        if len(labels) > len(set(labels)):
            return fail("Sequence labels cannot be duplicated.")
        for i, seq in enumerate(sequences):
            if seq.goto:
                if seq.goto not in labels:
                    return fail(
                        "Seq ({}): cannot go to an undefined seq ({})., sequence: {!r}".format(
                            i, seq.goto, seq
                        )
                    )
                if seq.Nrep == 0:
                    return fail(
                        f"Seq ({i}): Goto ({seq.goto}) cannot be used with infinite loop."
                        + f" sequence: {seq!r}"
                    )
            if seq.jumpto:
                if seq.jumpto not in labels:
                    return fail(
                        "Seq ({}): cannot jump to an undefined seq ({})., sequence: {!r}".format(
                            i, seq.jumpto, seq
                        )
                    )

        return self._adjust_blocks(blocks, freq)

    def _configure_tree_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_positive: bool = True,
        scaffold_name: str | None = None,
        endless: bool = True,
    ) -> list | None:
        self.offsets = self.validate_blocks(blocks, freq)
        if self.offsets is None:
            return None
        self.length = blocks.total_length()

        if scaffold_name is None:
            scaffold_name = self.SCAFFOLD
        scaffold_path = path.join(self.LOCAL_DIR, scaffold_name)

        tree = self.generate_tree(
            blocks,
            freq,
            trigger_positive=trigger_positive,
            scaffold_path=scaffold_path,
            endless=endless,
        )

        self.logger.info(f"DTG tree prepared. Total length: {self.length}")

        return tree

    def _build_seq(
        self, blockseq: BlockSeq, endless: bool
    ) -> tuple[Blocks[Block], list[SubSequence], list[Sequence]]:
        """Build DTG subsequences and sequences (intermediate repr) from BlockSeq."""

        blocks = blockseq.unique_blocks()
        sequences = []
        subsequences = []
        subseq_names = []
        # expand top-level Nrep here with total_sequence()
        for bs in blockseq.total_sequence():
            if isinstance(bs, Block):
                sequences.append(Sequence(bs.name, bs.Nrep, trigger=bs.trigger))
            elif isinstance(bs, BlockSeq):
                if bs.name not in subseq_names:
                    steps = [Step(blk.name, blk.Nrep) for blk in bs.data]
                    subsequences.append(SubSequence(bs.name, steps))
                    subseq_names.append(bs.name)
                sequences.append(Sequence(bs.name, bs.Nrep, trigger=bs.trigger))
            else:
                raise TypeError(f"Unknown type {type(bs)} of BlockSeq element: {bs}")
        if endless:
            sequences[0].label = "START"
            sequences[-1].goto = "START"

        return blocks, subsequences, sequences

    def _configure_tree_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_positive: bool = True,
        scaffold_name: str | None = None,
        endless: bool = True,
    ) -> list | None:
        blocks, subsequences, sequences = self._build_seq(blockseq, endless)
        self.offsets = self.validate_blocks_seq(blocks, subsequences, sequences, freq)
        if self.offsets is None:
            return None
        self.length = blockseq.total_length()
        l = self._get_total_block_length_seq(blocks, subsequences, sequences)
        if l != self.length:
            msg = f"length mismatch: {l} != {self.length}. Check offsets."
            if self._strict:
                self.logger.error(msg)
                return None
            else:
                self.logger.warn(msg)

        if scaffold_name is None:
            scaffold_name = self.SCAFFOLD
        scaffold_path = path.join(self.LOCAL_DIR, scaffold_name)

        tree = self.generate_tree_seq(
            blocks,
            subsequences,
            sequences,
            freq,
            trigger_positive=trigger_positive,
            scaffold_path=scaffold_path,
        )

        self.logger.info(f"DTG tree prepared. Total length: {self.length}")

        return tree
