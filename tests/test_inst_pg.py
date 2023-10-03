#!/usr/bin/env python3

from mahos.msgs.inst_pg_msgs import Block, BlockSeq


def ch_upper(channels):
    return [ch.upper() for ch in channels]


def test_block():
    block0 = Block(
        "block0", [(None, 0), (("b",), 5), (("aaaaa",), 0), (("b",), 3), (("ccccc",), 0)]
    )
    assert block0.simplify() == Block("block0", [("b", 8)])

    block1 = Block("block1", [("a", 5), ("b", 3), (("b", "c"), 7), ("d", 5)], Nrep=10)
    assert block1.raw_length() == 20
    assert block1.total_length() == 200
    assert len(block1) == 200
    block1_r = Block(
        "block1", [("a", 5), (("b0", "b1"), 3), (("b0", "b1", "c"), 7), ("d0", 5)], Nrep=10
    )
    assert block1.replace({"b": ("b0", "b1"), "d": ("d0",)}) == block1_r
    assert block1.apply(ch_upper) == Block(
        "block1", [("A", 5), (("B"), 3), (("B", "C"), 7), ("D", 5)], Nrep=10
    )

    assert block1.scale(3) == Block(
        "block1", [("a", 15), ("b", 9), (("b", "c"), 21), ("d", 15)], Nrep=10
    )


def test_seq():
    block0 = Block("block0", [("a", 5), ("b", 3), (("b", "c"), 2)], Nrep=3)  # len = 30, num = 3
    block1 = Block("block1", [("a", 2), ("d", 3)], Nrep=2)  # len = 10, num = 2
    seq0 = BlockSeq("seq0", [block0, block1], Nrep=2)
    seq1 = BlockSeq("seq1", [seq0, block0])
    seq2 = BlockSeq("seq2", [block0, seq0, seq1], Nrep=3)
    seq3 = BlockSeq("seq3", [seq2, seq0, seq1, block0], Nrep=3)

    assert seq0.channels() == ["a", "b", "c", "d"]
    assert seq0.nest_depth() == 1
    assert seq0.total_length() == 80
    assert seq0.total_pattern_num() == 26
    assert seq1.nest_depth() == 2
    assert seq1.total_length() == seq0.total_length() + block0.total_length()
    assert seq1.total_pattern_num() == seq0.total_pattern_num() + block0.total_pattern_num()
    assert seq2.nest_depth() == 3
    assert seq2.total_length() == 3 * (
        block0.total_length() + seq0.total_length() + seq1.total_length()
    )
    assert seq3.nest_depth() == 4

    assert seq0.equivalent(seq0.collapse())
    assert seq1.equivalent(seq1.collapse())
    assert seq2.equivalent(seq2.collapse())
    assert seq3.equivalent(seq3.collapse())


def test_seq_uniq():
    def sorted_names(blocks):
        return sorted([b.name for b in blocks])

    b0 = Block("b0", [])
    b1 = Block("b1", [])
    b2 = Block("b2", [])
    seq0 = BlockSeq("seq0", [b0, b1, b0, b1])
    seq1 = BlockSeq("seq1", [b0, seq0, b2, b0])

    assert sorted_names(seq0.unique_blocks()) == ["b0", "b1"]
    assert sorted_names(seq1.unique_blocks()) == ["b0", "b1", "b2"]
