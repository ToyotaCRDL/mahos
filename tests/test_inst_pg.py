#!/usr/bin/env python3

from mahos.msgs.inst_pg_msgs import Block


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
