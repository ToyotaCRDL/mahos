#!/usr/bin/env python3

"""
Tests for PosTweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.meas.pos_tweaker import PosTweakerClient
from util import get_some, expect_value

from fixtures import ctx, gconf, server, pos_tweaker, pos_tweaker_conf


def expect_target(cli: PosTweakerClient, targets: dict, poll_timeout_ms):
    def get():
        status = cli.get_status()
        if status is None:
            return None
        states = status.axis_states
        d = {}
        for key in targets:
            if key in states and states[key] is not None:
                d[key] = states[key]["target"]
        return d

    return expect_value(get, targets, poll_timeout_ms, trials=500)


def test_pos_tweaker(server, pos_tweaker, pos_tweaker_conf):
    poll_timeout_ms = pos_tweaker_conf["poll_timeout_ms"]

    pos_tweaker.wait()

    axis_states = get_some(pos_tweaker.get_status, poll_timeout_ms).axis_states
    assert sorted(list(axis_states.keys())) == ["pos_x", "pos_y"]
    assert pos_tweaker.home("pos_x")
    assert pos_tweaker.home("pos_y")
    assert pos_tweaker.home_all()
    targets = {"pos_x": 1, "pos_y": 2}
    assert pos_tweaker.set_target(targets)
    assert expect_target(pos_tweaker, targets, poll_timeout_ms)
