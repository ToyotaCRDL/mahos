#!/usr/bin/env python3

"""
Tests for Chrono.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.meas.chrono import ChronoClient, ChronoIO
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, chrono, server_conf, chrono_conf


def expect_chrono(cli: ChronoClient, poll_timeout_ms, inst: str, length: int = 10):
    def get():
        data = cli.get_data()
        if data is not None and data.has_data():
            return len(data.get_ydata(inst)) >= length
        else:
            return None

    return expect_value(get, True, poll_timeout_ms, trials=500)


def test_chrono(server, chrono, server_conf, chrono_conf):
    poll_timeout_ms = chrono_conf["poll_timeout_ms"]

    chrono.wait()

    assert get_some(chrono.get_status, poll_timeout_ms).state == BinaryState.IDLE

    params = chrono.get_param_dict("all")
    assert chrono.change_state(BinaryState.ACTIVE, params)
    assert expect_chrono(chrono, poll_timeout_ms, "dmm0")
    data = get_some(chrono.get_data, poll_timeout_ms)
    assert chrono.change_state(BinaryState.IDLE)

    save_load_test(ChronoIO(), data)
