#!/usr/bin/env python3

"""
Tests for Sweeper.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.meas.sweeper import SweeperClient, SweeperIO
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, sweeper, server_conf, sweeper_conf


def expect_sweeper(cli: SweeperClient, poll_timeout_ms, sweeps: int):
    def get():
        data = cli.get_data()
        if data is not None and data.has_data():
            return data.sweeps() >= sweeps
        else:
            return None

    return expect_value(get, True, poll_timeout_ms, trials=500)


def test_sweeper(server, sweeper, server_conf, sweeper_conf):
    poll_timeout_ms = sweeper_conf["poll_timeout_ms"]

    sweeper.wait()

    assert get_some(sweeper.get_status, poll_timeout_ms).state == BinaryState.IDLE

    # need to configure the mock dmm (sweep instrument)
    label = "ch1_dcv"
    dmm_params = server.get_param_dict("dmm0", label)
    assert server.configure("dmm0", dmm_params, label)

    params = sweeper.get_param_dict()
    assert params["start"].step() == sweeper_conf["x"]["step"]
    assert params["stop"].step() == sweeper_conf["x"]["step"]
    assert params["start"].adaptive_step() == sweeper_conf["x"]["adaptive_step"]
    assert params["stop"].adaptive_step() == sweeper_conf["x"]["adaptive_step"]
    invalid_log_params = {
        "start": 0.0,
        "stop": 1.0,
        "num": 11,
        "delay": 0.0,
        "sweeps": 1,
        "log": True,
    }
    assert not sweeper.change_state(BinaryState.ACTIVE, invalid_log_params)
    assert get_some(sweeper.get_status, poll_timeout_ms).state == BinaryState.IDLE

    params["sweeps"].set(3)
    params["delay"].set(1e-5)
    assert sweeper.change_state(BinaryState.ACTIVE, params)
    assert expect_sweeper(sweeper, poll_timeout_ms, 2)
    data = get_some(sweeper.get_data, poll_timeout_ms)
    assert sweeper.change_state(BinaryState.IDLE)

    save_load_test(SweeperIO(), data)
