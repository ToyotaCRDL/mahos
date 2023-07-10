#!/usr/bin/env python3

from mahos.meas.hbt import HBTClient, HBTIO
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, hbt, server_conf, hbt_conf


def expect_hbt(cli: HBTClient, params: dict, poll_timeout_ms):
    def get():
        data = cli.get_data()
        if data is not None and data.data is not None:
            return len(data.data)
        else:
            return None

    _range = int(round(params["range"] / params["bin"]))
    return expect_value(get, _range, poll_timeout_ms, trials=500)


def test_hbt(server, hbt, server_conf, hbt_conf):
    poll_timeout_ms = hbt_conf["poll_timeout_ms"]

    hbt.wait()

    assert get_some(hbt.get_status, poll_timeout_ms).state == BinaryState.IDLE

    params = hbt.get_param_dict("hbt")
    params["range"].set(50e-9)
    params["bin"].set(0.2e-9)
    assert hbt.change_state(BinaryState.ACTIVE, params)
    assert expect_hbt(hbt, params.unwrap(), poll_timeout_ms)
    data = get_some(hbt.get_data, poll_timeout_ms)
    assert hbt.change_state(BinaryState.IDLE)

    save_load_test(HBTIO(), data)
