#!/usr/bin/env python3

from mahos.meas.odmr import ODMRClient, ODMRIO
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, odmr, server_conf, odmr_conf


def expect_odmr(cli: ODMRClient, params: dict, poll_timeout_ms):
    def get():
        data = cli.get_data()
        if data is not None and data.data is not None:
            return data.data.shape[0]
        else:
            return None

    v = params["num"]
    return expect_value(get, v, poll_timeout_ms, trials=500)


def test_odmr(server, odmr, server_conf, odmr_conf):
    poll_timeout_ms = odmr_conf["poll_timeout_ms"]
    power_bounds = tuple(server_conf["instrument"]["sg"]["conf"]["power_bounds"])

    odmr.wait()

    params = odmr.get_param_dict("cw")
    params["num"].set(10)
    power = params["power"]
    assert (power.minimum(), power.maximum()) == power_bounds
    assert odmr.get_param_dict("non-existent-name") == None
    assert get_some(odmr.get_status, poll_timeout_ms).state == BinaryState.IDLE

    assert odmr.change_state(BinaryState.ACTIVE, params)
    assert expect_odmr(odmr, params.unwrap(), poll_timeout_ms)
    data = get_some(odmr.get_data, poll_timeout_ms)

    assert odmr.change_state(BinaryState.IDLE)

    save_load_test(ODMRIO(), data)
