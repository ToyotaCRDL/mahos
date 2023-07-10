#!/usr/bin/env python3

from mahos.meas.iodmr import IODMRClient, IODMRIO
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, iodmr, server_conf, iodmr_conf


def expect_iodmr(cli: IODMRClient, params: dict, poll_timeout_ms):
    def get():
        data = cli.get_data()
        if data is not None and data.data_sum is not None:
            return data.data_sum.shape[0]
        else:
            return None

    v = params["num"]
    return expect_value(get, v, poll_timeout_ms, trials=500)


def test_iodmr(server, iodmr, server_conf, iodmr_conf):
    poll_timeout_ms = iodmr_conf["poll_timeout_ms"]
    power_bounds = tuple(server_conf["instrument"]["sg"]["conf"]["power_bounds"])

    iodmr.wait()

    params = iodmr.get_param_dict()
    power = params["power"]
    assert (power.minimum(), power.maximum()) == power_bounds
    assert get_some(iodmr.get_status, poll_timeout_ms).state == BinaryState.IDLE

    # params = iodmr_params()
    params["num"].set(10)
    # should be zero or tiny value because mock actually make delay.
    params["exposure_delay"].set(0.0)
    params["sweeps"].set(3)
    assert iodmr.change_state(BinaryState.ACTIVE, params)
    assert expect_iodmr(iodmr, params.unwrap(), poll_timeout_ms)
    data = get_some(iodmr.get_data, poll_timeout_ms)

    assert iodmr.change_state(BinaryState.IDLE)

    save_load_test(IODMRIO(), data)
