#!/usr/bin/env python3

from mahos.meas.spectroscopy import SpectroscopyClient, SpectroscopyIO
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, spectroscopy, server_conf, spectroscopy_conf


def expect_spectroscopy(cli: SpectroscopyClient, params: dict, poll_timeout_ms):
    def get():
        data = cli.get_data()
        if data is not None and data.data is not None and data.xdata is not None:
            return True
        else:
            return None

    return expect_value(get, True, poll_timeout_ms, trials=500)


def test_spectroscopy(server, spectroscopy, server_conf, spectroscopy_conf):
    poll_timeout_ms = spectroscopy_conf["poll_timeout_ms"]

    spectroscopy.wait()

    assert get_some(spectroscopy.get_status, poll_timeout_ms).state == BinaryState.IDLE

    params = spectroscopy.get_param_dict("spectroscopy")
    params["exposure_time"].set(1.0)
    assert spectroscopy.change_state(BinaryState.ACTIVE, params)
    assert expect_spectroscopy(spectroscopy, params.unwrap(), poll_timeout_ms)
    data = get_some(spectroscopy.get_data, poll_timeout_ms)

    assert spectroscopy.change_state(BinaryState.IDLE)

    save_load_test(SpectroscopyIO(), data)
