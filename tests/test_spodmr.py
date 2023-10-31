#!/usr/bin/env python3

"""
Tests for meas.spodmr.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.meas.spodmr import SPODMRClient, SPODMRIO
from mahos.msgs.spodmr_msgs import SPODMRData
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, spodmr, server_conf, spodmr_conf


def expect_spodmr(cli: SPODMRClient, num: int, poll_timeout_ms):
    def get():
        data: SPODMRData = cli.get_data()
        if data is not None and data.data0 is not None:
            return data.data0.shape[0]
        else:
            return None

    return expect_value(get, num, poll_timeout_ms, trials=500)


def test_spodmr(server, spodmr, server_conf, spodmr_conf):
    poll_timeout_ms = spodmr_conf["poll_timeout_ms"]

    spodmr.wait()

    assert get_some(spodmr.get_status, poll_timeout_ms).state == BinaryState.IDLE
    for m in (
        "rabi",
        "fid",
        "spinecho",
        "trse",
        "cp",
        "cpmg",
        "xy4",
        "xy8",
        "xy16",
        "180train",
        "se90sweep",
        # "recovery",
        "spinlock",
        "xy8cl",
        "xy8cl1flip",
        "ddgate",
    ):
        print(m)
        params = spodmr.get_param_dict(m)
        params["num"].set(2)  # small num for quick test
        params["accum_window"].set(1e-4)  # small window for quick test
        if "90pulse" in params:
            params["90pulse"].set(10e-9)
        if "180pulse" in params:
            params["180pulse"].set(20e-9)  # default negative value causes error in se90sweep
        assert spodmr.validate(params)
        assert spodmr.start(params)
        assert expect_spodmr(spodmr, params["num"].value(), poll_timeout_ms)
        data = get_some(spodmr.get_data, poll_timeout_ms)
        assert spodmr.stop()
        if m == "rabi":
            save_load_test(SPODMRIO(), data)

    for m in ("cpN", "cpmgN", "xy4N", "xy8N", "xy16N", "xy8clNflip", "ddgateN"):
        params = spodmr.get_param_dict(m)
        params["Nnum"].set(3)  # small num for quick test
        params["accum_window"].set(1e-4)  # small window for quick test
        assert spodmr.validate(params)
        assert spodmr.start(params)
        assert expect_spodmr(spodmr, params["Nnum"].value(), poll_timeout_ms)
        assert spodmr.stop()
