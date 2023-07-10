#!/usr/bin/env python3

import copy

import numpy as np

from mahos.meas.podmr import PODMRClient, PODMRIO
from mahos.msgs.podmr_msgs import PODMRData
from mahos.msgs.common_msgs import BinaryState
from mahos.meas.podmr_generator.generator import make_generators
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, podmr, server_conf, podmr_conf
from podmr_patterns import patterns


def expect_podmr(cli: PODMRClient, num: int, poll_timeout_ms):
    def get():
        data: PODMRData = cli.get_data()
        if data is not None and data.data0 is not None:
            return len(data.data0)
        else:
            return None

    return expect_value(get, num, poll_timeout_ms, trials=500)


def pattern_equal(pattern0, pattern1):
    blocks0, freq0, laser_timing0 = pattern0
    blocks1, freq1, laser_timing1 = pattern1
    return blocks0 == blocks1 and abs(freq0 - freq1) < 1e-5 and laser_timing0 == laser_timing1


def pattern_equivalent(pattern0, pattern1):
    blocks0, freq0, laser_timing0 = pattern0
    blocks1, freq1, laser_timing1 = pattern1
    return (
        blocks0.equivalent(blocks1)
        and abs(freq0 - freq1) < 1e-5
        and laser_timing0 == laser_timing1
    )


def test_podmr_pattern_divide():
    params = {
        "base_width": 320e-9,
        "laser_delay": 45e-9,
        "laser_width": 5e-6,
        "mw_delay": 1e-6,
        "trigger_width": 20e-9,
        "init_delay": 0.0,
        "final_delay": 5e-6,
        "90pulse": 10e-9,
        "180pulse": 20e-9,
        "tauconst": 150e-9,
        "tau2const": 160e-9,
        "Nconst": 2,
        "N2const": 2,
        "N3const": 2,
        "ddphase": "Y:X:Y:X,Y:X:Y:iX",
        "supersample": 1,
        "iq_delay": 16e-9,
        "partial": -1,
        "nomw": False,
        "readY": False,
        "invertinit": False,
        "reinitX": False,
        "start": 100e-9,
        "num": 2,
        "step": 100e-9,
        "divide_block": False,
    }
    params_divide = params.copy()
    params_divide["divide_block"] = True

    tau = np.array([100e-6, 200e-6])
    generators = make_generators()
    # for meth in ("rabi", "fid", "spinecho", "trse", "cp", "cpmg", "xy4", "xy8", "xy16", "180train",
    #              "se90sweep", "recovery", "spinlock", "xy8cl", "xy8cl1flip", "ddgate"):
    for meth in ("rabi", "spinecho"):
        ptn = generators[meth].generate(tau, params)
        ptn_divide = generators[meth].generate(tau, params_divide)
        assert pattern_equivalent(ptn, ptn_divide)


def test_podmr_patterns():
    tau = np.array([100e-9, 200e-9])
    Ns = [1, 2]

    params = {
        "base_width": 320e-9,
        "laser_delay": 45e-9,
        "laser_width": 5e-6,
        "mw_delay": 1e-6,
        "trigger_width": 20e-9,
        "init_delay": 0.0,
        "final_delay": 5e-6,
        "90pulse": 10e-9,
        "180pulse": 20e-9,
        "tauconst": 150e-9,
        "tau2const": 160e-9,
        "Nconst": 2,
        "N2const": 2,
        "N3const": 2,
        "ddphase": "Y:X:Y:X,Y:X:Y:iX",
        "supersample": 1,
        "iq_delay": 16e-9,
        "partial": -1,
        "nomw": False,
        "readY": False,
        "invertinit": False,
        "reinitX": False,
        "start": 100e-9,
        "num": 2,
        "step": 100e-9,
        "divide_block": False,
    }

    generators = make_generators()

    # normal (sweep tau)
    for meth in (
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
        "recovery",
        "spinlock",
        "xy8cl",
        "xy8cl1flip",
        "ddgate",
    ):
        ptn = generators[meth].generate(tau, params)
        assert pattern_equal(ptn, patterns[meth])

    for meth in ("xy8", "xy16"):
        ps = copy.copy(params)
        ps["supersample"] = 2
        ps["method"] = meth
        data = PODMRData(ps)
        ptn = generators[meth].generate(data.xdata, ps)
        assert pattern_equal(ptn, patterns[meth + "ss"])

    # sweepN
    for meth in ("cpN", "cpmgN", "xy4N", "xy8N", "xy16N", "xy8clNflip", "ddgateN"):
        ptn = generators[meth].generate(Ns, params)
        assert pattern_equal(ptn, patterns[meth])

    # partial
    for partial in (0, 1):
        ps = copy.copy(params)
        ps["partial"] = partial
        ptn = generators["rabi"].generate(tau, ps)
        assert pattern_equal(ptn, patterns[f"rabi-p{partial:d}"])


def test_podmr(server, podmr, server_conf, podmr_conf):
    poll_timeout_ms = podmr_conf["poll_timeout_ms"]

    podmr.wait()

    assert get_some(podmr.get_status, poll_timeout_ms).state == BinaryState.IDLE
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
        "recovery",
        "spinlock",
        "xy8cl",
        "xy8cl1flip",
        "ddgate",
    ):
        print(m)
        params = podmr.get_param_dict(m)
        params["num"].set(2)  # small num for quick test
        if "90pulse" in params:
            params["90pulse"].set(10e-9)
        if "180pulse" in params:
            params["180pulse"].set(20e-9)  # default negative value causes error in se90sweep
        assert podmr.validate(params)
        assert podmr.start(params)
        assert expect_podmr(podmr, params["num"].value(), poll_timeout_ms)
        data = get_some(podmr.get_data, poll_timeout_ms)
        assert podmr.stop()
        if m == "rabi":
            save_load_test(PODMRIO(), data)

    for m in ("cpN", "cpmgN", "xy4N", "xy8N", "xy16N", "xy8clNflip", "ddgateN"):
        params = podmr.get_param_dict(m)
        params["Nnum"].set(3)  # small num for quick test
        assert podmr.validate(params)
        assert podmr.start(params)
        assert expect_podmr(podmr, params["Nnum"].value(), poll_timeout_ms)
        assert podmr.stop()
