#!/usr/bin/env python3

import time
import uuid
from io import BytesIO

from mahos.meas.confocal import ConfocalClient, ConfocalIO
from mahos.msgs.common_msgs import BinaryState
from mahos.msgs.confocal_msgs import ConfocalState, Axis, ScanDirection, ScanMode, LineMode
from mahos.msgs.confocal_tracker_msgs import OptMode
from mahos.inst.overlay.confocal_scanner_mock import DUMMY_CAPABILITY
from mahos.util.comp import dict_equal_inspect

from util import get_some, expect_value
from fixtures import ctx, gconf, server, confocal, tracker, manager, confocal_conf


def scan_params():
    p = {
        "xmin": 10.0,
        "xmax": 20.0,
        "ymin": 5.0,
        "ymax": 25.0,
        "xnum": 10,
        "ynum": 20,
        "z": 50.0,
        "direction": ScanDirection.XY,
        "time_window": 2.0,
        "delay": 0.0,
        "ident": uuid.uuid4(),
        "mode": ScanMode.COM_NOWAIT,
        "line_mode": LineMode.ASCEND,
    }
    return p


def expect_scan(cli: ConfocalClient, params: dict, poll_timeout_ms):
    def get():
        im = cli.get_image()
        if im is not None and im.image is not None:
            return im.running, im.ident, im.image.shape
        else:
            return (None, None, None)

    v = False, params["ident"], (params["xnum"], params["ynum"])
    return expect_value(get, v, poll_timeout_ms, trials=300, verbose=True)


def save_load_test(image, trace):
    """Test save data to default (hdf5) format and load. Compare the data before and after."""

    io = ConfocalIO()
    f = BytesIO()
    assert io.save_image(f, image)
    loaded = io.load_image(f)
    assert loaded is not None
    assert dict_equal_inspect(image.__dict__, loaded.__dict__)

    f = BytesIO()
    assert io.save_trace(f, trace)
    loaded = io.load_trace(f)
    assert loaded is not None
    assert dict_equal_inspect(trace.__dict__, loaded.__dict__)


def test_confocal(server, confocal, confocal_conf):
    poll_timeout_ms = confocal_conf["poll_timeout_ms"]
    tracer_size = confocal_conf["tracer"]["size"]

    confocal.wait()

    scan = confocal.get_param_dict("scan")
    assert scan["mode"].options() == DUMMY_CAPABILITY
    assert confocal.get_param_dict("non-existent-param") == None
    assert get_some(confocal.get_status, poll_timeout_ms).state == ConfocalState.IDLE

    v = 12.0
    assert not confocal.move(Axis.X, v)  # fail to move if state is IDLE
    assert confocal.change_state(ConfocalState.INTERACT)
    assert confocal.move(Axis.X, v)
    assert expect_value(lambda: confocal.get_status().pos.x_tgt, v, poll_timeout_ms)
    trace = get_some(confocal.get_trace, poll_timeout_ms)
    assert len(trace.traces[0]) == tracer_size

    sparams = scan_params()
    assert confocal.change_state(ConfocalState.SCAN, sparams)
    # assert expect_value(lambda: not confocal.get_image().running, True, poll_timeout_ms)
    assert expect_scan(confocal, sparams, poll_timeout_ms)
    image = get_some(confocal.get_image, poll_timeout_ms)
    assert confocal.change_state(ConfocalState.IDLE)

    save_load_test(image, trace)


def track_params():
    p = {
        "mode": ScanMode.COM_NOWAIT,
        "line_mode": LineMode.ASCEND,
        "order": (ScanDirection.XY, ScanDirection.XZ),
        "interval_sec": 0.1,
        "time_window": 2.0,
        "delay": 0.0,
    }
    p[ScanDirection.XY] = {
        "xlen": 10.0,
        "ylen": 5.0,
        "xnum": 10,
        "ynum": 20,
        "opt_mode": OptMode.Gauss2D,
    }
    p[ScanDirection.XZ] = {
        "xlen": 8.0,
        "ylen": 20.0,
        "xnum": 15,
        "ynum": 5,
        "opt_mode": OptMode.Gauss2D,
    }
    return p


def expect_track_scan(cli: ConfocalClient, params: dict, poll_timeout_ms):
    def get():
        im = cli.get_image()
        if im is not None and im.image is not None:
            return im.running, im.image.shape
        else:
            return (None, None)

    v = False, (params["xnum"], params["ynum"])
    return expect_value(get, v, poll_timeout_ms, trials=300, verbose=True)


def test_tracker(server, confocal, tracker, manager, confocal_conf):
    poll_timeout_ms = confocal_conf["poll_timeout_ms"]

    confocal.wait()
    tracker.wait()

    assert confocal.change_state(ConfocalState.INTERACT)
    assert confocal.move(Axis.X, 50.0)
    assert confocal.move(Axis.Y, 50.0)
    assert confocal.move(Axis.Z, 50.0)

    tr_params = track_params()
    assert tracker.change_state(BinaryState.ACTIVE, tr_params)

    assert expect_track_scan(confocal, tr_params[ScanDirection.XY], poll_timeout_ms)
    time.sleep(tr_params["interval_sec"])
    pos = confocal.get_status().pos
    print(pos.x_tgt, pos.y_tgt, pos.z_tgt)

    assert expect_track_scan(confocal, tr_params[ScanDirection.XZ], poll_timeout_ms)
    time.sleep(tr_params["interval_sec"])
    pos = confocal.get_status().pos
    print(pos.x_tgt, pos.y_tgt, pos.z_tgt)
