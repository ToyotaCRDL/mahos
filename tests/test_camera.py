#!/usr/bin/env python3

from mahos.meas.camera import CameraClient, CameraIO
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, camera, server_conf, camera_conf


def expect_camera(cli: CameraClient, params: dict, poll_timeout_ms):
    def get():
        image = cli.get_data()
        if image is not None and image.count > 1:
            return True
        else:
            return None

    return expect_value(get, True, poll_timeout_ms, trials=500)


def test_camera(server, camera, server_conf, camera_conf):
    poll_timeout_ms = camera_conf["poll_timeout_ms"]

    camera.wait()

    assert get_some(camera.get_status, poll_timeout_ms).state == BinaryState.IDLE

    params = camera.get_param_dict("camera")
    assert camera.change_state(BinaryState.ACTIVE, params)
    assert expect_camera(camera, params, poll_timeout_ms)
    data = get_some(camera.get_data, poll_timeout_ms)

    assert camera.change_state(BinaryState.IDLE)
    save_load_test(CameraIO(), data)
