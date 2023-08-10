#!/usr/bin/env python3


from mahos.meas.qdyne import QdyneIO
from mahos.msgs.common_msgs import BinaryState
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, qdyne, server_conf, qdyne_conf


def test_qdyne(server, qdyne, server_conf, qdyne_conf):
    poll_timeout_ms = qdyne_conf["poll_timeout_ms"]

    qdyne.wait()

    assert get_some(qdyne.get_status, poll_timeout_ms).state == BinaryState.IDLE
    params = qdyne.get_param_dict("xy8")

    assert qdyne.validate(params)
    assert qdyne.start(params)
    assert expect_value(qdyne.get_state, BinaryState.ACTIVE, poll_timeout_ms)

    # On Qdyne, data is fetched and analyzed on stop().
    assert qdyne.stop()
    assert expect_value(qdyne.get_state, BinaryState.IDLE, poll_timeout_ms)
    data = get_some(qdyne.get_data, poll_timeout_ms)
    save_load_test(QdyneIO(), data)
