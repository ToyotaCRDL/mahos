#!/usr/bin/env python3

"""
Tests for meas.qdyne.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import numpy as np

from mahos.meas.qdyne import QdyneIO
from mahos.msgs.qdyne_msgs import QdyneData
from mahos.meas.qdyne_worker import QdyneAnalyzer
from mahos.msgs.common_msgs import BinaryState
import mahos.ext.cqdyne_analyzer as C
from util import get_some, expect_value, save_load_test
from fixtures import ctx, gconf, server, qdyne, server_conf, qdyne_conf


def test_cqdyne_analyzer():
    N = 3
    T = 5
    # xdata: [0, 5, 10]
    xdata = np.arange(0, N * T, T, dtype=np.uint64)
    head = 1
    tail = 3
    # should count events (raw_data) in following intervals
    # [1, 3], [6, 8], [11, 13] (including both edges)
    # Note the following assumptions for raw_data
    # - raw_data must be sorted (ascending).
    # - we must have at least one event exceeding the last edge (13).

    def do_test(raw, expect):
        raw_data = np.array(raw, dtype=np.uint64)
        data = np.zeros(N, dtype=np.uint64)
        C.analyze(raw_data, xdata, data, head, tail)
        assert np.all(data == np.array(expect, dtype=np.uint64))

    do_test([1, 6, 11, 14], [1, 1, 1])
    do_test([3, 3, 8, 8, 13, 13, 14], [2, 2, 2])
    do_test([0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 9, 11, 12, 13, 13, 14], [6, 0, 4])
    do_test([5, 6, 7, 8, 9, 14], [0, 3, 0])
    do_test([14], [0, 0, 0])


def test_qdyne_analyzer():
    analyzer = QdyneAnalyzer()

    def do_test(raw, expect):
        # mock data to repro test_cqdyne_analyzer
        data = QdyneData({"instrument": {"tbin": 0.2e-9, "pg_length": 1, "pg_freq": 1e9}})
        # head = 1, tail = 3
        data.marker_indices = np.array([[1], [3], [0], [0]], dtype=np.int64)
        # T = 5
        assert data.get_period_bins() == 5
        data.raw_data = np.array(raw, dtype=np.uint64)
        analyzer.analyze(data)
        assert np.all(data.data == np.array(expect, dtype=np.uint64))

        analyzer._analyze_py(data)
        assert np.all(data.data == np.array(expect, dtype=np.uint64))

    # same tests as test_cqdyne_analyzer (N = 3).
    do_test([1, 6, 11, 14], [1, 1, 1])
    do_test([3, 3, 8, 8, 13, 13, 14], [2, 2, 2])
    do_test([0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 9, 11, 12, 13, 13, 14], [6, 0, 4])
    do_test([5, 6, 7, 8, 9, 14], [0, 3, 0])
    do_test([14], [0, 0, 0])

    # if the latest event doesn't exceed the last edge (13), the length becomes shorter (N = 2).
    do_test([1, 6, 11], [1, 1])
    do_test([3, 3, 8, 8, 13, 13], [2, 2])
    do_test([0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 9, 11, 12, 13, 13], [6, 0])
    do_test([5, 6, 7, 8, 9], [0, 3])
    do_test([13], [0, 0])
    # shorter case (N = 1).
    do_test([1, 1, 1, 5], [3])
    do_test([5], [0])


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
