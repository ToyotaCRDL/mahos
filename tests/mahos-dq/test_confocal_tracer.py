#!/usr/bin/env python3

"""
Tests for confocal tracer worker internals.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import logging
import time

import numpy as np

from mahos_dq.meas.confocal_worker import Tracer


class DummyCLI:
    pass


class DummyPD:
    def __init__(self, data):
        self.data = data

    def pop_all_opt(self):
        data = self.data
        self.data = None
        return data


def make_tracer(pd_channels):
    tracer = Tracer.__new__(Tracer)
    tracer.logger = logging.getLogger(__name__)
    tracer.pd_names = [f"pd{i}" for i in range(len(pd_channels))]
    tracer.pd_channels = pd_channels
    tracer.size = 8
    tracer.cb_samples = 2
    tracer.time_window_sec = 0.1
    tracer.trace = None
    return tracer


def test_tracer_interp_stamps_preserves_nanosecond_precision():
    tracer = make_tracer([1])
    tracer.cb_samples = 5
    base = 1_780_000_000_000_000_000

    # Absolute epoch nanoseconds lose precision with floating-point interpolation.
    stamps = tracer._interp_stamps((base + 1_000, base + 2_000), np.int64(base))

    assert stamps.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(
        stamps.view(np.int64) - base,
        [200, 400, 600, 800, 1_000, 1_200, 1_400, 1_600, 1_800, 2_000],
    )


def test_tracer_get_data_expands_multi_channel_pd():
    stamp0 = time.time_ns()
    tracer = make_tracer([2])
    tracer.trace = tracer_trace(channels=2, size=8)
    tracer.pds = [
        DummyPD(
            [
                (
                    [
                        np.array([1.0, 2.0]),
                        np.array([10.0, 20.0]),
                    ],
                    stamp0,
                ),
                (
                    [
                        np.array([3.0, 4.0]),
                        np.array([30.0, 40.0]),
                    ],
                    stamp0 + 200_000_000,
                ),
            ]
        )
    ]

    tracer.get_data()

    np.testing.assert_allclose(tracer.trace.traces[0][-4:], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(tracer.trace.traces[1][-4:], [10.0, 20.0, 30.0, 40.0])
    assert np.all(tracer.trace.stamps[0][-4:].view(np.int64) > 0)
    assert np.all(tracer.trace.stamps[1][-4:].view(np.int64) > 0)


def test_tracer_get_data_keeps_single_channel_pd_behavior():
    stamp0 = time.time_ns()
    tracer = make_tracer([1, 1])
    tracer.trace = tracer_trace(channels=2, size=8)
    tracer.pds = [
        DummyPD([(np.array([1.0, 2.0]), stamp0)]),
        DummyPD([(np.array([3.0, 4.0]), stamp0)]),
    ]

    tracer.get_data()

    np.testing.assert_allclose(tracer.trace.traces[0][-2:], [1.0, 2.0])
    np.testing.assert_allclose(tracer.trace.traces[1][-2:], [3.0, 4.0])


def tracer_trace(channels: int, size: int):
    from mahos_dq.msgs.confocal_msgs import Trace

    return Trace(size=size, channels=channels)
