#!/usr/bin/env python3

"""
Tests for segment utility functions.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos_dq.util.segments import (
    round_duration_for_segment_samples,
    round_segment_samples_down,
    round_segment_samples_up,
    valid_segment_samples,
)


def test_round_segment_samples_up():
    assert round_segment_samples_up(1) == 16
    assert round_segment_samples_up(16) == 16
    assert round_segment_samples_up(17) == 32
    assert round_segment_samples_up(32) == 32
    assert round_segment_samples_up(33) == 48
    assert round_segment_samples_up(33, granularity=2048) == 2048


def test_round_segment_samples_down():
    assert round_segment_samples_down(16) == 16
    assert round_segment_samples_down(4097, granularity=2048) == 4096
    assert round_segment_samples_down(2048, granularity=2048) == 2048

    try:
        round_segment_samples_down(1024, granularity=2048)
    except ValueError:
        pass
    else:
        raise AssertionError("rounding down below minimum should fail")


def test_valid_segment_samples():
    assert not valid_segment_samples(15)
    assert valid_segment_samples(16)
    assert valid_segment_samples(32)
    assert not valid_segment_samples(40)
    assert valid_segment_samples(48)
    assert not valid_segment_samples(48, granularity=2048)
    assert valid_segment_samples(2048, granularity=2048)


def test_round_duration_for_segment_samples():
    duration, samples = round_duration_for_segment_samples(
        duration=10, duration_step=4, sample_factor=3, sample_divisor=2, granularity=32
    )

    assert duration >= 10
    assert duration % 4 == 0
    assert samples == duration * 3 // 2
    assert valid_segment_samples(samples, granularity=32)
