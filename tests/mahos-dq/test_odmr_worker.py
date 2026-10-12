#!/usr/bin/env python3

"""
Tests for ODMR worker and helper logic.

New optional features have separate test files:

- pd_trace: test_odmr_trace.py
- pd_chop: test_odmr_chop.py

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import logging

import numpy as np
import pytest

from mahos_dq.meas.odmr_pd import configure_analog_pds, configure_apds, sum_pd_channels
from mahos_dq.meas.odmr_worker import Sweeper, SweeperOverlay
from mahos_dq.inst.overlay.odmr_sweeper import ODMRSweeperPG
from mahos_dq.msgs.odmr_msgs import ODMRData


def _overlay_worker(num: int, background: bool = False) -> SweeperOverlay:
    worker = SweeperOverlay.__new__(SweeperOverlay)
    worker.data = ODMRData({"num": num, "background": background}, "cw")
    return worker


def _direct_worker(num: int, background: bool = False) -> Sweeper:
    worker = Sweeper.__new__(Sweeper)
    worker._sg_first = False
    worker._pg_immediate = False
    worker.data = ODMRData({"num": num, "background": background}, "cw")
    return worker


class _Clock:
    def __init__(self, success=True):
        self.success = success
        self.params = None
        self.started = False

    def configure(self, params):
        self.params = params
        return self.success

    def get_internal_output(self):
        return "clock-output"

    def start(self):
        self.started = True
        return self.success


class _PD:
    def __init__(self, success=True):
        self.success = success
        self.params = None
        self.label = None
        self.started = False

    def configure(self, params, label=""):
        self.params = params
        self.label = label
        return self.success

    def start(self):
        self.started = True
        return self.success


def _analog_params():
    return {
        "timing": {"time_window": 10e-6},
        "pd": {
            "rate": 1e6,
            "buffer_size_coeff": 3,
            "bounds": (-5.0, 5.0),
        },
    }


@pytest.mark.parametrize("worker_factory", (_overlay_worker, _direct_worker))
def test_append_line_without_background(worker_factory):
    worker = worker_factory(3)

    worker.append_line(np.array([1.0, 2.0, 3.0]))
    worker.append_line(np.array([4.0, 5.0, 6.0]))

    np.testing.assert_array_equal(
        worker.data.data,
        np.array(
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ]
        ),
    )
    assert worker.data.bg_data is None


@pytest.mark.parametrize("worker_factory", (_overlay_worker, _direct_worker))
def test_append_line_splits_interleaved_background(worker_factory):
    worker = worker_factory(3, background=True)

    worker.append_line(np.array([1.0, 10.0, 2.0, 20.0, 3.0, 30.0]))
    worker.append_line(np.array([4.0, 40.0, 5.0, 50.0, 6.0, 60.0]))

    np.testing.assert_array_equal(
        worker.data.data,
        np.array(
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        worker.data.bg_data,
        np.array(
            [
                [10.0, 40.0],
                [20.0, 50.0],
                [30.0, 60.0],
            ]
        ),
    )


@pytest.mark.parametrize(("sg_first", "pg_immediate"), ((True, False), (False, True)))
def test_direct_append_line_restores_frequency_order(sg_first, pg_immediate):
    worker = _direct_worker(3, background=True)
    worker._sg_first = sg_first
    worker._pg_immediate = pg_immediate

    worker.append_line(np.array([3.0, 30.0, 1.0, 10.0, 2.0, 20.0]))

    np.testing.assert_array_equal(worker.data.data[:, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(worker.data.bg_data[:, 0], np.array([10.0, 20.0, 30.0]))


def test_append_point_starts_new_partial_line_without_background():
    worker = _overlay_worker(3)

    for value in (1.0, 2.0, 3.0, 4.0):
        worker.append_point(np.array([value]))

    np.testing.assert_array_equal(worker.data.data[:, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(worker.data.data[0, :], np.array([1.0, 4.0]))
    assert np.isnan(worker.data.data[1:, 1]).all()


def test_append_point_starts_new_partial_line_with_background():
    worker = _overlay_worker(2, background=True)

    for point in ((1.0, 10.0), (2.0, 20.0), (3.0, 30.0)):
        worker.append_point(np.array(point))

    np.testing.assert_array_equal(worker.data.data[:, 0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(worker.data.bg_data[:, 0], np.array([10.0, 20.0]))
    assert worker.data.data[0, 1] == 3.0
    assert worker.data.bg_data[0, 1] == 30.0
    assert np.isnan(worker.data.data[1, 1])
    assert np.isnan(worker.data.bg_data[1, 1])


def test_sum_pd_channels_flattens_multi_channel_blocks():
    channel = np.array([1.0, 2.0, 3.0])

    result = sum_pd_channels([channel, [2.0 * channel, 3.0 * channel]])

    np.testing.assert_array_equal(result, 6.0 * channel)


def test_configure_apds():
    pd = _PD()

    assert configure_apds(
        [pd],
        pd_clock="gate-in",
        time_window=2e-6,
        point_count=6,
        buffer_size_coeff=4,
        drop_first=1,
    )

    assert pd.params == {
        "clock": "gate-in",
        "cb_samples": 6,
        "samples": 24,
        "buffer_size": 24,
        "rate": 1e6,
        "finite": False,
        "every": False,
        "drop_first": 1,
        "gate": True,
        "time_window": 2e-6,
    }
    assert pd.label == ""


def test_configure_clocked_analog_pds():
    clock = _Clock()
    pd = _PD()

    assert configure_analog_pds(
        clock,
        [pd],
        "gate-in",
        _analog_params(),
        {},
        False,
        point_count=4,
        drop_first=1,
        data_transfer="dma",
        logger=logging.getLogger("test_configure_clocked_analog_pds"),
    )

    assert clock.params == {
        "freq": 1e6,
        "samples": 10,
        "finite": True,
        "trigger_source": "gate-in",
        "trigger_dir": True,
        "retriggerable": True,
    }
    assert pd.params == {
        "trigger_source": "clock-output",
        "clock": "clock-output",
        "cb_samples": 4,
        "samples": 12,
        "buffer_size": 12,
        "rate": 1e6,
        "bounds": (-5.0, 5.0),
        "finite": False,
        "every": False,
        "drop_first": 1,
        "oversample": 10,
        "clock_mode": True,
        "clock_dir": True,
        "trigger_dir": True,
        "data_transfer": "dma",
    }
    assert pd.label == "triggered"


def test_configure_spectrum_analog_pds_adjusts_oversample():
    params = _analog_params()
    params["timing"]["time_window"] = 21e-6
    pd = _PD()

    assert configure_analog_pds(
        None,
        [pd],
        "external-trigger",
        params,
        {"pd_segment_granularity": 16, "pd_segment_offset": 4},
        True,
        point_count=2,
        drop_first=0,
        data_transfer=None,
        logger=logging.getLogger("test_configure_spectrum_analog_pds"),
    )

    assert pd.params["oversample"] == 12
    assert pd.params["trigger_source"] == "external-trigger"
    assert pd.params["clock"] == "external-trigger"


def test_configure_analog_pds_stops_after_clock_failure():
    clock = _Clock(success=False)
    pd = _PD()

    assert not configure_analog_pds(
        clock,
        [pd],
        "gate-in",
        _analog_params(),
        {},
        False,
        point_count=1,
        drop_first=0,
        data_transfer=None,
        logger=logging.getLogger("test_configure_analog_pds_clock_failure"),
    )
    assert pd.params is None


def test_direct_pd_wrappers_apply_sweep_point_and_drop_policy():
    params_apd = {
        "num": 3,
        "background": True,
        "timing": {"time_window": 2e-6},
    }
    pd_apd = _PD()
    sweeper_apd = Sweeper.__new__(Sweeper)
    sweeper_apd.conf = {"buffer_size_coeff": 5}
    sweeper_apd._pd_clock = "gate-in"
    sweeper_apd._sg_first = True
    sweeper_apd._pg_immediate = False
    sweeper_apd.pds = [pd_apd]

    assert sweeper_apd.start_apd(params_apd, "cw")
    assert pd_apd.params["cb_samples"] == 6
    assert pd_apd.params["drop_first"] == 1
    assert pd_apd.started

    params_analog = _analog_params()
    params_analog.update(num=3, background=True)
    clock = _Clock()
    pd_analog = _PD()
    sweeper_analog = Sweeper.__new__(Sweeper)
    sweeper_analog.conf = {}
    sweeper_analog.logger = logging.getLogger("test_direct_analog_pd_policy")
    sweeper_analog._pd_clock = "gate-in"
    sweeper_analog._pd_data_transfer = None
    sweeper_analog._pd_trace = sweeper_analog._pd_spectrum = False
    sweeper_analog._sg_first = True
    sweeper_analog._pg_immediate = False
    sweeper_analog.clock = clock
    sweeper_analog.pds = [pd_analog]

    assert sweeper_analog.start_analog_pd(params_analog, "cw")
    assert pd_analog.params["cb_samples"] == 6
    assert pd_analog.params["drop_first"] == 1
    assert clock.started
    assert pd_analog.started


def test_overlay_pd_wrappers_apply_point_policy_without_starting():
    params_apd = {
        "background": True,
        "timing": {"time_window": 2e-6},
        "pd": {"buffer_size_coeff": 3, "dummy_points": 2},
    }
    pd_apd = _PD()
    sweeper_apd = ODMRSweeperPG.__new__(ODMRSweeperPG)
    sweeper_apd._closed = True
    sweeper_apd.conf = {"buffer_size_coeff": 5}
    sweeper_apd._pd_clock = "gate-in"
    sweeper_apd.pds = [pd_apd]

    assert sweeper_apd.configure_apd(params_apd, "cw")
    assert pd_apd.params["cb_samples"] == 2
    assert pd_apd.params["buffer_size"] == 6
    assert pd_apd.params["drop_first"] == 2
    assert not pd_apd.started

    params_analog = _analog_params()
    params_analog["background"] = True
    params_analog["pd"]["dummy_points"] = 2
    clock = _Clock()
    pd_analog = _PD()
    sweeper_analog = ODMRSweeperPG.__new__(ODMRSweeperPG)
    sweeper_analog._closed = True
    sweeper_analog.conf = {}
    sweeper_analog.logger = logging.getLogger("test_overlay_analog_pd_policy")
    sweeper_analog._pd_clock = "gate-in"
    sweeper_analog._pd_data_transfer = None
    sweeper_analog._pd_trace = sweeper_analog._pd_spectrum = False
    sweeper_analog.clock = clock
    sweeper_analog.pds = [pd_analog]

    assert sweeper_analog.configure_analog_pd(params_analog, "cw")
    assert pd_analog.params["cb_samples"] == 2
    assert pd_analog.params["drop_first"] == 2
    assert not clock.started
    assert not pd_analog.started


def test_overlay_dummy_points_must_fit_detector_buffer():
    sweeper = ODMRSweeperPG.__new__(ODMRSweeperPG)
    sweeper._closed = True
    sweeper.conf = {"buffer_size_coeff": 5}
    params = {"pd": {"dummy_points": 3, "buffer_size_coeff": 3}}

    assert sweeper.validate(params, "cw") == (True, "", "")

    params["pd"]["dummy_points"] = 4
    success, message, summary = sweeper.validate(params, "cw")

    assert not success
    assert message == "pd.dummy_points (4) must not exceed pd.buffer_size_coeff (3)."
    assert summary == ""
