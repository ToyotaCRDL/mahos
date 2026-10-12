#!/usr/bin/env python3

"""
Tests for mahos_dq.msgs.apodmr_msgs.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import numpy as np
import pytest

from mahos_dq.msgs.apodmr_msgs import APODMRData, update_data
from mahos_dq.meas.apodmr_io import APODMRIO
from mahos_dq.meas.apodmr_worker import APODMRDataOperator
from util import save_load_test


def make_data(num_pattern: int, plotmode: str, *, partial: int = -1) -> APODMRData:
    params = {
        "num_pattern": num_pattern,
        "partial": partial,
        "start": 1.0e-9,
        "num": 2,
        "step": 1.0e-9,
        "log": False,
        "invert_sweep": False,
        "roi_head": 4.0e-9,
        "roi_tail": 2.0e-9,
        "trigger_width": 2.0e-9,
        "sweeps_per_record": 2,
        "pulse": {},
        "plot": {
            "plotmode": plotmode,
            "taumode": "raw",
            "refmode": "ignore",
            "refaverage": False,
            "flipY": False,
        },
        "instrument": {
            "tbin": 2.0e-9,
            "trange": 12.0e-9,
            "samples_per_trace": 6,
            "pg_freq": 2.0e9,
            "length": 100,
            "pd_rate": 500e6,
        },
    }
    data = APODMRData(params, "rabi")
    data.trace_laser_timing = params["roi_head"]
    data.trigger_timing = np.arange(num_pattern * params["num"]) * 10.0e-9
    data.laser_timing = data.trigger_timing + params["roi_head"]
    return data


def set_pattern_data(data: APODMRData, signals: list[np.ndarray], refs: list[np.ndarray]):
    for i, (s, r) in enumerate(zip(signals, refs)):
        data.set_data(i, s)
        data.set_data_ref(i, r)


def test_apodmr_raw_xdata_is_trigger_local():
    data = make_data(2, "data01")
    x = data.get_raw_xdata()
    assert np.array_equal(x, np.arange(6) * 2.0e-9)


def test_apodmr_complementary_n2_diff():
    data = make_data(2, "diff")
    set_pattern_data(
        data,
        [np.array([5.0, 7.0]), np.array([1.0, 2.0])],
        [np.array([10.0, 10.0]), np.array([10.0, 10.0])],
    )

    y, y1 = data.get_ydata()
    assert y1 is None
    assert np.array_equal(y, np.array([4.0, 5.0]))


def test_apodmr_complementary_n4_concatenate():
    data = make_data(4, "concatenate")
    set_pattern_data(
        data,
        [
            np.array([10.0, 20.0]),
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        ],
        [
            np.array([100.0, 200.0]),
            np.array([10.0, 20.0]),
            np.array([30.0, 40.0]),
            np.array([50.0, 60.0]),
        ],
    )

    y, y1 = data.get_ydata()
    assert y1 is None
    assert np.array_equal(y, np.array([10.0, 1.0, 3.0, 5.0, 20.0, 2.0, 4.0, 6.0]))


def test_apodmr_save_load_roundtrip():
    data = make_data(2, "data01")
    data.raw_data = np.arange(24, dtype=np.float64).reshape(1, 4, 6)
    data.raw_data_sum = data.raw_data[0].copy()
    data.records = 1
    assert data.records == 1
    assert data.retained_records() == 1
    assert data.sweeps() == 2
    assert np.isclose(data.measurement_time(), 100e-9)
    set_pattern_data(
        data,
        [np.array([5.0, 7.0]), np.array([1.0, 2.0])],
        [np.array([10.0, 10.0]), np.array([10.0, 10.0])],
    )

    save_load_test(APODMRIO(), data)


def make_history_data(num_pattern=2, partial=-1, enabled=True):
    data = make_data(num_pattern, "data01", partial=partial)
    data.params["max_records"] = 2
    if not enabled:
        data.params["save_history"] = False
    data.params["plot"].update(sigdelay=0.0, sigwidth=0.0, refdelay=2e-9, refwidth=0.0)
    return data


def append_history(data, count=1):
    op = APODMRDataOperator()
    traces = data.params["num"] * (1 if data.is_partial() else data.num_pattern())
    for _ in range(count):
        raw = np.arange(traces * 6, dtype=np.float64).reshape(traces, 6) + data.records * 100
        op.append_record(data, raw)
        op.get_marker_indices(data)
        assert op.analyze(data)


@pytest.mark.parametrize("num_pattern,partial", [(2, -1), (4, -1), (4, 2)])
def test_history_individual_records(num_pattern, partial):
    data = make_history_data(num_pattern, partial)
    append_history(data, 5)
    traces = data.params["num"] * (1 if partial >= 0 else num_pattern)
    expected = np.arange(traces)[None, :] * 6 + np.arange(5)[:, None] * 100
    assert data.signal_history.dtype == np.float64
    assert data.reference_history.dtype == np.float64
    np.testing.assert_array_equal(data.signal_history, expected + 2)
    np.testing.assert_array_equal(data.reference_history, expected + 3)
    assert data.history_start_record == 0
    assert data.retained_records() == 2
    for pattern in [partial] if partial >= 0 else range(num_pattern):
        averaged = (expected + 2).mean(axis=0)
        if partial < 0:
            averaged = averaged[pattern::num_pattern]
        np.testing.assert_array_equal(getattr(data, f"data{pattern}"), averaged)
    original = data.signal_history
    assert APODMRDataOperator().analyze(data)
    assert data.signal_history is original
    period = data.params["instrument"]["length"] / data.params["instrument"]["pg_freq"]
    assert data.get_sampling_interval() == pytest.approx(2 * period, abs=0.0)
    assert np.isclose(data.measurement_time(), data.records * 2 * period)
    data.params["sweeps_per_record"] = 1
    assert data.get_sampling_interval() == pytest.approx(period, abs=0.0)
    assert np.isclose(data.measurement_time(), data.records * period)


def test_history_plot_changes_and_offline_reanalysis():
    data = make_history_data()
    append_history(data, 5)
    op = APODMRDataOperator()
    original = data.signal_history
    assert not op.update_plot_params(data, {"sigdelay": 0.0})
    for params in ({"plotmode": "diff"}, {"refmode": "divide"}, {"flipY": True}):
        assert op.update_plot_params(data, params)
        assert op.analyze(data)
        assert data.signal_history is original
    data.running = True
    assert op.update_plot_params(data, {"sigdelay": 2e-9})
    assert data.history_start_record == 3
    np.testing.assert_array_equal(data.signal_history, original[-2:] + 1)
    append_history(data)
    assert data.signal_history.shape == (3, 4)
    data.running = False
    APODMRIO().reanalyze_data({"sigdelay": 0.0}, data)
    assert data.history_start_record == 4
    assert data.signal_history.shape == (2, 4)


def test_history_invalid_windows_and_gap_recovery():
    data = make_history_data()
    append_history(data, 4)
    op = APODMRDataOperator()
    op.update_plot_params(data, {"refwidth": 100e-9})
    assert data.signal_history is data.reference_history is None
    assert data.history_start_record == data.records
    op.update_plot_params(data, {"refwidth": 0.0})
    assert data.history_start_record == 2
    for _ in range(3):
        op.append_record(data, np.zeros((4, 6)))
    assert op.analyze(data)
    assert data.history_start_record == 5
    assert data.signal_history.shape == (2, 4)
    data.raw_data = None
    assert op.analyze(data)
    assert data.signal_history is data.reference_history is None
    assert data.history_start_record == data.records


def test_history_resume_and_opt_out():
    data = make_history_data(enabled=False)
    append_history(data, 3)
    assert data.signal_history is data.reference_history is None
    assert data.history_start_record == 3
    op = APODMRDataOperator()
    params = {**data.params, "save_history": True}
    assert data.can_resume(params, data.label)
    op.update_params(data, params)
    assert data.history_start_record == 1
    original = data.signal_history
    op.update_params(data, None)
    op.update_params(data, data.params.copy())
    assert data.signal_history is original
    op.update_params(data, {"plot": {"sigdelay": 2e-9}})
    np.testing.assert_array_equal(data.signal_history, original + 1)
    op.update_params(data, {"save_history": False})
    assert data.signal_history is data.reference_history is None
    assert data.history_start_record == data.records
    default_data = make_history_data()
    assert default_data.can_resume({**default_data.params, "save_history": True}, data.label)
    assert default_data.can_resume({**default_data.params, "save_history": False}, data.label)
