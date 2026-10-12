#!/usr/bin/env python3

"""
Tests for mahos_dq.meas.apodmr.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import logging

import numpy as np
import pytest

from mahos.msgs.common_msgs import BinaryState
from mahos_dq.meas.apodmr import APODMRClient
from mahos_dq.meas.apodmr_io import APODMRIO
from mahos_dq.meas.apodmr_worker import APODMRBlockBuilder, APODMRDataOperator, Pulser
from mahos_dq.msgs.apodmr_msgs import APODMRData, MWMode
from mahos_dq.meas.podmr_generator.generator import make_generators
from mahos_dq.meas.podmr_generator import generator_kernel as K
from util import expect_value, get_some, save_load_test
from fixtures import ctx, gconf, server, apodmr, server_conf, apodmr_conf


def expect_apodmr(cli: APODMRClient, num: int, poll_timeout_ms):
    def get():
        data = cli.get_data()
        if data is not None and data.data0 is not None:
            return len(data.data0)
        return None

    return expect_value(get, num, poll_timeout_ms, trials=500)


def _apodmr_params() -> dict:
    return {
        "base_width": 320e-9,
        "laser_delay": 200e-9,
        "laser_width": 3e-6,
        "mw_delay": 1e-6,
        "trigger_width": 20e-9,
        "init_delay": 0.0,
        "final_delay": 5e-6,
        "enable_reduce": False,
        "partial": -1,
        "start": 100e-9,
        "num": 2,
        "step": 100e-9,
        "log": False,
        "invert_sweep": False,
        "divide_block": False,
        "roi_head": 100e-9,
        "roi_tail": 50e-9,
        "pd_rate": 500e3,
        "sweeps_per_record": 2,
        "max_records": 0,
        "burst_num": 1,
        "point_init_delay": 0.0,
        "pulse": {},
        "plot": {
            "plotmode": "data01",
            "taumode": "raw",
            "logX": False,
            "logY": False,
            "sigdelay": 50e-9,
            "sigwidth": 100e-9,
            "refdelay": 150e-9,
            "refwidth": 100e-9,
            "refmode": "ignore",
            "refaverage": False,
            "flipY": False,
        },
    }


def test_apodmr_sweep_validation_and_remaining_records(caplog):
    pulser = Pulser.__new__(Pulser)
    pulser.logger = logging.getLogger("test_apodmr_sweep_validation")

    assert pulser._validate_sweep_params(
        {"sweeps": 4, "sweeps_per_record": 2, "hardware_sweep_limit": True}
    )
    assert pulser._validate_sweep_params({"sweeps": 0, "sweeps_per_record": 3})
    assert pulser._validate_sweep_params({"sweeps": 5, "sweeps_per_record": 2})
    assert not pulser._validate_sweep_params(
        {"sweeps": 5, "sweeps_per_record": 2, "hardware_sweep_limit": True}
    )
    assert "sweeps=5, sweeps_per_record=2" in caplog.text
    assert not pulser._validate_sweep_params({"sweeps": 0, "sweeps_per_record": 0})

    params = {"sweeps": 8, "sweeps_per_record": 2, "hardware_sweep_limit": True}
    assert pulser._remaining_records(params, 1) == 3
    assert pulser._remaining_records(params, 4) == 0
    assert pulser._remaining_records(params, 5) == 0
    assert pulser._remaining_records({"sweeps": 0, "sweeps_per_record": 2}, 100) is None
    assert pulser._remaining_records({"sweeps": 8, "sweeps_per_record": 2}, 1) is None


def test_apodmr_start_rejects_nondivisible_sweeps_before_locking():
    class Operator:
        def update_axes(self, data):
            pass

    pulser = Pulser.__new__(Pulser)
    pulser.logger = logging.getLogger("test_apodmr_start_validation")
    pulser.data = APODMRData()
    pulser.op = Operator()
    locked = []
    pulser.lock_instruments = lambda: locked.append(True) or True
    params = _apodmr_params()
    params.update({"sweeps": 5, "sweeps_per_record": 2, "hardware_sweep_limit": True})

    assert not pulser.start(params, "rabi")
    assert not locked


def test_apodmr_pd_finite_samples_include_drop_first():
    class PD:
        def configure_triggered(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return True

        def start(self):
            return True

    pulser = Pulser.__new__(Pulser)
    pulser.data = APODMRData()
    pulser.data.params = {
        "pd": {"rate": 1.0e6, "buffer_size_coeff": 20, "drop_first": 2},
        "sweeps_per_record": 2,
        "hardware_sweep_limit": True,
        "burst_num": 1,
        "pulse": {},
    }
    pulser.trace_count = 4
    pulser.samples_per_trace = 8
    pulser.clock = None
    pulser._pd_trigger = "trigger"
    pulser._pd_data_transfer = None
    pulser.conf = {}
    pulser.pds = [PD()]

    assert pulser.init_start_pds(2)
    assert pulser.pds[0].args[1] == 32
    assert pulser.pds[0].args[2] == 32 * (2 + 2)
    assert pulser.pds[0].kwargs["finite"] is True

    assert pulser.init_start_pds(None)
    assert pulser.pds[0].args[2] == 32 * 20
    assert pulser.pds[0].kwargs["finite"] is False


def test_apodmr_finite_acquisition_finishes_on_pd_queue_overflow(caplog):
    class PD:
        def pop_block_with_status(self):
            return np.zeros(1), True

        def pop_block(self):
            raise AssertionError("finite acquisition must use overflow-aware pop")

    params = _apodmr_params()
    params.update({"sweeps": 2, "hardware_sweep_limit": True})
    pulser = Pulser.__new__(Pulser)
    pulser.logger = logging.getLogger("test_apodmr_queue_overflow")
    pulser.data = APODMRData(params, "rabi")
    pulser.data.start()
    pulser.pd_names = ["pd0"]
    pulser.pds = [PD()]
    pulser._acquisition_failed = False

    assert not pulser.update()
    assert pulser._acquisition_failed
    assert pulser.is_finished()
    assert pulser.data.records == 0
    assert "PD queue overflowed during finite acquisition (pd0)" in caplog.text


def test_apodmr_builder_inserts_trigger_before_each_laser():
    params = _apodmr_params()
    params["burst_num"] = 3
    xdata = np.array([100e-9, 200e-9])
    generators = make_generators()
    blocks, freq, common_pulses = generators["rabi"].generate_raw_blocks(xdata, params)
    unit_lengths = [
        blks[i].total_length() + blks[i + 1].total_length()
        for blks in blocks
        for i in range(0, len(blks), 2)
    ]
    builder = APODMRBlockBuilder(
        minimum_block_length=1000,
        block_base=4,
        mw_modes=(MWMode.QPSK,),
        iq_amplitude=0.0,
        channel_remap=None,
    )
    built, laser_timing, trigger_timing, trace_length_ticks = builder.build_blocks(
        blocks, freq, common_pulses, params, num_mw=1
    )
    base_width, _laser_delay, laser_width, _mw_delay, _tw, init_delay, final_delay = common_pulses
    expected_init_length = K.offset_base_inc(
        max(init_delay + laser_width, builder.minimum_block_length), base_width
    )
    expected_final_length = K.offset_base_inc(
        max(final_delay, builder.minimum_block_length), base_width
    )

    trigger = built.decode_digital("trigger")
    for t in trigger_timing:
        assert trigger[t]
    assert built[0].name == "INIT"
    assert built[-1].name == "FINAL"
    assert built[0].total_length() == expected_init_length
    assert built[-1].total_length() == expected_final_length
    assert {"laser", "sync", "mw_i", "mw_q"}.issubset(built[0].digital_channels())
    assert {"sync", "mw_i", "mw_q"}.issubset(built[-1].digital_channels())
    assert "laser" not in built[-1].digital_channels()
    assert len(laser_timing) == len(trigger_timing) == 4
    assert all(
        lt - tt == round(params["roi_head"] * freq) for lt, tt in zip(laser_timing, trigger_timing)
    )
    assert [blk.total_length() for blk in built[1:-1]] == [
        L * params["burst_num"] for L in unit_lengths
    ]
    assert [blk.Nrep for blk in built[1:-1]] == [params["burst_num"]] * len(unit_lengths)
    assert trace_length_ticks == round(
        (params["roi_head"] + params["laser_width"] + params["roi_tail"]) * freq
    )


def test_apodmr_builder_inserts_point_initialization():
    params = _apodmr_params()
    params["burst_num"] = 3
    params["point_init_delay"] = 2e-6
    xdata = np.array([100e-9, 200e-9])
    blocks, freq, common_pulses = make_generators()["rabi"].generate_raw_blocks(xdata, params)
    builder = APODMRBlockBuilder(
        minimum_block_length=1000,
        block_base=4,
        mw_modes=(MWMode.QPSK,),
        iq_amplitude=0.0,
        channel_remap=None,
    )

    built, laser_timing, trigger_timing, trace_length_ticks = builder.build_blocks(
        blocks, freq, common_pulses, params, num_mw=1
    )

    base_width, _ld, laser_width, _md, _tw, init_delay, _final_delay = common_pulses
    point_delay = round(params["point_init_delay"] * freq)
    point_inits = built[0:-1:2]
    acquisitions = built[1:-1:2]
    assert [block.name for block in point_inits] == [f"POINT_INIT{i}" for i in range(4)]
    assert len(acquisitions) == len(laser_timing) == len(trigger_timing) == 4
    delays = [point_delay + init_delay, point_delay, point_delay, point_delay]
    expected_dark_durations = [
        K.offset_base_inc(max(delay + laser_width, builder.minimum_block_length), base_width)
        - laser_width
        for delay in delays
    ]
    assert [block.pattern[0].duration for block in point_inits] == expected_dark_durations
    for block in point_inits:
        assert "trigger" not in block.digital_channels()
        laser_pulses = [pulse for pulse in block.pattern if "laser" in pulse.channels]
        assert len(laser_pulses) == 1
        assert laser_pulses[0].duration == laser_width
        assert block.pattern[-1] == laser_pulses[0]
    assert [block.Nrep for block in acquisitions] == [params["burst_num"]] * 4
    assert len(builder.all_trigger_timing) == 4 * params["burst_num"]
    assert trace_length_ticks == round(
        (params["roi_head"] + params["laser_width"] + params["roi_tail"]) * freq
    )


def test_apodmr_builder_rejects_invalid_point_init_delay():
    params = _apodmr_params()
    blocks, freq, common_pulses = make_generators()["rabi"].generate_raw_blocks(
        np.array([100e-9, 200e-9]), params
    )
    builder = APODMRBlockBuilder(1000, 4, (MWMode.QPSK,), 0.0, None)

    params["point_init_delay"] = -1e-9
    with pytest.raises(ValueError, match="non-negative"):
        builder.build_blocks(blocks, freq, common_pulses, params, num_mw=1)

    params["point_init_delay"] = 0.1 / freq
    with pytest.raises(ValueError, match="at least one pulse-generator tick"):
        builder.build_blocks(blocks, freq, common_pulses, params, num_mw=1)


def test_apodmr_builder_requires_deadtime_after_trace():
    builder = APODMRBlockBuilder(1000, 4, (MWMode.QPSK,), 0.0, None)
    builder.all_trigger_timing = [0, 110]
    builder.eos_deadtime_ticks = 10

    assert builder.check_sample_duration(100)

    builder.all_trigger_timing = [0, 109]

    assert not builder.check_sample_duration(100)


def test_apodmr_analyze_rejects_out_of_range_markers():
    data = APODMRData(_apodmr_params(), "rabi")
    data.raw_data_sum = np.zeros((4, 6), dtype=np.float64)
    data.records = 1
    data.marker_indices = np.array([0, 2, 3, 10], dtype=np.int64)
    op = APODMRDataOperator()

    assert op._analysis_error(data) is not None
    assert op.analyze_with_error(data) is not None
    assert op.analyze(data) is False


def test_apodmr_append_record_limits_retained_records():
    data = APODMRData(_apodmr_params(), "rabi")
    data.params["max_records"] = 2
    op = APODMRDataOperator()
    records = [np.full((4, 6), i, dtype=np.float64) for i in range(3)]

    for record in records:
        op.append_record(data, record)

    assert data.records == 3
    assert data.retained_records() == 2
    assert np.array_equal(data.raw_data, np.stack(records[1:]))
    assert np.array_equal(data.raw_data_sum, np.sum(records, axis=0))
    assert data.sweeps() == 6


def test_apodmr_analyze_uses_all_captured_records_with_limited_retention():
    data = APODMRData(_apodmr_params(), "rabi")
    data.params["num_pattern"] = 2
    data.params["max_records"] = 2
    data.marker_indices = np.array([0, 0, 1, 1], dtype=np.int64)
    op = APODMRDataOperator()

    op.append_record(
        data,
        np.array(
            [
                [1.0, 10.0, 0.0],
                [2.0, 20.0, 0.0],
                [3.0, 30.0, 0.0],
                [4.0, 40.0, 0.0],
            ]
        ),
    )
    op.append_record(data, np.zeros((4, 3), dtype=np.float64))
    op.append_record(data, np.zeros((4, 3), dtype=np.float64))

    assert data.retained_records() == 2
    assert op.analyze(data)
    assert np.allclose(data.data0, np.array([1.0, 3.0]) / 3.0)
    assert np.allclose(data.data1, np.array([2.0, 4.0]) / 3.0)
    assert np.allclose(data.data0ref, np.array([10.0, 30.0]) / 3.0)
    assert np.allclose(data.data1ref, np.array([20.0, 40.0]) / 3.0)


def test_apodmr(server, apodmr, server_conf, apodmr_conf):
    poll_timeout_ms = apodmr_conf["poll_timeout_ms"]
    expected_mw_modes = [MWMode.parse(m).name for m in apodmr_conf["pulser"]["mw_modes"]]

    apodmr.wait()

    assert get_some(apodmr.get_status, poll_timeout_ms).state == BinaryState.IDLE

    params = apodmr.get_param_dict("rabi")
    assert params["save_history"].value() is True
    assert "head" not in params["plot"]["taumode"].options()
    params["num"].set(2)
    params["sweeps"].set(4)
    params["sweeps_per_record"].set(2)
    params["max_records"].set(1)
    params["burst_num"].set(2)
    params["plot"]["sigdelay"].set(0.0)
    params["plot"]["sigwidth"].set(1e-9)
    params["plot"]["refdelay"].set(1e-9)
    params["plot"]["refwidth"].set(1e-9)
    params["sweeps"].set(5)
    params["hardware_sweep_limit"].set(True)
    assert not apodmr.validate(params, "rabi")
    assert not apodmr.start(params, "rabi")
    assert apodmr.get_state() == BinaryState.IDLE
    params["sweeps"].set(4)
    assert apodmr.validate(params, "rabi")
    assert apodmr.start(params, "rabi")
    assert expect_apodmr(apodmr, params["num"].value(), poll_timeout_ms)
    assert expect_value(apodmr.get_state, BinaryState.IDLE, poll_timeout_ms, trials=500)

    data = get_some(apodmr.get_data, poll_timeout_ms)
    assert data is not None
    assert data.raw_data is not None
    assert data.raw_data.shape[0] == 1
    assert data.retained_records() == 1
    assert data.records == 2
    assert data.history_start_record == 0
    assert data.signal_history.shape == (2, data.raw_data.shape[1])
    assert data.reference_history.shape == data.signal_history.shape
    assert int(data.sweeps()) == 4
    assert int(data.records * data.get_sweeps_per_record()) == 4
    assert data.raw_data_sum is not None
    assert data.raw_data_sum.shape == data.raw_data.shape[1:]
    assert data.get_samples_per_trace() == data.raw_data.shape[2]
    assert data.marker_indices is not None
    assert data.marker_indices.shape == (4,)
    assert data.params["instrument"]["mw_modes"] == expected_mw_modes
    assert np.isclose(data.trace_laser_timing, params["roi_head"].value())
    assert np.isclose(data.laser_timing[0] - data.trigger_timing[0], params["roi_head"].value())
    save_load_test(APODMRIO(), data)
