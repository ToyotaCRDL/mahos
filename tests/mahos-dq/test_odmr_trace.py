#!/usr/bin/env python3

"""
Tests for pd_trace logic option in ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import logging

import numpy as np
import pytest

from mahos.msgs import param_msgs as P
from mahos_dq.meas.odmr_pd import (
    configure_trace_pds,
    make_pd_param_dict,
    marker_indices,
    reduce_pd_blocks,
    reduce_traces,
    trace_samples,
    validate_trace_params,
)
from mahos_dq.meas.odmr_pg import ODMRPGMixin
from mahos_dq.meas.odmr_worker import Sweeper, SweeperBase, SweeperOverlay
from mahos_dq.inst.overlay.odmr_sweeper import ODMRSweeperPG
from mahos_dq.msgs.odmr_msgs import ODMRData


def _bounds():
    return {"freq": (1.0, 10.0), "power": (-30.0, 10.0)}


def _trace_params(rate=1.0e9, background=False):
    return {
        "start": 2.0,
        "stop": 3.0,
        "num": 2,
        "power": 0.0,
        "background": background,
        "delay": 0.0,
        "background_delay": 0.0,
        "final_delay": 0.0,
        "timing": {
            "laser_delay": 6e-9,
            "laser_width": 8e-9,
            "mw_delay": 20e-9,
            "mw_width": 3e-9,
            "trigger_width": 2e-9,
            "mw_offset": 0.0,
            "burst_num": 3,
            "roi_head": 4e-9,
            "roi_tail": 8e-9,
            "sig_delay": 0.0,
            "sig_width": 2e-9,
            "ref_delay": 1e-9,
            "ref_width": 2e-9,
            "refmode": "divide",
        },
        "pd": {
            "rate": rate,
            "buffer_size_coeff": 7,
            "bounds": (-5.0, 5.0),
            "eos_deadtime": 2e-9,
        },
    }


class _Clock:
    def configure(self, params):
        self.params = params
        return True

    def get_internal_output(self):
        return "clock-output"


class _PD:
    def __init__(self, data=None):
        self.data = data

    def configure(self, params, label):
        self.params = params
        self.label = label
        return True

    def pop_block(self):
        return self.data

    def start(self):
        return True


class _PGBuilder(ODMRPGMixin):
    def __init__(self, block_base=1):
        self._minimum_block_length = 1
        self._block_base = block_base
        self._channel_remap = None
        self._pd_chop = False


class _PGConfig:
    def configure_blocks(self, blocks, freq, **kwargs):
        return True


def _rising_edges(blocks, channel):
    values = blocks.decode_digital(channel)
    return np.flatnonzero(np.diff(np.r_[0, values]) == 1)


def _trace_data():
    traces = np.zeros((2, 20))
    traces[0, 4:7] = 8.0
    traces[0, 7:10] = 2.0
    traces[1, 4:7] = 15.0
    traces[1, 7:10] = 3.0
    return traces


def _direct_trace_sweeper(params, traces, *, sg_first=False):
    sweeper = Sweeper.__new__(Sweeper)
    sweeper.logger = logging.getLogger("test_direct_trace_sweeper")
    sweeper._pd_trace = True
    sweeper._samples_per_trace = traces.shape[1]
    sweeper._sg_first = sg_first
    sweeper._pg_immediate = False
    sweeper.pds = [_PD(traces.ravel())]
    sweeper.data = ODMRData(params, "pulse")
    sweeper.data.start()
    return sweeper


def test_direct_sweeper_reduces_and_sums_complete_trace_lines():
    params = _trace_params()
    sweeper = _direct_trace_sweeper(params, _trace_data())

    sweeper.work()
    sweeper.work()

    traces = _trace_data()
    assert sweeper.data.data.shape == (2, 2)
    assert np.allclose(sweeper.data.data, [[4.0, 4.0], [5.0, 5.0]])
    assert np.array_equal(sweeper.data.raw_data_sum, 2.0 * traces)


def test_overlay_worker_sums_raw_trace_points_by_frequency():
    params = _trace_params(background=True)

    class Overlay:
        def __init__(self):
            self.index = 0

        def get_point(self):
            point = np.array([self.index + 1.0, self.index + 2.0])
            traces = np.full((2, 20), self.index + 1.0)
            self.index = (self.index + 1) % params["num"]
            return point, traces

    worker = SweeperOverlay.__new__(SweeperOverlay)
    worker.logger = logging.getLogger("test_overlay_raw_trace_sum")
    worker._pd_trace = True
    worker.sweeper = Overlay()
    worker.data = ODMRData(params, "pulse")
    worker.data.start()

    for _ in range(2 * params["num"]):
        worker._work_point()

    expected = np.repeat(np.array([2.0, 2.0, 4.0, 4.0])[:, np.newaxis], 20, axis=1)
    assert np.array_equal(worker.data.raw_data_sum, expected)
    assert worker.data.data.shape == worker.data.bg_data.shape == (2, 2)


@pytest.mark.parametrize(
    ("spectrum", "point_count", "background", "expected_samples"),
    ((False, 2, False, 20), (False, 4, True, 20), (True, 2, False, 32)),
)
def test_configure_trace_pds(spectrum, point_count, background, expected_samples):
    params = _trace_params(background=background)
    params["pd"]["hardware_average"] = False
    conf = {"buffer_size_coeff": 5, "pd_segment_granularity": 16, "pg_freq_pulse": 1e9}
    clock = None if spectrum else _Clock()
    pd = _PD()

    result = configure_trace_pds(
        clock,
        [pd],
        "gate-in",
        params,
        conf,
        spectrum,
        point_count,
        1,
        "dma",
        logging.getLogger("test_configure_trace_pds"),
    )

    assert result == expected_samples
    assert pd.label == "triggered"
    assert pd.params["trigger_source"] == ("gate-in" if spectrum else "clock-output")
    assert pd.params["clock"] == ("gate-in" if spectrum else "clock-output")
    assert pd.params["cb_samples"] == point_count * expected_samples
    assert (
        pd.params["samples"]
        == pd.params["buffer_size"]
        == point_count * expected_samples * params["pd"]["buffer_size_coeff"]
    )
    assert pd.params["drop_first"] == 1
    assert pd.params["oversample"] == 1
    assert pd.params["block_samples"] == expected_samples
    assert pd.params["block_reduce_factor"] == params["timing"]["burst_num"]
    assert pd.params["block_reduce_op"] == "mean"
    assert pd.params["hardware_average"] is False
    if clock is not None:
        assert clock.params["samples"] == expected_samples
        assert clock.params["retriggerable"] is True


def test_trace_pulse_blocks_have_preinit_and_per_laser_gates():
    builder = _PGBuilder()
    blocks = builder._make_blocks_pulse_trace_nobg(
        delay=0,
        final_delay=0,
        laser_delay=6,
        laser_width=8,
        mw_delay=20,
        mw_width=3,
        trigger_width=2,
        roi_head=4,
        burst_num=3,
        mw_offset=0,
    )

    assert [block.name for block in blocks] == ["INIT", "MAIN", "FINAL"]
    assert blocks[0].pattern == [(("laser",), 8)]
    assert blocks[1].Nrep == 3
    assert len(_rising_edges(blocks, "laser")) == 4
    gates = _rising_edges(blocks, "gate")
    lasers = _rising_edges(blocks, "laser")[1:]
    assert len(gates) == 3
    assert np.array_equal(lasers - gates, [4, 4, 4])

    main = blocks[1]
    mw = main.decode_digital("mw")
    laser = main.decode_digital("laser")
    assert not any(mw[:20])
    assert all(mw[20:23])
    assert not any(mw[23:29])
    assert all(laser[29:37])


def test_trace_background_blocks_preserve_gates_and_remove_mw():
    builder = _PGBuilder()
    blocks = builder._make_blocks_pulse_trace_bg(
        delay=0,
        bg_delay=5,
        final_delay=0,
        laser_delay=6,
        laser_width=8,
        mw_delay=20,
        mw_width=3,
        trigger_width=2,
        roi_head=4,
        burst_num=2,
        mw_offset=0,
    )

    assert [block.name for block in blocks] == ["INIT", "MAIN", "INIT-BG", "MAIN-BG", "FINAL"]
    assert blocks[1].Nrep == blocks[3].Nrep == 2
    assert "mw" in blocks[1].digital_channels()
    assert "mw" not in blocks[3].digital_channels()
    assert len(_rising_edges(blocks, "gate")) == 4
    gates = _rising_edges(blocks, "gate")
    lasers = _rising_edges(blocks, "laser")
    acquisition_lasers = np.delete(lasers, [0, 3])
    assert np.array_equal(acquisition_lasers - gates, [4, 4, 4, 4])


def test_validate_trace_params_rejects_invalid_gate_windows_and_overlap():
    params = _trace_params()
    validate_trace_params(params, {"pg_freq_pulse": 1.0e9}, False)

    params["timing"]["laser_delay"] = 0.0
    validate_trace_params(params, {"pg_freq_pulse": 1.0e9}, False)

    params["timing"]["roi_head"] = 1e-9
    with pytest.raises(ValueError, match="trigger_width <= roi_head"):
        validate_trace_params(params, {"pg_freq_pulse": 1.0e9}, False)

    params = _trace_params()
    params["timing"]["roi_head"] = 30e-9
    with pytest.raises(ValueError, match="complete pre-laser sequence"):
        validate_trace_params(params, {"pg_freq_pulse": 1.0e9}, False)

    params = _trace_params()
    params["timing"]["ref_width"] = 100e-9
    with pytest.raises(ValueError, match="reference window"):
        validate_trace_params(params, {"pg_freq_pulse": 1.0e9}, False)

    params = _trace_params(rate=40e6)
    params["timing"]["mw_delay"] = 5e-9
    with pytest.raises(ValueError, match="overlaps the next detector trigger"):
        validate_trace_params(params, {"pg_freq_pulse": 1.0e9}, False)


def test_trace_parameter_dictionary_and_legacy_regressions():
    worker = SweeperBase.__new__(SweeperBase)
    worker.conf = {}
    worker.logger = logging.getLogger("test_odmr_trace_params")

    legacy = P.unwrap(worker._make_param_dict("pulse", _bounds(), True, False))
    assert {"time_window", "gate_delay", "post_gate_delay"} <= set(legacy["timing"])
    assert "roi_head" not in legacy["timing"]

    trace = P.unwrap(worker._make_param_dict("pulse", _bounds(), True, True))
    timing = trace["timing"]
    assert set(timing) == {
        "laser_delay",
        "laser_width",
        "mw_delay",
        "mw_width",
        "trigger_width",
        "mw_offset",
        "burst_num",
        "roi_head",
        "roi_tail",
        "sig_delay",
        "sig_width",
        "ref_delay",
        "ref_width",
        "refmode",
    }
    assert timing["trigger_width"] == 20e-9
    assert timing["roi_head"] == 20e-9
    assert timing["roi_tail"] == 100e-9
    assert timing["sig_delay"] == timing["ref_delay"] == 0.0
    assert timing["sig_width"] == timing["ref_width"] == 100e-9
    assert timing["refmode"] == "divide"

    apd = P.unwrap(worker._make_param_dict("pulse", _bounds(), False, False))
    assert "burst_num" in apd["timing"]
    assert "roi_head" not in apd["timing"]
    cw_legacy = P.unwrap(worker._make_param_dict("cw", _bounds(), True, False))
    cw_trace = P.unwrap(worker._make_param_dict("cw", _bounds(), True, True))
    assert cw_trace == cw_legacy


def test_overlay_trace_validation_uses_overlay_configuration():
    sweeper = ODMRSweeperPG.__new__(ODMRSweeperPG)
    sweeper._closed = True
    sweeper.logger = logging.getLogger("test_odmr_overlay_trace_validation")
    sweeper._pd_trace = sweeper._pd_spectrum = sweeper._pd_analog = True
    sweeper._pd_chop = False
    sweeper.conf = {
        "pg_freq_pulse": 1.0e9,
        "pd_segment_granularity": 64,
        "pd_segment_offset": 0,
    }
    params = _trace_params()

    success, _, _ = sweeper.get("validate", params, "pulse")
    assert not success

    sweeper.conf["pd_segment_granularity"] = 16
    success, _, _ = sweeper.get("validate", params, "pulse")
    assert success

    params["timing"]["trigger_width"] = 0.4e-9
    success, _, _ = sweeper.get("validate", params, "pulse")
    assert not success


def test_trace_parameter_defaults_pass_validation():
    worker = SweeperBase.__new__(SweeperBase)
    worker.conf = {}
    params = P.unwrap(worker._make_param_dict("pulse", _bounds(), True, True))
    params["pd"] = P.unwrap(make_pd_param_dict({}, pd_trace=True))

    validate_trace_params(params, {"pg_freq_pulse": 1e9}, False)
    validate_trace_params(params, {"pd_segment_granularity": 16, "pg_freq_pulse": 1e9}, True)


def test_direct_sweeper_rolls_background_raw_traces_into_frequency_order():
    params = _trace_params(background=True)
    traces = np.repeat(np.arange(1.0, 5.0)[:, np.newaxis], 20, axis=1)
    sweeper = _direct_trace_sweeper(params, traces, sg_first=True)

    sweeper.work()

    assert np.array_equal(sweeper.data.raw_data_sum, np.roll(traces, -2, axis=0))


def test_overlay_sweeper_reduces_background_trace_point():
    params = _trace_params(background=True)
    params["num"] = 1
    traces = _trace_data()
    sweeper = ODMRSweeperPG.__new__(ODMRSweeperPG)
    sweeper._closed = True
    sweeper._pd_trace = True
    sweeper._label = "pulse"
    sweeper._samples_per_trace = 20
    sweeper.params = params
    sweeper.pds = [_PD(traces.ravel())]

    point, raw_traces = sweeper.get_pd_data()

    assert np.allclose(point, [4.0, 5.0])
    assert np.array_equal(raw_traces, traces)


def test_validate_trace_params_includes_eos_deadtime():
    params = _trace_params()
    params["pd"]["eos_deadtime"] = 17e-9
    validate_trace_params(params, {"pg_freq_pulse": 1.0e9}, False)

    params["pd"]["eos_deadtime"] = 18e-9
    with pytest.raises(ValueError, match="including eos_deadtime"):
        validate_trace_params(params, {"pg_freq_pulse": 1.0e9}, False)

    params = _trace_params()
    params["pd"]["eos_deadtime"] = 5e-9
    validate_trace_params(params, {"pg_freq_pulse": 1.0e9, "pd_segment_granularity": 16}, True)

    params["pd"]["eos_deadtime"] = 6e-9
    with pytest.raises(ValueError, match="including eos_deadtime"):
        validate_trace_params(params, {"pg_freq_pulse": 1.0e9, "pd_segment_granularity": 16}, True)


def test_direct_parameter_exposures():
    class SG:
        def get_bounds(self):
            return _bounds()

    sweeper = Sweeper.__new__(Sweeper)
    sweeper.conf = {}
    sweeper.logger = logging.getLogger("test_direct_spectrum_params")
    sweeper.sg = SG()
    sweeper._pd_analog = sweeper._pd_spectrum = sweeper._pd_trace = True
    sweeper._pd_chop = False

    cw = P.unwrap(sweeper.get_param_dict("cw"))
    pulse = P.unwrap(sweeper.get_param_dict("pulse"))

    assert cw["pd"]["rate"] == 400e3
    assert pulse["pd"]["rate"] == 250e6
    assert "hardware_average" not in cw["pd"]
    assert pulse["pd"]["hardware_average"] is True
    assert "eos_deadtime" not in cw["pd"]
    assert "eos_deadtime" in pulse["pd"]


def test_overlay_parameter_exposures():
    class Overlay:
        def get_bounds(self):
            return _bounds()

        def get_param_dict(self, label):
            return overlay.get_param_dict(label)

    overlay = ODMRSweeperPG.__new__(ODMRSweeperPG)
    overlay._closed = True
    overlay.conf = {}
    overlay._pd_analog = overlay._pd_spectrum = overlay._pd_trace = True

    worker = SweeperOverlay.__new__(SweeperOverlay)
    worker.conf = {}
    worker.logger = logging.getLogger("test_odmr_overlay_spectrum_params")
    worker._class_name = "ODMRSweeperPG"
    worker._pd_analog = True
    worker._pd_trace = True
    worker._pd_chop = False
    worker.sweeper = Overlay()

    cw = P.unwrap(worker.get_param_dict("cw"))
    pulse = P.unwrap(worker.get_param_dict("pulse"))

    assert cw["pd"]["rate"] == 400e3
    assert pulse["pd"]["rate"] == 250e6
    assert "hardware_average" not in cw["pd"]
    assert pulse["pd"]["hardware_average"] is True
    assert "eos_deadtime" not in cw["pd"]
    assert "eos_deadtime" in pulse["pd"]


def test_trace_samples_and_markers_follow_apodmr_rounding():
    params = _trace_params()
    samples = trace_samples(params, {}, False)
    spectrum = trace_samples(params, {"pd_segment_granularity": 16, "pd_segment_offset": 0}, True)
    offset_spectrum = trace_samples(
        params, {"pd_segment_granularity": 16, "pd_segment_offset": 16}, True
    )

    assert samples.logical == samples.realized == 20
    assert spectrum.logical == 20
    assert spectrum.realized == 32
    assert offset_spectrum.realized == 16
    assert np.array_equal(marker_indices(params["timing"], params["pd"]["rate"]), [4, 6, 7, 9])


def test_trace_pg_warns_when_background_delay_is_shorter_than_realized_trace(caplog):
    builder = _PGBuilder()
    builder.conf = {"pg_freq_pulse": 1.0e9}
    builder.logger = logging.getLogger("test_odmr_trace_background_delay")
    builder._pd_spectrum = False
    builder.pg = _PGConfig()

    params = _trace_params(background=True)
    with caplog.at_level(logging.WARNING):
        assert builder.configure_pg_pulse_trace(params, None)

    assert "recommended minimum 8.0 ns" in caplog.text

    caplog.clear()
    params["background_delay"] = 8e-9
    with caplog.at_level(logging.WARNING):
        assert builder.configure_pg_pulse_trace(params, None)

    assert "recommended minimum" not in caplog.text


def test_trace_gate_may_overlap_mw_with_zero_laser_delay():
    builder = _PGBuilder(block_base=4)
    blocks = builder._make_blocks_pulse_trace_nobg(
        delay=0,
        final_delay=0,
        laser_delay=0,
        laser_width=8,
        mw_delay=20,
        mw_width=3,
        trigger_width=2,
        roi_head=4,
        burst_num=2,
        mw_offset=0,
    )

    main = blocks[1]
    gates = _rising_edges(blocks, "gate")
    lasers = _rising_edges(blocks, "laser")[1:]
    assert main.raw_length() % builder._block_base == 0
    assert np.array_equal(lasers - gates, [4, 4])
    assert np.any(np.asarray(main.decode_digital("gate")) & np.asarray(main.decode_digital("mw")))


def test_reduce_pd_blocks_sums_detectors_and_channels():
    params = _trace_params()
    params["timing"]["refmode"] = "subtract"
    channel = np.zeros((2, 20))
    channel[:, 4:7] = [[5.0], [8.0]]
    channel[:, 7:10] = [[2.0], [3.0]]

    result = reduce_pd_blocks(
        [channel.ravel(), [channel.ravel(), 2.0 * channel.ravel()]], params, 2, 20
    )

    assert np.allclose(result, [12.0, 20.0])


def test_reduce_pd_blocks_combines_channels_before_division():
    params = _trace_params()
    channel = np.zeros((2, 20))
    channel[:, 4:7] = [[8.0], [12.0]]
    channel[:, 7:10] = [[8.0], [12.0]]

    result = reduce_pd_blocks([channel.ravel(), channel.ravel()], params, 2, 20)

    assert np.allclose(result, [1.0, 1.0])


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("subtract", [6.0, 12.0]), ("divide", [4.0, 4.0]), ("ignore", [8.0, 16.0])),
)
def test_reduce_traces(mode, expected):
    params = _trace_params()
    params["timing"]["refmode"] = mode
    traces = np.zeros((2, 20))
    traces[0, 4:7] = 8.0
    traces[0, 7:10] = 2.0
    traces[1, 4:7] = 16.0
    traces[1, 7:10] = 4.0

    assert np.allclose(reduce_traces(traces, params["timing"], params["pd"]["rate"]), expected)
