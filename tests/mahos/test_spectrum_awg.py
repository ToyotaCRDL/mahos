#!/usr/bin/env python3

"""
Tests for mahos.inst.awg.spectrum.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import h5py
import numpy as np
import pytest

from mahos.inst.awg_file import save_waveforms
from mahos.inst.awg.spectrum import Spectrum_AWG, Waveform, pack_waveform, unpack_waveform
from mahos.msgs.inst.awg_msgs import TriggerType


@pytest.fixture
def awg():
    inst = Spectrum_AWG(
        "awg",
        {
            "mock": True,
            "memory_granularity": 8,
            "amplitude_mV": {0: 100, 1: 80},
        },
    )
    yield inst
    inst.close()


def waveform_params(**overrides):
    params = {
        "analog": {0: np.zeros(64)},
        "digital": {},
        "rate": 5e9,
        "trigger_type": TriggerType.IMMEDIATE,
        "n_runs": None,
    }
    params.update(overrides)
    return params


def test_pack_waveform_preserves_markers_and_quantized_analog():
    analog = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    laser = np.array([False, True, False, True, False])
    trigger = np.array([True, False, False, True, True])
    waveform = Waveform(1e9, analog, {"laser": laser, "trigger": trigger})

    packed, bit_map = pack_waveform(waveform, marker_order=("laser", "trigger"))
    restored, markers = unpack_waveform(packed, n_markers=2)

    assert packed.dtype == np.int16
    assert bit_map == {"laser": 15, "trigger": 14}
    np.testing.assert_array_equal(markers[0], laser)
    np.testing.assert_array_equal(markers[1], trigger)
    np.testing.assert_allclose(restored, analog, atol=4.0 / np.iinfo(np.int16).max)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"analog": [0.0, 1.1]},
        {"analog": [0.0, np.nan]},
        {"analog": np.zeros((2, 2))},
        {"analog": np.zeros(2), "markers": {"laser": [True]}},
    ],
)
def test_waveform_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        Waveform(sample_rate=1e9, **kwargs)


@pytest.mark.parametrize(
    ("sample_rate", "expected"),
    [
        (4e9, 1e9),
        (5e9, 1.25e9),
        (5e9 + 1.0, (5e9 + 1.0) / 8.0),
        (10e9, 1.25e9),
    ],
)
def test_get_digital_rate_uses_spectrum_divisors(awg, sample_rate, expected):
    assert awg.get("digital_rate", sample_rate) == expected


def test_configure_immediate_waveform_and_lifecycle(awg):
    laser = np.zeros(64, dtype=bool)
    laser[16:32] = True

    assert awg.configure(
        waveform_params(digital={"laser": [(False, 16), (True, 16), (False, 32)]}),
        "waveforms",
    )

    assert awg._core.config["trigger"]["source"] == "software"
    assert awg._core.last_upload["replay_mode"] == "single"
    assert awg._core.last_upload["loops"] == 0
    assert awg.get("length") == 64
    assert awg.get("offsets") == [0]
    info = awg.get("waveform_info")
    assert info["analog_channels"] == [0]
    assert info["marker_bits"] == {"trigger": 15, "laser": 14}
    assert info["marker_lines"] == {"trigger": 0, "laser": 1}

    _, markers = unpack_waveform(awg._core.last_upload["samples"][0], n_markers=2)
    np.testing.assert_array_equal(markers[0], np.zeros(64, dtype=bool))
    np.testing.assert_array_equal(markers[1], laser)

    assert awg.get("finished")
    assert awg.start()
    assert awg.get("status")["running"]
    assert not awg.get("finished")
    assert awg.stop()
    assert awg.get("finished")
    assert awg.reset()
    assert awg.get("length") == 0
    assert awg.get("offsets") == []


@pytest.mark.parametrize(
    ("trigger_type", "n_runs", "source", "edge", "replay_mode", "loops"),
    [
        (TriggerType.SOFTWARE, 3, "none", True, "singlerestart", 3),
        (TriggerType.HARDWARE_RISING, 2, "ext0", True, "singlerestart", 2),
        (TriggerType.HARDWARE_FALLING, None, "ext0", False, "singlerestart", 0),
    ],
)
def test_configure_trigger_modes(awg, trigger_type, n_runs, source, edge, replay_mode, loops):
    assert awg.configure(
        waveform_params(trigger_type=trigger_type, n_runs=n_runs),
        "waveforms",
    )

    assert awg._core.config["trigger"]["source"] == source
    assert awg._core.config["trigger"]["edge"] is edge
    assert awg._core.last_upload["replay_mode"] == replay_mode
    assert awg._core.last_upload["loops"] == loops


def test_configure_two_channels_routes_markers_to_their_sources():
    inst = Spectrum_AWG(
        "awg",
        {
            "mock": True,
            "memory_granularity": 8,
            "markers": {
                "laser": {"line": 0, "source": 0},
                "trigger": {"line": 1, "source": 1},
            },
        },
    )
    try:
        laser = np.arange(64) < 16
        trigger = np.arange(64) >= 48
        assert inst.configure(
            {
                "analog": {0: np.zeros(64), 1: np.linspace(-0.5, 0.5, 64)},
                "digital": {"laser": laser, "trigger": trigger},
                "rate": 5e9,
                "trigger_type": TriggerType.IMMEDIATE,
                "n_runs": 1,
            },
            "waveforms",
        )

        assert set(inst._core.last_upload["samples"]) == {0, 1}
        assert inst.get("waveform_info")["analog_channels"] == [0, 1]
        assert inst.get("waveform_info")["marker_bits"] == {"laser": 15, "trigger": 15}
        assert inst.get("waveform_info")["marker_sources"] == {"laser": 0, "trigger": 1}
        _, marker0 = unpack_waveform(inst._core.last_upload["samples"][0], n_markers=1)
        _, marker1 = unpack_waveform(inst._core.last_upload["samples"][1], n_markers=1)
        np.testing.assert_array_equal(marker0[0], laser)
        np.testing.assert_array_equal(marker1[0], trigger)
    finally:
        inst.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"analog": {0: np.zeros(63)}},
        {"analog": {0: np.zeros(64), 1: np.zeros(56)}},
        {"analog": {0: np.full(64, 1.01)}},
        {"digital": {"laser": np.zeros(63, dtype=bool)}},
        {"digital": {"laser": [(False, 32), (True, 31)]}},
        {"digital": {"laser": [(2, 64)]}},
        {"digital": {"unknown": np.zeros(64, dtype=bool)}},
        {"n_runs": 0},
        {"rate": 3e9},
    ],
)
def test_configure_waveforms_rejects_invalid_inputs(awg, overrides):
    assert not awg.configure(waveform_params(**overrides), "waveforms")


def test_configure_waveforms_file(tmp_path):
    inst = Spectrum_AWG(
        "awg",
        {
            "mock": True,
            "memory_granularity": 8,
            "file_transport_dir": str(tmp_path),
        },
    )
    try:
        path = tmp_path / "waveforms.h5"
        laser = [(False, 16), (True, 16), (False, 32)]
        save_waveforms(str(path), {0: np.linspace(-0.5, 0.5, 64)}, {"laser": laser})

        assert inst.configure(
            {
                "file_name": path.name,
                "rate": 5e9,
                "trigger_type": TriggerType.SOFTWARE,
                "n_runs": 3,
            },
            "waveforms_file",
        )
        assert inst.get("bounds")["file_transport"]
        assert inst._core.last_upload["loops"] == 3
        _, markers = unpack_waveform(inst._core.last_upload["samples"][0], n_markers=2)
        expected = np.repeat([False, True, False], [16, 16, 32])
        np.testing.assert_array_equal(markers[1], expected)

        assert not inst.configure({"file_name": "../waveforms.h5", "rate": 5e9}, "waveforms_file")
        assert not inst.configure({"file_name": r"..\waveforms.h5", "rate": 5e9}, "waveforms_file")

        with h5py.File(path, "r+") as f:
            f.attrs["version"] = 99
        assert not inst.configure({"file_name": path.name, "rate": 5e9}, "waveforms_file")
    finally:
        inst.close()


def test_bounds_report_waveform_capabilities(awg):
    bounds = awg.get("bounds")

    assert bounds["analog_channels"] == (0, 1)
    assert bounds["sample_rate"] == (0.0, 10e9)
    assert bounds["amplitude_mV"] == (1, 100)
    assert bounds["granularity"] == (8, 8)
    assert bounds["digital_lines"] == {
        "trigger": {"line": 0, "source": 0},
        "laser": {"line": 1, "source": 0},
    }
    assert bounds["digital_min_duration"] == 4e-9
    assert not bounds["has_sequence_mode"]
    assert not bounds["file_transport"]
