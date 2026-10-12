#!/usr/bin/env python3

"""
Tests for mahos_dq.meas.awg_renderer.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import os

import numpy as np
import pytest

from mahos.inst.awg_file import load_waveforms
from mahos.msgs.inst.awg_msgs import TriggerType
from mahos.msgs.inst.pg_msgs import AnalogChannel as A
from mahos.msgs.inst.pg_msgs import Block, Blocks
from mahos_dq.meas.awg_renderer import AWGRenderer
from mahos_dq.meas.awg_renderer import renderer_kernel as K


class RecordingAWG:
    def __init__(self):
        self.amplitudes = []
        self.waveforms = []
        self.file_waveforms = []
        self.set_amplitude_result = True
        self.configure_result = True

    def set_amplitude(self, channel, amplitude_mV):
        self.amplitudes.append((channel, amplitude_mV))
        return self.set_amplitude_result

    def configure_waveforms(
        self, analog, digital, rate, trigger_type=TriggerType.IMMEDIATE, n_runs=None
    ):
        self.waveforms.append(
            {
                "analog": analog,
                "digital": digital,
                "rate": rate,
                "trigger_type": trigger_type,
                "n_runs": n_runs,
            }
        )
        return self.configure_result

    def configure_waveforms_file(
        self, file_name, rate, trigger_type=TriggerType.IMMEDIATE, n_runs=None
    ):
        analog, digital = load_waveforms(os.path.join(self.file_transport_dir, file_name))
        self.file_waveforms.append(
            {
                "file_name": file_name,
                "analog": analog,
                "digital": digital,
                "rate": rate,
                "trigger_type": trigger_type,
                "n_runs": n_runs,
            }
        )
        return self.configure_result


def make_bounds(**overrides):
    bounds = {
        "analog_channels": (0, 1),
        "sample_rate": (1.0, 10.0e9),
        "amplitude_mV": (1, 500),
        "load_impedance": 50.0,
        "memory_samples": 1024,
        "granularity": (1, 1),
        "digital_min_duration": None,
    }
    bounds.update(overrides)
    return bounds


def test_render_flat_synthesizes_phase_coherent_tone_and_digital_track():
    blocks = Blocks(
        [
            Block(
                "pulse",
                [(("mw", "laser", A("mw_phase", 0.0)), 2), (None, 2)],
            )
        ]
    )
    params = K.RenderParams(
        tones=[K.MWTone("mw", "mw_phase", freq=1.0, power=-10.0)],
        analog_channels=(0,),
        num_logical_mw=1,
    )

    result = K.render_flat(blocks, pg_freq=4.0, rate=8.0, params=params, granularity=(1, 1))

    expected = np.zeros(8)
    expected[:4] = np.sin(2.0 * np.pi * np.arange(4) / 8.0)
    np.testing.assert_allclose(result.analog[0], expected, atol=1e-15)
    np.testing.assert_array_equal(result.digital["laser"], [True] * 4 + [False] * 4)
    assert result.digital_rle()["laser"] == [(True, 4), (False, 4)]
    assert result.rendered_samples == 8
    assert result.actual_samples == 8
    assert result.max_rounding_error_sec == 0.0


def test_render_flat_resets_local_phase_at_each_mw_pulse():
    blocks = Blocks(
        [
            Block(
                "pulses",
                [
                    (("mw", A("mw_phase", 30.0)), 2),
                    (None, 1),
                    (("mw", A("mw_phase", 30.0)), 2),
                ],
            )
        ]
    )
    params = K.RenderParams(
        tones=[K.MWTone("mw", "mw_phase", freq=1.0, power=-10.0)],
        local_phase=True,
    )

    result = K.render_flat(blocks, pg_freq=10.0, rate=10.0, params=params, granularity=(1, 1))

    pulse = np.sin(2.0 * np.pi * np.arange(2) / 10.0 + np.pi / 6.0)
    np.testing.assert_allclose(result.analog[0], [pulse[0], pulse[1], 0.0, *pulse], atol=1e-15)


def test_render_flat_keeps_local_carrier_contiguous_across_blocks_and_phase_changes():
    blocks = Blocks(
        [
            Block("first", [(("mw", A("mw_phase", 0.0)), 2)]),
            Block("second", [(("mw", A("mw_phase", 90.0)), 2)]),
        ]
    )
    params = K.RenderParams(
        tones=[K.MWTone("mw", "mw_phase", freq=1.0, power=-10.0)],
        local_phase=True,
    )

    result = K.render_flat(blocks, pg_freq=8.0, rate=8.0, params=params, granularity=(1, 1))

    expected = np.concatenate(
        (
            np.sin(2.0 * np.pi * np.arange(2) / 8.0),
            np.sin(2.0 * np.pi * np.arange(2, 4) / 8.0 + np.pi / 2.0),
        )
    )
    np.testing.assert_allclose(result.analog[0], expected, atol=1e-15)


def test_render_flat_resets_logical_mw_tones_independently():
    blocks = Blocks(
        [
            Block(
                "two_tones",
                [
                    (("mw", A("mw_phase", 0.0)), 1),
                    (("mw", "mw1", A("mw_phase", 0.0), A("mw1_phase", 0.0)), 2),
                ],
            )
        ]
    )
    params = K.RenderParams(
        tones=[
            K.MWTone("mw", "mw_phase", freq=1.0, power=-10.0, awg_channel=0),
            K.MWTone("mw1", "mw1_phase", freq=1.0, power=-10.0, awg_channel=1),
        ],
        analog_channels=(0, 1),
        num_logical_mw=2,
        local_phase=True,
    )

    result = K.render_flat(blocks, pg_freq=8.0, rate=8.0, params=params, granularity=(1, 1))

    phase = 2.0 * np.pi * np.arange(3) / 8.0
    np.testing.assert_allclose(result.analog[0], np.sin(phase), atol=1e-15)
    np.testing.assert_allclose(result.analog[1], [0.0, 0.0, np.sin(np.pi / 4.0)], atol=1e-15)


def test_render_flat_supports_two_physical_channels_and_logical_mw_channels():
    blocks = Blocks(
        [
            Block(
                "two_tones",
                [
                    (("mw", "mw1", A("mw_phase", 0.0), A("mw1_phase", 90.0)), 4),
                ],
            )
        ]
    )
    params = K.RenderParams(
        tones=[
            K.MWTone("mw", "mw_phase", freq=1.0, power=-10.0, awg_channel=0),
            K.MWTone("mw1", "mw1_phase", freq=1.0, power=-10.0, awg_channel=1),
        ],
        analog_channels=(0, 1),
        num_logical_mw=2,
    )

    result = K.render_flat(blocks, pg_freq=4.0, rate=4.0, params=params, granularity=(1, 1))

    phase = 2.0 * np.pi * np.arange(4) / 4.0
    np.testing.assert_allclose(result.analog[0], np.sin(phase), atol=1e-15)
    np.testing.assert_allclose(result.analog[1], np.sin(phase + np.pi / 2.0), atol=1e-15)
    assert result.amplitude_mV[0] == pytest.approx(100.0)
    assert result.amplitude_mV[1] == pytest.approx(100.0)


def test_render_flat_pads_to_granularity_and_encodes_padding_low():
    blocks = Blocks([Block("digital", [("laser", 2), (None, 2)])])

    result = K.render_flat(
        blocks,
        pg_freq=1.0,
        rate=1.0,
        params=K.RenderParams(tones=[], analog_channels=(1,), num_logical_mw=1),
        granularity=(8, 4),
    )

    assert result.rendered_samples == 4
    assert result.actual_samples == 8
    np.testing.assert_array_equal(result.analog[1], np.zeros(8))
    assert result.digital_rle()["laser"] == [(True, 2), (False, 6)]


def test_render_flat_raises_when_positive_interval_collapses():
    blocks = Blocks([Block("collapse", [("laser", 1), (None, 1)])])

    with pytest.raises(ValueError, match="positive interval collapsed"):
        K.render_flat(
            blocks,
            pg_freq=2.0,
            rate=1.0,
            params=K.RenderParams(tones=[]),
            granularity=(1, 1),
        )


@pytest.mark.parametrize(
    ("blocks", "params", "message"),
    [
        (
            Blocks([Block("unknown", [("mw_i", 1)])]),
            K.RenderParams(tones=[]),
            "unknown channel",
        ),
        (
            Blocks([Block("logical", [("mw1", 1)])]),
            K.RenderParams(
                tones=[K.MWTone("mw1", "mw1_phase", 1.0, -10.0)],
                num_logical_mw=1,
            ),
            "invalid logical MW channels",
        ),
    ],
)
def test_render_flat_rejects_invalid_channels(blocks, params, message):
    with pytest.raises(ValueError, match=message):
        K.render_flat(blocks, pg_freq=2.0, rate=2.0, params=params, granularity=(1, 1))


def test_renderer_renders_and_uploads_two_channels():
    awg = RecordingAWG()
    renderer = AWGRenderer(awg, channels=(1, 0))
    blocks = Blocks(
        [
            Block(
                "pulse",
                [
                    (("mw", "laser", A("mw_phase", 0.0)), 2),
                    (("mw1", "trigger", A("mw1_phase", 90.0)), 2),
                    (None, 4),
                ],
            )
        ]
    )
    tones = [
        K.MWTone("mw", "mw_phase", 1.0, -10.0, awg_channel=0),
        K.MWTone("mw1", "mw1_phase", 2.0, -10.0, awg_channel=1),
    ]

    meta = renderer.render(
        blocks,
        pg_freq=8.0,
        tones=tones,
        num_logical_mw=2,
        params={"awg": {"rate": 8.0}},
        bounds=make_bounds(granularity=(8, 8)),
    )

    assert meta == {
        "form": "waveforms",
        "analog_channels": (0, 1),
        "amplitude_mV": {0: 100, 1: 100},
        "sample_rate": 8.0,
        "rendered_samples": 8,
        "actual_samples": 8,
        "rendered_duration": 1.0,
        "actual_duration": 1.0,
        "max_rounding_error_sec": 0.0,
    }
    assert renderer.upload(trigger_type=TriggerType.SOFTWARE, n_runs=3)
    assert awg.amplitudes == [(0, 100), (1, 100)]
    assert len(awg.waveforms) == 1
    upload = awg.waveforms[0]
    assert tuple(upload["analog"]) == (0, 1)
    assert upload["digital"]["laser"] == [(True, 2), (False, 6)]
    assert upload["digital"]["trigger"] == [(False, 2), (True, 2), (False, 4)]
    assert upload["rate"] == 8.0
    assert upload["trigger_type"] == TriggerType.SOFTWARE
    assert upload["n_runs"] == 3
    assert renderer.get_meta_data()["trigger_type"] == str(TriggerType.SOFTWARE)
    assert renderer.get_meta_data()["n_runs"] == 3


def test_renderer_validates_amplitude_and_total_memory():
    blocks = Blocks([Block("pulse", [(("mw", A("mw_phase", 0.0)), 8)])])
    tones = [K.MWTone("mw", "mw_phase", 1.0, -10.0)]

    with pytest.raises(ValueError, match="exceeds the amplitude limit"):
        AWGRenderer(RecordingAWG()).render(
            blocks,
            pg_freq=8.0,
            tones=tones,
            num_logical_mw=1,
            params={"awg": {"rate": 8.0}},
            bounds=make_bounds(amplitude_mV=(1, 99)),
        )

    with pytest.raises(ValueError, match="across 2 channels"):
        AWGRenderer(RecordingAWG(), channels=(0, 1)).render(
            blocks,
            pg_freq=8.0,
            tones=tones,
            num_logical_mw=1,
            params={"awg": {"rate": 8.0}},
            bounds=make_bounds(memory_samples=15),
        )


def test_renderer_validates_digital_min_duration_across_replay_boundary():
    renderer = AWGRenderer(RecordingAWG())
    bounds = make_bounds(digital_min_duration=2e-9)

    with pytest.raises(ValueError, match="shorter than digital_min_duration"):
        renderer.render(
            Blocks([Block("short", [("laser", 1), (None, 7)])]),
            pg_freq=1e9,
            tones=[],
            num_logical_mw=1,
            params={"awg": {"rate": 1e9}},
            bounds=bounds,
        )

    meta = renderer.render(
        Blocks([Block("wrapped", [(None, 1), ("laser", 6), (None, 1)])]),
        pg_freq=1e9,
        tones=[],
        num_logical_mw=1,
        params={"awg": {"rate": 1e9}},
        bounds=bounds,
    )
    assert meta["actual_samples"] == 8


def test_renderer_upload_requires_render_and_propagates_instrument_failure():
    awg = RecordingAWG()
    renderer = AWGRenderer(awg)
    assert not renderer.upload()

    renderer.render(
        Blocks([Block("digital", [("laser", 8)])]),
        pg_freq=8.0,
        tones=[],
        num_logical_mw=1,
        params={"awg": {"rate": 8.0}},
        bounds=make_bounds(),
    )
    awg.configure_result = False
    assert not renderer.upload()


@pytest.mark.parametrize("remove_file", [True, False])
def test_renderer_file_transport_and_cleanup(tmp_path, remove_file):
    awg = RecordingAWG()
    awg.file_transport_dir = str(tmp_path)
    renderer = AWGRenderer(
        awg,
        channels=(0, 1),
        file_transport_dir=str(tmp_path),
        remove_transport_file=remove_file,
    )
    renderer.render(
        Blocks([Block("digital", [("laser", 4), (None, 4)])]),
        pg_freq=8.0,
        tones=[],
        num_logical_mw=1,
        params={"awg": {"rate": 8.0}},
        bounds=make_bounds(),
    )

    assert renderer.upload(file_transport=True)
    assert renderer.get_meta_data()["transport"] == "file"
    upload = awg.file_waveforms[0]
    assert tuple(upload["analog"]) == (0, 1)
    assert upload["digital"]["laser"] == [(True, 4), (False, 4)]
    assert (tmp_path / upload["file_name"]).exists() is (not remove_file)


def test_renderer_file_transport_cleanup_on_failure(tmp_path):
    awg = RecordingAWG()
    awg.file_transport_dir = str(tmp_path)
    awg.configure_result = False
    renderer = AWGRenderer(awg, file_transport_dir=str(tmp_path), remove_transport_file=True)
    renderer.render(
        Blocks([Block("digital", [(None, 8)])]),
        pg_freq=8.0,
        tones=[],
        num_logical_mw=1,
        params={"awg": {"rate": 8.0}},
        bounds=make_bounds(),
    )

    assert not renderer.upload(file_transport=True)
    assert not list(tmp_path.iterdir())
