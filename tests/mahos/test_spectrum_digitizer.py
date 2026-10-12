#!/usr/bin/env python3

"""
Tests for Spectrum digitizer internals.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import numpy as np
import pytest

from mahos.inst.digitizer import Spectrum_AnalogIn
from mahos.util.queue import RollingQueue


class DummyUnits:
    s = 1.0
    V = 1.0
    mV = 1.0
    percent = 1.0


class DummyChannel:
    def amp(self, value, return_unit):
        self.amp_value = value
        return value

    def offset(self, value, return_unit):
        self.offset_value = value
        return value

    def convert_data(self, raw, *args, **kwargs):
        return raw


class DummyChannels(list):
    def __init__(self, card, card_enable):
        super().__init__([DummyChannel()])

    def path(self, value):
        self._path_value = value

    def termination(self, value):
        self._termination_value = value

    def coupling(self, value):
        self._coupling_value = value


class DummyCard:
    def reset(self):
        self.was_reset = True

    def card_mode(self, mode):
        self.mode = mode

    def timeout(self, timeout):
        self.timeout_value = timeout

    def loops(self, loops):
        self.loop_count = loops

    def start(self, *commands):
        self.start_commands = commands

    def stop(self, *commands):
        self.stop_commands = commands

    def close(self):
        pass


class DummyTransfer:
    def __init__(self, bytes_per_sample=2, channels=1):
        self.bytes_per_sample = bytes_per_sample
        self.channels = channels

    def samples_to_bytes(self, samples):
        return int(samples) * self.bytes_per_sample * self.channels

    def start_buffer_transfer(self, *commands):
        self.commands = commands


class DummyTriggeredTransfer(DummyTransfer):
    def __init__(self, card):
        super().__init__()
        self.card = card

    def averages(self, averages):
        return averages

    def allocate_buffer(self, segment_samples, num_segments):
        self.buffer = np.zeros((num_segments, segment_samples, 1))

    def notify_samples(self, samples):
        self.notified = samples

    def to_transfer_samples(self, samples):
        self.transfer_samples = samples

    def post_trigger(self, samples):
        self.post_trigger_samples = samples + getattr(self.card, "post_trigger_offset", 0)
        return self.post_trigger_samples


class DummySpcm:
    SPC_REC_FIFO_AVERAGE = 1
    SPC_REC_FIFO_MULTI = 2
    M2CMD_DATA_STARTDMA = 4
    M2CMD_DATA_STOPDMA = 8
    M2CMD_CARD_ENABLETRIGGER = 16
    M2CMD_CARD_FORCETRIGGER = 32
    COUPLING_DC = 0
    COUPLING_AC = 1
    units = DummyUnits()

    Channels = DummyChannels
    Multi = DummyTriggeredTransfer
    BlockAverage = DummyTriggeredTransfer


class DummyThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


def make_inst(conf=None):
    inst = Spectrum_AnalogIn("digitizer", {"mock": True, "lines": [0], **(conf or {})})
    inst._spcm = DummySpcm
    return inst


def make_triggered_params(**overrides):
    return {
        "trigger_source": "ext0",
        "cb_samples": 64,
        "samples": 192,
        "rate": 1.0e6,
        "finite": True,
        "block_samples": 32,
        **overrides,
    }


def configure_triggered(
    inst,
    monkeypatch,
    *,
    post_trigger_offset=0,
    expect_success=True,
    use_real_reset=False,
    **overrides,
):
    card = DummyCard()
    card.post_trigger_offset = post_trigger_offset
    params = make_triggered_params(**overrides)
    monkeypatch.setattr(inst, "_import_spcm", lambda: DummySpcm)
    if not use_real_reset:
        monkeypatch.setattr(inst, "_reset_card", lambda: card)
    monkeypatch.setattr(inst, "_open_card", lambda: card)
    monkeypatch.setattr(inst, "_setup_trigger", lambda _params: True)
    monkeypatch.setattr(inst, "_setup_clock", lambda _params: True)
    assert inst.configure_triggered(params) is expect_success
    return card, inst._transfer


@pytest.mark.parametrize(
    ("bounds", "expected_amp", "expected_offset"),
    [
        ((0.0, 0.6), 500.0, -60.0),
        ((-0.6, 0.0), 500.0, 60.0),
        ((-0.6, 0.6), 1000.0, 0.0),
        ((0.5, 1.0), 1000.0, -75.0),
        ((-1.0, -0.5), 1000.0, 75.0),
        ((-0.499, 0.501), 1000.0, 0.0),
    ],
)
def test_setup_channels_computes_offset_from_selected_amplitude(
    monkeypatch, bounds, expected_amp, expected_offset
):
    inst = make_inst()
    monkeypatch.setattr(inst, "_import_spcm", lambda: DummySpcm)
    monkeypatch.setattr(inst, "_open_card", lambda: object())

    assert inst._setup_channels({"bounds": bounds})
    channel = inst._channels[0]
    assert channel.amp_value == expected_amp
    assert channel.offset_value == expected_offset


def test_align_notify_samples_uses_transfer_byte_size():
    inst = make_inst()
    inst._transfer = DummyTransfer(bytes_per_sample=2, channels=1)

    assert inst._align_notify_samples(5) == 2048

    inst._transfer = DummyTransfer(bytes_per_sample=2, channels=2)

    assert inst._align_notify_samples(5) == 1024


def test_align_notify_samples_preserves_segment_boundary():
    inst = make_inst()
    inst._transfer = DummyTransfer(bytes_per_sample=2, channels=1)

    samples = inst._align_notify_samples(1500, step_samples=512)

    assert samples >= 1500
    assert samples % 512 == 0
    assert inst._samples_to_bytes(samples) % 4096 == 0


def test_align_segment_samples_uses_physical_byte_alignment():
    inst = make_inst({"segment_alignment_bytes": 4096})
    inst._transfer = DummyTransfer(bytes_per_sample=2, channels=1)

    assert inst._align_segment_samples(2032) == 2048
    assert inst._align_segment_samples(2048) == 4096

    inst._transfer = DummyTransfer(bytes_per_sample=2, channels=2)

    assert inst._align_segment_samples(1008) == 1024


def test_align_segment_samples_can_disable_byte_alignment():
    inst = make_inst({"notify_alignment_bytes": 4096, "segment_alignment_bytes": 1})
    inst._transfer = DummyTransfer(bytes_per_sample=2, channels=1)

    assert inst._align_segment_samples(16) == 32


def test_align_segment_samples_uses_configured_hardware_constraints():
    inst = make_inst(
        {
            "segment_alignment_bytes": 1,
            "trigger_sample_granularity": 64,
            "min_segment_samples": 128,
            "min_pre_trigger_samples": 32,
        }
    )
    inst._transfer = DummyTransfer(bytes_per_sample=2, channels=1)

    assert inst._align_segment_samples(64) == 128
    assert inst._align_segment_samples(128) == 192


def test_align_buffer_samples_is_notify_multiple():
    inst = make_inst()

    assert inst._align_buffer_samples(5000, 2048) == 6144


def test_finite_notify_samples_is_aligned_exact_divisor():
    inst = make_inst({"notify_alignment_bytes": 256})
    inst._transfer = DummyTransfer(bytes_per_sample=2)

    assert inst._finite_notify_samples(64, 256, 32) == 128


def test_finite_notify_samples_rejects_transfer_without_valid_divisor():
    inst = make_inst({"notify_alignment_bytes": 256})
    inst._transfer = DummyTransfer(bytes_per_sample=2)

    with pytest.raises(ValueError, match="no valid notify size"):
        inst._finite_notify_samples(64, 192, 32)


def test_convert_segment_removes_pre_trigger_samples():
    inst = make_inst()
    inst._channels = [DummyChannel()]
    inst._averages = 1
    inst._logical_segment_samples = 16
    inst._segment_samples = 32
    inst._pre_trigger_samples = 16

    converted = inst._convert_segment(np.arange(32.0)[:, np.newaxis])

    np.testing.assert_allclose(converted[0], np.arange(16.0, 32.0))


def test_append_stream_samples_reblocks_aligned_notifications():
    inst = make_inst()
    inst.queue = RollingQueue(100)
    inst._line_num = 1
    inst._stamp = False
    inst._oversample = 2
    inst._stream_samples = 4
    inst._pending = [[]]

    inst._append_stream_samples([np.arange(10.0)])

    np.testing.assert_allclose(inst.pop_opt(), np.array([0.5, 2.5]))
    np.testing.assert_allclose(inst.pop_opt(), np.array([4.5, 6.5]))
    assert inst.pop_opt() is None
    np.testing.assert_allclose(inst._pending[0][0], np.array([8.0, 9.0]))


def test_append_stream_samples_handles_multi_channel_blocks():
    inst = make_inst({"lines": [0, 1]})
    inst.queue = RollingQueue(100)
    inst._line_num = 2
    inst._stamp = False
    inst._oversample = 1
    inst._stream_samples = 3
    inst._pending = [[], []]

    inst._append_stream_samples([np.array([1.0, 2.0]), np.array([10.0, 20.0])])
    assert inst.pop_opt() is None

    inst._append_stream_samples([np.array([3.0, 4.0]), np.array([30.0, 40.0])])
    data = inst.pop_opt()

    np.testing.assert_allclose(data[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(data[1], np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(inst._pending[0][0], np.array([4.0]))
    np.testing.assert_allclose(inst._pending[1][0], np.array([40.0]))


def test_append_stream_samples_uses_sample_derived_stamps(monkeypatch):
    inst = make_inst()
    inst.queue = RollingQueue(100)
    inst._line_num = 1
    inst._stamp = True
    inst._oversample = 1
    inst._sampling_rate = 100
    inst._stream_samples = 5
    inst._stream_epoch_ns = None
    inst._stream_emitted_samples = 0
    inst._pending = [[]]
    monkeypatch.setattr("mahos.inst.digitizer.spectrum.time.time_ns", lambda: 1_000_000_000)

    inst._note_stream_notification(12)
    inst._append_stream_samples([np.arange(12.0)])
    inst._note_stream_notification(12)
    inst._append_stream_samples([np.arange(12.0, 24.0)])

    items = inst.pop_all_opt()
    chunks, stamps = zip(*items)
    np.testing.assert_allclose(chunks[0], np.arange(5.0))
    np.testing.assert_allclose(chunks[1], np.arange(5.0, 10.0))
    np.testing.assert_allclose(chunks[2], np.arange(10.0, 15.0))
    np.testing.assert_allclose(chunks[3], np.arange(15.0, 20.0))
    np.testing.assert_array_equal(stamps, [930_000_000, 980_000_000, 1_030_000_000, 1_080_000_000])
    np.testing.assert_allclose(inst._pending[0][0], np.arange(20.0, 24.0))


@pytest.mark.parametrize("trigger_source", ["software", "soft"])
def test_configure_triggered_rejects_software_trigger(monkeypatch, trigger_source):
    inst = make_inst()
    errors = []
    params = make_triggered_params(trigger_source=trigger_source)
    monkeypatch.setattr(inst, "_reset_card", lambda: pytest.fail("card should not be reset"))
    monkeypatch.setattr(inst, "fail_with", lambda msg: errors.append(msg) or False)

    assert not inst.configure_triggered(params)
    assert errors == ["software trigger is not supported in triggered mode."]


@pytest.mark.parametrize("samples", [0, -64, 65])
def test_configure_triggered_rejects_invalid_finite_samples(monkeypatch, samples):
    inst = make_inst()
    errors = []
    params = make_triggered_params(samples=samples)
    monkeypatch.setattr(inst, "fail_with", lambda msg: errors.append(msg) or False)

    assert not inst.configure_triggered(params)
    assert errors


@pytest.mark.parametrize("block_samples", [1, 15, 17, 24])
def test_configure_triggered_rejects_invalid_logical_segment(monkeypatch, block_samples):
    inst = make_inst()
    errors = []
    params = make_triggered_params(samples=64, block_samples=block_samples)
    monkeypatch.setattr(inst, "_import_spcm", lambda: DummySpcm)
    monkeypatch.setattr(inst, "_reset_card", lambda: DummyCard())
    monkeypatch.setattr(inst, "fail_with", lambda msg: errors.append(msg) or False)

    assert not inst.configure_triggered(params)
    assert "logical segment samples" in errors[0]


def test_configure_triggered_uses_configured_min_post_trigger(monkeypatch):
    inst = make_inst({"min_post_trigger_samples": 64})
    errors = []
    params = make_triggered_params(samples=64, block_samples=32)
    monkeypatch.setattr(inst, "_reset_card", lambda: DummyCard())
    monkeypatch.setattr(inst, "fail_with", lambda msg: errors.append(msg) or False)

    assert not inst.configure_triggered(params)
    assert errors == [
        "derived logical segment samples must be an integer multiple of 16 and at least 64: 32"
    ]


@pytest.mark.parametrize(
    ("block_reduce_factor", "hardware_average", "reduce_factor"),
    [(2, False, 1), (2, True, 1), (1, False, 2)],
)
def test_configure_triggered_rejects_nondivisible_cb_samples(
    monkeypatch, block_reduce_factor, hardware_average, reduce_factor
):
    inst = make_inst()
    errors = []
    params = make_triggered_params(
        cb_samples=24,
        samples=24,
        block_reduce_factor=block_reduce_factor,
        hardware_average=hardware_average,
        reduce_factor=reduce_factor,
    )

    monkeypatch.setattr(inst, "_import_spcm", lambda: object())
    monkeypatch.setattr(inst, "_reset_card", lambda: object())
    monkeypatch.setattr(inst, "fail_with", lambda msg: errors.append(msg) or False)

    assert not inst.configure_triggered(params)
    assert errors == ["cb_samples must be integer multiple of block_samples."]


def test_configure_triggered_accepts_sixteen_logical_samples(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})

    _card, transfer = configure_triggered(
        inst, monkeypatch, cb_samples=32, samples=32, block_samples=16
    )

    assert inst._logical_segment_samples == 16
    assert inst._segment_samples == 32
    assert transfer.post_trigger_samples == 16
    assert inst._pre_trigger_samples == 16


def test_configure_triggered_rejects_mismatched_realized_post_trigger(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})

    _card, transfer = configure_triggered(
        inst, monkeypatch, post_trigger_offset=16, expect_success=False
    )

    assert transfer.post_trigger_samples == 48
    assert inst._mode == inst.Mode.UNCONFIGURED


def test_failed_triggered_reconfiguration_cannot_be_started(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})
    inst._mode = inst.Mode.STREAM

    _card, transfer = configure_triggered(
        inst,
        monkeypatch,
        post_trigger_offset=16,
        expect_success=False,
        use_real_reset=True,
    )

    assert transfer is not None
    assert inst._mode == inst.Mode.UNCONFIGURED
    assert not inst.start()


def test_configure_triggered_infinite_uses_zero_loops(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})
    card, transfer = configure_triggered(inst, monkeypatch, finite=False)

    assert card.loop_count == 0
    assert not hasattr(transfer, "transfer_samples")


def test_configure_triggered_finite_multi_counts_output_segments(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})
    card, transfer = configure_triggered(inst, monkeypatch)

    assert card.mode == DummySpcm.SPC_REC_FIFO_MULTI
    assert card.loop_count == 6
    assert transfer.transfer_samples == 288


def test_configure_triggered_finite_average_excludes_average_factor(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})
    card, transfer = configure_triggered(
        inst, monkeypatch, block_reduce_factor=5, hardware_average=True
    )

    assert card.mode == DummySpcm.SPC_REC_FIFO_AVERAGE
    assert card.loop_count == 6
    assert transfer.transfer_samples == 288


def test_configure_triggered_finite_multi_includes_software_block_factor(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})
    card, transfer = configure_triggered(
        inst, monkeypatch, block_reduce_factor=3, hardware_average=False
    )

    assert card.loop_count == 18
    assert transfer.transfer_samples == 864


def test_configure_triggered_increases_buffer_to_finite_notify_size(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 256})

    _card, transfer = configure_triggered(inst, monkeypatch, samples=256, buffer_size=64)

    assert transfer.notified == 256
    assert inst._buffer_samples == 256


def test_configure_triggered_rounds_buffer_up_to_whole_records(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})

    configure_triggered(inst, monkeypatch, buffer_size=100)

    assert inst._buffer_samples == 192


def test_triggered_fifo_status_distinguishes_logical_and_physical_samples(monkeypatch):
    inst = make_inst({"notify_alignment_bytes": 1})
    configure_triggered(inst, monkeypatch)

    status = inst.get_fifo_status()

    assert status["segment_samples"] == 48
    assert status["logical_segment_samples"] == 32
    assert status["pre_trigger_samples"] == 16
    assert status["post_trigger_samples"] == 32
    assert status["record_samples"] == 64


@pytest.mark.parametrize(
    ("trigger_source", "expected_commands"),
    [
        ("software", (DummySpcm.M2CMD_CARD_ENABLETRIGGER, DummySpcm.M2CMD_CARD_FORCETRIGGER)),
        ("ext0", (DummySpcm.M2CMD_CARD_ENABLETRIGGER,)),
    ],
)
def test_start_forces_only_software_trigger(monkeypatch, trigger_source, expected_commands):
    inst = make_inst()
    inst._card = DummyCard()
    inst._transfer = DummyTransfer()
    inst._mode = inst.Mode.STREAM
    inst._trigger_source = trigger_source
    monkeypatch.setattr("mahos.inst.digitizer.spectrum.threading.Thread", DummyThread)

    assert inst.start()
    assert inst._transfer.commands == (DummySpcm.M2CMD_DATA_STARTDMA,)
    assert inst._card.start_commands == expected_commands
    assert inst.stop()
    assert inst._card.stop_commands == (DummySpcm.M2CMD_DATA_STOPDMA,)
