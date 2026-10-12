#!/usr/bin/env python3

"""
Tests for pd_chop logic option in ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import logging

import numpy as np
import pytest

from mahos.msgs import param_msgs as P
from mahos_dq.inst.overlay.odmr_sweeper import ODMRSweeperCommandBase, ODMRSweeperPG
from mahos_dq.meas.odmr_pg import ODMRPGMixin
from mahos_dq.meas.odmr_worker import SweeperBase


def _bounds():
    return {"freq": (1.0, 10.0), "power": (-30.0, 10.0)}


def _pulse_params(background=False):
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
            "laser_delay": 5e-9,
            "laser_width": 10e-9,
            "mw_delay": 8e-9,
            "mw_width": 2e-9,
            "trigger_width": 2e-9,
            "mw_offset": 0.0,
            "burst_num": 3,
            "chop_delay": 12e-9,
            "chop_width": 4e-9,
        },
    }


class _PG:
    def configure_blocks(self, blocks, freq, **kwargs):
        self.blocks = blocks
        self.freq = freq
        return True


class _Builder(ODMRPGMixin):
    def __init__(self, pd_chop=True):
        self.conf = {"pg_freq_cw": 1e6, "pg_freq_pulse": 1e9}
        self.logger = logging.getLogger("test_odmr_chop")
        self._minimum_block_length = 1
        self._block_base = 1
        self._channel_remap = None
        self._pd_analog = False
        self._pd_trace = False
        self._pd_chop = pd_chop
        self.pg = _PG()


class _PD:
    def configure(self, params):
        self.params = params
        return True


def _rising_edges(blocks, channel):
    values = blocks.decode_digital(channel)
    return np.flatnonzero(np.diff(np.r_[0, values]) == 1)


def test_chop_parameter_dictionary_is_opt_in_for_apd():
    worker = SweeperBase.__new__(SweeperBase)
    worker.conf = {}
    worker.logger = logging.getLogger("test_odmr_chop_params")

    legacy = P.unwrap(worker._make_param_dict("pulse", _bounds(), False, False, False))
    chopped = P.unwrap(worker._make_param_dict("pulse", _bounds(), False, False, True))
    analog = P.unwrap(worker._make_param_dict("pulse", _bounds(), True, False, True))

    assert "chop_delay" not in legacy["timing"]
    assert chopped["timing"]["chop_delay"] == 0.0
    assert chopped["timing"]["chop_width"] == 100e-9
    assert "chop_delay" not in analog["timing"]


def test_odmr_sweeper_capability_is_returned_as_one_dictionary():
    command = ODMRSweeperCommandBase.__new__(ODMRSweeperCommandBase)
    command._closed = True
    command._pd_analog = True
    assert command.get("capability") == {
        "pd_analog": True,
        "pd_trace": False,
        "pd_chop": False,
    }

    pg = ODMRSweeperPG.__new__(ODMRSweeperPG)
    pg._closed = True
    pg._pd_analog = False
    pg._pd_trace = False
    pg._pd_chop = True
    assert pg.get("capability") == {
        "pd_analog": False,
        "pd_trace": False,
        "pd_chop": True,
    }


def test_chop_validation_allows_detector_window_after_laser_edge():
    builder = _Builder()
    params = _pulse_params()

    success, message, _ = builder.validate_pulse_params(params)

    assert success, message


def test_chop_validation_rejects_subtick_and_unit_overrun():
    builder = _Builder()
    params = _pulse_params()
    params["timing"]["chop_width"] = 0.4e-9

    success, message, _ = builder.validate_pulse_params(params)

    assert not success
    assert "at least one PG tick" in message

    params = _pulse_params()
    params["timing"]["chop_delay"] = 14e-9

    success, message, _ = builder.validate_pulse_params(params)

    assert success, message

    params["timing"]["chop_delay"] = 15e-9

    success, message, _ = builder.validate_pulse_params(params)

    assert not success
    assert "must not exceed" in message


def test_chop_is_overlaid_on_each_foreground_main_unit():
    builder = _Builder()
    params = _pulse_params()

    assert builder.configure_pg_pulse_apd(params, None)

    blocks = builder.pg.blocks
    assert [block.name for block in blocks] == ["INIT", "MAIN", "FINAL"]
    assert "chop" not in blocks[0].digital_channels()
    assert "chop" not in blocks[2].digital_channels()
    main = blocks[1]
    assert main.Nrep == params["timing"]["burst_num"]
    assert np.array_equal(_rising_edges(blocks, "chop"), [37, 62, 87])
    assert main.raw_channel_length("chop", True) == 4


def test_chop_is_overlaid_on_foreground_and_background_main_units():
    builder = _Builder()
    params = _pulse_params(background=True)

    assert builder.configure_pg_pulse_apd(params, None)

    blocks = builder.pg.blocks
    assert [block.name for block in blocks] == [
        "INIT",
        "MAIN",
        "FINAL",
        "INIT-BG",
        "MAIN-BG",
        "FINAL-BG",
    ]
    assert blocks[1].Nrep == blocks[4].Nrep == 3
    assert blocks[1].raw_channel_length("chop", True) == 4
    assert blocks[4].raw_channel_length("chop", True) == 4
    for index in (0, 2, 3, 5):
        assert "chop" not in blocks[index].digital_channels()


@pytest.mark.parametrize("background", (False, True))
def test_cw_apd_holds_chop_high_for_entire_pg_cycle(background):
    builder = _Builder()
    params = {
        "background": background,
        "delay": 3e-6,
        "background_delay": 2e-6,
        "final_delay": 4e-6,
        "timing": {"time_window": 10e-6, "gate_delay": 2e-6, "post_gate_delay": 3e-6},
    }

    assert builder.configure_pg_CW_apd(params, None)

    chop = builder.pg.blocks.decode_digital("chop")
    assert len(chop)
    assert all(chop)


def test_apd_count_rate_uses_chop_width_only_when_enabled():
    params = _pulse_params()
    builder = _Builder(pd_chop=True)
    legacy = _Builder(pd_chop=False)

    assert builder.apd_time_window(params, "pulse") == pytest.approx(12e-9)
    assert legacy.apd_time_window(params, "pulse") == pytest.approx(30e-9)

    sweeper = ODMRSweeperPG.__new__(ODMRSweeperPG)
    sweeper._closed = True
    sweeper._pd_chop = True
    sweeper._pd_clock = "gate-in"
    sweeper.conf = {"buffer_size_coeff": 5}
    sweeper.pds = [_PD()]

    assert sweeper.configure_apd(params, "pulse")
    assert sweeper.pds[0].params["time_window"] == pytest.approx(12e-9)
    assert sweeper.pds[0].params["rate"] == pytest.approx(2.0 / 12e-9)
