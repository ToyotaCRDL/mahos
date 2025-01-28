#!/usr/bin/env python3

"""
Worker for ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import time

import numpy as np

from ..msgs.odmr_msgs import ODMRData
from ..msgs import param_msgs as P
from ..msgs.inst.pg_msgs import Block, Blocks, TriggerType
from ..inst.sg_interface import SGInterface
from ..inst.pg_interface import PGInterface
from ..inst.pd_interface import PDInterface
from ..inst.daq_interface import ClockSourceInterface
from ..inst.overlay.odmr_sweeper_interface import ODMRSweeperInterface
from ..util.conf import PresetLoader
from .common_worker import Worker


_MOD_LABELS = ["iq_ext", "am_ext", "fm_ext", "iq_int", "am_int", "fm_int"]


class SweeperBase(Worker):
    def data_msg(self) -> ODMRData:
        return self.data

    def is_finished(self) -> bool:
        if not self.data.has_params() or not self.data.has_data():
            return False
        if self.data.params.get("sweeps", 0) <= 0:
            return False  # no sweeps limit defined.
        return self.data.sweeps() >= self.data.params["sweeps"]

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        params = P.unwrap(params)
        if params["start"] >= params["stop"]:
            self.logger.error("stop must be greater than start")
            return False
        return True

    def get_param_dict_labels(self) -> list:
        return ["cw", "pulse"] + _MOD_LABELS

    def _make_param_dict(self, label, bounds) -> P.ParamDict[str, P.PDValue] | None:
        if label in ["cw"] + _MOD_LABELS:
            timing = P.ParamDict(
                time_window=P.FloatParam(
                    self.conf.get("time_window", 10e-3), 0.1e-3, 1.0, unit="s", SI_prefix=True
                ),
                gate_delay=P.FloatParam(
                    self.conf.get("gate_delay", 0.0), 0.0, 1.0, unit="s", SI_prefix=True
                ),
            )
        elif label == "pulse":
            timing = P.ParamDict(
                laser_delay=P.FloatParam(
                    100e-9, 0.0, 1e-3, unit="s", SI_prefix=True, doc="delay before laser"
                ),
                laser_width=P.FloatParam(
                    300e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="width of laser"
                ),
                mw_delay=P.FloatParam(
                    1e-6,
                    0.0,
                    1e-3,
                    unit="s",
                    SI_prefix=True,
                    doc="delay for microwave (>= trigger_width)",
                ),
                mw_width=P.FloatParam(
                    1e-6, 0.0, 1e-3, unit="s", SI_prefix=True, doc="width of microwave"
                ),
                trigger_width=P.FloatParam(
                    100e-9,
                    0.0,
                    10e-6,
                    unit="s",
                    SI_prefix=True,
                    doc="width of trigger (<= mw_delay)",
                ),
                burst_num=P.IntParam(100, 1, 100_000, doc="number of bursts at each freq."),
            )
        else:
            self.logger.error(f"Unknown param dict label: {label}")
            return None

        if bounds is None:
            self.logger.error("Could not get SG bounds.")
            return None
        f_min, f_max = bounds["freq"]
        p_min, p_max = bounds["power"]
        f_start = max(min(self.conf.get("start", 2.74e9), f_max), f_min)
        f_stop = max(min(self.conf.get("stop", 3.00e9), f_max), f_min)
        d = P.ParamDict(
            start=P.FloatParam(f_start, f_min, f_max),
            stop=P.FloatParam(f_stop, f_min, f_max),
            num=P.IntParam(self.conf.get("num", 101), 2, 10000),
            power=P.FloatParam(self.conf.get("power", p_min), p_min, p_max),
            sweeps=P.IntParam(0, 0, 1_000_000_000),
            timing=timing,
            background=P.BoolParam(False, doc="take background data"),
            delay=P.FloatParam(
                0.0,
                0.0,
                1.0,
                unit="s",
                SI_prefix=True,
                doc="delay after PG trigger before the measurement",
            ),
            background_delay=P.FloatParam(
                0.0,
                0.0,
                1.0,
                unit="s",
                SI_prefix=True,
                doc="delay between normal and background (reference) measurements",
            ),
            resume=P.BoolParam(False),
            continue_mw=P.BoolParam(False),
            ident=P.UUIDParam(optional=True, enable=False),
        )

        mod = P.ParamDict()
        if label in ("am_ext", "am_int"):
            mod["am_depth"] = P.FloatParam(self.conf.get("am_depth", 0.1), doc="depth of AM")
            mod["am_log"] = P.BoolParam(
                self.conf.get("am_log", False), doc="True indicates log scale AM depth"
            )
            if label == "am_int":
                mod["am_rate"] = P.FloatParam(
                    self.conf.get("am_rate", 400.0),
                    unit="Hz",
                    SI_prefix=True,
                    doc="rate (baseband frequency) of AM",
                )
        elif label in ("fm_ext", "fm_int"):
            mod["fm_deviation"] = P.FloatParam(
                self.conf.get("fm_deviation", 1e3),
                unit="Hz",
                SI_prefix=True,
                doc="deviation of FM",
            )
            if label == "fm_int":
                mod["fm_rate"] = P.FloatParam(
                    self.conf.get("fm_rate", 400.0),
                    unit="Hz",
                    SI_prefix=True,
                    doc="rate (baseband frequency) of FM",
                )
        # TODO: more additional params for iq_int, am_int, and fm_int?
        d["mod"] = mod

        return d


class SweeperOverlay(SweeperBase):
    """Sweeper using Overlay.

    Refer to :mod:`mahos.inst.overlay.odmr_sweeper` for docs of target overlay.

    """

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger, conf)
        self.sweeper = ODMRSweeperInterface(cli, "sweeper")
        self.add_instruments(self.sweeper)

        self._class_name = cli.class_name("sweeper")

        self.data = ODMRData()

    def get_param_dict_labels(self) -> list[str]:
        if self._class_name.startswith("ODMRSweeperCommand"):
            return ["cw"] + _MOD_LABELS
        else:
            return ["cw", "pulse"] + _MOD_LABELS

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        if self._class_name.startswith("ODMRSweeperCommand") and label == "pulse":
            return None

        bounds = self.sweeper.get_bounds()
        if bounds is None:
            self.logger.error("Failed to get bounds from sweeper.")
            return None

        d = self._make_param_dict(label, bounds)
        d["pd"] = self.sweeper.get_param_dict("pd")

        if label != "pulse" and self._class_name.endswith("AnalogPD"):
            d["timing"] = P.ParamDict(
                time_window=P.FloatParam(
                    self.conf.get("time_window", 10e-3), 0.1e-3, 1.0, unit="s", SI_prefix=True
                ),
            )
        elif label != "pulse" and self._class_name.endswith("AnalogPDMM"):
            d["timing"] = P.ParamDict()

        return d

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str = ""
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)
        success = self.sweeper.lock()

        if params is not None:
            success &= self.sweeper.configure(params, label)
        if not success:
            return self.fail_with_release("Error configuring sweeper.")

        success &= self.sweeper.start()
        if not success:
            return self.fail_with_release("Error starting sweeper.")

        if params is not None and not params["resume"]:
            # new measurement.
            self.data = ODMRData(params, label)
            self.data.start()
            self.data.yunit = self.sweeper.get_unit()
            self.logger.info("Started sweeper.")
        else:
            # resume.
            self.data.update_params(params)
            self.data.resume()
            self.logger.info("Resuming sweeper.")

        return True

    def _append_line_nobg(self, data, line):
        if data is None:
            return np.array(line, ndmin=2).T
        else:
            return np.append(data, np.array(line, ndmin=2).T, axis=1)

    def _append_line_bg(self, data, bg_data, line):
        l_data = line[0::2]
        l_bg = line[1::2]
        if data is None:
            return np.array(l_data, ndmin=2).T, np.array(l_bg, ndmin=2).T
        else:
            return (
                np.append(data, np.array(l_data, ndmin=2).T, axis=1),
                np.append(bg_data, np.array(l_bg, ndmin=2).T, axis=1),
            )

    def append_line(self, line):
        if not self.data.measure_background():
            self.data.data = self._append_line_nobg(self.data.data, line)
        else:
            self.data.data, self.data.bg_data = self._append_line_bg(
                self.data.data, self.data.bg_data, line
            )

    def work(self):
        if not self.data.running:
            return  # or raise Error?

        line = self.sweeper.get_line()
        if line is None:
            self.logger.error("Got None from sweeper.get_line()")
            return

        self.append_line(line)

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.sweeper.stop() and self.sweeper.release()

        self.data.finalize()
        if success:
            self.logger.info("Stopped sweeper.")
        else:
            self.logger.error("Error stopping sweeper.")
        return success


class Sweeper(SweeperBase):
    """Worker for fast ODMR sweep using mutual triggering between SG and PG.

    :param sweeper.pd_clock: DAQ terminal name for PD's clock (gate)
    :type sweeper.pd_clock: str
    :param sweeper.pd_names: (default: ["pd0", "pd1"]) PD names in target.servers.
    :type sweeper.pd_names: list[str]
    :param sweeper.pd_analog: (default: False) set True if PD is AnalogIn-based.
    :type sweeper.pd_analog: bool
    :param sweeper.channel_remap: mapping to fix default channel names.
    :type sweeper.channel_remap: dict[str | int, str | int]

    :param sweeper.start_delay: (default: 0.0) delay time (sec.) before starting SG/PG output.
    :type sweeper.start_delay: float
    :param sweeper.sg_first: (has preset) if True, turn on SG first and PG second.
        This mode is for SGs which won't start the point sweep by software command.
    :type sweeper.sg_first: bool

    :param sweeper.pg_freq_cw: (has preset) PG frequency for CW mode.
    :type sweeper.pg_freq_cw: float
    :param sweeper.pg_freq_pulse: (has preset) PG frequency for Pulse mode.
    :type sweeper.pg_freq_pulse: float
    :param sweeper.minimum_block_length: (has preset) minimum block length in generated blocks
    :type sweeper.minimum_block_length: int
    :param sweeper.block_base: (has preset) block base granularity of pulse generator.
    :type sweeper.block_base: int

    :param sweeper.start: (default param) start frequency in Hz.
    :type sweeper.start: float
    :param sweeper.stop: (default param) stop frequency in Hz.
    :type sweeper.stop: float
    :param sweeper.num: (default param) number of frequency points.
    :type sweeper.num: int
    :param sweeper.power: (default param) SG output power in dBm.
    :type sweeper.power: float
    :param sweeper.time_window: (default param) time window for cw mode.
    :type sweeper.time_window: float
    :param sweeper.pd_rate: (default param) analog PD sampling rate.
    :type sweeper.pd_rate: float

    """

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger, conf)
        self.load_pg_conf_preset(cli)
        self.load_sg_conf_preset(cli)

        self.sg = SGInterface(cli, "sg")
        self.pg = PGInterface(cli, "pg")
        self.pd_names = self.conf.get("pd_names", ["pd0", "pd1"])
        self.pds = [PDInterface(cli, n) for n in self.pd_names]
        self._pd_analog = self.conf.get("pd_analog", False)
        if self._pd_analog:
            self.clock = ClockSourceInterface(cli, self.conf.get("clock_name", "clock"))
        else:
            self.clock = None
        self.add_instruments(self.sg, self.pg, *self.pds)

        self.check_required_conf(["pd_clock", "block_base", "minimum_block_length"])
        self._pd_clock = self.conf["pd_clock"]
        self._pd_data_transfer = self.conf.get("pd_data_transfer")
        self._minimum_block_length = self.conf["minimum_block_length"]
        self._block_base = self.conf["block_base"]
        self._start_delay = self.conf.get("start_delay", 0.0)
        self._sg_first = self.conf.get("sg_first", False)
        self._channel_remap = self.conf.get("channel_remap")
        self._continue_mw = False

        self.data = ODMRData()

    def load_pg_conf_preset(self, cli):
        loader = PresetLoader(self.logger, PresetLoader.Mode.FORWARD)
        loader.add_preset(
            "DTG",
            [
                ("block_base", 4),
                ("pg_freq_cw", 1.0e6),
                ("pg_freq_pulse", 2.0e9),
                ("minimum_block_length", 1000),
            ],
        )
        loader.add_preset(
            "PulseStreamer",
            [
                ("block_base", 8),
                ("pg_freq_cw", 1.0e9),
                ("pg_freq_pulse", 1.0e9),
                ("minimum_block_length", 1),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("pg"))

    def load_sg_conf_preset(self, cli):
        loader = PresetLoader(self.logger, PresetLoader.Mode.FORWARD)
        loader.add_preset(
            "N5182B",
            [
                ("sg_first", False),
            ],
        )
        loader.add_preset(
            "MG3710E",
            [
                ("sg_first", True),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("sg"))

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        d = self._make_param_dict(label, self.sg.get_bounds())
        if self._pd_analog:
            d["pd"] = P.ParamDict()
            d["pd"]["rate"] = P.FloatParam(
                self.conf.get("pd_rate", 400e3), 1e3, 10000e3, doc="PD sampling rate"
            )
            lb, ub = self.conf.get("pd_bounds", (-10.0, 10.0))
            d["pd"]["bounds"] = [
                P.FloatParam(lb, -10.0, 10.0, unit="V"),
                P.FloatParam(ub, -10.0, 10.0, unit="V"),
            ]
        return d

    def configure_sg(self, params: dict, label: str) -> bool:
        p = params
        success = self.sg.configure_point_trig_freq_sweep(
            p["start"], p["stop"], p["num"], p["power"]
        )
        mod = params.get("mod", {})
        if label == "iq_ext":
            success &= self.sg.configure_iq_ext()
        elif label == "iq_int":
            success &= self.sg.configure_iq_int()
        elif label == "fm_ext":
            success &= self.sg.configure_fm_ext(mod["fm_deviation"])
        elif label == "fm_int":
            success &= self.sg.configure_fm_int(mod["fm_deviation"], mod["fm_rate"])
        elif label == "am_ext":
            success &= self.sg.configure_am_ext(mod["am_depth"], mod["am_log"])
        elif label == "am_int":
            success &= self.sg.configure_am_int(mod["am_depth"], mod["am_log"], mod["am_rate"])
        success &= self.sg.get_opc()
        return success

    def _adjust_block(self, block: Block, index: int):
        """Mutate block so that block's total_length is integer multiple of block base."""

        duration = block.total_length()
        if M := duration % self._block_base:
            ch, d = block.pattern[index].channels, block.pattern[index].duration
            block.pattern[index] = (ch, d + self._block_base - M)

    def configure_pg_CW_analog(self, params: dict) -> bool:
        freq = self.conf["pg_freq_cw"]
        # gate / trigger pulse width
        unit = round(freq * 1.0e-6)
        window = round(freq * params["timing"]["time_window"])
        gate_delay = round(freq * params["timing"].get("gate_delay", 0.0))
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        background = params.get("background", False)
        if background and gate_delay:
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    (("laser", "mw"), gate_delay),
                    (("laser", "mw", "gate"), unit),
                    # As measurement window is defined by DAQ sampling side,
                    # here we give long enough laser / mw pulse width (no "window - unit" below).
                    (("laser", "mw"), window),
                    (None, max(unit, bg_delay)),
                    ("laser", gate_delay),
                    (("laser", "gate"), unit),
                    ("laser", window),
                    ("trigger", unit),
                ],
                trigger=True,
            )
        elif background:  # no gate_delay
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    ("gate", unit),
                    (("laser", "mw"), window),
                    (None, max(unit, bg_delay)),
                    ("gate", unit),
                    ("laser", window),
                    ("trigger", unit),
                ],
                trigger=True,
            )
        elif gate_delay:  # no background
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    (("laser", "mw"), gate_delay),
                    (("laser", "mw", "gate"), unit),
                    (("laser", "mw"), window),
                    ("trigger", unit),
                ],
                trigger=True,
            )
        else:  # no gate_delay, no background
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    ("gate", unit),
                    (("laser", "mw"), window),
                    ("trigger", unit),
                ],
                trigger=True,
            )
        self._adjust_block(b, 0)
        blocks = Blocks([b])
        if self._channel_remap is not None:
            blocks = blocks.replace(self._channel_remap)
        blocks = blocks.simplify()
        return self.pg.configure_blocks(
            blocks, freq, trigger_type=TriggerType.HARDWARE_RISING, n_runs=1
        )

    def configure_pg_CW_apd(self, params: dict) -> bool:
        freq = self.conf["pg_freq_cw"]
        # gate / trigger pulse width
        unit = round(freq * 1.0e-6)
        window = round(freq * params["timing"]["time_window"])
        gate_delay = round(freq * params["timing"].get("gate_delay", 0.0))
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        background = params.get("background", False)
        if background and gate_delay:
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    (("laser", "mw"), gate_delay),
                    # here we define measurement window using interval between gate pulses
                    (("laser", "mw", "gate"), unit),
                    (("laser", "mw"), window - unit),
                    (("laser", "mw", "gate"), unit),
                    (None, max(unit, bg_delay)),
                    ("laser", gate_delay),
                    (("laser" "gate"), unit),
                    ("laser", window - unit),
                    (("laser" "gate"), unit),
                    ("trigger", unit),
                ],
                trigger=True,
            )
        elif background:  # no gate_delay
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    ("gate", unit),
                    # here we define measurement window using laser / mw pulse width
                    # (no "window - unit" below)
                    (("laser", "mw"), window),
                    ("gate", unit),
                    (None, max(unit, bg_delay)),
                    ("gate", unit),
                    ("laser", window),
                    (("gate", "trigger"), unit),
                ],
                trigger=True,
            )
        elif gate_delay:  # no background
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    (("laser", "mw"), gate_delay),
                    (("laser", "mw", "gate"), unit),
                    (("laser", "mw"), window - unit),
                    (("laser", "mw", "gate"), unit),
                    ("trigger", unit),
                ],
                trigger=True,
            )
        else:  # no gate_delay, no background
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    ("gate", unit),
                    (("laser", "mw"), window),
                    (("gate", "trigger"), unit),
                ],
                trigger=True,
            )
        self._adjust_block(b, 0)
        blocks = Blocks([b])
        if self._channel_remap is not None:
            blocks = blocks.replace(self._channel_remap)
        blocks = blocks.simplify()
        return self.pg.configure_blocks(
            blocks, freq, trigger_type=TriggerType.HARDWARE_RISING, n_runs=1
        )

    def _make_blocks_pulse_apd_nobg(
        self, delay, laser_delay, laser_width, mw_delay, mw_width, trigger_width, burst_num
    ):
        min_len = self._minimum_block_length

        init = Block(
            "INIT",
            [
                (None, max(delay, min_len - laser_width - mw_delay)),
                ("laser", laser_width),
                ("gate", trigger_width),
                (None, mw_delay - trigger_width),
            ],
            trigger=True,
        )

        main = Block(
            "MAIN",
            [
                ("mw", mw_width),
                (None, laser_delay),
                ("laser", laser_width),
                (None, mw_delay),
            ],
            Nrep=burst_num,
        )

        final = Block(
            "FINAL",
            [
                (["gate", "trigger"], trigger_width),
                (None, max(0, min_len - trigger_width)),
            ],
        )

        for b in (init, final):
            self._adjust_block(b, 0)
        self._adjust_block(main, -1)
        blocks = Blocks([init, main, final])
        if self._channel_remap is not None:
            blocks = blocks.replace(self._channel_remap)
        return blocks.simplify()

    def _make_blocks_pulse_apd_bg(
        self,
        delay,
        bg_delay,
        laser_delay,
        laser_width,
        mw_delay,
        mw_width,
        trigger_width,
        burst_num,
    ):
        min_len = self._minimum_block_length

        init = Block(
            "INIT",
            [
                (None, max(delay, min_len - laser_width - mw_delay)),
                ("laser", laser_width),
                ("gate", trigger_width),
                (None, mw_delay - trigger_width),
            ],
            trigger=True,
        )

        main = Block(
            "MAIN",
            [
                ("mw", mw_width),
                (None, laser_delay),
                ("laser", laser_width),
                (None, mw_delay),
            ],
            Nrep=burst_num,
        )

        final = Block(
            "FINAL",
            [
                ("gate", trigger_width),
                (None, max(0, min_len - trigger_width)),
            ],
        )

        init_bg = Block(
            "INIT-BG",
            [
                (None, max(bg_delay, min_len - laser_width - mw_delay)),
                ("laser", laser_width),
                ("gate", trigger_width),
                (None, mw_delay - trigger_width),
            ],
        )

        main_bg = Block(
            "MAIN-BG",
            [
                (None, mw_width),
                (None, laser_delay),
                ("laser", laser_width),
                (None, mw_delay),
            ],
            Nrep=burst_num,
        )

        final_bg = Block(
            "FINAL-BG",
            [
                (["gate", "trigger"], trigger_width),
                (None, max(0, min_len - trigger_width)),
            ],
        )

        for b in (init, final, init_bg, final_bg):
            self._adjust_block(b, 0)
        for b in (main, main_bg):
            self._adjust_block(b, -1)
        blocks = Blocks([init, main, final, init_bg, main_bg, final_bg])
        if self._channel_remap is not None:
            blocks = blocks.replace(self._channel_remap)
        return blocks.simplify()

    def configure_pg_pulse_apd(self, params: dict) -> bool:
        freq = self.conf["pg_freq_pulse"]
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        laser_delay, laser_width, mw_delay, mw_width, trigger_width = [
            round(params["timing"][k] * freq)
            for k in ("laser_delay", "laser_width", "mw_delay", "mw_width", "trigger_width")
        ]
        burst_num = params["timing"]["burst_num"]

        if mw_delay < trigger_width:
            self.logger.error("mw_delay >= trigger_width must be satisfied.")
            return False

        if params.get("background", False):
            blocks = self._make_blocks_pulse_apd_bg(
                delay,
                bg_delay,
                laser_delay,
                laser_width,
                mw_delay,
                mw_width,
                trigger_width,
                burst_num,
            )
        else:
            blocks = self._make_blocks_pulse_apd_nobg(
                delay, laser_delay, laser_width, mw_delay, mw_width, trigger_width, burst_num
            )

        return self.pg.configure_blocks(
            blocks, freq, trigger_type=TriggerType.HARDWARE_RISING, n_runs=1
        )

    def configure_pg(self, params: dict, label: str) -> bool:
        if not (self.pg.stop() and self.pg.clear()):
            return False
        if self._pd_analog:
            if label != "pulse":
                return self.configure_pg_CW_analog(params)
            else:
                self.logger.error("Pulse for Analog PD is not implemented yet.")
                return False
        else:
            if label != "pulse":
                return self.configure_pg_CW_apd(params)
            else:
                return self.configure_pg_pulse_apd(params)

    def start_apd(self, params: dict, label: str) -> bool:
        if label != "pulse":
            time_window = params["timing"]["time_window"]
        else:
            # time_window is used to compute APD's count rate.
            # gate is opened for whole burst sequence, but meaning time window for APD
            # is just burst_num * laser_width.
            t = params["timing"]
            time_window = t["burst_num"] * t["laser_width"]

        # max. expected sampling rate. double expected freq due to gate mode.
        # this max rate is achieved if freq switching time was zero (it's non-zero in reality).
        rate = 2.0 / time_window
        num = params["num"]
        if params.get("background", False):
            num *= 2
        buffer_size = num * self.conf.get("buffer_size_coeff", 20)
        params_pd = {
            "clock": self._pd_clock,
            "cb_samples": num,
            "samples": buffer_size,
            "buffer_size": buffer_size,
            "rate": rate,
            "finite": False,
            "every": False,
            # drop the first line because it contains invalid data at the first point
            # (line will be [f_N, f_0, f_1, ..., f_N-1) in general, however,
            #  SG is unknown state at the first point of the first line)
            "drop_first": int(self._sg_first),
            "gate": True,
            "time_window": time_window,
        }

        success = all([pd.configure(params_pd) for pd in self.pds]) and all(
            [pd.start() for pd in self.pds]
        )
        return success

    def start_analog_pd(self, params: dict, label: str) -> bool:
        rate = params["pd"]["rate"]
        if label != "pulse":
            oversamp = round(params["timing"]["time_window"] * rate)
        else:
            # t = params["timing"]
            # oversamp = round(t["laser_width"] * rate * t["burst_num"])
            # won't reach here but just in case
            self.logger.error("Pulse for Analog PD is not implemented yet.")
            return False

        self.logger.info(f"Analog PD oversample: {oversamp}")

        params_clock = {
            "freq": rate,
            "samples": oversamp,
            "finite": True,
            "trigger_source": self._pd_clock,
            "trigger_dir": True,
            "retriggerable": True,
        }
        if not self.clock.configure(params_clock):
            return self.fail_with("failed to configure clock.")
        clock_pd = self.clock.get_internal_output()

        num = params["num"]
        if params.get("background", False):
            num *= 2
        buffer_size = num * self.conf.get("buffer_size_coeff", 20)
        params_pd = {
            "clock": clock_pd,
            "cb_samples": num,
            "samples": buffer_size,
            "buffer_size": buffer_size,
            "rate": rate,
            "finite": False,
            "every": False,
            # drop the first line because it contains invalid data at the first point
            # (line will be [f_N, f_0, f_1, ..., f_N-1) in general, however,
            #  SG is unknown state at the first point of the first line)
            "drop_first": int(self._sg_first),
            "clock_mode": True,
            "oversample": oversamp,
            "bounds": params["pd"].get("bounds", (-10.0, 10.0)),
        }
        if self._pd_data_transfer:
            params_pd["data_transfer"] = self._pd_data_transfer

        success = (
            all([pd.configure(params_pd) for pd in self.pds])
            and self.clock.start()
            and all([pd.start() for pd in self.pds])
        )
        return success

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str = ""
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)
            self._continue_mw = params.get("continue_mw", False)
        resume = params is None or params.get("resume")

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not resume:
            self.data = ODMRData(params, label)
        else:
            # TODO: check ident if resume?
            self.data.update_params(params)
        self.data.yunit = self.pds[0].get_unit()

        if not self.configure_sg(self.data.params, self.data.label):
            return self.fail_with_release("Failed to configure SG.")
        if not self.configure_pg(self.data.params, self.data.label):
            return self.fail_with_release("Failed to configure PG.")
        if self._pd_analog:
            if not self.start_analog_pd(self.data.params, self.data.label):
                return self.fail_with_release("Failed to start PD (Analog).")
        else:
            if not self.start_apd(self.data.params, self.data.label):
                return self.fail_with_release("Failed to start APD.")

        time.sleep(self._start_delay)

        if self._sg_first:
            if not (self.sg.set_output(True) and self.sg.start() and self.sg.get_opc()):
                return self.fail_with_release("Failed to start SG.")
            if not (self.pg.start() and self.pg.get_opc()):
                return self.fail_with_release("Failed to start PG.")
            self.pg.trigger()
        else:
            if not (self.pg.start() and self.pg.get_opc()):
                return self.fail_with_release("Failed to start PG.")
            if not (self.sg.set_output(True) and self.sg.start()):
                return self.fail_with_release("Failed to start SG.")

        if resume:
            self.data.resume()
            self.logger.info("Resumed sweeper.")
        else:
            self.data.start()
            self.logger.info("Started sweeper.")
        return True

    def _roll_line(self, line):
        """Fix the rolling of data due to sg_first operation.

        When sg_first is True, the data will be like
        [f_N, f0, f1, ..., f_N-1] instead of [f0, f1, ..., f_N].
        Fix the former to the latter by rolling the array.

        """

        if self._sg_first:
            return np.roll(line, -1)
        else:
            return line

    def _append_line_nobg(self, data, line):
        line = self._roll_line(line)
        if data is None:
            return np.array(line, ndmin=2).T
        else:
            return np.append(data, np.array(line, ndmin=2).T, axis=1)

    def _append_line_bg(self, data, bg_data, line):
        l_data = self._roll_line(line[0::2])
        l_bg = self._roll_line(line[1::2])
        if data is None:
            return np.array(l_data, ndmin=2).T, np.array(l_bg, ndmin=2).T
        else:
            return (
                np.append(data, np.array(l_data, ndmin=2).T, axis=1),
                np.append(bg_data, np.array(l_bg, ndmin=2).T, axis=1),
            )

    def append_line(self, line):
        if not self.data.measure_background():
            self.data.data = self._append_line_nobg(self.data.data, line)
        else:
            self.data.data, self.data.bg_data = self._append_line_bg(
                self.data.data, self.data.bg_data, line
            )

    def work(self):
        if not self.data.running:
            return  # or raise Error?

        lines = []
        for pd in self.pds:
            ls = pd.pop_block()
            if isinstance(ls, list):
                # PD has multi channel
                lines.extend(ls)
            else:
                # single channel, assume ls is np.ndarray
                lines.append(ls)

        line = np.sum(lines, axis=0)
        self.append_line(line)

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.sg.stop()

        if self._continue_mw:
            self.logger.warn("Skipping to turn off MW output")
        else:
            success &= self.sg.set_output(False)

        success &= all([pd.stop() for pd in self.pds])
        if self._pd_analog:
            success &= self.clock.stop()
        success &= self.pg.stop()
        success &= self.release_instruments()

        self.data.finalize()

        if success:
            self.logger.info("Stopped sweeper.")
        else:
            self.logger.error("Error stopping sweeper.")
        return success
