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
from ..util.conf import PresetLoader
from .common_worker import Worker


class Sweeper(Worker):
    """Worker for ODMR Sweep."""

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
        self._trigger_ch = self.conf.get("trigger_channel", "trigger")
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

    def get_param_dict_labels(self) -> list:
        return ["cw", "pulse"]

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        if label == "cw":
            timing = P.ParamDict(
                time_window=P.FloatParam(self.conf.get("time_window", 10e-3), 0.1e-3, 1.0)
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

        bounds = self.sg.get_bounds()
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
            sg_modulation=P.BoolParam(
                self.conf.get("sg_modulation", False), doc="enable external IQ modulation for SG"
            ),
            resume=P.BoolParam(False),
            continue_mw=P.BoolParam(False),
            ident=P.UUIDParam(optional=True, enable=False),
        )

        if self._pd_analog:
            d["pd_rate"] = P.FloatParam(
                self.conf.get("pd_rate", 400e3), 1e3, 10000e3, doc="PD sampling rate"
            )
            lb, ub = self.conf.get("pd_bounds", (-10.0, 10.0))
            d["pd_bounds"] = [
                P.FloatParam(lb, -10.0, 10.0, unit="V"),
                P.FloatParam(ub, -10.0, 10.0, unit="V"),
            ]
        return d

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        params = P.unwrap(params)
        if params["start"] >= params["stop"]:
            self.logger.error("stop must be greater than start")
            return False
        return True

    def configure_sg(self, params: dict) -> bool:
        p = params
        success = self.sg.configure_point_trig_freq_sweep(
            p["start"], p["stop"], p["num"], p["power"]
        )
        if params.get("sg_modulation", False):
            success &= (
                self.sg.set_modulation(True)
                and self.sg.set_dm_source("EXT")
                and self.sg.set_dm(True)
            )
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
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        if params.get("background", False):
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    ("gate", unit),
                    (("laser", "mw"), window),
                    (None, max(unit, bg_delay)),
                    ("gate", unit),
                    ("laser", window),
                    (self._trigger_ch, unit),
                ],
                trigger=True,
            )
        else:
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    ("gate", unit),
                    (("laser", "mw"), window),
                    (self._trigger_ch, unit),
                ],
                trigger=True,
            )
        self._adjust_block(b, 0)
        blocks = Blocks([b]).simplify()
        return self.pg.configure_blocks(
            blocks, freq, trigger_type=TriggerType.HARDWARE_RISING, n_runs=1
        )

    def configure_pg_CW_apd(self, params: dict) -> bool:
        freq = self.conf["pg_freq_cw"]
        # gate / trigger pulse width
        unit = round(freq * 1.0e-6)
        window = round(freq * params["timing"]["time_window"])
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        if params.get("background", False):
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    ("gate", unit),
                    (("laser", "mw"), window),
                    ("gate", unit),
                    (None, max(unit, bg_delay)),
                    ("gate", unit),
                    ("laser", window),
                    (("gate", self._trigger_ch), unit),
                ],
                trigger=True,
            )
        else:
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    ("gate", unit),
                    (("laser", "mw"), window),
                    (("gate", self._trigger_ch), unit),
                ],
                trigger=True,
            )
        self._adjust_block(b, 0)
        blocks = Blocks([b]).simplify()
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
                (["gate", self._trigger_ch], trigger_width),
                (None, max(0, min_len - trigger_width)),
            ],
        )

        for b in (init, final):
            self._adjust_block(b, 0)
        self._adjust_block(main, -1)
        return Blocks([init, main, final]).simplify()

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
                (["gate", self._trigger_ch], trigger_width),
                (None, max(0, min_len - trigger_width)),
            ],
        )

        for b in (init, final, init_bg, final_bg):
            self._adjust_block(b, 0)
        for b in (main, main_bg):
            self._adjust_block(b, -1)
        return Blocks([init, main, final, init_bg, main_bg, final_bg]).simplify()

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
            if label == "cw":
                return self.configure_pg_CW_analog(params)
            else:
                self.logger.error("Pulse for Analog PD is not implemented yet.")
                return False
        else:
            if label == "cw":
                return self.configure_pg_CW_apd(params)
            else:
                return self.configure_pg_pulse_apd(params)

    def start_apd(self, params: dict, label: str) -> bool:
        if label == "cw":
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
        rate = params["pd_rate"]
        if label == "cw":
            oversamp = round(params["timing"]["time_window"] * params["pd_rate"])
        else:
            # t = params["timing"]
            # oversamp = round(t["laser_width"] * params["pd_rate"] * t["burst_num"])
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
            "bounds": params.get("pd_bounds", (-10.0, 10.0)),
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

        if not self.configure_sg(self.data.params):
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

    def is_finished(self) -> bool:
        if not self.data.has_params() or not self.data.has_data():
            return False
        if self.data.params.get("sweeps", 0) <= 0:
            return False  # no sweeps limit defined.
        return self.data.sweeps() >= self.data.params["sweeps"]

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

    def data_msg(self) -> ODMRData:
        return self.data
