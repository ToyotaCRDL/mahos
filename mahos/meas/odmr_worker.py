#!/usr/bin/env python3

"""
Worker for ODMR.

.. This file is a part of Mahos project.

"""

from __future__ import annotations

import time

import numpy as np

from ..msgs.odmr_msgs import ODMRData
from ..msgs import param_msgs as P
from ..msgs.inst_pg_msgs import Block, Blocks
from ..inst.sg_interface import SGInterface
from ..inst.pg_interface import PGInterface
from ..inst.daq_interface import APDCounterInterface
from .common_worker import Worker


class Sweeper(Worker):
    """Worker for ODMR Sweep."""

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger)
        self.sg = SGInterface(cli, "sg")
        self.pg = PGInterface(cli, "pg")
        self.pds = [APDCounterInterface(cli, n) for n in conf.get("pds", ["pd0", "pd1"])]
        self.add_instruments(self.sg, self.pg, *self.pds)

        if "pd_clock" not in conf:
            raise KeyError("sweeper.pd_clock must be given")
        self._pd_clock = conf["pd_clock"]
        self._minimum_block_length = conf.get("minimum_block_length", 1000)
        self._start_delay = conf.get("start_delay", 0.0)
        self._drop_first = conf.get("drop_first", 0)
        self._sg_first = conf.get("sg_first", False)
        self._continue_mw = False

        self.data = ODMRData()

    def get_param_dict_names(self) -> list:
        return ["cw", "pulse"]

    def get_param_dict(self, method: str) -> P.ParamDict[str, P.PDValue] | None:
        if method == "cw":
            timing = P.ParamDict(time_window=P.FloatParam(10e-3, 0.1e-3, 1.0))
        elif method == "pulse":
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
                # TODO: is this necessarily choice ?
                burst_num=P.IntChoiceParam(
                    100, (100, 500, 1000, 5000, 10000), doc="number of bursts at each freq."
                ),
            )
        else:
            self.logger.error(f"Unknown param dict name: {method}")
            return None

        bounds = self.sg.get_bounds()
        if bounds is None:
            self.logger.error("Could not get SG bounds.")
            return None
        f_min, f_max = bounds["freq"]
        p_min, p_max = bounds["power"]
        d = P.ParamDict(
            method=P.StrChoiceParam(method, ("cw", "pulse")),
            start=P.FloatParam(2.74e9, f_min, f_max),
            stop=P.FloatParam(3.00e9, f_min, f_max),
            num=P.IntParam(101, 2, 10000),
            power=P.FloatParam(p_min, p_min, p_max),
            timing=timing,
            background=P.BoolParam(False, doc="take background data"),
            resume=P.BoolParam(False),
            continue_mw=P.BoolParam(False),
            ident=P.UUIDParam(optional=True, enable=False),
        )
        return d

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]
    ) -> bool:
        params = P.unwrap(params)
        if params["start"] >= params["stop"]:
            self.logger.error("stop must be greater than start")
            return False
        return True

    def configure_sg(self, params: dict) -> bool:
        p = params
        success = (
            self.sg.configure_point_trig_freq_sweep(p["start"], p["stop"], p["num"], p["power"])
            and self.sg.get_opc()
        )
        return success

    def configure_pg_CW(self, params: dict) -> bool:
        freq = 1.0e6
        window = round(params["timing"]["time_window"] * freq)
        if params.get("background", False):
            b = Block(
                "CW-ODMR",
                [
                    (None, 6),
                    ("gate", 1),
                    (("laser", "mw"), window),
                    ("gate", 1),
                    (None, 6),
                    ("gate", 1),
                    ("laser", window),
                    (("gate", "trigger"), 1),
                ],
                trigger=True,
            )
        else:
            b = Block(
                "CW-ODMR",
                [(None, 6), ("gate", 1), (("laser", "mw"), window), (("gate", "trigger"), 1)],
                trigger=True,
            )
        blocks = Blocks([b]).simplify()
        return self.pg.configure({"blocks": blocks, "freq": freq})

    def _make_blocks_pulse_nobg(
        self, laser_delay, laser_width, mw_delay, mw_width, trigger_width, burst_num
    ):
        min_len = self._minimum_block_length

        init = Block(
            "INIT",
            [
                (None, max(0, min_len - laser_width - mw_delay)),
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

        return Blocks([init, main, final]).simplify()

    def _make_blocks_pulse_bg(
        self, laser_delay, laser_width, mw_delay, mw_width, trigger_width, burst_num
    ):
        min_len = self._minimum_block_length

        init = Block(
            "INIT",
            [
                (None, max(0, min_len - laser_width - mw_delay)),
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
                (None, max(0, min_len - laser_width - mw_delay)),
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

        return Blocks([init, main, final, init_bg, main_bg, final_bg]).simplify()

    def configure_pg_pulse(self, params: dict) -> bool:
        freq = 1.0e9
        laser_delay, laser_width, mw_delay, mw_width, trigger_width = [
            round(params["timing"][k] * freq)
            for k in ("laser_delay", "laser_width", "mw_delay", "mw_width", "trigger_width")
        ]
        burst_num = params["timing"]["burst_num"]

        if mw_delay < trigger_width:
            self.logger.error("mw_delay >= trigger_width must be satisfied.")
            return False

        if params.get("background", False):
            blocks = self._make_blocks_pulse_bg(
                laser_delay, laser_width, mw_delay, mw_width, trigger_width, burst_num
            )
        else:
            blocks = self._make_blocks_pulse_nobg(
                laser_delay, laser_width, mw_delay, mw_width, trigger_width, burst_num
            )

        return self.pg.configure({"blocks": blocks, "freq": freq})

    def configure_pg(self, params: dict) -> bool:
        if not (self.pg.stop() and self.pg.clear()):
            return False
        if params["method"] == "cw":
            return self.configure_pg_CW(params)
        else:
            return self.configure_pg_pulse(params)

    def start_counter(self, params: dict) -> bool:
        if params["method"] == "cw":
            time_window = params["timing"]["time_window"]
        else:
            # time_window is used to compute APD's count rate.
            # gate is opened for whole burst sequence, but meaning time window for APD
            # is just burst_num * laser_width.
            t = params["timing"]
            time_window = t["burst_num"] * t["laser_width"]

        freq = 1.0 / time_window
        rate = freq * 2  # max. expected sampling rate. double expected freq for safety
        num = params["num"]
        drop = self._drop_first * 2  # double drops due to gate mode
        if params.get("background", False):
            num *= 2
            drop *= 2
        total_samples = num * 10  # just a reference value for infinite measurement
        params_pd = {
            "clock": self._pd_clock,
            "cb_samples": num,
            "samples": total_samples,
            "rate": rate,
            "time_window": time_window,
            "finite": False,
            "every": True,
            "gate": True,
            "drop_first": drop,
        }

        success = all([pd.configure(params_pd) for pd in self.pds]) and all(
            [pd.start() for pd in self.pds]
        )
        return success

    def start(self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is not None:
            params = P.unwrap(params)
            self._continue_mw = params.get("continue_mw", False)
        resume = params is None or params.get("resume")

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not resume:
            self.data = ODMRData(params)
        else:
            # TODO: check ident if resume?
            self.data.update_params(params)

        if not self.configure_sg(self.data.params):
            return self.fail_with_release("Failed to configure SG.")
        if not self.configure_pg(self.data.params):
            return self.fail_with_release("Failed to configure PG.")
        if not self.start_counter(self.data.params):
            return self.fail_with_release("Failed to start PD.")

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

    def get_line(self):
        lines = [pd.pop_block() for pd in self.pds]
        data = np.sum(lines, axis=0)
        return data

    def append_line(self, line):
        if not self.data.measure_background():
            if not self.data.has_data():
                self.data.data = np.array(line, ndmin=2).T
            else:
                self.data.data = np.append(self.data.data, np.array(line, ndmin=2).T, axis=1)
        else:
            l_data = line[0::2]
            l_bg = line[1::2]
            if not self.data.has_data():
                self.data.data = np.array(l_data, ndmin=2).T
                self.data.bg_data = np.array(l_bg, ndmin=2).T
            else:
                self.data.data = np.append(self.data.data, np.array(l_data, ndmin=2).T, axis=1)
                self.data.bg_data = np.append(self.data.bg_data, np.array(l_bg, ndmin=2).T, axis=1)

    def work(self):
        if not self.data.running:
            return  # or raise Error?

        line = self.get_line()
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

        success &= (
            all([pd.stop() for pd in self.pds]) and self.pg.stop() and self.release_instruments()
        )

        self.data.finalize()

        if success:
            self.logger.info("Stopped sweeper.")
        else:
            self.logger.error("Error stopping sweeper.")
        return success

    def data_msg(self) -> ODMRData:
        return self.data
