#!/usr/bin/env python3

"""
Worker for Pulse ODMR with Slow detectors.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np

from ..msgs.spodmr_msgs import SPODMRData, is_sweepN, is_CPlike, is_correlation
from ..msgs.pulse_msgs import PulsePattern
from ..msgs import param_msgs as P
from ..inst.sg_interface import SGInterface
from ..inst.pg_interface import PGInterface, Block, Blocks
from ..inst.pd_interface import PDInterface
from ..inst.fg_interface import FGInterface
from .common_worker import Worker

from .podmr_generator.generator import make_generators
from .podmr_generator import generator_kernel as K


class SPODMRDataOperator(object):
    """Operations (set / get / analyze) on SPODMRData."""

    def set_instrument_params(self, data: SPODMRData, pg_freq, offsets):
        if "instrument" in data.params:
            return
        data.params["instrument"] = {}
        data.params["instrument"]["pg_freq"] = pg_freq
        data.params["instrument"]["offsets"] = offsets

    def _append_line_single(self, data, line):
        if data is None:
            return np.array(line, ndmin=2).T
        else:
            return np.append(data, np.array(line, ndmin=2).T, axis=1)

    def _append_line_double(self, data0, data1, line):
        l0 = line[0::2]
        l1 = line[1::2]
        if data0 is None:
            return np.array(l0, ndmin=2).T, np.array(l1, ndmin=2).T
        else:
            return (
                np.append(data0, np.array(l0, ndmin=2).T, axis=1),
                np.append(data1, np.array(l1, ndmin=2).T, axis=1),
            )

    def update(self, data: SPODMRData, line):
        if data.partial() in (0, 2):
            self._append_line_single(data.data0, line)
        elif data.partial() == 1:
            self._append_line_single(data.data1, line)
        else:  # -1
            self._append_line_double(data.data0, data.data1, line)

    def update_plot_params(self, data, plot_params: dict[str, P.RawPDValue]) -> bool:
        """update plot_params. returns True if param is actually updated."""

        if not data.has_params():
            return False
        updated = not P.isclose(data.params["plot"], plot_params)
        if "plot" not in data.params:
            data.params["plot"] = plot_params
        else:
            data.params["plot"].update(plot_params)
        if updated:
            self.update_axes(data)
        return updated

    def update_axes(self, data):
        plot = data.params["plot"]

        data.ylabel = "Intensity ({})".format(plot["plotmode"])
        data.yunit = ""

        taumode = plot["taumode"]
        if taumode == "raw":
            if data.is_sweepN():
                data.xlabel, data.xunit = "N", "pulses"
            else:
                data.xlabel, data.xunit = "tau", "s"
        elif taumode == "total":
            data.xlabel, data.xunit = "total precession time", "s"
        elif taumode == "freq":
            data.xlabel, data.xunit = "detecting frequency", "Hz"
        elif taumode == "index":
            data.xlabel, data.xunit = "sweep index", "#"
        elif taumode == "head":
            data.xlabel, data.xunit = "signal head time", "s"

        # transform
        if plot.get("xlogscale"):
            data.xscale = "log"
        else:
            data.xscale = "linear"

        if plot.get("ylogscale"):
            data.yscale = "log"
        else:
            data.yscale = "linear"


class Bounds(object):
    def __init__(self):
        self._sg = None
        self._fg = None

    def has_sg(self):
        return self._sg is not None

    def sg(self):
        return self._sg

    def set_sg(self, sg_bounds):
        self._sg = sg_bounds

    def has_fg(self):
        return self._fg is not None

    def fg(self):
        return self._fg

    def set_fg(self, fg_bounds):
        self._fg = fg_bounds


class BlocksBuilder(object):
    """Build the PG Blocks for SPODMR from PODMR's Blocks."""

    def build_complementary(
        self, blocks: list[Blocks[Block]], freq: float, half_flip_period: int, params: dict
    ):
        pd_period = int(round(freq / params["pd_rate"]))
        pd_rep = half_flip_period // pd_period
        residual = half_flip_period % pd_period
        w = pd_period // 2
        gate0_blk = Block(
            "gate",
            [("sync", residual)] + [(("sync", "gate"), w), ("sync", pd_period - w)] * pd_rep,
        )
        gate1_blk = Block(
            "gate", [(None, residual)] + [("gate", w), (None, pd_period - w)] * pd_rep
        )

        extended_blks = Blocks()
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            assert len(blks) == 4
            blks0, blks1 = blks[:2], blks[2:]
            T = blks0.total_length()
            assert T == blks1.total_length()
            rep, residual = half_flip_period // T, half_flip_period % T
            blk0 = blks0.repeat(rep).collapse()
            blk1 = blks1.repeat(rep).collapse()
            if residual:
                blk0.insert(0, (None, residual))
                blk1.insert(0, (None, residual))
            extended_blks.extend([blk0.union(gate0_blk), blk1.union(gate1_blk)])
            markers.append(extended_blks.total_length())

        return extended_blks, markers, pd_rep

    def build_partial(
        self, blocks: list[Blocks[Block]], freq: float, half_flip_period: int, params: dict
    ):
        pd_period = int(round(freq / params["pd_rate"]))
        pd_rep = half_flip_period // pd_period
        residual = half_flip_period % pd_period
        w = pd_period // 2
        gate_blks = Block(
            "gates", [(None, residual)] + [("gate", w), (None, pd_period - w)] * pd_rep
        )

        extended_blks = Blocks()
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            assert len(blks) == 2
            T = blks.total_length()
            rep, residual = half_flip_period // T, half_flip_period % T
            blk = blks.repeat(rep).collapse()
            if residual:
                blk.insert(0, (None, residual))
            extended_blks.append(blk.union(gate_blks))
            markers.append(extended_blks.total_length())

        return extended_blks, markers, pd_rep

    def build_lockin(
        self, blocks: list[Blocks[Block]], freq: float, half_flip_period: int, params: dict
    ):
        flip_rep = params["flip_rep"]
        pd_period = int(round(freq / params["pd_rate"]))
        pd_rep = half_flip_period // pd_period
        residual = half_flip_period % pd_period
        w = pd_period // 2
        gate0_blk = Block(
            "gate",
            [("sync", residual)] + [(("sync", "gate"), w), ("sync", pd_period - w)] * pd_rep,
        )
        gate1_blk = Block(
            "gate", [(None, residual)] + [("gate", w), (None, pd_period - w)] * pd_rep
        )

        extended_blks = Blocks()
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            assert len(blks) == 4
            blks0, blks1 = blks[:2], blks[2:]
            T = blks0.total_length()
            assert T == blks1.total_length()
            rep, residual = half_flip_period // T, half_flip_period % T
            blk0 = blks0.repeat(rep).collapse()
            blk1 = blks1.repeat(rep).collapse()
            if residual:
                blk0.insert(0, (None, residual))
                blk1.insert(0, (None, residual))
            blks = Blocks([blk0.union(gate0_blk), blk1.union(gate1_blk)])
            extended_blks.append(blks.repeat(flip_rep))
            markers.append(extended_blks.total_length())

        return extended_blks, markers, pd_rep * flip_rep

    def build_blocks(self, blocks: list[Blocks[Block]], freq: float, common_pulses, params):
        invertY = params.get("invertY", False)

        (
            base_width,
            laser_delay,
            laser_width,
            mw_delay,
            trigger_width,
            init_delay,
            final_delay,
        ) = common_pulses

        flip_period = int(round(freq / params["flip_rate"]))
        # flip_period should be even int
        if flip_period % 2:
            flip_period += 1
        half_flip_period = flip_period // 2

        partial = params["partial"]
        if partial == -1:
            blocks, markers, oversample = self.build_complementary(
                blocks, freq, half_flip_period, params
            )
        elif partial in (0, 1):
            blocks, markers, oversample = self.build_partial(
                blocks, freq, half_flip_period, params
            )
        elif partial == 2:
            blocks, markers, oversample = self.build_lockin(blocks, freq, half_flip_period, params)
        else:
            raise ValueError(f"Invalid partial {partial}")

        # shaping blocks
        blocks = blocks.simplify()
        blocks = K.encode_mw_phase(blocks)
        if invertY:
            blocks = K.invert_y_phase(blocks)

        return blocks, markers, oversample


class Pulser(Worker):
    def __init__(self, cli, logger, has_fg: bool, conf: dict):
        """Worker for Pulse ODMR with Slow detectors.

        Function generator is an option (fg may be None).

        """

        Worker.__init__(self, cli, logger)
        self.sg = SGInterface(cli, "sg")
        self.pg = PGInterface(cli, "pg")
        self.pd_names = conf.get("pd_names", ["pd0", "pd1"])
        self.pds = [PDInterface(cli, n) for n in self.pd_names]
        if has_fg:
            self.fg = FGInterface(cli, "fg")
        else:
            self.fg = None
        self.add_instruments(self.sg, self.pg, self.fg, *self.pds)

        self.length = self.offsets = self.freq = self.oversample = None

        mbl = conf.get("minimum_block_length", 1000)
        bb = conf.get("block_base", 4)
        self.generators = make_generators(
            freq=conf.get("freq", 2.0e9),
            reduce_start_divisor=conf.get("reduce_start_divisor", 2),
            split_fraction=conf.get("split_fraction", 4),
            minimum_block_length=mbl,
            block_base=bb,
            print_fn=self.logger.info,
        )
        self.builder = BlocksBuilder()

        self.data = SPODMRData()
        self.op = SPODMRDataOperator()
        self.bounds = Bounds()
        self.pulse_pattern = None

    def init_inst(self, params: dict) -> bool:
        # SG
        success = (
            self.sg.configure_cw(params["freq"], params["power"])
            and self.sg.set_modulation(True)
            and self.sg.set_dm_source("EXT")
            and self.sg.set_dm(True)
            and self.sg.get_opc()
        )
        if not success:
            self.logger.error("Error initializing SG.")
            return False

        if not self.init_fg(params):
            self.logger.error("Error initializing FG.")
            return False

        # PG
        if not self.init_pg(params):
            self.logger.error("Error initializing PG.")
            return False

        self.op.set_instrument_params(self.data, self.freq, self.offsets)

        return True

    def init_start_pds(self, params: dict):
        rate = params["pd_rate"]
        num = params["num"]
        samples = num * 10  # large samples to assure enough buffer size
        params_pd = {
            "clock": self._pd_clock,
            "cb_samples": num,
            "samples": samples,
            "rate": rate,
            "finite": False,
            "every": self._conf.get("every", False),
            "clock_mode": True,
            "oversample": self.oversample,
            "bounds": params.get("pd_bounds", (-10.0, 10.0)),
        }

        if not (
            all([pd.configure(params_pd) for pd in self.pds])
            and all([pd.start() for pd in self.pds])
        ):
            self.logger.error("Error starting PDs.")
            return False

        return True

    def _fg_enabled(self, params: dict) -> bool:
        return "fg" in params and params["fg"] is not None and params["fg"]["mode"] != "disable"

    def init_fg(self, params: dict) -> bool:
        if not self._fg_enabled(params):
            return True
        if self.fg is None:
            self.logger.error("FG is required but not enabled.")
            return False

        c = params["fg"]
        # TODO: let some kwargs loaded from Pulser conf.
        if c["mode"] == "cw":
            self.fg.configure_cw(c["wave"], c["freq"], c["ampl"], reset=True)
        elif c["mode"] == "gate":
            self.fg.configure_gate(c["wave"], c["freq"], c["ampl"], c["phase"], reset=True)
        else:
            self.logger.error("Unknown mode {} for fg.".format(c["mode"]))
            return False
        return True

    def init_pg(self, params: dict) -> bool:
        if not (self.pg.stop() and self.pg.clear()):
            self.logger.error("Error stopping PG.")
            return False

        blocks, self.freq, markers, self.oversample = self.generate_blocks()
        self.pulse_pattern = PulsePattern(blocks, self.freq, markers=markers)
        pg_params = {"blocks": blocks, "freq": self.freq}

        if not (self.pg.configure(pg_params) and self.pg.get_opc()):
            self.logger.error("Error configuring PG.")
            return False

        self.length = self.pg.get_length()
        self.offsets = self.pg.get_offsets()

        return True

    def generate_blocks(self, data: SPODMRData | None = None):
        if data is None:
            data = self.data
        generate = self.generators[data.params["method"]].generate_raw_blocks

        params = data.get_pulse_params()
        # fill unused params
        params["trigger_width"] = params["init_delay"] = params["final_delay"] = 0.0

        blocks, freq, common_pulses = generate(data.xdata, params)
        blocks, markers, oversample = self.builder.build_blocks(
            blocks, freq, common_pulses, params
        )
        return blocks, freq, markers, oversample

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]
    ) -> bool:
        params = P.unwrap(params)
        d = SPODMRData(params)
        blocks, freq, markers, oversample = self.generate_blocks(d)
        offsets = self.pg.validate_blocks(blocks, freq)
        return offsets is not None

    def update_plot_params(self, params: dict) -> bool:
        if not self.data.has_params():
            return False
        if self.op.update_plot_params(self.data, params):
            self.data.clear_fit_data()
        return True

    def start(self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is not None:
            params = P.unwrap(params)
        resume = params is None or ("resume" in params and params["resume"])
        if not resume:
            self.data = SPODMRData(params)
            self.op.update_axes(self.data)
        else:
            self.data.update_params(params)

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not resume and not self.init_inst(self.data.params):
            return self.fail_with_release("Error initializing instruments.")

        # start instruments
        if not self.init_start_pds(params):
            return self.fail_with_release("Error initializing or starting PDs.")

        success = self.sg.set_output(not self.data.params.get("nomw", False))
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)
        # time.sleep(1)
        success &= self.pg.start()

        if not success:
            return self.fail_with_release("Error starting pulser.")

        if resume:
            self.data.resume()
            self.logger.info("Resumed pulser.")
        else:
            self.data.start()
            self.logger.info("Started pulser.")
        return True

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

        success = (
            self.pg.stop()
            and self.pg.release()
            and self.sg.set_output(False)
            and self.sg.release()
        )
        success &= all([pd.stop() for pd in self.pds])
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(False)
        success &= self.fg.release()

        self.data.finalize()

        if success:
            self.logger.info("Stopped pulser.")
        else:
            self.logger.error("Error stopping pulser.")
        return success

    def _get_param_dict_pulse(self, method: str, d: dict):
        ## common_pulses
        d["base_width"] = P.FloatParam(320e-9, 1e-9, 1e-4)
        d["laser_delay"] = P.FloatParam(45e-9, 0.0, 1e-4)
        d["laser_width"] = P.FloatParam(5e-6, 1e-9, 1e-4)
        d["mw_delay"] = P.FloatParam(1e-6, 0.0, 1e-4)
        # below are unused
        # d["trigger_width"] = P.FloatParam(20e-9, 1e-9, 1e-6)
        # d["init_delay"] = P.FloatParam(0.0, 0.0, 1e-6)
        # d["final_delay"] = P.FloatParam(5e-6, 0.0, 1e-4)

        ## common switches
        d["invertsweep"] = P.BoolParam(False)
        d["nomw"] = P.BoolParam(False)
        d["enable_reduce"] = P.BoolParam(False)
        d["divide_block"] = P.BoolParam(True)
        d["partial"] = P.IntChoiceParam(
            -1, (-1, 0, 1, 2), doc="-1: complementary, 0/1: 0/1 only, 2: lockin"
        )

        ## sweep params (tau / N)
        if self.generators[method].is_sweepN():
            d["Nstart"] = P.IntParam(1, 1, 10000)
            d["Nnum"] = P.IntParam(50, 1, 10000)
            d["Nstep"] = P.IntParam(1, 1, 10000)
        else:
            d["start"] = P.FloatParam(1.0e-9, 1.0e-9, 1.0e-3)
            d["num"] = P.IntParam(50, 1, 10000)
            d["step"] = P.FloatParam(1.0e-9, 1.0e-9, 1.0e-3)
            d["log"] = P.BoolParam(False)

        return d

    def _get_param_dict_pulse_opt(self, method: str, d: dict):
        pulse_params = self.generators[method].pulse_params()

        if "supersample" in pulse_params:
            d["supersample"] = P.IntParam(1, 1, 1000)
        if "90pulse" in pulse_params:
            d["90pulse"] = P.FloatParam(10e-9, 1e-9, 1000e-9)
        if "180pulse" in pulse_params:
            d["180pulse"] = P.FloatParam(-1.0e-9, -1.0e-9, 1000e-9)

        if "tauconst" in pulse_params:
            d["tauconst"] = P.FloatParam(1.0e-9, 1.0e-9, 1.0e-3)
        if "tau2const" in pulse_params:
            d["tau2const"] = P.FloatParam(1.0e-9, 1.0e-9, 1.0e-3)
        if "iq_delay" in pulse_params:
            d["iq_delay"] = P.FloatParam(10e-9, 1e-9, 1000e-9)

        if "Nconst" in pulse_params:
            d["Nconst"] = P.IntParam(4, 1, 10000)
        if "N2const" in pulse_params:
            d["N2const"] = P.IntParam(2, 1, 10000)
        if "N3const" in pulse_params:
            d["N3const"] = P.IntParam(2, 1, 10000)
        if "ddphase" in pulse_params:
            d["ddphase"] = P.StrParam("Y:X:Y:X,Y:X:Y:iX")

        if "invertinit" in pulse_params:
            d["invertinit"] = P.BoolParam(False)

        if "readY" in pulse_params:
            d["readY"] = P.BoolParam(False)
            d["invertY"] = P.BoolParam(False)
        if "reinitX" in pulse_params:
            d["reinitX"] = P.BoolParam(False)

        return d

    def get_param_dict_labels(self) -> list:
        return list(self.generators.keys())

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        if label not in self.generators:
            self.logger.error(f"Unknown method {label}")
            return None

        if self.bounds.has_sg():
            sg = self.bounds.sg()
        else:
            sg = self.sg.get_bounds()
            if sg is None:
                self.logger.error("Failed to get SG bounds.")
                return None
            self.bounds.set_sg(sg)

        f_min, f_max = sg["freq"]
        p_min, p_max = sg["power"]

        # fundamentals
        d = P.ParamDict(
            method=P.StrChoiceParam(label, list(self.generators.keys())),
            resume=P.BoolParam(False),
            freq=P.FloatParam(2.80e9, f_min, f_max),
            power=P.FloatParam(p_min, p_min, p_max),
            timebin=P.FloatParam(3.2e-9, 0.1e-9, 100e-9),
            interval=P.FloatParam(1.0, 0.1, 10.0),
            sweeps=P.IntParam(0, 0, 9999999),
            ident=P.UUIDParam(optional=True, enable=False),
        )

        self._get_param_dict_pulse(label, d)
        self._get_param_dict_pulse_opt(label, d)

        if self.fg is not None:
            if self.bounds.has_fg():
                fg = self.bounds.fg()
            else:
                fg = self.fg.get_bounds()
                if fg is None:
                    self.logger.error("Failed to get FG bounds.")
                    return None
                self.bounds.set_fg(fg)

            f_min, f_max = fg["freq"]
            a_min, a_max = fg["ampl"]
            d["fg"] = {
                "mode": P.StrChoiceParam("disable", ("disable", "cw", "gate")),
                "wave": P.StrParam("sinusoid"),
                "freq": P.FloatParam(1e6, f_min, f_max),
                "ampl": P.FloatParam(a_min, a_min, a_max),
                "phase": P.FloatParam(0.0, -180.0, 180.0),
            }

        taumodes = ["raw", "total", "freq", "index"]
        if not (is_CPlike(label) or is_correlation(label) or label in ("spinecho", "trse")):
            taumodes.remove("total")
        if not ((is_CPlike(label) and not is_sweepN(label)) or label == "spinecho"):
            taumodes.remove("freq")

        d["plot"] = {
            "plotmode": P.StrChoiceParam(
                "data01",
                ("data01", "data0", "data1", "diff", "average", "normalize", "concatenate"),
            ),
            "taumode": P.StrChoiceParam("raw", taumodes),
            "xlogscale": P.BoolParam(False),
            "ylogscale": P.BoolParam(False),
            "fft": P.BoolParam(False),
        }
        return d

    def work(self) -> bool:
        if not self.data.running:
            return False

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
        self.op.update(self.data, line)

        return True

    def data_msg(self) -> SPODMRData:
        return self.data

    def pulse_msg(self) -> PulsePattern | None:
        return self.pulse_pattern
