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
from ..inst.pg_interface import PGInterface, Block, Blocks, BlockSeq
from ..inst.pd_interface import PDInterface
from ..inst.fg_interface import FGInterface
from ..inst.daq_interface import ClockSourceInterface
from .common_worker import Worker

from .podmr_generator.generator import make_generators
from .podmr_generator import generator_kernel as K


class SPODMRDataOperator(object):
    """Operations (set / get / analyze) on SPODMRData."""

    def set_laser_duties(self, data: SPODMRData, laser_duties):
        if data.laser_duties is not None:
            return
        data.laser_duties = np.array(laser_duties)

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
            data.data0 = self._append_line_single(data.data0, line)
        elif data.partial() == 1:
            data.data1 = self._append_line_single(data.data1, line)
        else:  # -1
            data.data0, data.data1 = self._append_line_double(data.data0, data.data1, line)

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

    def __init__(self, trigger_width: float, trigger_channel: str):
        self.trigger_width = trigger_width
        self.trigger_channel = trigger_channel

    def build_complementary(
        self,
        blocks: list[Blocks[Block]],
        accum_window: int,
        accum_rep: int,
        drop_rep: int,
        laser_width: int,
        trigger_width: int,
    ):
        trigger0_blk = Block(
            "trigger0",
            [
                (("sync", self.trigger_channel), trigger_width),
                ("sync", accum_window - trigger_width),
            ],
        )
        sync0_blk = Block("sync0", [("sync", accum_window)])
        # drop1_blk = Block("drop1", [(None, accum_window)])
        trigger1_blk = Block(
            "trigger1",
            [(self.trigger_channel, trigger_width), (None, accum_window - trigger_width)],
        )
        # rest1_blk = Block("rest1", [(None, accum_window)])

        extended_blks = Blocks()
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("sync")
            assert len(blks) == 4
            blks0: Blocks[Block] = blks[:2]
            blks1: Blocks[Block] = blks[2:]
            T = blks0.total_length()
            assert T == blks1.total_length()
            rep, residual = accum_window // T, accum_window % T
            blk0 = blks0.repeat(rep).collapse()
            blk1 = blks1.repeat(rep).collapse()
            if residual:
                blk0.insert(0, (blk0.pattern[0].channels, residual))
                blk1.insert(0, (blk1.pattern[0].channels, residual))
            extended_blks.extend(
                [
                    blk0.union(sync0_blk).suffix("_d").repeat(drop_rep),
                    blk0.union(trigger0_blk).suffix("_t"),
                    blk0.union(sync0_blk).repeat(accum_rep - 1),
                    blk1.suffix("_d").repeat(drop_rep),  # no need to union(drop1_blk)
                    blk1.union(trigger1_blk).suffix("_t"),
                    blk1.repeat(accum_rep - 1),  # no need to union(rest1)
                ]
            )
            # residual is dark: duty becomes slightly lower than laser_width / T
            laser_duties.append(laser_width * rep / accum_window)
            markers.append(extended_blks.total_length())

        return extended_blks, laser_duties, markers

    def build_partial(
        self,
        blocks: list[Blocks[Block]],
        accum_window: int,
        accum_rep: int,
        drop_rep: int,
        laser_width: int,
        trigger_width: int,
    ):
        trigger_blk = Block(
            "trigger",
            [(self.trigger_channel, trigger_width), (None, accum_window - trigger_width)],
        )

        extended_blks = Blocks()
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("sync")
            assert len(blks) == 2
            T = blks.total_length()
            rep, residual = accum_window // T, accum_window % T
            blk = blks.repeat(rep).collapse()
            if residual:
                blk.insert(0, (blk.pattern[0].channels, residual))
            extended_blks.extend(
                [
                    blk.suffix("_d").repeat(drop_rep),
                    blk.union(trigger_blk).suffix("_t"),
                    blk.repeat(accum_rep - 1),
                ]
            )
            # residual is dark: duty becomes slightly lower than laser_width / T
            laser_duties.append(laser_width * rep / accum_window)
            markers.append(extended_blks.total_length())

        return extended_blks, laser_duties, markers

    def build_lockin(
        self,
        blocks: list[Blocks[Block]],
        accum_window: int,
        accum_rep: int,
        lockin_rep: int,
        drop_rep: int,
        laser_width: int,
        trigger_width: int,
    ):
        trigger0_blk = Block(
            "trigger0",
            [
                (("sync", self.trigger_channel), trigger_width),
                ("sync", accum_window - trigger_width),
            ],
        )
        sync0_blk = Block("sync0", [("sync", accum_window)])

        extended_blks = Blocks()
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("sync")
            assert len(blks) == 4
            blks0: Blocks[Block] = blks[:2]
            blks1: Blocks[Block] = blks[2:]
            T = blks0.total_length()
            assert T == blks1.total_length()
            rep, residual = accum_window // T, accum_window % T
            blk0 = blks0.repeat(rep).collapse()
            blk1 = blks1.repeat(rep).collapse()
            if residual:
                blk0.insert(0, (blk0.pattern[0].channels, residual))
                blk1.insert(0, (blk1.pattern[0].channels, residual))
            # TODO: using concatenate() collapses the repeat (Nrep) of
            # blk0.union(gate0_blk).repeat(accum_rep) and blk1.union(...).repeat(...) here,
            # and can increase memory requirement in PG.
            # Nested-Blocks (type to express nested repeat) would resolve this.
            trig = (
                blk0.union(trigger0_blk)
                .suffix("_t")
                .concatenate(blk0.union(sync0_blk).repeat(accum_rep - 1))
                .concatenate(blk1.repeat(accum_rep))
            )
            normal = blk0.union(sync0_blk).repeat(accum_rep).concatenate(blk1.repeat(accum_rep))
            extended_blks.extend(
                [normal.suffix("_d").repeat(drop_rep), trig, normal.repeat(lockin_rep - 1)]
            )
            # residual is dark: duty becomes slightly lower than laser_width / T
            laser_duties.append(laser_width * rep / accum_window)
            markers.append(extended_blks.total_length())

        return extended_blks, laser_duties, markers

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

        accum_rep = params["accum_rep"]
        drop_rep = params["drop_rep"]
        accum_window = int(round(freq * params["accum_window"]))
        trigger_width = int(round(freq * self.trigger_width))
        pd_period = int(round(freq / params["pd_rate"]))

        partial = params["partial"]
        if partial == -1:
            blocks, laser_duties, markers = self.build_complementary(
                blocks, accum_window, accum_rep, drop_rep, laser_width, trigger_width
            )
            oversample = (accum_window * accum_rep) // pd_period
        elif partial in (0, 1):
            blocks, laser_duties, markers = self.build_partial(
                blocks, accum_window, accum_rep, drop_rep, laser_width, trigger_width
            )
            oversample = (accum_window * accum_rep) // pd_period
        elif partial == 2:
            lockin_rep = params["lockin_rep"]
            blocks, laser_duties, markers = self.build_lockin(
                blocks, accum_window, accum_rep, lockin_rep, drop_rep, laser_width, trigger_width
            )
            oversample = (2 * accum_window * accum_rep * lockin_rep) // pd_period
        else:
            raise ValueError(f"Invalid partial {partial}")

        laser_duties = np.array(laser_duties, dtype=np.float64)

        # shaping blocks
        blocks = blocks.simplify()
        blocks = K.encode_mw_phase(blocks)
        if invertY:
            blocks = K.invert_y_phase(blocks)

        return blocks, laser_duties, markers, oversample


class BlockSeqBuilder(object):
    """Build the PG BlockSeq for SPODMR from PODMR's Blocks."""

    def __init__(self, trigger_width: float, trigger_channel: str):
        self.trigger_width = trigger_width
        self.trigger_channel = trigger_channel

    def build_complementary(
        self,
        blocks: list[Blocks[Block]],
        accum_window: int,
        accum_rep: int,
        drop_rep: int,
        laser_width: int,
        trigger_width: int,
    ):
        seqs = []
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("sync")
            assert len(blks) == 4
            blks0: Blocks[Block] = blks[:2]
            blks1: Blocks[Block] = blks[2:]
            T = blks0.total_length()
            assert T == blks1.total_length()
            rep, residual = accum_window // T, accum_window % T
            blk0 = blks0.collapse()  # Pi-0, readi-0 -> Pi-0
            blk0 = blk0.union(Block("sync", [("sync", blk0.total_length())]))
            blk1 = blks1.collapse()  # Pi-1, readi-1 -> Pi-1
            blk0R = blk0.suffix("R")
            blk1R = blk1.suffix("R")
            if residual:
                blk0R.insert(0, (blk0R.pattern[0].channels, residual))
                blk1R.insert(0, (blk1R.pattern[0].channels, residual))
            blk0T = blk0R.suffix("T").union(
                Block(
                    "trig",
                    [
                        (self.trigger_channel, trigger_width),
                        (None, blk0R.total_length() - trigger_width),
                    ],
                )
            )
            blk1T = blk1R.suffix("T").union(
                Block(
                    "trig",
                    [
                        (self.trigger_channel, trigger_width),
                        (None, blk1R.total_length() - trigger_width),
                    ],
                )
            )
            # units having duration of accum_window
            blk0acc = [blk0R, blk0.repeat(rep - 1)]
            blk0accT = [blk0T, blk0.repeat(rep - 1)]
            blk1acc = [blk1R, blk1.repeat(rep - 1)]
            blk1accT = [blk1T, blk1.repeat(rep - 1)]
            seqs.extend(
                [
                    BlockSeq(blk0.name + "_SeqD", blk0acc, drop_rep),
                    BlockSeq(blk0.name + "_SeqT", blk0accT),
                    BlockSeq(blk0.name + "_SeqM", blk0acc, accum_rep - 1),
                    BlockSeq(blk1.name + "_SeqD", blk1acc, drop_rep),
                    BlockSeq(blk1.name + "_SeqT", blk1accT),
                    BlockSeq(blk1.name + "_SeqM", blk1acc, accum_rep - 1),
                ]
            )
            # residual is dark: duty becomes slightly lower than laser_width / T
            laser_duties.append(laser_width * rep / accum_window)
            markers.append(sum([s.total_length() for s in seqs]))

        return BlockSeq("top", seqs), laser_duties, markers

    def build_partial(
        self,
        blocks: list[Blocks[Block]],
        accum_window: int,
        accum_rep: int,
        drop_rep: int,
        laser_width: int,
        trigger_width: int,
    ):
        seqs = []
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("sync")
            assert len(blks) == 2
            T = blks.total_length()
            rep, residual = accum_window // T, accum_window % T
            blk = blks.collapse()
            blkR = blk.suffix("R")
            if residual:
                blkR.insert(0, (blkR.pattern[0].channels, residual))
            blkT = blkR.suffix("T").union(
                Block(
                    "trig",
                    [
                        (self.trigger_channel, trigger_width),
                        (None, blkR.total_length() - trigger_width),
                    ],
                )
            )
            # units having duration of accum_window
            blkacc = [blkR, blk.repeat(rep - 1)]
            blkaccT = [blkT, blk.repeat(rep - 1)]
            seqs.extend(
                [
                    BlockSeq(blk.name + "_SeqD", blkacc, drop_rep),
                    BlockSeq(blk.name + "_SeqT", blkaccT),
                    BlockSeq(blk.name + "_SeqM", blkacc, accum_rep - 1),
                ]
            )
            # residual is dark: duty becomes slightly lower than laser_width / T
            laser_duties.append(laser_width * rep / accum_window)
            markers.append(sum([s.total_length() for s in seqs]))

        return BlockSeq("top", seqs), laser_duties, markers

    def build_lockin(
        self,
        blocks: list[Blocks[Block]],
        accum_window: int,
        accum_rep: int,
        lockin_rep: int,
        drop_rep: int,
        laser_width: int,
        trigger_width: int,
    ):
        seqs = []
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("sync")
            assert len(blks) == 4
            blks0: Blocks[Block] = blks[:2]
            blks1: Blocks[Block] = blks[2:]
            T = blks0.total_length()
            assert T == blks1.total_length()
            rep, residual = accum_window // T, accum_window % T
            blk0 = blks0.collapse()  # Pi-0, readi-0 -> Pi-0
            blk0 = blk0.union(Block("sync", [("sync", blk0.total_length())]))
            blk1 = blks1.collapse()  # Pi-1, readi-1 -> Pi-1
            blk0R = blk0.suffix("R")
            blk1R = blk1.suffix("R")
            if residual:
                blk0R.insert(0, (blk0R.pattern[0].channels, residual))
                blk1R.insert(0, (blk1R.pattern[0].channels, residual))
            blk0T = blk0R.suffix("T").union(
                Block(
                    "trig",
                    [
                        (self.trigger_channel, trigger_width),
                        (None, blk0R.total_length() - trigger_width),
                    ],
                )
            )
            # units having duration of accum_window
            blk0acc = [blk0R, blk0.repeat(rep - 1)]
            blk0accT = [blk0T, blk0.repeat(rep - 1)]
            blk1acc = [blk1R, blk1.repeat(rep - 1)]
            # blkXacc * accum_rep here may be inefficient (blockseq data may be long),
            # however, accum_rep won't be huge (order of 10 - 100).
            # If this is a problem, we can try deeper nest of BlockSeq.
            seqs.extend(
                [
                    BlockSeq(
                        blk0.name + "_SeqD",
                        blk0acc * accum_rep + blk1acc * accum_rep,
                        drop_rep,
                    ),
                    BlockSeq(
                        blk0.name + "_SeqT",
                        blk0accT + blk0acc * (accum_rep - 1) + blk1acc * accum_rep,
                    ),
                    BlockSeq(
                        blk0.name + "_SeqM",
                        blk0acc * accum_rep + blk1acc * accum_rep,
                        lockin_rep - 1,
                    ),
                ]
            )
            # residual is dark: duty becomes slightly lower than laser_width / T
            laser_duties.append(laser_width * rep / accum_window)
            markers.append(sum([s.total_length() for s in seqs]))

        return BlockSeq("top", seqs), laser_duties, markers

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

        accum_rep = params["accum_rep"]
        drop_rep = params["drop_rep"]
        accum_window = int(round(freq * params["accum_window"]))
        trigger_width = int(round(freq * self.trigger_width))
        pd_period = int(round(freq / params["pd_rate"]))

        partial = params["partial"]
        if partial == -1:
            blockseq, laser_duties, markers = self.build_complementary(
                blocks, accum_window, accum_rep, drop_rep, laser_width, trigger_width
            )
            oversample = (accum_window * accum_rep) // pd_period
        elif partial in (0, 1):
            blockseq, laser_duties, markers = self.build_partial(
                blocks, accum_window, accum_rep, drop_rep, laser_width, trigger_width
            )
            oversample = (accum_window * accum_rep) // pd_period
        elif partial == 2:
            lockin_rep = params["lockin_rep"]
            blockseq, laser_duties, markers = self.build_lockin(
                blocks, accum_window, accum_rep, lockin_rep, drop_rep, laser_width, trigger_width
            )
            oversample = (2 * accum_window * accum_rep * lockin_rep) // pd_period
        else:
            raise ValueError(f"Invalid partial {partial}")

        laser_duties = np.array(laser_duties, dtype=np.float64)

        # shaping blockseq
        blockseq = blockseq.simplify()
        blockseq = K.encode_mw_phase(blockseq)
        if invertY:
            blockseq = K.invert_y_phase(blockseq)

        return blockseq, laser_duties, markers, oversample


class Pulser(Worker):
    def __init__(self, cli, logger, has_fg: bool, conf: dict):
        """Worker for Pulse ODMR with Slow detectors.

        Function generator is an option (fg may be None).

        """

        Worker.__init__(self, cli, logger)
        self.sg = SGInterface(cli, "sg")
        self.pg = PGInterface(cli, "pg")
        self.pd_names = conf.get("pd_names", ["pd0"])
        self.pds = [PDInterface(cli, n) for n in self.pd_names]
        self.clock = ClockSourceInterface(cli, conf.get("clock_name", "clock"))
        if has_fg:
            self.fg = FGInterface(cli, "fg")
        else:
            self.fg = None
        self.add_instruments(self.sg, self.pg, self.fg, *self.pds)

        self.length = self.offsets = self.freq = self.oversample = None

        if "pd_trigger" not in conf:
            raise KeyError("pulser.pd_trigger must be given")
        self._pd_trigger = conf["pd_trigger"]
        self.conf = conf

        self.block_base = conf.get("block_base", 4)
        self.generators = make_generators(
            freq=conf.get("pg_freq", 2.0e9),
            reduce_start_divisor=conf.get("reduce_start_divisor", 2),
            split_fraction=conf.get("split_fraction", 4),
            minimum_block_length=conf.get("minimum_block_length", 1000),
            block_base=self.block_base,
            print_fn=self.logger.info,
        )
        # disable recovery measurement because Pattern0 and Pattern1 lengths can be different
        # in current pattern generator.
        del self.generators["recovery"]

        self.builder = BlockSeqBuilder(
            conf.get("trigger_width", 1e-6), conf.get("trigger_channel", "trigger")
        )
        # for debug using old builder
        # self._builder = BlocksBuilder(
        #     conf.get("trigger_width", 1e-6), conf.get("trigger_channel", "trigger")
        # )

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

        # FG
        if not self.init_fg(params):
            self.logger.error("Error initializing FG.")
            return False

        # PG
        if not self.init_pg(params):
            self.logger.error("Error initializing PG.")
            return False

        self.op.set_instrument_params(self.data, self.freq, self.offsets)

        return True

    def init_start_pds(self) -> bool:
        params = self.data.params
        rate = params["pd_rate"]

        params_clock = {
            "freq": rate,
            "samples": self.oversample,
            "finite": True,
            "trigger_source": self._pd_trigger,
            "trigger_dir": True,
            "retriggerable": True,
        }
        if not self.clock.configure(params_clock):
            return self.fail_with("failed to configure clock.")
        clock_pd = self.clock.get_internal_output()

        num = self.data.get_num()
        if params["partial"] == -1:
            num *= 2
        buffer_size = num * self.conf.get("buffer_size_coeff", 20)
        params_pd = {
            "clock": clock_pd,
            "cb_samples": num,
            "samples": buffer_size,
            "buffer_size": buffer_size,
            "rate": rate,
            "finite": False,
            "every": self.conf.get("every", False),
            "clock_mode": True,
            "oversample": self.oversample,
            "bounds": params.get("pd_bounds", (-10.0, 10.0)),
        }

        if not (
            all([pd.configure(params_pd) for pd in self.pds])
            and self.clock.start()
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

        blockseq, self.freq, laser_duties, markers, self.oversample = self.generate_blocks()
        self.op.set_laser_duties(self.data, laser_duties)
        self.pulse_pattern = PulsePattern(blockseq, self.freq, markers=markers)
        self.logger.info(f"Initialized PG. PD oversample: {self.oversample}")

        if not (self.pg.configure_blockseq(blockseq, self.freq) and self.pg.get_opc()):
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
        params["base_width"] = params["trigger_width"] = 0.0
        params["init_delay"] = params["final_delay"] = 0.0
        params["fix_base_width"] = self.block_base

        blocks, freq, common_pulses = generate(data.xdata, params)
        blockseq, laser_duties, markers, oversample = self.builder.build_blocks(
            blocks, freq, common_pulses, params
        )

        # for debug using old builder
        # _blocks, _laser_duties, _markers, _oversample = self._builder.build_blocks(
        #     blocks, freq, common_pulses, params
        # )
        # assert np.allclose(laser_duties, _laser_duties)
        # assert oversample == _oversample
        # assert markers == _markers
        # assert blockseq.equivalent(_blocks)

        self.logger.info(f"Built BlockSeq. total pattern #: {blockseq.total_pattern_num()}")

        return blockseq, freq, laser_duties, markers, oversample

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]
    ) -> bool:
        params = P.unwrap(params)
        d = SPODMRData(params)
        blockseq, freq, laser_duties, markers, oversample = self.generate_blocks(d)
        offsets = self.pg.validate_blockseq(blockseq, freq)
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
        if not self.init_start_pds():
            return self.fail_with_release("Error initializing or starting PDs.")

        success = self.sg.set_output(not self.data.params.get("nomw", False))
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)
        # time.sleep(1)
        success &= self.pg.start()

        if not success:
            self.pg.stop()
            if self._fg_enabled(self.data.params):
                self.fg.set_output(False)
            self.sg.set_output(False)
            for pd in self.pds:
                pd.stop()
            self.clock.stop()
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
        success &= all([pd.stop() for pd in self.pds]) and all([pd.release() for pd in self.pds])
        success &= self.clock.stop()
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(False)
        if self.fg is not None:
            success &= self.fg.release()

        self.data.finalize()

        if success:
            self.logger.info("Stopped pulser.")
        else:
            self.logger.error("Error stopping pulser.")
        return success

    def _get_param_dict_pulse(self, method: str, d: dict):
        ## common_pulses
        d["laser_delay"] = P.FloatParam(45e-9, 0.0, 1e-4)
        d["laser_width"] = P.FloatParam(5e-6, 1e-9, 1e-4)
        d["mw_delay"] = P.FloatParam(1e-6, 0.0, 1e-4)
        # below are unused
        # d["base_width"] = P.FloatParam(320e-9, 1e-9, 1e-4)
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
        sg_freq = max(min(self.conf.get("sg_freq", 2.8e9), f_max), f_min)
        # fundamentals
        d = P.ParamDict(
            method=P.StrChoiceParam(label, list(self.generators.keys())),
            resume=P.BoolParam(False),
            freq=P.FloatParam(sg_freq, f_min, f_max),
            power=P.FloatParam(p_min, p_min, p_max),
            sweeps=P.IntParam(0, 0, 9999999),
            ident=P.UUIDParam(optional=True, enable=False),
            pd_rate=P.FloatParam(
                self.conf.get("pd_rate", 500e3), 1e3, 10000e3, doc="PD sampling rate"
            ),
            accum_window=P.FloatParam(
                self.conf.get("accum_window", 1e-3), 1e-5, 1.0, doc="accumulation time window"
            ),
            accum_rep=P.IntParam(
                self.conf.get("accum_rep", 10), 1, 10000, doc="number of accumulation repetitions"
            ),
            drop_rep=P.IntParam(
                self.conf.get("drop_rep", 1),
                0,
                100,
                doc="number of dummy (dropped) accum. pattern repetitions",
            ),
            lockin_rep=P.IntParam(
                self.conf.get("lockin_rep", 1), 1, 10000, doc="number of lockin repetitions"
            ),
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
            "normalize": P.BoolParam(True, doc="normalize data using laser duties"),
            "complex_conv": P.StrChoiceParam("real", ("real", "imag", "abs", "angle")),
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


class DebugPulser(Pulser):
    def init_inst(self, params: dict) -> bool:
        # PG
        if not self.init_pg(params):
            self.logger.error("Error initializing PG.")
            return False

        self.op.set_instrument_params(self.data, self.freq, self.offsets)

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

        # Skip PD, SG, and FG starts

        # time.sleep(1)
        success = self.pg.start()

        if not success:
            return self.fail_with_release("Error starting pulser.")

        if resume:
            self.data.resume()
            self.logger.info("Resumed pulser.")
        else:
            self.data.start()
            self.logger.info("Started pulser.")
        return True

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.pg.stop()
        self.release_instruments()

        self.data.finalize()

        if success:
            self.logger.info("Stopped pulser.")
        else:
            self.logger.error("Error stopping pulser.")
        return success

    def work(self) -> bool:
        if not self.data.running:
            return False

        return True
