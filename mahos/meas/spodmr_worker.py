#!/usr/bin/env python3

"""
Worker for Pulse ODMR with Slow detectors.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

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
from ..util.conf import PresetLoader
from .common_worker import Worker

from .podmr_generator.generator import make_generators
from .podmr_generator import generator_kernel as K


class SPODMRDataOperator(object):
    """Operations (set / get / analyze) on SPODMRData."""

    def set_laser_duties(self, data: SPODMRData, laser_duties):
        if data.laser_duties is not None:
            return
        data.laser_duties = np.array(laser_duties)

    def set_instrument_params(self, data: SPODMRData, pg_freq, length, offsets):
        if "instrument" in data.params:
            return
        data.params["instrument"] = {}
        data.params["instrument"]["pg_freq"] = pg_freq
        data.params["instrument"]["length"] = length
        if all([ofs == 0 for ofs in offsets]):
            data.params["instrument"]["offsets"] = []
        else:
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
        if plot.get("logX"):
            data.xscale = "log"
        else:
            data.xscale = "linear"

        if plot.get("logY"):
            data.yscale = "log"
        else:
            data.yscale = "linear"


class Bounds(object):
    def __init__(self):
        self._sg = None
        self._sg2 = None
        self._fg = None

    def has_sg(self):
        return self._sg is not None

    def sg(self):
        return self._sg

    def set_sg(self, sg_bounds):
        self._sg = sg_bounds

    def has_sg2(self):
        return self._sg2 is not None

    def sg2(self):
        return self._sg2

    def set_sg2(self, sg2_bounds):
        self._sg2 = sg2_bounds

    def has_fg(self):
        return self._fg is not None

    def fg(self):
        return self._fg

    def set_fg(self, fg_bounds):
        self._fg = fg_bounds


class BlockSeqBuilder(object):
    """Build the PG BlockSeq for SPODMR from PODMR's Blocks."""

    def __init__(
        self,
        trigger_width: float,
        trigger_channel: str,
        nest: bool,
        block_base: int,
        eos_margin: int,
    ):
        self.trigger_width = trigger_width
        self.trigger_channel = trigger_channel
        self.nest = nest
        self.block_base = block_base
        self.eos_margin = eos_margin

    def fix_block_base(self, blk: Block, idx: int = 0) -> Block:
        res = blk.total_length() % self.block_base
        if not res:
            return blk

        pulse = blk.pattern[idx]
        blk.update_duration(idx, pulse.duration + res)
        return blk

    def _append_eos(self, seqs: list):
        if self.eos_margin:
            # for PulseBlaster, this is in order to avoid loop unrolling
            ch = seqs[-1].last_pulse().channels
            seqs.append(Block("eos", [(ch, self.eos_margin)]))

    def _encode_mw_phase(self, bs: BlockSeq, params) -> BlockSeq:
        # Fixed version of K.encode_mw_phase for double mw mode.
        # (i, q), (i2, q2) is same phase now.
        # Maybe we could change relative readout phase.

        nomw = params.get("nomw", False)
        nomw2 = "nomw2" not in params or params["nomw2"]
        if nomw2:
            return K.encode_mw_phase(bs)
        elif nomw:  # mw2 only
            iq_phase_dict = {
                "mw_x": ("mw_i2", "mw_q2"),
                "mw_y": ("mw_q2",),
                "mw_x_inv": (),
                "mw_y_inv": ("mw_i2",),
            }
        else:  # both mw and mw2
            iq_phase_dict = {
                "mw_x": ("mw_i", "mw_q", "mw_i2", "mw_q2"),
                "mw_y": ("mw_q", "mw_q2"),
                "mw_x_inv": (),
                "mw_y_inv": ("mw_i", "mw_i2"),
            }

        return bs.replace(iq_phase_dict)

    def build_complementary(
        self,
        blocks: list[Blocks[Block]],
        accum_window: int,
        accum_rep: int,
        drop_rep: int,
        laser_width: int,
        trigger_width: int,
        sync_mode: str,
    ):
        seqs = []
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("trigger")
            if sync_mode != "laser":
                blks = blks.remove("sync")
            assert len(blks) == 4
            blk0 = self.fix_block_base(blks[:2].collapse())  # Pi-0, readi-0 -> Pi-0
            if sync_mode == "lockin":
                blk0 = blk0.union(Block("sync", [("sync", blk0.total_length())]))
            blk1 = self.fix_block_base(blks[2:].collapse())  # Pi-1, readi-1 -> Pi-1
            blk0R = blk0.suffix("R")
            blk1R = blk1.suffix("R")

            T = blk0.total_length()
            assert T == blk1.total_length()
            rep, residual = accum_window // T, accum_window % T
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
            if self.nest:
                seqs.extend(
                    [
                        BlockSeq(blk0.name + "SeqD", blk0acc, drop_rep),
                        BlockSeq(blk0.name + "SeqT", blk0accT),
                        BlockSeq(blk0.name + "SeqM", blk0acc, accum_rep - 1),
                        BlockSeq(blk1.name + "SeqD", blk1acc, drop_rep),
                        BlockSeq(blk1.name + "SeqT", blk1accT),
                        BlockSeq(blk1.name + "SeqM", blk1acc, accum_rep - 1),
                    ]
                )
            else:
                seqs.extend(
                    blk0acc * drop_rep
                    + blk0accT
                    + blk0acc * (accum_rep - 1)
                    + blk1acc * drop_rep
                    + blk1accT
                    + blk1acc * (accum_rep - 1)
                )
            # residual is dark: duty becomes slightly lower than laser_width / T
            lw = Blocks(blk0acc).total_channel_length("laser", True)
            assert lw == Blocks(blk1acc).total_channel_length("laser", True)
            laser_duties.append(lw / accum_window)
            markers.append(sum([s.total_length() for s in seqs]))

        self._append_eos(seqs)

        return BlockSeq("top", seqs), laser_duties, markers

    def build_partial(
        self,
        blocks: list[Blocks[Block]],
        accum_window: int,
        accum_rep: int,
        drop_rep: int,
        laser_width: int,
        trigger_width: int,
        sync_mode: str,
    ):
        seqs = []
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("trigger")
            if sync_mode != "laser":
                blks = blks.remove("sync")
            assert len(blks) == 2
            blk = self.fix_block_base(blks.collapse())
            blkR = blk.suffix("R")

            T = blk.total_length()
            rep, residual = accum_window // T, accum_window % T
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
            if self.nest:
                seqs.extend(
                    [
                        BlockSeq(blk.name + "SeqD", blkacc, drop_rep),
                        BlockSeq(blk.name + "SeqT", blkaccT),
                        BlockSeq(blk.name + "SeqM", blkacc, accum_rep - 1),
                    ]
                )
            else:
                seqs.extend(blkacc * drop_rep + blkaccT + blkacc * (accum_rep - 1))
            # residual is dark: duty becomes slightly lower than laser_width / T
            lw = Blocks(blkacc).total_channel_length("laser", True)
            laser_duties.append(lw / accum_window)
            markers.append(sum([s.total_length() for s in seqs]))

        self._append_eos(seqs)

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
        sync_mode: str,
    ):
        seqs = []
        laser_duties = []
        # markers just for PulseMonitor visualization
        markers = [0]

        for blks in blocks:
            blks = blks.remove("trigger")
            if sync_mode != "laser":
                blks = blks.remove("sync")
            assert len(blks) == 4
            blk0 = self.fix_block_base(blks[:2].collapse())  # Pi-0, readi-0 -> Pi-0
            if sync_mode == "lockin":
                blk0 = blk0.union(Block("sync", [("sync", blk0.total_length())]))
            blk1 = self.fix_block_base(blks[2:].collapse())  # Pi-1, readi-1 -> Pi-1
            blk0R = blk0.suffix("R")
            blk1R = blk1.suffix("R")

            T = blk0.total_length()
            assert T == blk1.total_length()
            rep, residual = accum_window // T, accum_window % T
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
            if self.nest:
                seqs.extend(
                    [
                        BlockSeq(
                            blk0.name + "SeqD",
                            [
                                BlockSeq(blk0.name + "SeqD0", blk0acc, accum_rep),
                                BlockSeq(blk0.name + "SeqD1", blk1acc, accum_rep),
                            ],
                            drop_rep,
                        ),
                        BlockSeq(
                            blk0.name + "SeqT",
                            [
                                BlockSeq(blk0.name + "SeqTT", blk0accT),
                                BlockSeq(blk0.name + "SeqT0", blk0acc, accum_rep - 1),
                                BlockSeq(blk0.name + "SeqT1", blk1acc, accum_rep),
                            ],
                        ),
                        BlockSeq(
                            blk0.name + "SeqM",
                            [
                                BlockSeq(blk0.name + "SeqM0", blk0acc, accum_rep),
                                BlockSeq(blk0.name + "SeqM1", blk1acc, accum_rep),
                            ],
                            lockin_rep - 1,
                        ),
                    ]
                )
            else:
                seqs.extend(
                    (blk0acc * accum_rep + blk1acc * accum_rep) * drop_rep
                    + (blk0accT + blk0acc * (accum_rep - 1) + blk1acc * accum_rep)
                    + (blk0acc * accum_rep + blk1acc * accum_rep) * (lockin_rep - 1)
                )
            # residual is dark: duty becomes slightly lower than laser_width / T
            lw = Blocks(blk0acc).total_channel_length("laser", True)
            assert lw == Blocks(blk1acc).total_channel_length("laser", True)
            laser_duties.append(lw / accum_window)
            markers.append(sum([s.total_length() for s in seqs]))

        self._append_eos(seqs)

        return BlockSeq("top", seqs), laser_duties, markers

    def build_blocks(
        self, blocks: list[Blocks[Block]], freq: float, common_pulses, params, sync_mode
    ):
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
        accum_window = K.offset_base_inc(
            int(round(freq * params["accum_window"])), self.block_base
        )
        trigger_width = int(round(freq * self.trigger_width))
        pd_period = int(round(freq / params["pd_rate"]))

        partial = params["partial"]
        if partial == -1:
            blockseq, laser_duties, markers = self.build_complementary(
                blocks, accum_window, accum_rep, drop_rep, laser_width, trigger_width, sync_mode
            )
            oversample = (accum_window * accum_rep) // pd_period
        elif partial in (0, 1):
            blockseq, laser_duties, markers = self.build_partial(
                blocks, accum_window, accum_rep, drop_rep, laser_width, trigger_width, sync_mode
            )
            oversample = (accum_window * accum_rep) // pd_period
        elif partial == 2:
            lockin_rep = params["lockin_rep"]
            blockseq, laser_duties, markers = self.build_lockin(
                blocks,
                accum_window,
                accum_rep,
                lockin_rep,
                drop_rep,
                laser_width,
                trigger_width,
                sync_mode,
            )
            oversample = (2 * accum_window * accum_rep * lockin_rep) // pd_period
        else:
            raise ValueError(f"Invalid partial {partial}")

        laser_duties = np.array(laser_duties, dtype=np.float64)

        # shaping blockseq
        blockseq = blockseq.simplify()
        if invertY:
            blockseq = K.invert_y_phase(blockseq)
        blockseq = self._encode_mw_phase(blockseq, params)

        return blockseq, laser_duties, markers, oversample


class Pulser(Worker):
    def __init__(self, cli, logger, conf):
        """Worker for Pulse ODMR with Slow detectors.

        Function generator is an option (fg may be None).

        """

        Worker.__init__(self, cli, logger, conf)
        self.load_conf_preset(cli)

        self.sg = SGInterface(cli, "sg")
        if "sg2" in cli.insts():
            self.sg2 = SGInterface(cli, "sg2")
        else:
            self.sg2 = None
        self.pg = PGInterface(cli, "pg")
        self.pd_names = self.conf.get("pd_names", ["pd0"])
        self.pds = [PDInterface(cli, n) for n in self.pd_names]
        self.clock = ClockSourceInterface(cli, self.conf.get("clock_name", "clock"))
        if "fg" in cli:
            self.fg = FGInterface(cli, "fg")
        else:
            self.fg = None
        self.add_instruments(self.sg, self.sg2, self.pg, self.fg, *self.pds)

        self.length = self.offsets = self.freq = self.oversample = None

        self.check_required_conf(
            ["pd_trigger", "block_base", "pg_freq", "reduce_start_divisor", "minimum_block_length"]
        )
        self._pd_trigger = self.conf["pd_trigger"]
        self._pd_data_transfer = self.conf.get("pd_data_transfer")
        self._quick_resume = self.conf.get("quick_resume", True)

        self.generators = make_generators(
            freq=self.conf["pg_freq"],
            reduce_start_divisor=self.conf["reduce_start_divisor"],
            split_fraction=self.conf.get("split_fraction", 4),
            minimum_block_length=self.conf["minimum_block_length"],
            block_base=self.conf["block_base"],
            print_fn=self.logger.info,
        )

        self.builder = BlockSeqBuilder(
            self.conf.get("trigger_width", 1e-6),
            self.conf.get("trigger_channel", "gate"),
            self.conf.get("nest_blockseq", False),
            self.conf["block_base"],
            self.conf.get("eos_margin", 0),
        )

        self.data = SPODMRData()
        self.op = SPODMRDataOperator()
        self.bounds = Bounds()
        self.pulse_pattern = None

    def load_conf_preset(self, cli):
        loader = PresetLoader(self.logger, PresetLoader.Mode.FORWARD)
        loader.add_preset(
            "DTG",
            [
                ("block_base", 4),
                ("pg_freq", 2.0e9),
                ("reduce_start_divisor", 2),
                ("minimum_block_length", 1000),
                ("nest_blockseq", False),
            ],
        )
        loader.add_preset(
            "PulseStreamer",
            [
                ("block_base", 1),
                ("pg_freq", 1.0e9),
                ("reduce_start_divisor", 10),
                ("minimum_block_length", 1),
                ("nest_blockseq", False),
            ],
        )
        loader.add_preset(
            "SpinCore_PulseBlaster",
            [
                ("block_base", 1),
                ("pg_freq", 0.5e9),
                ("reduce_start_divisor", 5),
                ("minimum_block_length", 5),
                ("nest_blockseq", True),
                ("eos_margin", 5),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("pg"))

    def init_inst(self, params: dict) -> bool:
        # SG
        if not self.sg.configure_cw_iq(params["freq"], params["power"]):
            self.logger.error("Error initializing SG.")
            return False
        if self.sg2 is not None and not self.sg2.configure_cw_iq(
            params["freq2"], params["power2"]
        ):
            self.logger.error("Error initializing SG2.")
            return False

        # FG
        if not self.init_fg(params):
            self.logger.error("Error initializing FG.")
            return False

        # PG
        if not self.init_pg():
            self.logger.error("Error initializing PG.")
            return False

        self.op.set_instrument_params(self.data, self.freq, self.length, self.offsets)

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
            self.logger.error("failed to configure clock.")
            return False
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
        if self._pd_data_transfer:
            params_pd["data_transfer"] = self._pd_data_transfer

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

    def init_pg(self) -> bool:
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

    def _sync_mode(self, params: dict) -> str:
        gated_fg = "fg" in params and params["fg"]["mode"] == "gate"
        lockin = params["partial"] == 2
        if gated_fg and lockin:
            # This case is TODO
            msg = "Cannot determine default sync_mode as FG is gate and partial is lockin."
            msg += "defaulting to laser, but consider fixing code (to use multiple sync ch)!"
            self.logger.warn(msg)
            default = "laser"
        elif gated_fg:
            default = "laser"
        else:  # lockin or unused
            default = "lockin"

        if "sync_mode" not in params or params["sync_mode"] == "default":
            self.logger.debug(f"Defaulting sync_mode to {default}")
            return default
        if params["sync_mode"] == default:
            return default

        msg = f"Overriding default sync_mode = {default} with params['sync_mode'] = "
        msg += params["sync_mode"]
        self.logger.warn(msg)
        return params["sync_mode"]

    def generate_blocks(self, data: SPODMRData | None = None):
        if data is None:
            data = self.data
        generate = self.generators[data.label].generate_raw_blocks

        params = data.get_params()
        # fill unused params
        params["base_width"] = params["trigger_width"] = 0.0
        params["init_delay"] = params["final_delay"] = 0.0
        params["fix_base_width"] = 1  # ignore base_width
        if self.sg2 is not None:
            # disable nomw feature at generator
            # because nomw = True, nomw2 = False is possible usage (using SG2 only)
            params["nomw"] = False

        blocks, freq, common_pulses = generate(data.xdata, params)
        sync_mode = self._sync_mode(params)
        blockseq, laser_duties, markers, oversample = self.builder.build_blocks(
            blocks, freq, common_pulses, params, sync_mode
        )

        self.logger.info(f"Built BlockSeq. total pattern #: {blockseq.total_pattern_num()}")

        return blockseq, freq, laser_duties, markers, oversample

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        params = P.unwrap(params)
        d = SPODMRData(params, label)
        blockseq, freq, laser_duties, markers, oversample = self.generate_blocks(d)
        offsets = self.pg.validate_blockseq(blockseq, freq)
        return offsets is not None

    def update_plot_params(self, params: dict) -> bool:
        if not self.data.has_params():
            return False
        if self.op.update_plot_params(self.data, params):
            self.data.clear_fit_data()
        return True

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)
        resume = params is None or ("resume" in params and params["resume"])
        if params is None:
            quick_resume = resume and self._quick_resume
        else:
            quick_resume = resume and params.get("quick_resume", self._quick_resume)
        if not resume:
            self.data = SPODMRData(params, label)
            self.op.update_axes(self.data)
        else:
            self.data.update_params(params)

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if quick_resume:
            self.logger.info("Quick resume enabled: skipping initial inst configurations.")
        if not quick_resume and not self.init_inst(self.data.params):
            return self.fail_with_release("Error initializing instruments.")

        # start instruments
        if not self.init_start_pds():
            return self.fail_with_release("Error initializing or starting PDs.")

        success = self.sg.set_output(not self.data.params.get("nomw", False))
        if self.sg2 is not None:
            success &= self.sg2.set_output(not self.data.params.get("nomw2", False))
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)
        # time.sleep(1)
        success &= self.pg.start()

        if not success:
            self.pg.stop()
            if self._fg_enabled(self.data.params):
                self.fg.set_output(False)
            self.sg.set_output(False)
            if self.sg2 is not None:
                self.sg2.set_output(False)
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
        if self.sg2 is not None:
            success &= self.sg2.set_output(False) and self.sg2.release()
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

    def _get_param_dict_pulse(self, label: str, d: dict):
        ## common_pulses
        d["laser_delay"] = P.FloatParam(45e-9, 0.0, 1e-4)
        d["laser_width"] = P.FloatParam(3e-6, 1e-9, 1e-4)
        d["mw_delay"] = P.FloatParam(1e-6, 0.0, 1e-4)
        # below are unused
        # d["base_width"] = P.FloatParam(320e-9, 1e-9, 1e-4)
        # d["trigger_width"] = P.FloatParam(20e-9, 1e-9, 1e-6)
        # d["init_delay"] = P.FloatParam(0.0, 0.0, 1e-6)
        # d["final_delay"] = P.FloatParam(5e-6, 0.0, 1e-4)

        ## common switches
        d["invert_sweep"] = P.BoolParam(False)
        d["enable_reduce"] = P.BoolParam(False)
        d["divide_block"] = P.BoolParam(True)
        d["partial"] = P.IntChoiceParam(
            -1, (-1, 0, 1, 2), doc="-1: complementary, 0/1: 0/1 only, 2: lockin"
        )

        ## sweep params (tau / N)
        if self.generators[label].is_sweepN():
            d["Nstart"] = P.IntParam(1, 1, 10000)
            d["Nnum"] = P.IntParam(50, 1, 10000)
            d["Nstep"] = P.IntParam(1, 1, 10000)
        else:
            d["start"] = P.FloatParam(1.0e-9, 1.0e-9, 1.0e-3)
            d["num"] = P.IntParam(50, 1, 10000)
            d["step"] = P.FloatParam(1.0e-9, 1.0e-9, 1.0e-3)
            d["log"] = P.BoolParam(False)

        return d

    def get_param_dict_labels(self) -> list:
        return list(self.generators.keys())

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        if label not in self.generators:
            self.logger.error(f"Unknown label {label}")
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
            resume=P.BoolParam(False),
            quick_resume=P.BoolParam(False),
            freq=P.FloatParam(sg_freq, f_min, f_max),
            power=P.FloatParam(p_min, p_min, p_max),
            nomw=P.BoolParam(False),
            sweeps=P.IntParam(0, 0, 9999999),
            ident=P.UUIDParam(optional=True, enable=False),
            pd_rate=P.FloatParam(
                self.conf.get("pd_rate", 500e3), 1e3, 10000e3, doc="PD sampling rate"
            ),
            pd_bounds=[
                P.FloatParam(-10.0, -10.0, +10.0, doc="PD voltage lower bound"),
                P.FloatParam(+10.0, -10.0, +10.0, doc="PD voltage upper bound"),
            ],
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
            sync_mode=P.StrChoiceParam(
                "default", ("default", "lockin", "laser"), doc="mode of sync channel"
            ),
        )

        if self.sg2 is not None:
            if self.bounds.has_sg2():
                sg2 = self.bounds.sg2()
            else:
                sg2 = self.sg2.get_bounds()
                if sg2 is None:
                    self.logger.error("Failed to get SG2 bounds.")
                    return None
                self.bounds.set_sg2(sg2)
            f_min, f_max = sg2["freq"]
            p_min, p_max = sg2["power"]
            sg2_freq = max(min(self.conf.get("sg2_freq", 2.8e9), f_max), f_min)
            d["freq2"] = P.FloatParam(sg2_freq, f_min, f_max)
            d["power2"] = P.FloatParam(p_min, p_min, p_max)
            d["nomw2"] = P.BoolParam(False)

        self._get_param_dict_pulse(label, d)
        d["pulse"] = self.generators[label].pulse_params()

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
                "phase": P.FloatParam(0.0, 0.0, 360.0),
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
            "logX": P.BoolParam(False),
            "logY": P.BoolParam(False),
            "flipY": P.BoolParam(False),
            "normalize": P.BoolParam(True, doc="normalize data using laser duties"),
            "offset": P.FloatParam(0.0, SI_prefix=True, doc="offset value for normalization"),
            "complex_conv": P.StrChoiceParam("real", ("real", "imag", "abs", "angle")),
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
        if not self.init_pg():
            self.logger.error("Error initializing PG.")
            return False

        self.op.set_instrument_params(self.data, self.freq, self.length, self.offsets)

        return True

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)
        resume = params is None or ("resume" in params and params["resume"])
        if not resume:
            self.data = SPODMRData(params, label)
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
