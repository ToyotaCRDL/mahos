#!/usr/bin/env python3

"""
Worker for Qdyne.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os
import time
from itertools import chain

import numpy as np

from ..util.timer import IntervalTimer, StopWatch
from ..util.io import load_h5
from ..util.conv import real_fft
from ..msgs import param_msgs as P
from ..msgs.pulse_msgs import PulsePattern
from ..msgs.inst.pg_msgs import Block, Blocks
from ..msgs.inst.tdc_msgs import RawEvents
from ..msgs.qdyne_msgs import QdyneData, TDCStatus
from ..inst.sg_interface import SGInterface
from ..inst.pg_interface import PGInterface
from ..inst.tdc_interface import TDCInterface
from ..inst.fg_interface import FGInterface
from ..util.conf import PresetLoader
from .common_worker import Worker

from .podmr_generator import generator_kernel as K
from .podmr_generator.generator import make_generators
from ..ext import cqdyne_analyzer as C


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
    """Build the PG Blocks for Qdyne from PODMR's Blocks."""

    def __init__(self, minimum_block_length, block_base):
        self.minimum_block_length = minimum_block_length
        self.block_base = block_base

    def build_blocks(self, blocks, common_pulses, params):
        divide = params.get("divide_block", False)
        invertY = params.get("invertY", False)
        minimum_block_length = self.minimum_block_length
        block_base = self.block_base

        (
            base_width,
            laser_delay,
            laser_width,
            mw_delay,
            trigger_width,
            init_delay,
            final_delay,
        ) = common_pulses

        # flatten list[list[Block]] to Blocks[Block]
        blocks = Blocks(chain.from_iterable(blocks))
        assert len(blocks) == 2

        # swap [P0-0, read0-0] -> [read0-0, P0-0]
        blocks[0], blocks[1] = blocks[1], blocks[0]

        # inject trigger at head of the first block (read0-0: laser)
        b = blocks[0]
        assert len(b.pattern) == 1
        p = b.pattern[0][0]
        t = b.pattern[0][1]
        assert t >= trigger_width
        pattern = [(p + ("trigger",), trigger_width), (p, t - trigger_width)]
        blocks[0] = Block(b.name, pattern, Nrep=b.Nrep, trigger=b.trigger)

        # only one laser timing at the head
        laser_timing = [0]

        # shaping blocks
        if divide:
            blocks = K.divide_long_operation(blocks, minimum_block_length, block_base)
            # divide_long_laser doesn't work due to trigger injection above.
            # blocks = K.divide_long_laser(blocks, minimum_block_length)
        blocks = K.merge_short_blocks(blocks, minimum_block_length)

        blocks = blocks.simplify()
        blocks = K.encode_mw_phase(blocks)
        if invertY:
            blocks = K.invert_y_phase(blocks)

        return blocks, laser_timing


class QdyneAnalyzer(object):
    """Analyzer for QdyneData."""

    def _analyze_py(self, data: QdyneData) -> bool:
        """Analyze data.raw_data and set data.data and data.xdata.

        This is Python implementation and thus deprecated:
            analyze() using C++ extension is way faster.

        """

        if not data.has_raw_data() or data.marker_indices is None:
            return False
        signal_head, signal_tail, reference_head, reference_tail = data.marker_indices.T[0]

        # T: period of measurements
        T = data.get_period_bins()
        # mx: the last measured data point
        mx = int(np.max(data.raw_data))
        # N: number of measurements performed
        N = mx // T
        # head_last: head of the last period including mx
        head_last = N * T
        if head_last + signal_tail < mx:
            # we can use the last period
            N += 1

        data.xdata = np.arange(0, N * T, T)
        data.data = np.zeros(N, dtype=np.uint64)
        rd = data.raw_data

        idx = 0
        each = 1000
        w = StopWatch()
        for i, t in enumerate(data.xdata):
            # head, tail = t + signal_head, t + signal_tail
            # data.data[i] = np.count_nonzero(np.logical_and(head <= rd, rd <= tail))

            head, tail = t + signal_head, t + signal_tail
            head_idx = np.argmax(head <= rd[idx:])
            tail_idx = np.argmax(tail < rd[idx + head_idx :])
            data.data[i] = tail_idx
            idx += head_idx + tail_idx

            if not (i % each):
                print(f"[{i}/{len(data.xdata)}] {i/len(data.xdata):.1%} {w.elapsed_str()}")
        return True

    def analyze(self, data: QdyneData) -> bool:
        """Analyze data.raw_data and set data.data and data.xdata."""

        if not data.has_raw_data() or data.marker_indices is None:
            return False
        signal_head, signal_tail, reference_head, reference_tail = data.marker_indices.T[0]

        # T: period of measurements
        T = data.get_period_bins()
        # mx: the last measured data point
        mx = int(np.max(data.raw_data))
        # N: number of measurements performed
        N = mx // T
        # head_last: head of the last period including mx
        head_last = N * T
        if head_last + signal_tail < mx:
            # we can use the last period
            N += 1

        data.xdata = np.arange(0, N * T, T, dtype=np.uint64)
        data.data = np.zeros(N, dtype=np.uint64)
        C.analyze(data.raw_data, data.xdata, data.data, signal_head, signal_tail)
        return True

    def fft(self, data: QdyneData) -> bool:
        if not data.has_data():
            return False
        x = data.get_xdata() * data.get_bin()
        y = data.get_ydata()
        data.fft_xdata, data.fft_data = real_fft(x, y)
        return True


class Pulser(Worker):
    def __init__(self, cli, logger, conf: dict):
        """Worker for Qdyne.

        Function generator is an option (fg may be None).

        """

        Worker.__init__(self, cli, logger, conf)
        self.load_conf_preset(cli)

        self.sg = SGInterface(cli, "sg")
        self.pg = PGInterface(cli, "pg")
        self.tdc = TDCInterface(cli, "tdc")
        if "fg" in cli:
            self.fg = FGInterface(cli, "fg")
        else:
            self.fg = None
        self.add_instruments(self.sg, self.pg, self.tdc, self.fg)

        self.timer = None
        self.length = self.offsets = self.freq = self.laser_timing = None

        self.check_required_conf(
            ["block_base", "pg_freq", "reduce_start_divisor", "minimum_block_length"]
        )
        self._raw_events_dir = self.conf.get("raw_events_dir", "")
        self._remove_raw_events = self.conf.get("remove_raw_events", True)
        self._start_delay = self.conf.get("start_delay", 0.0)

        mbl = self.conf["minimum_block_length"]
        bb = self.conf["block_base"]
        self.generators = make_generators(
            freq=self.conf["pg_freq"],
            reduce_start_divisor=self.conf["reduce_start_divisor"],
            split_fraction=self.conf.get("split_fraction", 4),
            minimum_block_length=mbl,
            block_base=bb,
            print_fn=self.logger.info,
        )
        self.builder = BlocksBuilder(mbl, bb)

        self.data = QdyneData()
        self.analyzer = QdyneAnalyzer()
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
                ("divide_block", True),
            ],
        )
        loader.add_preset(
            "PulseStreamer",
            [
                ("block_base", 1),
                ("pg_freq", 1.0e9),
                ("reduce_start_divisor", 10),
                ("minimum_block_length", 1),
                ("divide_block", False),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("pg"))

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

        # Detector
        trange = self.length / self.freq
        save_file = "qdyne_" + self.data.ident.hex
        if not self.tdc.configure_raw_events("qdyne", save_file, trange=trange, tbin=0.0):
            self.logger.error("Error configuring TDC.")
            return False
        if params.get("sweeps", 0) and not self.tdc.set_sweeps(params["sweeps"]):
            self.logger.error("Error setting sweeps limit for TDC.")
            return False
        if params.get("duration", 0.0) and not self.tdc.set_duration(params["duration"]):
            self.logger.error("Error setting duration limit for TDC.")
            return False
        tbin = self.tdc.get_timebin()

        self.data.set_instrument_params(tbin, self.freq, self.length, self.offsets)
        self.data.set_marker_indices()

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

        blocks, self.freq, laser_timing = self.generate_blocks()
        self.data.set_laser_timing(np.array(laser_timing) / self.freq)
        self.pulse_pattern = PulsePattern(blocks, self.freq, markers=laser_timing)
        pg_params = {"blocks": blocks, "freq": self.freq}

        if not (self.pg.configure(pg_params) and self.pg.get_opc()):
            self.logger.error("Error configuring PG.")
            return False

        self.length = self.pg.get_length()
        self.offsets = self.pg.get_offsets()

        return True

    def generate_blocks(self, data: QdyneData | None = None):
        if data is None:
            data = self.data
        generate = self.generators[data.label].generate_raw_blocks

        # fill unused parameters
        xdata = [data.params["tauconst"]]
        params = data.get_pulse_params()
        params["init_delay"] = params["final_delay"] = 0.0

        blocks, freq, common_pulses = generate(xdata, params)
        blocks, laser_timing = self.builder.build_blocks(blocks, common_pulses, params)
        return blocks, freq, laser_timing

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> tuple[bool, Blocks[Block], float, list[float], list[int]]:
        params = P.unwrap(params)
        d = QdyneData(params, label)
        blocks, freq, laser_timing = self.generate_blocks(d)
        offsets = self.pg.validate_blocks(blocks, freq)
        self.pulse_pattern = PulsePattern(blocks, freq, markers=laser_timing)
        valid = offsets is not None
        return valid, blocks, freq, laser_timing, offsets

    def update(self) -> bool:
        if not self.data.running:
            return False

        self.data.set_status(self.get_tdc_status())

        return True

    def fetch_data(self) -> bool:
        self.logger.info("Fetching raw events data.")
        raw_events = self.tdc.get_raw_events()
        self.tdc.release()

        if raw_events is None:
            self.logger.error("Failed to fetch raw events data (tdc side).")
            return True  # return True anyway to finalize measurement

        if isinstance(raw_events, str):
            raw_events_path = os.path.join(self._raw_events_dir, raw_events)
            self.logger.info(f"Start loading {raw_events_path}")
            raw_events = load_h5(raw_events_path, RawEvents, self.logger)
            if self._remove_raw_events:
                if os.path.exists(raw_events_path):
                    os.remove(raw_events_path)
                    self.logger.debug(f"Removed {raw_events_path}")

        if raw_events is None:
            self.logger.error("Failed to fetch raw events data (load failure).")
            return True  # return True anyway to finalize measurement

        self.data.set_raw_data(raw_events.data)

        self.logger.info("Analyzing fetched raw data.")
        self.analyzer.analyze(self.data)
        self.analyzer.fft(self.data)
        self.logger.info("Finished analyzing raw data.")

        if self.data.params.get("remove_raw_data", True):
            self.data.remove_raw_data()
        return True  # return True anyway to finalize measurement

    def start(
        self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)
        resume = params is None or ("resume" in params and params["resume"])
        if not resume:
            self.data = QdyneData(params, label)
        else:
            self.data.update_params(params)

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not resume and not self.init_inst(self.data.params):
            return self.fail_with_release("Error initializing instruments.")

        # start instruments
        if resume:
            success = self.tdc.resume()
        else:
            success = self.tdc.stop()
            success &= self.tdc.clear()
            success &= self.tdc.start()

        success &= self.sg.set_output(not self.data.params.get("nomw", False))
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)

        time.sleep(self._start_delay)

        success &= self.pg.start()

        if not success:
            return self.fail_with_release("Error starting pulser.")

        self.timer = IntervalTimer(self.data.params["interval"])

        if resume:
            self.data.resume()
            self.logger.info("Resumed pulser.")
        else:
            self.data.start()
            self.logger.info("Started pulser.")
        return True

    def discard(self) -> bool:
        if not self.data.running:
            return False
        return self.tdc.stop() and self.wait_tdc_stop() and self.tdc.clear() and self.tdc.resume()

    def is_finished(self) -> bool:
        # TDC may stop running by itself if duration limit is set
        return not self.get_tdc_running()

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = (
            self.tdc.stop()
            and self.wait_tdc_stop()
            and self.pg.stop()
            and self.pg.release()
            and self.sg.set_output(False)
            and self.sg.release()
        )
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(False)
        success &= self.fg.release()

        success &= self.fetch_data()

        if success:
            self.timer = None
            self.data.finalize()
            self.logger.info("Stopped pulser.")
        else:
            self.logger.error("Error stopping pulser.")
        return success

    def get_tdc_status(self) -> TDCStatus:
        """Get status from TDC."""

        st0 = self.tdc.get_status(0)
        st1 = self.tdc.get_status(1)

        return TDCStatus(round(st0.runtime), st0.starts, st0.total, st1.total)

    def get_tdc_running(self) -> bool:
        """return True if TDC is running."""

        st0 = self.tdc.get_status(0)
        return st0.running if st0 is not None else False

    def wait_tdc_stop(self, timeout_sec=10.0, interval_sec=0.2) -> bool:
        """Wait for TDC status become not-running (stopped)."""

        self.logger.debug("Waiting TDC stop")
        for i in range(int(round(timeout_sec / interval_sec))):
            if not self.get_tdc_running():
                return True
            time.sleep(interval_sec)

        self.logger.error(f"Timeout ({timeout_sec} sec) encountered in wait_tdc_stop!")
        return False

    def _get_param_dict_pulse(self, label: str, d: dict):
        ## common_pulses
        d["base_width"] = P.FloatParam(320e-9, 0.5e-9, 1e-4, unit="s", SI_prefix=True)
        d["laser_delay"] = P.FloatParam(45e-9, 0.0, 1e-4, unit="s", SI_prefix=True)
        d["laser_width"] = P.FloatParam(3e-6, 1e-9, 1e-4, unit="s", SI_prefix=True)
        d["mw_delay"] = P.FloatParam(1e-6, 0.0, 1e-4, unit="s", SI_prefix=True)
        d["trigger_width"] = P.FloatParam(20e-9, 1e-9, 1e-6, unit="s", SI_prefix=True)
        # these are unused
        # d["init_delay"] = P.FloatParam(0.0, 0.0, 1e-6)
        # d["final_delay"] = P.FloatParam(5e-6, 0.0, 1e-4)

        ## common switches
        d["nomw"] = P.BoolParam(False)
        d["enable_reduce"] = P.BoolParam(False)
        d["divide_block"] = P.BoolParam(self.conf.get("divide_block", False))
        # partial == -1 (complementary) is not allowed for Qdyne.
        d["partial"] = P.IntParam(0, 0, 1)

        return d

    def _get_param_dict_pulse_opt(self, label: str, d: dict):
        pulse_params = self.generators[label].pulse_params()

        d["tauconst"] = P.FloatParam(1.0e-9, 1.0e-9, 1.0e-3, unit="s", SI_prefix=True)
        d["Nconst"] = P.IntParam(1, 1, 10000)

        if "90pulse" in pulse_params:
            d["90pulse"] = P.FloatParam(10e-9, 1e-9, 1000e-9, unit="s", SI_prefix=True)
        if "180pulse" in pulse_params:
            d["180pulse"] = P.FloatParam(-1.0e-9, -1.0e-9, 1000e-9, unit="s", SI_prefix=True)

        if "readY" in pulse_params:
            d["readY"] = P.BoolParam(True)
            d["invertY"] = P.BoolParam(False)

        return d

    def get_param_dict_labels(self) -> list:
        return ["cp", "cpmg", "xy4", "xy8", "xy16"]

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

        # fundamentals
        d = P.ParamDict(
            resume=P.BoolParam(False),
            freq=P.FloatParam(2.80e9, f_min, f_max, unit="Hz", SI_prefix=True),
            power=P.FloatParam(p_min, p_min, p_max, unit="dBm"),
            interval=P.FloatParam(1.0, 0.1, 10.0, unit="s"),
            # TODO: we won't respect sweeps for the time being.
            # we must implement finite n_runs (using subsequencing?) for PG.
            sweeps=P.IntParam(0, 0, 9999999, doc="limit number of sweeps"),
            duration=P.FloatParam(0.0, 0.0, 9999999, unit="s", doc="limit measurement duration"),
            ident=P.UUIDParam(optional=True, enable=False),
            remove_raw_data=P.BoolParam(
                True, doc="If True, remove raw_data after finishing measurement"
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
                "freq": P.FloatParam(1e6, f_min, f_max, unit="Hz", SI_prefix=True),
                "ampl": P.FloatParam(a_min, a_min, a_max, unit="Vpp"),
                "phase": P.FloatParam(0.0, -180.0, 180.0),
            }

        d["plot"] = {
            "sigdelay": P.FloatParam(200e-9, 0.0, 10e-6, unit="s", SI_prefix=True),
            "sigwidth": P.FloatParam(300e-9, 1e-9, 10e-6, unit="s", SI_prefix=True),
            "refdelay": P.FloatParam(2200e-9, 1e-9, 100e-6, unit="s", SI_prefix=True),
            "refwidth": P.FloatParam(2400e-9, 1e-9, 10e-6, unit="s", SI_prefix=True),
        }
        return d

    def work(self):
        if not self.data.running:
            return

        if self.timer.check():
            self.update()

    def data_msg(self) -> QdyneData | None:
        """Get QdyneData message to publish.

        None is returned if raw_data is retained, because it can be massive.

        """

        if self.data.has_raw_data():
            return None
        else:
            return self.data

    def pulse_msg(self) -> PulsePattern | None:
        return self.pulse_pattern
