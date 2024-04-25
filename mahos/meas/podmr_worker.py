#!/usr/bin/env python3

"""
Worker for Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import time

import numpy as np

from ..util.timer import IntervalTimer
from ..msgs.podmr_msgs import PODMRData, TDCStatus, is_sweepN, is_CPlike, is_correlation
from ..msgs.pulse_msgs import PulsePattern
from ..msgs import param_msgs as P
from ..inst.sg_interface import SGInterface
from ..inst.pg_interface import PGInterface
from ..inst.tdc_interface import TDCInterface
from ..inst.fg_interface import FGInterface
from ..util.conf import PresetLoader
from .common_worker import Worker

from .podmr_generator.generator import make_generators


class PODMRDataOperator(object):
    """Operations (set / get / analyze) on PODMRData."""

    def set_laser_timing(self, data: PODMRData, laser_timing):
        if data.laser_timing is not None:
            return
        data.laser_timing = np.array(laser_timing)  # unit is [sec]

    def set_instrument_params(self, data: PODMRData, trange, tbin, pg_freq, offsets):
        if "instrument" in data.params:
            return
        data.params["instrument"] = {}
        data.params["instrument"]["trange"] = trange
        data.params["instrument"]["tbin"] = tbin
        data.params["instrument"]["pg_freq"] = pg_freq
        data.params["instrument"]["offsets"] = offsets

    def update(self, data: PODMRData, data_new, tdc_status):
        data.raw_data = data_new
        data.tdc_status = tdc_status

    def get_marker_indices(self, data: PODMRData):
        """get marker indices, that is the analysis timings in unit of time bins."""

        tbin = data.get_bin()
        if data.params is None or tbin is None:
            return None

        sigdelay, sigwidth, refdelay, refwidth = [
            data.params["plot"][k] for k in ("sigdelay", "sigwidth", "refdelay", "refwidth")
        ]

        signal_head = data.laser_timing + sigdelay
        signal_tail = signal_head + sigwidth
        reference_head = signal_tail + refdelay
        reference_tail = reference_head + refwidth

        # sec to time bin index
        signal_head = np.round(signal_head / tbin).astype(np.int64)
        signal_tail = np.round(signal_tail / tbin).astype(np.int64)
        reference_head = np.round(reference_head / tbin).astype(np.int64)
        reference_tail = np.round(reference_tail / tbin).astype(np.int64)

        data.marker_indices = np.vstack((signal_head, signal_tail, reference_head, reference_tail))
        return data.marker_indices

    def analyze(self, data: PODMRData) -> bool:
        if not data.has_raw_data() or data.marker_indices is None or data.tdc_status.sweeps < 1:
            return False

        if data.is_partial():
            if data.has_roi():
                return self._analyze_partial_roi(data)
            else:
                return self._analyze_partial_noroi(data)
        else:
            if data.has_roi():
                return self._analyze_complementary_roi(data)
            else:
                return self._analyze_complementary_noroi(data)

    def _analyze_partial_roi(self, data: PODMRData) -> bool:
        sig = np.zeros(len(data.xdata))
        ref = np.zeros(len(data.xdata))
        sig_head, sig_tail, ref_head, ref_tail = data.marker_indices

        for i, d in enumerate(data.raw_data):
            s, _ = data.get_roi(i)
            try:
                sig[i] = np.mean(d[sig_head[i] - s : sig_tail[i] - s + 1])
            except IndexError as e:
                print("analyze_sig (sig %d): %r" % (i, e))

            try:
                ref[i] = np.mean(d[ref_head[i] - s : ref_tail[i] - s + 1])
            except IndexError as e:
                print("analyze_sig (ref %d): %r" % (i, e))

        sweeps = data.tdc_status.sweeps
        if data.partial() == 0:
            data.data0 = sig / sweeps
            data.data0ref = ref / sweeps
        else:  # assert data.partial() == 1
            data.data1 = sig / sweeps
            data.data1ref = ref / sweeps

        return True

    def _analyze_partial_noroi(self, data: PODMRData) -> bool:
        sig = np.zeros(len(data.xdata))
        ref = np.zeros(len(data.xdata))
        sig_head, sig_tail, ref_head, ref_tail = data.marker_indices

        for i in range(len(data.xdata)):
            try:
                sig[i] = np.mean(data.raw_data[sig_head[i] : sig_tail[i] + 1])
            except IndexError as e:
                print("analyze_sig (sig %d): %r" % (i, e))

            try:
                ref[i] = np.mean(data.raw_data[ref_head[i] : ref_tail[i] + 1])
            except IndexError as e:
                print("analyze_sig (ref %d): %r" % (i, e))

        sweeps = data.tdc_status.sweeps
        if data.partial() == 0:
            data.data0 = sig / sweeps
            data.data0ref = ref / sweeps
        else:  # assert data.partial() == 1
            data.data1 = sig / sweeps
            data.data1ref = ref / sweeps

        return True

    def _analyze_complementary_roi(self, data: PODMRData) -> bool:
        sig = np.zeros(len(data.xdata) * 2)
        ref = np.zeros(len(data.xdata) * 2)
        sig_head, sig_tail, ref_head, ref_tail = data.marker_indices

        for i, d in enumerate(data.raw_data):
            s, _ = data.get_roi(i)
            try:
                sig[i] = np.mean(d[sig_head[i] - s : sig_tail[i] - s + 1])
            except IndexError as e:
                print("analyze_sig (sig %d): %r" % (i, e))

            try:
                ref[i] = np.mean(d[ref_head[i] - s : ref_tail[i] - s + 1])
            except IndexError as e:
                print("analyze_sig (ref %d): %r" % (i, e))

        sweeps = data.tdc_status.sweeps
        data.data0 = sig[0::2] / sweeps
        data.data1 = sig[1::2] / sweeps
        data.data0ref = ref[0::2] / sweeps
        data.data1ref = ref[1::2] / sweeps

        return True

    def _analyze_complementary_noroi(self, data: PODMRData) -> bool:
        sig = np.zeros(len(data.xdata) * 2)
        ref = np.zeros(len(data.xdata) * 2)
        sig_head, sig_tail, ref_head, ref_tail = data.marker_indices

        for i in range(len(data.xdata) * 2):
            try:
                sig[i] = np.mean(data.raw_data[sig_head[i] : sig_tail[i] + 1])
            except IndexError as e:
                print("analyze_sig (sig %d): %r" % (i, e))

            try:
                ref[i] = np.mean(data.raw_data[ref_head[i] : ref_tail[i] + 1])
            except IndexError as e:
                print("analyze_sig (ref %d): %r" % (i, e))

        sweeps = data.tdc_status.sweeps
        data.data0 = sig[0::2] / sweeps
        data.data1 = sig[1::2] / sweeps
        data.data0ref = ref[0::2] / sweeps
        data.data1ref = ref[1::2] / sweeps

        return True

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


class Pulser(Worker):
    def __init__(self, cli, logger, conf: dict):
        """Worker for Pulse ODMR.

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
        self.generators = make_generators(
            freq=self.conf["pg_freq"],
            reduce_start_divisor=self.conf["reduce_start_divisor"],
            split_fraction=self.conf.get("split_fraction", 4),
            minimum_block_length=self.conf["minimum_block_length"],
            block_base=self.conf["block_base"],
            print_fn=self.logger.info,
        )
        self.eos_margin = self.conf.get("eos_margin", 1e-6)

        self.data = PODMRData()
        self.op = PODMRDataOperator()
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
        trange = self.length / self.freq - self.eos_margin
        if not self.tdc.configure_histogram("podmr", trange, params["timebin"]):
            self.logger.error("Error configuring TDC.")
            return False
        if params.get("sweeps", 0) and not self.tdc.set_sweeps(params["sweeps"]):
            self.logger.error("Error setting sweeps limit for TDC.")
            return False
        if params.get("duration", 0.0) and not self.tdc.set_duration(params["duration"]):
            self.logger.error("Error setting duration limit for TDC.")
            return False
        d = self.tdc.get_range_bin()

        self.op.set_instrument_params(self.data, d["range"], d["bin"], self.freq, self.offsets)

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
        self.op.set_laser_timing(self.data, np.array(laser_timing) / self.freq)
        self.pulse_pattern = PulsePattern(blocks, self.freq, markers=laser_timing)
        pg_params = {"blocks": blocks, "freq": self.freq}

        if not (self.pg.configure(pg_params) and self.pg.get_opc()):
            self.logger.error("Error configuring PG.")
            return False

        self.length = self.pg.get_length()
        self.offsets = self.pg.get_offsets()

        return True

    def generate_blocks(self, data: PODMRData | None = None):
        if data is None:
            data = self.data
        generate = self.generators[data.params["method"]].generate
        params = data.get_pulse_params()
        if not self.conf.get("divide_block", False) and params["divide_block"]:
            self.logger.warn("divide_block is recommended to be False.")
        return generate(data.xdata, params)

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]
    ) -> bool:
        params = P.unwrap(params)
        d = PODMRData(params)
        blocks, freq, laser_timing = self.generate_blocks(d)
        offsets = self.pg.validate_blocks(blocks, freq)
        return offsets is not None

    def update(self) -> bool:
        if not self.data.running:
            return False

        if self.data.has_roi():
            roi = self.data.get_rois()
            data0 = self.tdc.get_data_roi(0, roi)
            data1 = self.tdc.get_data_roi(1, roi)
            # because length of each ROI fragments are all same,
            # we can convert list[ndarray] to 2D ndarray.
            if data0 is not None:
                data0 = np.array(data0)
            if data1 is not None:
                data1 = np.array(data1)
        else:
            data0 = self.tdc.get_data(0)
            data1 = self.tdc.get_data(1)

        if data0 is not None and data1 is not None:
            self.op.update(self.data, data0 + data1, self.get_tdc_status())
            self.op.get_marker_indices(self.data)
            self.op.analyze(self.data)

        return True

    def update_plot_params(self, params: dict) -> bool:
        if self.data.params is None:
            return False
        if self.op.update_plot_params(self.data, params):
            self.data.clear_fit_data()
        if not self.data.running:
            # when measument is running, re-analysis is done on next data update.
            # re-analyze here when measurement isn't running (viewing finished / loaded data).
            self.op.get_marker_indices(self.data)
            self.op.analyze(self.data)
        return True

    def start(self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is not None:
            params = P.unwrap(params)
        resume = params is None or ("resume" in params and params["resume"])
        if not resume:
            self.data = PODMRData(params)
            self.op.update_axes(self.data)
        else:
            self.data.update_params(params)

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not resume and not self.init_inst(self.data.params):
            return self.fail_with_release("Error initializing instruments.")

        # start instruments
        if resume:
            # update duration here because duration means duration of additional measurement.
            # this treatise is different from sweeps (sweeps limit is considered total).
            if params.get("duration", 0.0):
                success = self.tdc.set_duration(params["duration"])
            else:
                success = True
            success &= self.tdc.resume()
        else:
            success = self.tdc.stop()
            success &= self.tdc.clear()
            success &= self.tdc.start()

        success &= self.sg.set_output(not self.data.params.get("nomw", False))
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)
        # time.sleep(1)
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
        if not self.data.has_params() or not self.data.has_data():
            return False
        if self.data.params.get("sweeps", 0) > 0:
            return self.data.sweeps() >= self.data.params["sweeps"]
        # TDC may stop running by itself if duration limit is set
        return not self.get_tdc_running()

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = (
            self.pg.stop()
            and self.pg.release()
            and self.sg.set_output(False)
            and self.sg.release()
            and self.tdc.stop()
            and self.wait_tdc_stop()
            and self.update()
            and self.tdc.release()
        )
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(False)
        if self.fg is not None:
            success &= self.fg.release()

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

    def wait_tdc_stop(self, timeout_sec=60.0, interval_sec=0.2) -> bool:
        """Wait for TDC status become not-running (stopped)."""

        self.logger.debug("Waiting TDC stop")
        for i in range(int(round(timeout_sec / interval_sec))):
            if not self.get_tdc_running():
                return True
            time.sleep(interval_sec)

        self.logger.error(f"Timeout ({timeout_sec} sec) encountered in wait_tdc_stop!")
        return False

    def _get_param_dict_pulse(self, method: str, d: dict):
        ## common_pulses
        d["base_width"] = P.FloatParam(320e-9, 1e-9, 1e-4)
        d["laser_delay"] = P.FloatParam(45e-9, 0.0, 1e-4)
        d["laser_width"] = P.FloatParam(5e-6, 1e-9, 1e-4)
        d["mw_delay"] = P.FloatParam(1e-6, 0.0, 1e-4)
        d["trigger_width"] = P.FloatParam(20e-9, 1e-9, 1e-6)
        d["init_delay"] = P.FloatParam(0.0, 0.0, 1e-6)
        d["final_delay"] = P.FloatParam(5e-6, 0.0, 1e-4)

        ## common switches
        d["invertsweep"] = P.BoolParam(False)
        d["nomw"] = P.BoolParam(False)
        d["enable_reduce"] = P.BoolParam(False)
        d["divide_block"] = P.BoolParam(self.conf.get("divide_block", False))
        d["partial"] = P.IntParam(-1, -1, 1)

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
        if "flip_head" in pulse_params:
            d["flip_head"] = P.BoolParam(False)

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
            sweeps=P.IntParam(0, 0, 9999999, doc="limit number of sweeps"),
            duration=P.FloatParam(0.0, 0.0, 9999999, unit="s", doc="limit measurement duration"),
            ident=P.UUIDParam(optional=True, enable=False),
            roi_head=P.FloatParam(
                -1e-9,
                -1e-9,
                10e6,
                unit="s",
                doc="margin at head of ROI. negative value disables ROI.",
            ),
            roi_tail=P.FloatParam(
                -1e-9,
                -1e-9,
                10e6,
                unit="s",
                doc="margin at tail of ROI. negative value disables ROI.",
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

        taumodes = ["raw", "total", "freq", "index", "head"]
        if not (is_CPlike(label) or is_correlation(label) or label in ("spinecho", "trse")):
            taumodes.remove("total")
        if not ((is_CPlike(label) and not is_sweepN(label)) or label == "spinecho"):
            taumodes.remove("freq")

        d["plot"] = {
            "plotmode": P.StrChoiceParam(
                "data01",
                ("data01", "data0", "data1", "diff", "average", "normalize", "concatenate", "ref"),
            ),
            "taumode": P.StrChoiceParam("raw", taumodes),
            "logX": P.BoolParam(False),
            "logY": P.BoolParam(False),
            "fft": P.BoolParam(False),
            "sigdelay": P.FloatParam(200e-9, 0.0, 10e-6),
            "sigwidth": P.FloatParam(300e-9, 1e-9, 10e-6),
            "refdelay": P.FloatParam(2200e-9, 1e-9, 100e-6),
            "refwidth": P.FloatParam(2400e-9, 1e-9, 10e-6),
            "refmode": P.StrChoiceParam("subtract", ("subtract", "divide", "ignore")),
            "refaverage": P.BoolParam(False),
            "flipY": P.BoolParam(False),
        }
        return d

    def work(self):
        if not self.data.running:
            return

        if self.timer.check():
            self.update()

    def data_msg(self) -> PODMRData:
        return self.data

    def pulse_msg(self) -> PulsePattern | None:
        return self.pulse_pattern
