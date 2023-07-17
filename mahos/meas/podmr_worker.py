#!/usr/bin/env python3

"""
Worker for Pulse ODMR.

.. This file is a part of MAHOS project.

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

        # [sec] ==> index
        signal_head = np.round(signal_head / tbin).astype(np.int64)
        signal_tail = np.round(signal_tail / tbin).astype(np.int64)
        reference_head = np.round(reference_head / tbin).astype(np.int64)
        reference_tail = np.round(reference_tail / tbin).astype(np.int64)

        data.marker_indices = np.vstack((signal_head, signal_tail, reference_head, reference_tail))
        return data.marker_indices

    def analyze(self, data: PODMRData) -> bool:
        if not data.has_raw_data() or data.marker_indices is None or data.tdc_status.sweeps < 1.0:
            return False

        if data.is_partial():
            return self._analyze_partial(data)
        else:
            return self._analyze_complementary(data)

    def _analyze_partial(self, data: PODMRData) -> bool:
        signal = np.zeros(len(data.xdata))
        reference = np.zeros(len(data.xdata))
        signal_head, signal_tail, reference_head, reference_tail = data.marker_indices

        for i in range(len(data.xdata)):
            try:
                signal[i] = np.mean(data.raw_data[signal_head[i] : signal_tail[i] + 1])
            except IndexError as e:
                print("analyze_sig (sig %d): %r" % (i, e))
                signal[i] = 0

            try:
                reference[i] = np.mean(data.raw_data[reference_head[i] : reference_tail[i] + 1])
            except IndexError as e:
                print("analyze_sig (ref %d): %r" % (i, e))
                reference[i] = 0

        sweeps = data.tdc_status.sweeps
        if data.partial() == 0:
            data.data0 = signal / sweeps
            data.data0ref = reference / sweeps
        else:  # assert data.partial() == 1
            data.data1 = signal / sweeps
            data.data1ref = reference / sweeps

        return True

    def _analyze_complementary(self, data: PODMRData) -> bool:
        signal = np.zeros(len(data.xdata) * 2)
        reference = np.zeros(len(data.xdata) * 2)
        signal_head, signal_tail, reference_head, reference_tail = data.marker_indices

        for i in range(len(data.xdata) * 2):
            try:
                signal[i] = np.mean(data.raw_data[signal_head[i] : signal_tail[i] + 1])
            except IndexError as e:
                print("analyze_sig (sig %d): %r" % (i, e))
                signal[i] = 0

            try:
                reference[i] = np.mean(data.raw_data[reference_head[i] : reference_tail[i] + 1])
            except IndexError as e:
                print("analyze_sig (ref %d): %r" % (i, e))
                reference[i] = 0

        sweeps = data.tdc_status.sweeps
        data.data0 = signal[0::2] / sweeps
        data.data1 = signal[1::2] / sweeps
        data.data0ref = reference[0::2] / sweeps
        data.data1ref = reference[1::2] / sweeps

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


class Pulser(Worker):
    def __init__(self, cli, logger, has_fg: bool, conf: dict):
        """Worker for Pulse ODMR.

        Function generator is an option (fg may be None).

        """

        Worker.__init__(self, cli, logger)
        self.sg = SGInterface(cli, "sg")
        self.pg = PGInterface(cli, "pg")
        self.tdc = TDCInterface(cli, "tdc")
        if has_fg:
            self.fg = FGInterface(cli, "fg")
        else:
            self.fg = None
        self.add_instruments(self.sg, self.pg, self.tdc, self.fg)

        self.timer = None
        self.length = self.offsets = self.freq = self.laser_timing = None

        self.generators = make_generators(
            freq=conf.get("freq", 2.0e9),
            reduce_start_divisor=conf.get("reduce_start_divisor", 2),
            split_fraction=conf.get("split_fraction", 4),
            minimum_block_length=conf.get("minimum_block_length", 1000),
            block_base=conf.get("block_base", 4),
            print_fn=self.logger.info,
        )
        self.eos_margine = conf.get("eos_margine", 1e-6)

        self.data = PODMRData()
        self.op = PODMRDataOperator()
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

        # Detector
        tdc_params = {
            "file": "podmr",
            "bin": params["timebin"],
            "range": self.length / self.freq - self.eos_margine,
        }
        if not self.tdc.configure(tdc_params):
            self.logger.error("Error configuring TDC.")
            return False
        if params["sweeps"] and not self.tdc.set_sweeps(params["sweeps"]):
            self.logger.error("Error setting sweeps for TDC.")
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
            self.fg.configure_cw(c["wave"], c["freq"], c["ampl"])
        elif c["mode"] == "gate":
            self.fg.configure_gate(c["wave"], c["freq"], c["ampl"], c["phase"])
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
        return generate(data.xdata, data.get_pulse_params())

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
            success = self.tdc.resume()
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
        if self.data.params.get("sweeps", 0) <= 0:
            return False  # no sweeps limit defined.
        return self.data.sweeps() >= self.data.params["sweeps"]

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = (
            self.tdc.stop()
            and self.wait_tdc_stop()
            and self.update()
            and self.tdc.release()
            and self.pg.stop()
            and self.pg.release()
            and self.sg.set_output(False)
            and self.sg.release()
        )
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(False)
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

        return TDCStatus(st0.runtime, st0.sweeps, st0.starts, st0.totalsum, st1.totalsum)

    def get_tdc_running(self) -> bool:
        """return True if TDC is running."""

        st0 = self.tdc.get_status(0)
        return bool(st0.started)

    def wait_tdc_stop(self, timeout_sec=10.0, interval_sec=0.2) -> bool:
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
        d["divide_block"] = P.BoolParam(True)
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

        return d

    def get_param_dict_names(self) -> list:
        return list(self.generators.keys())

    def get_param_dict(self, method: str) -> P.ParamDict[str, P.PDValue] | None:
        if method not in self.generators:
            self.logger.error(f"Unknown method {method}")
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
            method=P.StrChoiceParam(method, list(self.generators.keys())),
            resume=P.BoolParam(False),
            freq=P.FloatParam(2.80e9, f_min, f_max),
            power=P.FloatParam(p_min, p_min, p_max),
            timebin=P.FloatParam(3.2e-9, 0.1e-9, 100e-9),
            interval=P.FloatParam(1.0, 0.1, 10.0),
            sweeps=P.IntParam(0, 0, 9999999),
            ident=P.UUIDParam(optional=True, enable=False),
        )

        self._get_param_dict_pulse(method, d)
        self._get_param_dict_pulse_opt(method, d)

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
        if not (is_CPlike(method) or is_correlation(method) or method in ("spinecho", "trse")):
            taumodes.remove("total")
        if not ((is_CPlike(method) and not is_sweepN(method)) or method == "spinecho"):
            taumodes.remove("freq")

        d["plot"] = {
            "plotmode": P.StrChoiceParam(
                "data01",
                ("data01", "data0", "data1", "diff", "average", "normalize", "concatenate", "ref"),
            ),
            "taumode": P.StrChoiceParam("raw", taumodes),
            "xlogscale": P.BoolParam(False),
            "ylogscale": P.BoolParam(False),
            "fft": P.BoolParam(False),
            "sigdelay": P.FloatParam(200e-9, 0.0, 10e-6),
            "sigwidth": P.FloatParam(300e-9, 1e-9, 10e-6),
            "refdelay": P.FloatParam(2200e-9, 1e-9, 100e-6),
            "refwidth": P.FloatParam(2400e-9, 1e-9, 10e-6),
            "refmode": P.StrChoiceParam("subtract", ("subtract", "divide", "ignore")),
            "refaverage": P.BoolParam(False),
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
