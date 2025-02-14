#!/usr/bin/env python3

"""
Worker for Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import time
import copy

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
        if all([ofs == 0 for ofs in offsets]):
            data.params["instrument"]["offsets"] = []
        else:
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
        self._sgs = {}
        self._fg = None

    def has_sg(self, i):
        return i in self._sgs and self._sgs[i] is not None

    def sg(self, i):
        return self._sgs[i]

    def set_sg(self, i, sg_bounds):
        self._sgs[i] = sg_bounds

    def has_fg(self):
        return self._fg is not None

    def fg(self):
        return self._fg

    def set_fg(self, fg_bounds):
        self._fg = fg_bounds


class Pulser(Worker):
    """Worker for Pulse ODMR.

    Function generator is an option (fg may be None).

    """

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger, conf)
        self.load_conf_preset(cli)

        self.sgs = {"sg": SGInterface(cli, "sg")}
        _default_channels = [{"sg": "sg"}]
        for i in range(1, 10):
            name = f"sg{i}"
            if name in cli.insts():
                self.sgs[name] = SGInterface(cli, name)
                _default_channels.append({"sg": name})

        self.mw_modes = tuple(self.conf.get("mw_modes", (0,) * len(self.sgs)))
        self.mw_channels = self.conf.get("mw_channels", _default_channels)

        self.pg = PGInterface(cli, "pg")
        self.tdc = TDCInterface(cli, "tdc")
        if "fg" in cli:
            self.fg = FGInterface(cli, "fg")
        else:
            self.fg = None
        self.add_instruments(self.pg, self.tdc, self.fg, *self.sgs.values())

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
            mw_modes=self.mw_modes,
            iq_amplitude=self.conf.get("iq_amplitude", 0.0),
            channel_remap=self.conf.get("channel_remap"),
            print_fn=self.logger.info,
        )
        self.eos_margin = self.conf.get("eos_margin", 1e-6)
        self._quick_resume = self.conf.get("quick_resume", True)
        self._resume_raw_data = None
        self._resume_tdc_status = None

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
        loader.add_preset(
            "SpinCore_PulseBlaster",
            [
                ("block_base", 1),
                ("pg_freq", 0.5e9),
                ("reduce_start_divisor", 5),
                ("minimum_block_length", 5),
                ("divide_block", False),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("pg"))

    def init_inst(self, params: dict) -> bool:
        # Generators
        if not self.init_sg(params):
            self.logger.error("Error initializing SG.")
            return False
        if not self.init_fg(params):
            self.logger.error("Error initializing FG.")
            return False
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

    def init_sg(self, params: dict) -> bool:
        configured = []
        for i, (channel, mode) in enumerate(zip(self.mw_channels, self.mw_modes)):
            sg: SGInterface = self.sgs[channel["sg"]]
            ch = channel.get("ch", 1)
            reset = channel["sg"] not in configured
            idx = "" if not i else i
            freq = params[f"freq{idx}"]
            power = params[f"power{idx}"]

            if mode in (0, 2):
                if not sg.configure_cw_iq_ext(freq, power, ch=ch, reset=reset):
                    self.logger.error(f"Error initializing SG{idx}.")
                    return False
            else:  # mode 1
                if not sg.configure_cw(freq, power, ch=ch, reset=reset):
                    self.logger.error(f"Error initializing SG{idx}.")
                    return False
            configured.append(channel["sg"])
        return True

    def start_sg(self, params: dict) -> bool:
        for i, channel in enumerate(self.mw_channels):
            sg: SGInterface = self.sgs[channel["sg"]]
            ch = channel.get("ch", 1)
            idx = "" if not i else i
            nomw = params.get(f"nomw{idx}", False)
            if not sg.set_output(not nomw, ch=ch):
                self.logger.error(f"Error starting SG{idx}.")
                return False
        return True

    def stop_sg(self) -> bool:
        success = True
        for i, channel in enumerate(self.mw_channels):
            sg: SGInterface = self.sgs[channel["sg"]]
            ch = channel.get("ch", 1)
            success &= sg.set_output(False, ch=ch) and sg.release()
        return success

    def get_sg_bounds(self, i: int):
        channel = self.mw_channels[i]
        sg: SGInterface = self.sgs[channel["sg"]]
        ch = channel.get("ch", 1)
        return sg.get_bounds(ch)

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
        generate = self.generators[data.label].generate
        params = data.get_params()
        if not self.conf.get("divide_block", False) and params["divide_block"]:
            self.logger.warn("divide_block is recommended to be False.")
        return generate(data.xdata, params)

    def validate_params(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str
    ) -> bool:
        params = P.unwrap(params)
        d = PODMRData(params, label)
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
            if self._resume_raw_data is not None:
                new_data = data0 + data1 + self._resume_raw_data
            else:
                new_data = data0 + data1
            self.op.update(self.data, new_data, self.get_tdc_status())
            self.op.get_marker_indices(self.data)
            self.op.analyze(self.data)

        return True

    def update_plot_params(self, params: dict) -> bool:
        if self.data.params is None:
            return False
        if self.op.update_plot_params(self.data, params):
            self.data.remove_fit_data()
        if not self.data.running:
            # when measument is running, re-analysis is done on next data update.
            # re-analyze here when measurement isn't running (viewing finished / loaded data).
            self.op.get_marker_indices(self.data)
            self.op.analyze(self.data)
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
            self.data = PODMRData(params, label)
            self.op.update_axes(self.data)
        else:
            _last_duration = self.data.params.get("duration", 0.0)
            self.data.update_params(params)

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if quick_resume:
            self.logger.info("Quick resume enabled: skipping initial inst configurations.")
        if not quick_resume and not self.init_inst(self.data.params):
            return self.fail_with_release("Error initializing instruments.")
        if resume:
            self._resume_raw_data = self.data.raw_data.copy()
            self._resume_tdc_status = copy.copy(self.data.tdc_status)
        else:
            self._resume_raw_data = None
            self._resume_tdc_status = None

        # update duration here because duration means duration of additional measurement.
        # this treatise is different from sweeps (sweeps limit is considered total).
        if (
            quick_resume
            and params is not None
            and params.get("duration", 0.0) != _last_duration
            and not self.tdc.set_duration(params["duration"])
        ):
            return self.fail_with_release("Failed to set tdc duration.")

        # start instruments
        success = self.tdc.stop() and self.tdc.clear() and self.tdc.start()
        success &= self.start_sg(self.data.params)
        if self._fg_enabled(self.data.params):
            success &= self.fg.set_output(True)
        # time.sleep(1)
        success &= self.pg.start()

        if not success:
            self.pg.stop()
            if self._fg_enabled(self.data.params):
                self.fg.set_output(False)
            self.stop_sg()
            self.tdc.stop()
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
            and self.stop_sg()
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

        if self._resume_tdc_status:
            r: TDCStatus = self._resume_tdc_status
            return TDCStatus(
                round(st0.runtime) + r[0], st0.starts + r[1], st0.total + r[2], st1.total + r[3]
            )
        else:
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

    def _get_param_dict_pulse(self, label: str, d: dict):
        ## common_pulses
        d["base_width"] = P.FloatParam(320e-9, 1e-9, 1e-4)
        d["laser_delay"] = P.FloatParam(45e-9, 0.0, 1e-4)
        d["laser_width"] = P.FloatParam(3e-6, 1e-9, 1e-4)
        d["mw_delay"] = P.FloatParam(1e-6, 0.0, 1e-4)
        d["trigger_width"] = P.FloatParam(20e-9, 1e-9, 1e-6)
        d["init_delay"] = P.FloatParam(0.0, 0.0, 1e-6)
        d["final_delay"] = P.FloatParam(5e-6, 0.0, 1e-4)

        ## common switches
        d["invert_sweep"] = P.BoolParam(False)
        d["enable_reduce"] = P.BoolParam(False)
        d["divide_block"] = P.BoolParam(self.conf.get("divide_block", False))
        d["partial"] = P.IntParam(-1, -1, 1)

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

        # fundamentals
        d = P.ParamDict(
            resume=P.BoolParam(False),
            quick_resume=P.BoolParam(False),
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

        for i in range(len(self.mw_channels)):
            idx = "" if not i else i
            if self.bounds.has_sg(i):
                sg = self.bounds.sg(i)
            else:
                sg = self.get_sg_bounds(i)
                if sg is None:
                    self.logger.error(f"Failed to get SG{idx} bounds.")
                    return None
                self.bounds.set_sg(i, sg)

            f_min, f_max = sg["freq"]
            p_min, p_max = sg["power"]
            sg_freq = max(min(self.conf.get(f"sg{idx}_freq", 2.8e9), f_max), f_min)
            d[f"freq{idx}"] = P.FloatParam(sg_freq, f_min, f_max)
            d[f"power{idx}"] = P.FloatParam(p_min, p_min, p_max)
            d[f"nomw{idx}"] = P.BoolParam(False)

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
