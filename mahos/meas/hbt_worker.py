#!/usr/bin/env python3

"""
Worker for HBT.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from ..util.timer import IntervalTimer
from ..msgs.hbt_msgs import HBTData
from ..msgs import param_msgs as P
from ..inst.tdc_interface import TDCInterface
from ..util.conf import PresetLoader
from .common_worker import Worker


class Listener(Worker):
    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger, conf)
        self.load_conf_preset(cli)

        self.tdc = TDCInterface(cli, "tdc")
        self.add_instruments(self.tdc)

        self.interval_sec = conf.get("interval_sec", 1.0)
        self.tdc_correlation = self.conf.get("tdc_correlation", False)
        self.tdc_normalize = self.conf.get("tdc_normalize", False)

        if self.tdc_correlation and "t0_ns" in conf:
            self.logger.warn("conf['t0_ns'] is irrelevant if conf['tdc_correlation'] is True.")
        elif "default_t0_ns" in conf:
            self.logger.warn("conf['default_t0_ns'] is deprecated. use 't0_ns' instead.")
            self._t0 = conf.get("default_t0_ns", 100.0) * 1e-9
        else:
            self._t0 = conf.get("t0_ns", 100.0) * 1e-9

        self.data = HBTData()
        self.timer = None

    def load_conf_preset(self, cli):
        loader = PresetLoader(self.logger, PresetLoader.Mode.FORWARD)
        loader.add_preset(
            "MCS",
            [
                ("tdc_correlation", False),
                ("tdc_normalize", False),
            ],
        )
        loader.add_preset(
            "TimeTagger",
            [
                ("tdc_correlation", True),
                ("tdc_normalize", True),
            ],
        )
        loader.load_preset(self.conf, cli.class_name("tdc"))

    def update_plot_params(self, params: dict) -> bool:
        if not self.data.has_params():
            return False
        self.data.update_plot_params(params)
        self.data.clear_fit_data()
        return True

    def get_param_dict(self) -> P.ParamDict[str, P.PDValue]:
        if self.tdc_correlation:
            t0 = P.FloatParam(0.0, 0.0, 0.0, unit="s", SI_prefix=True)
        else:
            t0 = P.FloatParam(self._t0, 0.0, 1e-6, unit="s", SI_prefix=True)
        if self.tdc_normalize:
            ref_start = P.FloatParam(0.0, 0.0, 0.0, unit="s", SI_prefix=True)
            ref_stop = P.FloatParam(0.0, 0.0, 0.0, unit="s", SI_prefix=True)
            bg_ratio = P.FloatParam(0.0, 0.0, 0.0)
        else:
            ref_start = P.FloatParam(-200e-9, -10e-6, 10e-6, unit="s", SI_prefix=True)
            ref_stop = P.FloatParam(0.0, -10e-6, 10e-6, unit="s", SI_prefix=True)
            bg_ratio = P.FloatParam(0.0, 0.0, 1.0)
        plot_param = P.ParamDict(t0=t0, ref_start=ref_start, ref_stop=ref_stop, bg_ratio=bg_ratio)
        d = P.ParamDict(
            bin=P.FloatParam(0.2e-9, 1e-12, 100e-9, unit="s", SI_prefix=True),
            range=P.FloatParam(
                self.conf.get("range_ns", 1011) * 1e-9, 0.1e-9, 100e-6, unit="s", SI_prefix=True
            ),
            plot=plot_param,
            resume=P.BoolParam(False),
            ident=P.UUIDParam(optional=True, enable=False),
        )
        return d

    def start(self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is not None:
            params = P.unwrap(params)
        resume = params is None or ("resume" in params and params["resume"])

        success = self.tdc.lock()
        if resume:
            success &= self.tdc.resume()
        else:
            if not ("range" in params and "bin" in params):
                self.logger.error("Required params: range and bin.")
                return False

            if self.tdc_correlation:
                success &= self.tdc.configure_correlation("hbt", params["range"], params["bin"])
            else:
                success &= self.tdc.configure_histogram("hbt", params["range"], params["bin"])
            timebin = self.tdc.get_timebin()
            success &= self.tdc.stop()
            # necessary to call clear() here?
            success &= self.tdc.start()

        if not success:
            return self.fail_with_release("Error starting listener.")

        self.timer = IntervalTimer(self.interval_sec)

        if resume:
            # TODO: check ident if resume?
            self.data.update_params(params)
            self.data.resume()
            self.logger.info("Resuming listener.")
        else:
            # new measurement.
            self.data = HBTData(params)
            self.data.set_bin(timebin)
            self.data.tdc_correlation = self.tdc_correlation
            self.data.start()
            self.logger.info("Started listener.")

        return True

    def work(self) -> bool:
        if not self.data.running:
            return False

        if self.timer.check():
            ch = self.conf.get("tdc_channel", 1)
            d = self.tdc.get_data(ch)
            if self.tdc_normalize:
                d_normalized = self.tdc.get_data_normalized(ch)
            else:
                d_normalized = None
            if d is not None:
                self.data.data = d
                self.data.data_normalized = d_normalized
                return True
        return False

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.tdc.stop() and self.tdc.release()

        self.timer = None
        self.data.finalize()

        if success:
            self.logger.info("Stopped listener.")
        else:
            self.logger.error("Error stopping listener.")
        return success

    def data_msg(self) -> HBTData:
        return self.data
