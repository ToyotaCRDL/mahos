#!/usr/bin/env python3

"""
Worker for HBT.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from ..util.timer import IntervalTimer
from ..msgs.hbt_msgs import HBTData
from ..msgs import param_msgs as P
from ..inst.tdc_interface import TDCInterface
from .common_worker import Worker


class Listener(Worker):
    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger)
        self.tdc = TDCInterface(cli, "tdc")
        self.add_instrument(self.tdc)

        self.interval_sec = conf.get("interval_sec", 1.0)
        self._default_t0 = conf.get("default_t0_ns", 100.0) * 1e-9

        self.data = HBTData()
        self.timer = None

    def update_plot_params(self, params: dict) -> bool:
        if not self.data.has_params():
            return False
        self.data.update_plot_params(params)
        self.data.clear_fit_data()
        return True

    def get_param_dict(self) -> P.ParamDict[str, P.PDValue]:
        plot_param = P.ParamDict(
            t0=P.FloatParam(self._default_t0, minimum=0.0, maximum=1e-6, unit="s", SI_prefix=True),
            ref_start=P.FloatParam(
                -200e-9, minimum=-10e-6, maximum=10e-6, unit="s", SI_prefix=True
            ),
            ref_stop=P.FloatParam(0.0, minimum=-10e-6, maximum=10e-6, unit="s", SI_prefix=True),
            bg_ratio=P.FloatParam(0.0, minimum=0.0, maximum=1.0),
        )
        d = P.ParamDict(
            bin=P.FloatParam(0.2e-9, minimum=1e-12, maximum=100e-9, unit="s", SI_prefix=True),
            range=P.FloatParam(1011e-9, minimum=1e-9, maximum=100e-6, unit="s", SI_prefix=True),
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
            c = {"file": "hbt", "range": params["range"], "bin": params["bin"]}

            success &= self.tdc.configure(c)
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
            self.data.start()
            self.logger.info("Started listener.")

        return True

    def work(self) -> bool:
        if not self.data.running:
            return False

        if self.timer.check():
            d = self.tdc.get_data(1)
            if d is not None:
                self.data.data = d
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
