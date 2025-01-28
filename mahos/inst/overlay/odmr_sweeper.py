#!/usr/bin/env python3

"""
InstrumentOverlay for sweeping ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import time
import threading

import numpy as np

from .overlay import InstrumentOverlay
from ...msgs import param_msgs as P
from ...util.locked_queue import LockedQueue


class ODMRSweeperCommandBase(InstrumentOverlay):
    """ODMRSweeperCommandBase provides primitive operations for ODMR sweep.

    This class performs the sweep by issuing SG / PD commands every step.
    Thus, sweep speed will not be very good.

    :param sg: The reference to SG Instrument.
    :param pd: The reference to PD Instrument.
    :param queue_size: (default: 8) Size of queue of scanned line data.
    :type queue_size: int

    """

    def __init__(self, name, conf, prefix=None):
        InstrumentOverlay.__init__(self, name, conf=conf, prefix=prefix)
        self.sg = self.conf.get("sg")
        self.pd = self.conf.get("pd")
        self.add_instruments(self.sg, self.pd)

        self._queue_size = self.conf.get("queue_size", 8)
        self._queue = LockedQueue(self._queue_size)
        self._stop_ev = self._thread = None
        self.running = False

    def _set_attrs(self, params):
        self.start_f, self.stop_f = params["start"], params["stop"]
        self.num = params["num"]
        self.freqs = np.linspace(self.start_f, self.stop_f, self.num)
        self.power = params["power"]
        self.delay = params["delay"]
        self.background = params.get("background", False)
        self.bg_delay = params.get("background_delay", 0.0)
        self.continue_mw = params.get("continue_mw", False)

    def get_line(self):
        return self._queue.pop_block()

    def sweep_loop(self, ev: threading.Event):
        while True:
            line = []
            for f in self.freqs:
                self.sg.set_freq_CW(f)
                time.sleep(self.delay)
                res = self.get_pd_data()
                if ev.is_set():
                    self.logger.info("Quitting sweep loop.")
                    return
                line.append(res)

                if self.background:
                    self.sg.set_output(False, silent=True)
                    time.sleep(self.bg_delay)
                    res = self.get_pd_data()
                    if ev.is_set():
                        self.logger.info("Quitting sweep loop.")
                        return
                    line.append(res)
                    self.sg.set_output(True, silent=True)

            self._queue.append(np.array(line))

    def configure_pd(self):
        raise NotImplementedError("configure_pd is not implemented.")

    def get_pd_data(self):
        raise NotImplementedError("get_pd_data is not implemented.")

    def get_pd_param_dict(self) -> P.ParamDict[str, P.PDValue] | None:
        raise NotImplementedError("get_pd_param_dict is not implemented.")

    # Standard API

    def get_param_dict_labels(self) -> list[str]:
        return ["pd"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        if label == "pd":
            return self.get_pd_param_dict()

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "pulse":
            return self.fail_with("label='pulse' is not supported.")
        if not self.check_required_params(params, ("start", "stop", "num", "power", "delay")):
            return False

        self._set_attrs(params)
        self.params = params
        self._queue = LockedQueue(self._queue_size)

        if not self.configure_pd():
            return self.fail_with("failed to configure PD.")

        mod = params.get("mod", {})
        success = self.sg.configure_cw(self.start_f, self.power)
        if label == "iq_ext":
            success &= self.sg.configure_iq_ext()
        elif label == "iq_int":
            success &= self.sg.configure_iq_int()
        elif label == "fm_ext":
            success &= self.sg.configure_fm_ext(mod["fm_deviation"])
        elif label == "fm_int":
            success &= self.sg.configure_fm_int(mod["fm_deviation"], mod["fm_rate"])
        elif label == "am_ext":
            success &= self.sg.configure_am_ext(mod["am_depth"], mod["am_log"])
        elif label == "am_int":
            success &= self.sg.configure_am_int(mod["am_depth"], mod["am_log"], mod["am_rate"])
        if not success:
            return self.fail_with("failed to configure SG.")

        return True

    def start(self, label: str = "") -> bool:
        if self.running:
            self.logger.warn("start() is called while running.")
            return True

        if not self.sg.set_output(True):
            return self.fail_with("Failed to start sg.")
        if not self.pd.start():
            return self.fail_with("Failed to start pd.")

        self._stop_ev = threading.Event()
        self._thread = threading.Thread(target=self.sweep_loop, args=(self._stop_ev,))
        self._thread.start()

        self.running = True

        return True

    def stop(self, label: str = "") -> bool:
        if not self.running:
            return True
        self.running = False

        self.logger.info("Stopping sweeper.")

        self._stop_ev.set()
        self._thread.join()

        if self.continue_mw:
            self.logger.warn("Skipping to turn off MW output")
            success = True
        else:
            success = self.sg.set_output(False)
        success &= self.pd.stop()

        if not success:
            return self.fail_with("failed to stop SG or PD.")

        return True

    def get(self, key: str, args=None, label: str = ""):
        if key == "line":
            return self.get_line()
        elif key == "bounds":
            return self.sg.get_bounds()
        elif key == "unit":
            return self.pd.get("unit")
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class ODMRSweeperCommandAnalogPD(ODMRSweeperCommandBase):
    """ODMRSweeperCommand for AnalogPD (Photo Diode read with NI-DAQ AnalogIn)."""

    def get_pd_param_dict(self) -> P.ParamDict[str, P.PDValue] | None:
        d = P.ParamDict()
        d["bounds"] = [
            P.FloatParam(-10.0, -10.0, 10.0, unit="V", doc="lower bound of expected voltage"),
            P.FloatParam(10.0, -10.0, 10.0, unit="V", doc="upper bound of expected voltage"),
        ]
        return d

    def configure_pd(self):
        t = self.params["timing"]["time_window"]
        # TODO we are not sure if AnalogIn samples at max rate for on demand readout.
        rate = self.pd.get_max_rate()
        self._oversample = int(round(t * rate))
        self.logger.info(f"AnalogPD oversample: {self._oversample}")

        return self.pd.configure_on_demand(self.params["pd"])

    def get_pd_data(self):
        return self.pd.read_on_demand(self._oversample)


class ODMRSweeperCommandAnalogPDMM(ODMRSweeperCommandBase):
    """ODMRSweeperCommand for AnalogPDMM (Photo Diode read with DMM)."""

    def get_pd_param_dict(self) -> P.ParamDict[str, P.PDValue] | None:
        return self.pd.get_param_dict("pd")

    def configure_pd(self):
        return self.pd.configure(self.params["pd"])

    def get_pd_data(self):
        return self.pd.get_data()
