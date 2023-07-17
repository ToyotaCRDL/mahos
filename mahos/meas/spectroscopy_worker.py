#!/usr/bin/env python3

"""
Worker for Spectroscopy.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np

from ..msgs.spectroscopy_msgs import SpectroscopyData
from ..msgs import param_msgs as P
from ..inst.spectrometer_interface import SpectrometerInterface
from .common_worker import Worker


class Repeater(Worker):
    """Worker for Repeated Spectroscopy."""

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger)
        self.spectrometer = SpectrometerInterface(cli, "spectrometer")
        self.add_instrument(self.spectrometer)

        self.data = SpectroscopyData()

    def get_param_dict(self) -> P.ParamDict[str, P.PDValue] | None:
        base_configs = self.spectrometer.get_base_configs()
        if base_configs is None:
            return None
        d = P.ParamDict(
            base_config=P.StrChoiceParam(base_configs[0], base_configs),
            exposure_time=P.FloatParam(1000.0, 1.0, 1e6, doc="exposure time in ms"),
            exposures=P.IntParam(0, 0, 10000, doc="number of exposures (0 for inf.)"),
            center_wavelength=P.FloatParam(700.0, 200.0, 1300.0, doc="center wavelength in nm"),
            resume=P.BoolParam(False),
            ident=P.UUIDParam(optional=True, enable=False),
        )
        return d

    def start(self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is not None:
            params = P.unwrap(params)
        success = self.spectrometer.lock()
        if params is not None:
            success &= self.spectrometer.configure(params)

        if not success:
            return self.fail_with_release("Error starting repeater.")

        if params is not None and not params["resume"]:
            # new measurement.
            self.data = SpectroscopyData(params)
            self.data.start()
            self.logger.info("Started repeater.")
        else:
            # TODO: check ident if resume?
            self.data.update_params(params)
            self.data.resume()
            self.logger.info("Resuming repeater.")

        return True

    def append_line(self, line):
        if not self.data.has_data():
            self.data.data = np.array(line, ndmin=2).T
        else:
            self.data.data = np.append(self.data.data, np.array(line, ndmin=2).T, axis=1)

    def work(self):
        if not self.data.running:
            return

        d = self.spectrometer.get_data()
        xdata, data = d[0], d[1]
        if self.data.xdata is None:
            self.data.xdata = xdata
        self.append_line(data)

    def is_finished(self) -> bool:
        if not self.data.has_params() or not self.data.has_data():
            return False
        if self.data.params.get("acquisitions", 0) <= 0:
            return False  # no acquisitions limit defined.
        return self.data.acquisitions() >= self.data.params["acquisitions"]

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.spectrometer.release()

        self.data.finalize()
        if success:
            self.logger.info("Stopped repeater.")
        else:
            self.logger.error("Error stopping repeater.")
        return success

    def data_msg(self) -> SpectroscopyData:
        return self.data
