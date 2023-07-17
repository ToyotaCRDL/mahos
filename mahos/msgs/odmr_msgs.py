#!/usr/bin/env python3

"""
Message Types for ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import msgpack

from .common_msgs import Request
from .common_meas_msgs import BasicMeasData
from .fit_msgs import PeakType
from ..util import conv


class ValidateReq(Request):
    """Validate Measurement Params Request"""

    def __init__(self, params: dict):
        self.params = params


class ODMRData(BasicMeasData):
    """Data type for ODMR measurement."""

    def __init__(self, params: dict | None = None):
        self.set_version(2)
        self.init_params(params)
        self.init_attrs()

        self.data = None
        self.bg_data = None

    def init_axes(self):
        self.xlabel: str = "Frequency"
        self.xunit: str = "Hz"
        self.ylabel: str = "Intensity"
        self.yunit: str = ""
        self.xscale: str = "linear"
        self.yscale: str = "linear"

    def has_data(self) -> bool:
        """return True if data is ready and valid data could be read out."""

        return self.data is not None

    def has_background(self) -> bool:
        return self.bg_data is not None and (self.bg_data > 0).all()

    def get_xdata(self):
        return np.linspace(self.params["start"], self.params["stop"], self.params["num"])

    def _get_ydata_nobg(self, last_n, normalize_n):
        ydata = np.mean(self.data[:, -last_n:], axis=1)
        if normalize_n:
            coeff = np.mean(np.sort(ydata)[-normalize_n:])
            if coeff != 0.0:
                ydata /= coeff
        return ydata, None

    def _get_ydata_bg(self, last_n, normalize_n):
        ydata = np.mean(self.data[:, -last_n:], axis=1)
        bg_ydata = np.mean(self.bg_data[:, -last_n:], axis=1)
        if normalize_n:
            return ydata / bg_ydata, None
        return ydata, bg_ydata

    def get_ydata(
        self, last_n: int = 0, normalize_n: int = 0
    ) -> tuple[NDArray | None, NDArray | None]:
        """get ydata.

        :returns: (raw_ydata, background_ydata) if normalize_n is 0 and background is available.
                  (raw_ydata, None) if normalize_n is 0 and background is not available.
                  (normalized_ydata, None) if normalize_n is positive.
                  (None, None) if data is not ready.

        """

        if not self.has_data():
            return None, None

        if self.has_background():
            return self._get_ydata_bg(last_n, normalize_n)
        else:
            return self._get_ydata_nobg(last_n, normalize_n)

    def get_fit_xdata(self):
        return self.fit_xdata

    def get_fit_ydata(self, last_n: int = 0, normalize_n: int = 0) -> NDArray | None:
        """get fit ydata."""

        if self.fit_data is None:
            return None

        if self.has_background():
            # fitting is done by using background-normalized data.
            # denormalized data is not available.
            if not normalize_n:
                return None
            return self.fit_data

        # no background
        # fitting is done by using raw data.
        if not normalize_n:
            return self.fit_data

        ## normalize by measured ydata.
        ydata = np.mean(self.data[:, -last_n:], axis=1)
        coeff = np.mean(np.sort(ydata)[-normalize_n:])
        if coeff == 0.0:
            return self.fit_data
        return self.fit_data / coeff

    def bounds(self) -> tuple[float, float] | None:
        if not self.has_params():
            return None
        return self.params["start"], self.params["stop"]

    def sweeps(self) -> int:
        if not self.has_data():
            return 0
        return self.data.shape[1]

    def measure_background(self) -> bool:
        if not self.has_params():
            return False
        return self.params.get("background", False)

    def step(self) -> float | None:
        if not self.has_params():
            return None
        return conv.num_to_step(self.params["start"], self.params["stop"], self.params["num"])

    def can_resume(self, params: dict | None) -> bool:
        """Check if the measurement can be resumed with given new_params."""

        if not self.has_params() or params is None:
            return False
        p = params
        return (
            self.params["start"] == p["start"]
            and self.params["stop"] == p["stop"]
            and self.params["num"] == p["num"]
        )

    def _h5_write_fit_params(self, val):
        d = val.copy()
        if "peak_type" in d:
            d["peak_type"] = d["peak_type"].value
        return np.void(msgpack.dumps(d))

    def _h5_read_fit_params(self, val):
        d = msgpack.loads(val.tobytes())
        if "peak_type" in d:
            d["peak_type"] = PeakType(d["peak_type"])
        return d

    def _h5_attr_writers(self) -> dict:
        return {"fit_params": self._h5_write_fit_params}

    def _h5_readers(self) -> dict:
        return {"fit_result": self._h5_read_fit_result, "fit_params": self._h5_read_fit_params}


def update_data(data: ODMRData):
    """update data to latest format"""

    if data.version() <= 0:
        # version 0 to 1
        ## add missing attributes
        data.clear_fit_data()
        data.init_axes()
        data._saved = True
        ## param change
        if data.has_params() and "pulsed" in data.params:
            data.params["method"] = "pulse" if data.params["pulsed"] else "cw"
            del data.params["pulsed"]
        data.set_version(1)

    if data.version() <= 1:
        # version 1 to 2
        if data.has_params():
            data.params["background"] = False
        data.bg_data = None
        data.set_version(2)

    return data
