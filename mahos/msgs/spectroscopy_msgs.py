#!/usr/bin/env python3

"""
Message Types for Spectroscopy.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np
import msgpack

from .common_meas_msgs import BasicMeasData
from .fit_msgs import PeakType

from ..util.stat import filter_outlier_2d


class SpectroscopyData(BasicMeasData):
    def __init__(self, params: dict | None = None):
        self.set_version(1)
        self.init_params(params)
        self.init_attrs()

        self.data = None
        self.xdata = None

    def init_axes(self):
        self.xlabel: str = "Wavelength"
        self.xunit: str = "nm"
        self.ylabel: str = "Intensity"
        self.yunit: str = ""
        self.xscale: str = "linear"
        self.yscale: str = "linear"

    def has_data(self) -> bool:
        """return True if current data is valid (not empty)."""

        return self.data is not None

    def acquisitions(self) -> int:
        if not self.has_data():
            return 0
        return self.data.shape[1]

    def get_xdata(self):
        return self.xdata

    def get_ydata(self, last_n: int = 0, filter_n: float = 0.0):
        d = self.data[:, -last_n:]
        if not filter_n:
            return np.mean(d, axis=1)
        df, outlier = filter_outlier_2d(d, n=filter_n, axis=1, both=False)
        if df.shape[1] == 0:  # all is outlier because filter_n is too small. fallback.
            return np.mean(d, axis=1)
        return np.mean(df, axis=1)

    def n_outliers(self, last_n: int = 0, filter_n: float = 0.0) -> int:
        d = self.data[:, -last_n:]
        if not filter_n:
            return 0
        df, outlier = filter_outlier_2d(d, n=filter_n, axis=1, both=False)
        if df.shape[1] == 0:  # all is outlier because filter_n is too small. fallback.
            return 0
        return outlier.shape[1]

    def get_fit_xdata(self):
        return self.fit_xdata

    def get_fit_ydata(self):
        return self.fit_data

    def can_resume(self, params: dict | None) -> bool:
        """Check if the measurement can be continued with given new_params."""

        if not self.has_params() or params is None:
            return False
        p = params
        return (
            self.params["base_config"] == p["base_config"]
            and self.params["center_wavelength"] == p["center_wavelength"]
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


def update_data(data: SpectroscopyData):
    """update data to latest format"""

    if data.version() == 0:
        # version 0 to 1
        ## add missing attributes
        data.clear_fit_data()
        data.init_axes()
        data._saved = True
        data.set_version(1)

    return data
