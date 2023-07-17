#!/usr/bin/env python3

"""
Message Types for HBT.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np

from .common_msgs import Request
from .common_meas_msgs import BasicMeasData


class UpdatePlotParamsReq(Request):
    """Request to update the plot params"""

    def __init__(self, params: dict):
        self.params = params


class HBTData(BasicMeasData):
    def __init__(self, params: dict | None = None):
        self.set_version(1)
        self.init_params(params)
        self.init_attrs()

        self.data = None
        self.timebin = None

    def init_axes(self):
        self.xlabel: str = "Time"
        self.xunit: str = "s"
        self.ylabel: str = "Events"
        self.yunit: str = ""
        self.xscale: str = "linear"
        self.yscale: str = "linear"

    def has_data(self) -> bool:
        """return True if current data is valid (not empty)."""

        return self.data is not None and self.timebin is not None

    def update_plot_params(self, plot_params: dict):
        if not self.has_params():
            return
        if "plot" not in self.params:
            self.params["plot"] = plot_params
        else:
            self.params["plot"].update(plot_params)

    def get_xdata(self, normalize: bool = False):
        if not self.has_data():
            return None

        x = np.linspace(0.0, self.timebin * (len(self.data) - 1), num=len(self.data))
        if normalize:
            return x - self.get_t0()
        else:
            return x

    def get_ydata(self, normalize: bool = False, fit: bool = False):
        if normalize:
            return self._get_normalized_ydata(fit)
        return self.fit_data if fit else self.data

    def _get_normalized_ydata(self, fit: bool):
        ref = self.get_reference()
        bg = ref * self.get_bg_ratio()
        y = self.fit_data if fit else self.data
        if y is None:
            return None
        return (y - bg) / (ref - bg)

    def get_fit_xdata(self, normalize: bool = False):
        if self.fit_xdata is None:
            return None

        if normalize:
            return self.fit_xdata - self.get_t0()
        else:
            return self.fit_xdata

    def get_fit_ydata(self, normalize: bool = False):
        return self.get_ydata(normalize=normalize, fit=True)

    def get_t0(self) -> float:
        return self.params["plot"]["t0"]

    def get_bg_ratio(self) -> float:
        return self.params["plot"]["bg_ratio"]

    def get_reference_window(self) -> tuple[float, float]:
        ref_start = self.params["plot"]["ref_start"]
        ref_stop = self.params["plot"]["ref_stop"]
        t = self.get_xdata()
        if ref_start <= 0.0:
            ref_start += np.max(t)
        if ref_stop <= 0.0:
            ref_stop += np.max(t)
        return sorted([ref_start, ref_stop])

    def get_reference(self) -> float:
        t = self.get_xdata()
        y = self.get_ydata()
        ref_start, ref_stop = self.get_reference_window()
        idx = np.where(np.logical_and((ref_start <= t), (t <= ref_stop)))
        yf = y[idx]
        if len(yf) == 0:
            return 1.0
        return np.mean(y[idx])

    def denormalize_xdata(self, xdata):
        return xdata + self.get_t0()

    def denormalize_ydata(self, ydata):
        ref = self.get_reference()
        bg = ref * self.get_bg_ratio()
        return (ref - bg) * ydata + bg

    def get_bin(self):
        return self.timebin

    def get_range(self):
        return len(self.get_xdata())

    def set_bin(self, timebin):
        self.timebin = timebin

    def can_resume(self, params: dict | None) -> bool:
        """Check if the measurement can be resumed with given new params."""

        if not self.has_params() or params is None:
            return False
        p = params
        return self.params["range"] == p["range"] and self.params["bin"] == p["bin"]


def update_data(data: HBTData):
    """update data to latest format"""

    if data.version() == 0:
        # version 0 to 1
        ## hold y only
        x = data.data[:, 0]
        data.data = data.data[:, 1]
        ## add missing attributes
        data.clear_fit_data()
        data.init_axes()
        data.set_bin(x[1])
        data.update_plot_params(
            {"ref_start": -200e-9, "ref_stop": 0.0, "bg_ratio": 0.0, "t0": 100e-9}
        )
        data._saved = True
        data.set_version(1)

    return data
