#!/usr/bin/env python3

"""
Message Types for Pulse ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import typing as T

import numpy as np
from numpy.typing import NDArray

from ..util.comp import dict_isclose
from .common_msgs import Request
from .common_meas_msgs import BasicMeasData


class ValidateReq(Request):
    """Validate Measurement Params Request"""

    def __init__(self, params: dict):
        self.params = params


class DiscardReq(Request):
    """Discard Data Request"""

    pass


class UpdatePlotParamsReq(Request):
    """Request to update the plot param"""

    def __init__(self, params: dict):
        self.params = params


class TDCStatus(T.NamedTuple):
    runtime: float
    sweeps: float
    starts: float
    stops0: float
    stops1: float


def is_sweepN(method: str) -> bool:
    return method in ("cpN", "cpmgN", "xy4N", "xy8N", "xy16N", "ddgateN", "xy8clNflip")


def is_CPlike(method: str) -> bool:
    return method in (
        "cp",
        "cpN",
        "cpmg",
        "cpmgN",
        "xy4",
        "xy4N",
        "xy8",
        "xy8N",
        "xy16",
        "xy16N",
        "ddgate",
        "ddgateN",
    )


def is_correlation(method: str) -> bool:
    return method in ("xy8cl", "xy8cl1flip", "xy8clNflip")


class PODMRData(BasicMeasData):
    def __init__(self, params: dict | None = None):
        self.set_version(2)
        self.init_params(params)
        self.init_attrs()

        self.init_xdata()

        self.tdc_status: TDCStatus | None = None

        self.data0 = None
        self.data1 = None
        self.data0ref = None
        self.data1ref = None

        self.raw_data = None
        self.raw_xdata = None

        self.marker_indices = None
        self.laser_timing = None

    def init_xdata(self):
        if not self.has_params():
            # dummy xdata
            self.xdata = np.arange(101) * 1e-9
            return

        # xdata is N (not tau)
        if self.is_sweepN():
            i = np.arange(self.params["Nnum"], dtype=np.int64)
            self.xdata = self.params["Nstart"] + i * self.params["Nstep"]
        else:
            # supersampling
            if self.is_supersampling():
                i = np.arange(self.params["num"])
                pulse_num = {"xy8": 8, "xy16": 16}[self.params["method"]] * self.params["Nconst"]
                sample = np.linspace(0, 1, num=self.params["supersample"] + 1)
                i_sub = (sample * pulse_num).astype(np.int64)[:-1] / pulse_num

                i = np.array([i_sub + j for j in i]).reshape(len(i) * len(i_sub))
                self.xdata = self.params["start"] + i * self.params["step"]

            # xlog
            elif self.params["log"]:
                start = np.log10(self.params["start"])
                stop = np.log10(
                    self.params["start"] + (self.params["num"] - 1) * self.params["step"]
                )
                x = np.logspace(start, stop, self.params["num"])
                if (self.params["start"] >= 1e-6) and (self.params["step"] >= 1e-6):
                    self.xdata = (x * 1e9 / 10).astype(np.int64) * 10 / 1e9  # round in [10 ns]
                else:
                    self.xdata = (x * 1e9).astype(np.int64) / 1e9  # round in [ns]

            # normal
            else:
                i = np.arange(self.params["num"])
                self.xdata = self.params["start"] + i * self.params["step"]

        if self.params.get("invertsweep", False):
            self.xdata = self.xdata[::-1]

    # h5

    def _h5_write_tdc_status(self, val):
        return np.array(val)

    def _h5_read_tdc_status(self, val):
        return TDCStatus(*val)

    def _h5_attr_writers(self) -> dict:
        return {"fit_result": self._h5_write_fit_result, "tdc_status": self._h5_write_tdc_status}

    def _h5_readers(self) -> dict:
        return {"fit_result": self._h5_read_fit_result, "tdc_status": self._h5_read_tdc_status}

    # getters

    def get_bin(self):
        try:
            return self.params["instrument"]["tbin"]
        except (KeyError, TypeError):
            return None

    def get_range(self):
        try:
            return self.params["instrument"]["trange"]
        except (KeyError, TypeError):
            return None

    def get_raw_xdata(self):
        if self.raw_xdata is not None:
            return self.raw_xdata

        if (self.raw_data is None) or ("instrument" not in self.params):
            return None

        self.raw_xdata = np.arange(len(self.raw_data)) * self.params["instrument"]["tbin"]
        return self.raw_xdata

    def get_fit_xdata(self) -> NDArray | None:
        return self.get_xdata(fit=True)

    def _concat_xdata(self, xdata):
        if not self.is_partial() and self.params["plot"]["plotmode"] == "concatenate":
            xdata = np.column_stack((xdata, xdata)).reshape(len(xdata) * 2)
        return xdata

    def get_total_scale(self):
        """Get the scaling parameters to get total precession time.

        :returns: (a, b) where total precession time is a * raw_xdata + b

        """

        m = self.get_method()
        a, b = 1, 0
        if m == "spinecho":
            a = 2
        elif m == "trse":
            b = self.params["tauconst"]
        elif m in ["cp", "cpmg", "ddgate"]:
            a = 2 * self.params["Nconst"]
        elif m == "xy4":
            a = 2 * self.params["Nconst"] * 4
        elif m == "xy8":
            a = 2 * self.params["Nconst"] * 8
        elif m == "xy16":
            a = 2 * self.params["Nconst"] * 16

        elif m in ["cpN", "cpmgN"]:
            a = 2 * self.params["tauconst"]
        elif m == "xy4N":
            a = 2 * self.params["tauconst"] * 4
        elif m == "xy8N":
            a = 2 * self.params["tauconst"] * 8
        elif m == "xy16N":
            a = 2 * self.params["tauconst"] * 16

        elif m == "xy8cl":
            a = 1
        elif m == "xy8cl1flip":
            a = 2
            b = self.params["180pulse"]
        elif m == "xy8clNflip":
            a = 2 * self.params["tauconst"] + self.params["180pulse"]
            b = 0
        else:
            raise ValueError(f"invalid method {m} for taumode == total")

        return a, b

    def _get_xdata_total(self, xdata):
        a, b = self.get_total_scale()
        return a * xdata + b

    def _get_xdata_freq(self, xdata):
        m = self.get_method()
        if self.is_CPlike() and not self.is_sweepN():
            pi = self.get_pulse_params()["180pulse"]
            # freq = 1 / (4 * tau + 2 * 180pulse)
            return 1 / (xdata * 4 + pi * 2)
        elif m == "spinecho":
            pi = self.get_pulse_params()["180pulse"]
            # freq = 1 / (2 * tau + 180pulse)
            return 1 / (xdata * 2 + pi)
        else:
            raise ValueError(f"invalid method {m} for taumode == freq")

    def _get_xdata_head(self, xdata):
        signal_head = self.marker_indices[0]
        tbin = self.params["instrument"]["tbin"]
        if self.is_partial() or self.params["plot"]["plotmode"] == "concatenate":
            return signal_head * tbin
        else:
            return signal_head[0::2] * tbin

    def get_xdata(self, fit=False, force_taumode: str = "") -> NDArray | None:
        """get analyzed xdata.

        :returns: if data is not ready, None is returned.
        :raises ValueError: when taumode is unknown or method is invalid for given taumode.

        """

        if fit:
            xdata = self.fit_xdata
        else:
            xdata = self.xdata
        if xdata is None:
            return None

        taumode = force_taumode or self.params["plot"]["taumode"]
        if taumode == "raw":
            return self._concat_xdata(xdata)
        elif taumode == "total":
            return self._concat_xdata(self._get_xdata_total(xdata))
        elif taumode == "freq":
            return self._concat_xdata(self._get_xdata_freq(xdata))
        elif taumode == "index":
            return self._concat_xdata(np.arange(len(xdata)))
        elif taumode == "head":
            if fit:  # not for fit
                return None
            return self._get_xdata_head(xdata)
        else:
            raise ValueError(f"unknown taumode {taumode}")

    def get_fit_ydata(self):
        return self.fit_data

    def get_ydata(self) -> tuple[NDArray | None, NDArray | None]:
        """get analyzed ydata.

        :returns: (ydata0, ydata1) if two types of data are available.
                  (ydata, None) if only one type of data is available.
                  (None, None) if data is not ready.
        :raises ValueError: when mode is unknown.

        """

        if not self.has_data():
            return None, None

        if self.is_partial():
            return self._get_ydata_partial()
        else:
            return self._get_ydata_complementary()

    def _get_ydata_partial(self):
        if self.partial() == 0:
            data = self.data0
            ref = self.data0ref
        else:  # assert self.partial() == 1
            data = self.data1
            ref = self.data1ref

        refmode = self.params["plot"]["refmode"]
        if refmode == "subtract":
            s = data - ref
        elif refmode == "divide":
            s = data / ref
        elif refmode == "ignore":
            s = data
        else:
            raise ValueError(f"unknown refmode {refmode}")

        plotmode = self.params["plot"]["plotmode"]
        if plotmode == "ref":
            return ref, None
        else:  # other plotmode is ignored
            return s, None

    def _get_ydata_complementary(self):
        if self.params["plot"]["refaverage"]:
            r = np.mean((self.data0ref, self.data1ref))
            r0, r1 = r, r
        else:
            r0, r1 = self.data0ref, self.data1ref

        refmode = self.params["plot"]["refmode"]
        if refmode == "subtract":
            s0 = self.data0 - r0
            s1 = self.data1 - r1
        elif refmode == "divide":
            s0 = self.data0 / r0
            s1 = self.data1 / r1
        elif refmode == "ignore":
            s0 = self.data0
            s1 = self.data1
        else:
            raise ValueError(f"unknown refmode {refmode}")

        plotmode = self.params["plot"]["plotmode"]
        if plotmode == "data01":
            return s0, s1
        elif plotmode == "data0":
            return s0, None
        elif plotmode == "data1":
            return s1, None
        elif plotmode == "diff":
            return s0 - s1, None
        elif plotmode == "average":
            return (s0 + s1) / 2, None
        elif plotmode == "normalize":
            return (s0 - s1) / (s0 + s1), None
        elif plotmode == "concatenate":
            return np.column_stack((s0, s1)).reshape(len(s0) * 2), None
        elif plotmode == "ref":
            return self.data0ref, self.data1ref
        else:
            raise ValueError(f"unknown plotmode {plotmode}")

    def get_sampling_rate(self):
        signal_head = self.marker_indices[0]
        dt = (signal_head[1] - signal_head[0]) * self.params["instrument"]["tbin"]
        total = dt * 2 * len(self.xdata)
        return dt, total

    def get_pulse_params(self) -> dict:
        if not ("90pulse" in self.params and "180pulse" in self.params):
            return self.params.copy()

        p90, p180 = [self.params[k] for k in ("90pulse", "180pulse")]

        if p180 <= 0:
            p180 = p90 * 2

        p = self.params.copy()
        p["90pulse"] = p90
        p["180pulse"] = p180

        return p

    def get_method(self) -> str | None:
        if not self.has_params():
            return None
        return self.params["method"]

    # helpers

    def sweeps(self) -> float:
        if self.tdc_status is None:
            return 0.0
        return self.tdc_status.sweeps

    def measurement_time(self) -> float:
        """calculate measurement time in sec."""

        trange, tbin = self.get_range(), self.get_bin()
        if trange is None or tbin is None:
            return 0.0
        return self.sweeps() * trange * tbin

    def has_raw_data(self) -> bool:
        """return True if current data is valid (not empty nor all zero)."""

        return self.raw_data is not None and any(self.raw_data > 0)

    def has_data(self) -> bool:
        partial = self.partial()

        if partial == 0:
            return all([d is not None for d in (self.data0, self.data0ref)])
        elif partial == 1:
            return all([d is not None for d in (self.data1, self.data1ref)])
        else:
            return all(
                [d is not None for d in (self.data0, self.data1, self.data0ref, self.data1ref)]
            )

    def partial(self) -> int | None:
        if not self.has_params():
            return None
        return self.params.get("partial", -1)

    def is_partial(self) -> bool:
        return self.has_params() and self.params.get("partial") in (0, 1)

    def is_sweepN(self):
        return self.has_params() and is_sweepN(self.params["method"])

    def is_CPlike(self) -> bool:
        return self.has_params() and is_CPlike(self.params["method"])

    def is_correlation(self):
        return self.has_params() and is_correlation(self.params["method"])

    def is_supersampling(self):
        return (
            self.params["method"] in ("xy8", "xy16")
            and "supersample" in self.params
            and self.params["supersample"] != 1
        )

    def can_resume(self, params: dict | None) -> bool:
        """Check if the measurement can be resumed with given new_params."""

        if not self.has_params() or params is None:
            return False
        p0 = self.params.copy()
        p1 = params.copy()
        for p in (p0, p1):
            for k in ("instrument", "plot", "resume"):
                if k in p:
                    del p[k]
        # parameters contains small float values (several nano-seconds: ~ 1e-9).
        # atol should be smaller than that.
        return dict_isclose(p0, p1, atol=1e-11)


def update_data(data: PODMRData):
    """update data to latest format"""

    if data.version() <= 0:
        # version 0 to 1
        ## tdc_status maybe raw tuple
        if data.tdc_status is not None:
            data.tdc_status = TDCStatus(*data.tdc_status)
        ## rename fit_param -> fit_params
        data.fit_params = data.fit_param
        del data.fit_param
        ## no None-initialization of fit_result
        if not hasattr(data, "fit_result"):
            data.fit_result = None
        ## rename param["trigger_length"] -> param["trigger_width"]
        data.params["trigger_width"] = data.params["trigger_length"]
        del data.params["trigger_length"]
        ## add missing attributes
        data._saved = True
        data.set_version(1)

    if data.version() <= 1:
        # version 1 to 2
        # store marker_indices as an array, instead of tuple of arrays.
        data.marker_indices = np.vstack(data.marker_indices)
        # fix bug of setting yunit = None
        if data.yunit is None:
            data.yunit = ""
        data.set_version(2)

    return data
