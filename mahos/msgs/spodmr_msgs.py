#!/usr/bin/env python3

"""
Message Types for Pulse ODMR with Slow detectors.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..util.comp import dict_isclose
from .common_msgs import Request, BinaryState, Status
from .common_meas_msgs import BasicMeasData
from .data_msgs import ComplexDataMixin


class ValidateReq(Request):
    """Validate Measurement Params Request"""

    def __init__(self, params: dict, label: str):
        self.params = params
        self.label = label


class UpdatePlotParamsReq(Request):
    """Request to update the plot param"""

    def __init__(self, params: dict):
        self.params = params


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


class SPODMRStatus(Status):
    def __init__(self, state: BinaryState, pg_freq: float):
        self.state = state
        self.pg_freq = pg_freq

    def __repr__(self):
        return f"SPODMRStatus({self.state}, {self.pg_freq*1e-9:.2f} GHz)"

    def __str__(self):
        return f"SPODMR({self.state.name}, {self.pg_freq*1e-9:.2f} GHz)"


class SPODMRData(BasicMeasData, ComplexDataMixin):
    def __init__(self, params: dict | None = None, label: str = ""):
        self.set_version(3)
        self.init_params(params, label)
        self.init_attrs()

        self.init_xdata()

        self.data0 = None
        self.data1 = None
        self.laser_duties = None

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
                pulse_num = {"xy8": 8, "xy16": 16}[self.label] * self.params["pulse"]["Nconst"]
                sample = np.linspace(0, 1, num=self.params["pulse"]["supersample"] + 1)
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

        if self.params.get("invert_sweep", False):
            self.xdata = self.xdata[::-1]

    # getters

    def _normalize_image(self, data):
        if self.params["plot"].get("normalize", True):
            offset = self.params["plot"].get("offset", 0.0)
            return ((data.T - offset) / self.laser_duties).T
        else:
            return data

    def get_image(self, last_n: int = 0) -> NDArray:
        if self.partial() in (0, 2):
            return self._normalize_image(self._conv_complex(self.data0))[:, -last_n:]
        elif self.partial() == 1:
            return self._normalize_image(self._conv_complex(self.data1))[:, -last_n:]
        else:  # -1
            i0 = self._normalize_image(self._conv_complex(self.data0))[:, -last_n:]
            i1 = self._normalize_image(self._conv_complex(self.data1))[:, -last_n:]

            plotmode = self.params["plot"]["plotmode"]
            flip = self.params["plot"].get("flipY", False)
            if plotmode == "data0":
                return i0
            elif plotmode == "data1":
                return i1
            elif plotmode == "average":
                return (i0 + i1) / 2
            elif plotmode == "normalize-data0":
                mean = np.mean(i0)
                return (i0 - mean) / mean
            elif plotmode == "normalize-data1":
                mean = np.mean(i1)
                return (i1 - mean) / mean
            elif plotmode == "normalize":
                if flip:
                    return (i1 - i0) / (i0 + i1) * 2
                else:
                    return (i0 - i1) / (i0 + i1) * 2
            elif plotmode == "normalize1":
                if flip:
                    return (i1 - i0) / i1
                else:
                    return (i0 - i1) / i1
            else:  # "diff". fall back "data01" to "diff" too.
                if flip:
                    return i1 - i0
                else:
                    return i0 - i1

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
        pp = self.get_pulse_params()
        a, b = 1, 0
        if m == "spinecho":
            a = 2
        elif m == "trse":
            b = pp["tauconst"]
        elif m in ["cp", "cpmg", "ddgate"]:
            a = 2 * pp["Nconst"]
        elif m == "xy4":
            a = 2 * pp["Nconst"] * 4
        elif m == "xy8":
            a = 2 * pp["Nconst"] * 8
        elif m == "xy16":
            a = 2 * pp["Nconst"] * 16

        elif m in ["cpN", "cpmgN"]:
            a = 2 * pp["tauconst"]
        elif m == "xy4N":
            a = 2 * pp["tauconst"] * 4
        elif m == "xy8N":
            a = 2 * pp["tauconst"] * 8
        elif m == "xy16N":
            a = 2 * pp["tauconst"] * 16

        elif m == "xy8cl":
            a = 1
        elif m == "xy8cl1flip":
            a = 2
            b = pp["180pulse"]
        elif m == "xy8clNflip":
            a = 2 * pp["tauconst"] + pp["180pulse"]
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
        else:
            raise ValueError(f"unknown taumode {taumode}")

    def get_fit_ydata(self):
        return self.fit_data

    def get_ydata(
        self, last_n: int = 0, std: bool = False
    ) -> tuple[NDArray | None, NDArray | None]:
        """get analyzed ydata.

        :param last_n: if nonzero, take last_n sweeps to make y data.
        :param std: if True, estimated std. dev. instead of mean along the accumulations.

        :returns: (ydata0, ydata1) if two types of data are available.
                  (ydata, None) if only one type of data is available.
                  (None, None) if data is not ready.
        :raises ValueError: when mode is unknown.

        """

        def conv(data):
            if data is None:
                return None
            if std:
                N = data.shape[1]
                if N == 1:
                    # avoid nan (zero division)
                    return np.zeros(data.shape[0])
                return np.std(data, axis=1, ddof=1) / np.sqrt(N)
            else:
                return np.mean(data, axis=1)

        if not self.has_data():
            return None, None

        if self.is_partial():
            return conv(self._get_ydata_partial(last_n)), None
        else:
            ydata0, ydata1 = self._get_ydata_complementary(last_n)
            return conv(ydata0), conv(ydata1)

    def _normalize(self, data):
        if self.params["plot"].get("normalize", True):
            offset = self.params["plot"].get("offset", 0.0)
            return ((data.T - offset) / self.laser_duties).T
        else:
            return data

    def _conv_complex(self, data):
        complex_conv = self.params["plot"].get("complex_conv", "real")
        return self.conv_complex(data, complex_conv)

    def _get_ydata_partial(self, last_n: int):
        if self.partial() in (0, 2):
            return self._normalize(self._conv_complex(self.data0)[:, -last_n:])
        else:  # assert self.partial() == 1
            return self._normalize(self._conv_complex(self.data1)[:, -last_n:])

    def _get_ydata_complementary(self, last_n: int):
        s0 = self._normalize(self._conv_complex(self.data0)[:, -last_n:])
        s1 = self._normalize(self._conv_complex(self.data1)[:, -last_n:])

        plotmode = self.params["plot"]["plotmode"]
        flip = self.params["plot"].get("flipY", False)
        if plotmode == "data01":
            return s0, s1
        elif plotmode == "data0":
            return s0, None
        elif plotmode == "data1":
            return s1, None
        elif plotmode == "diff":
            if flip:
                return s1 - s0, None
            else:
                return s0 - s1, None
        elif plotmode == "average":
            return (s0 + s1) / 2, None
        elif plotmode == "normalize-data0":
            mean = np.mean(s0)
            return (s0 - mean) / mean, None
        elif plotmode == "normalize-data1":
            mean = np.mean(s1)
            return (s1 - mean) / mean, None
        elif plotmode == "normalize":
            if flip:
                return (s1 - s0) / (s0 + s1) * 2, None
            else:
                return (s0 - s1) / (s0 + s1) * 2, None
        elif plotmode == "normalize1":
            if flip:
                return (s1 - s0) / s0, None
            else:
                return (s0 - s1) / s1, None
        elif plotmode == "concatenate":
            out = np.empty((s0.shape[0] * 2, s0.shape[1]))
            out[0::2, :] = s0
            out[1::2, :] = s1
            return out, None
        else:
            raise ValueError(f"unknown plotmode {plotmode}")

    def get_params(self) -> dict:
        if not self.has_params():
            return {}
        p = self.params.copy()
        p["pulse"] = self.get_pulse_params()
        return p

    def get_pulse_params(self) -> dict:
        if not self.has_params():
            return {}
        pp = self.params["pulse"]
        if not ("90pulse" in pp and "180pulse" in pp):
            return pp.copy()

        p90, p180 = [pp[k] for k in ("90pulse", "180pulse")]

        if p180 <= 0:
            p180 = p90 * 2

        p = pp.copy()
        p["90pulse"] = p90
        p["180pulse"] = p180

        return p

    def get_method(self) -> str | None:
        if not self.has_params():
            return None
        return self.label

    # helpers

    def sweeps(self) -> int:
        if not self.has_data():
            return 0
        if self.partial() == 1:
            return self.data1.shape[1]
        else:
            return self.data0.shape[1]

    def measurement_time(self) -> float:
        """calculate measurement time in sec."""

        try:
            freq = self.params["instrument"]["pg_freq"]
            length = self.params["instrument"]["length"]
            return self.sweeps() * length / freq
        except KeyError:
            return 0.0

    def has_data(self) -> bool:
        partial = self.partial()

        if partial in (0, 2):
            return self.data0 is not None
        elif partial == 1:
            return self.data1 is not None
        else:
            return all([d is not None for d in (self.data0, self.data1)])

    def partial(self) -> int | None:
        if not self.has_params():
            return None
        return self.params.get("partial", -1)

    def is_partial(self) -> bool:
        return self.has_params() and self.params.get("partial") in (0, 1, 2)

    def is_sweepN(self):
        return self.has_params() and is_sweepN(self.label)

    def get_num(self):
        if not self.has_params():
            return None
        if self.is_sweepN():
            return self.params["Nnum"]
        else:
            return self.params["num"]

    def is_CPlike(self) -> bool:
        return self.has_params() and is_CPlike(self.label)

    def is_correlation(self):
        return self.has_params() and is_correlation(self.label)

    def is_supersampling(self):
        return (
            self.label in ("xy8", "xy16")
            and "supersample" in self.params["pulse"]
            and self.params["pulse"]["supersample"] != 1
        )

    def is_complex(self) -> bool:
        if not self.has_data():
            return False
        d = self.data1 if self.partial() == 1 else self.data0
        return np.issubdtype(d.dtype, np.complexfloating)

    def can_resume(self, params: dict | None, label: str) -> bool:
        """Check if the measurement can be resumed with given new params."""

        if self.label != label:
            return False
        if not self.has_params() or params is None:
            return False
        p0 = self.params.copy()
        p1 = params.copy()
        for p in (p0, p1):
            for k in ("instrument", "plot", "resume", "quick_resume", "sweeps"):
                if k in p:
                    del p[k]
        # parameters contains small float values (several nano-seconds: ~ 1e-9).
        # atol should be smaller than that.
        return dict_isclose(p0, p1, atol=1e-11)


def update_data(data: SPODMRData):
    """update data to latest format"""

    if data.version() <= 0:
        # version 0 to 1
        plot = data.params["plot"]
        if "xlogscale" in plot:
            plot["logX"] = plot["xlogscale"]
            del plot["xlogscale"]
        if "ylogscale" in plot:
            plot["logY"] = plot["ylogscale"]
            del plot["ylogscale"]
        data.set_version(1)

    if data.version() <= 1:
        # version 1 to 2
        data.label = data.params["method"]
        del data.params["method"]
        if data.fit_params:
            data.fit_label = data.fit_params["method"]
            del data.fit_params["method"]
        data.set_version(2)

    if data.version() <= 2:
        # version 2 to 3
        ## fixed param name
        if "invertsweep" in data.params:
            data.params["invert_sweep"] = data.params["invertsweep"]
            del data.params["invertsweep"]
        ## fixed location of optional pulse params
        ## here's only keys actually used for experiments before patch
        keys = [
            "90pulse",
            "180pulse",
            "tauconst",
            "Nconst",
            "readY",
            "invertY",
            "supersample",
            "flip_head",
        ]
        data.params["pulse"] = {}
        for k in keys:
            if k in data.params:
                data.params["pulse"][k] = data.params[k]
                del data.params[k]

        ## fixed parameter names (SG2 -> SG1)
        if "freq2" in data.params:
            for k in ["freq", "power", "nomw"]:
                data.params[k + "1"] = data.params[k + "2"]
                del data.params[k + "2"]
        data.set_version(3)

    return data
