#!/usr/bin/env python3

"""
Message Types for Qdyne.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..util.comp import dict_isclose
from ..util.conv import real_fft
from .common_msgs import Request
from .common_meas_msgs import BasicMeasData
from .podmr_msgs import TDCStatus
from . import param_msgs as P


class ValidateReq(Request):
    """Validate Measurement Params Request"""

    def __init__(self, params: dict, label: str):
        self.params = params
        self.label = label


class DiscardReq(Request):
    """Discard Data Request"""

    pass


class QdyneData(BasicMeasData):
    def __init__(self, params: dict | None = None, label: str = ""):
        self.set_version(2)
        self.init_params(params, label)
        self.init_attrs()

        self.tdc_status: TDCStatus | None = None

        self.xdata = None
        self.data = None

        self.fft_xdata = None
        self.fft_data = None

        self.raw_data = None

        self.marker_indices = None
        self.laser_timing = None

    # h5

    def _h5_write_tdc_status(self, val):
        return np.array(val)

    def _h5_read_tdc_status(self, val):
        if len(val) == 5:
            # older version had 5 values
            return TDCStatus(val[0], *val[2:])
        return TDCStatus(*val)

    def _h5_attr_writers(self) -> dict:
        return {"fit_result": self._h5_write_fit_result, "tdc_status": self._h5_write_tdc_status}

    def _h5_readers(self) -> dict:
        return {"fit_result": self._h5_read_fit_result, "tdc_status": self._h5_read_tdc_status}

    # setters

    def set_laser_timing(self, laser_timing):
        if self.laser_timing is not None:
            return
        self.laser_timing = np.array(laser_timing)  # unit is [sec]

    def set_instrument_params(self, tbin, pg_freq, pg_length, pg_offsets):
        if "instrument" in self.params:
            return
        self.params["instrument"] = {}
        self.params["instrument"]["tbin"] = tbin
        self.params["instrument"]["pg_freq"] = pg_freq
        self.params["instrument"]["pg_length"] = pg_length
        if all([ofs == 0 for ofs in pg_offsets]):
            self.params["instrument"]["pg_offsets"] = []
        else:
            self.params["instrument"]["pg_offsets"] = pg_offsets

    def set_status(self, tdc_status: TDCStatus):
        self.tdc_status = tdc_status

    def set_raw_data(self, raw_data: np.ndarray):
        self.raw_data = raw_data

    def remove_raw_data(self):
        self.raw_data = None

    def set_marker_indices(self):
        tbin = self.get_bin()
        if self.params is None or tbin is None:
            return None

        sigdelay, sigwidth, refdelay, refwidth = [
            self.params["plot"][k] for k in ("sigdelay", "sigwidth", "refdelay", "refwidth")
        ]

        signal_head = self.laser_timing + sigdelay
        signal_tail = signal_head + sigwidth
        reference_head = signal_tail + refdelay
        reference_tail = reference_head + refwidth

        # [sec] ==> index
        signal_head = np.round(signal_head / tbin).astype(np.int64)
        signal_tail = np.round(signal_tail / tbin).astype(np.int64)
        reference_head = np.round(reference_head / tbin).astype(np.int64)
        reference_tail = np.round(reference_tail / tbin).astype(np.int64)

        self.marker_indices = np.vstack((signal_head, signal_tail, reference_head, reference_tail))

    def update_plot_params(self, plot_params: dict[str, P.RawPDValue]) -> bool:
        """update plot_params. returns True if param is actually updated."""

        if not self.has_params():
            return False
        updated = not P.isclose(self.params["plot"], plot_params)
        if "plot" not in self.params:
            self.params["plot"] = plot_params
        else:
            self.params["plot"].update(plot_params)
        return updated

    # getters

    def get_bin(self):
        try:
            return self.params["instrument"]["tbin"]
        except (KeyError, TypeError):
            return None

    def get_period(self) -> float | None:
        """Get measurement period in sec."""

        try:
            return self.params["instrument"]["pg_length"] / self.params["instrument"]["pg_freq"]
        except (KeyError, TypeError):
            return None

    def get_period_bins(self) -> int | None:
        """Get measurement period in bins."""

        try:
            T_sec = self.get_period()
            tbin = self.get_bin()
            if T_sec is None or tbin is None:
                return None
            return int(round(T_sec / tbin))
        except (KeyError, TypeError):
            return None

    def remove_fft_data(self):
        self.fft_xdata = self.fft_data = None

    def fft(self) -> bool:
        if not self.has_data():
            return False
        x = self.get_xdata() * self.get_bin()
        y = self.get_ydata()
        self.fft_xdata, self.fft_data = real_fft(x, y)
        return True

    def get_fft_xdata(self):
        if self.fft_xdata is None:
            self.fft()
        return self.fft_xdata

    def get_xdata(self, fft: bool = False) -> NDArray | None:
        """get analyzed xdata.

        :param freq: True (False) to get fft (time-domain) data.

        :returns: xdata if data is available.
                  None  if data is not ready.

        """

        return self.get_fft_xdata() if fft else self.xdata

    def get_fft_ydata(self):
        if self.fft_data is None:
            self.fft()
        return self.fft_data

    def get_ydata(self, fft: bool = False) -> NDArray | None:
        """get analyzed ydata.

        :param freq: True (False) to get fft (time-domain) data.

        :returns: ydata if data is available.
                  None  if data is not ready.

        """

        return self.get_fft_ydata() if fft else self.data

    def get_params(self) -> dict:
        if not self.has_params():
            return {}
        p = self.params.copy()
        # fill unused parameters
        p["init_delay"] = 0.0
        p["final_delay"] = 0.0

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
        # fill unused parameters
        p["supersample"] = 1

        return p

    def get_method(self) -> str | None:
        if not self.has_params():
            return None
        return self.label

    # helpers

    def has_raw_data(self) -> bool:
        """return True if current raw data is valid."""

        return self.raw_data is not None

    def has_data(self) -> bool:
        return self.data is not None

    def is_partial(self) -> bool:
        return self.has_params() and self.params.get("partial") in (0, 1)

    def can_resume(self, params: dict | None, label: str) -> bool:
        """Check if the measurement can be resumed with given new_params."""

        if self.label != label:
            return False
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


def update_data(data: QdyneData):
    """update data to latest format"""

    if data.version() <= 0:
        # version 0 to 1
        data.label = data.params["method"]
        del data.params["method"]
        if data.fit_params:
            data.fit_label = data.fit_params["method"]
            del data.fit_params["method"]
        data.set_version(1)

    if data.version() <= 1:
        # version 1 to 2
        ## fixed location of optional pulse params
        ## here's only keys actually used for experiments before patch
        keys = [
            "90pulse",
            "180pulse",
            "tauconst",
            "Nconst",
            "readY",
            "invertY",
            # "supersample",
            "flip_head",
        ]
        data.params["pulse"] = {}
        for k in keys:
            if k in data.params:
                data.params["pulse"][k] = data.params[k]
                del data.params[k]
        data.set_version(2)

    return data
