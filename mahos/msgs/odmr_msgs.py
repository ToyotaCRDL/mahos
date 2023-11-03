#!/usr/bin/env python3

"""
Message Types for ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import msgpack

from .common_msgs import Request
from .data_msgs import ComplexDataMixin
from .common_meas_msgs import BasicMeasData
from .fit_msgs import PeakType
from ..util import conv


class ValidateReq(Request):
    """Validate Measurement Params Request"""

    def __init__(self, params: dict):
        self.params = params


class ODMRData(BasicMeasData, ComplexDataMixin):
    """Data type for ODMR measurement."""

    def __init__(self, params: dict | None = None):
        self.set_version(2)
        self.init_params(params)
        self.init_attrs()

        self.data = None
        self.bg_data = None

        #: complex_conv used for loading data fit
        self.fit_complex_conv: str = ""

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

    def is_complex(self) -> bool:
        return self.has_data() and np.issubdtype(self.data.dtype, np.complexfloating)

    def get_xdata(self):
        return np.linspace(self.params["start"], self.params["stop"], self.params["num"])

    def _get_ydata_nobg(self, last_n, normalize_n, complex_conv):
        ydata = self.conv_complex(self.data, complex_conv)[:, -last_n:]
        if not normalize_n:
            return ydata, None

        if normalize_n > 0:
            coeff = np.mean(np.sort(np.mean(ydata, axis=1))[-normalize_n:])
        else:
            coeff = np.mean(np.sort(np.mean(ydata, axis=1))[:-normalize_n])
        if coeff == 0.0:
            coeff = 1.0
        return ydata / coeff, None

    def _get_ydata_bg(self, last_n, normalize_n, complex_conv):
        ydata = self.conv_complex(self.data, complex_conv)[:, -last_n:]
        bg_ydata = self.conv_complex(self.bg_data, complex_conv)[:, -last_n:]
        if normalize_n:
            return ydata / bg_ydata, None
        return ydata, bg_ydata

    def get_ydata(
        self,
        last_n: int = 0,
        normalize_n: int = 0,
        complex_conv: str = "real",
        std: bool = False,
    ) -> tuple[NDArray | None, NDArray | None]:
        """get ydata.

        :param last_n: if nonzero, take last_n sweeps to make y data.
        :param normalize_n: if non-zero, normalize the y data.
            if this has background data, any non-zero value invokes normalization
            using background data. if background data is not available and normalize_n is
            positive (negative), top (bottom) normalize_n data points are used to determine
            the baseline.
        :param complex_conv: (real | imag | abs[olute] | angle) conversion method for complex data.
            unused if data is not complex.
        :param std: if True, estimated std. dev. instead of mean along the accumulations.
        :returns: (raw_ydata, background_ydata) if normalize_n is 0 and background is available.
                  (raw_ydata, None) if normalize_n is 0 and background is not available.
                  (normalized_ydata, None) if normalize_n is non-zero.
                  (None, None) if data is not ready.

        """

        def conv(data):
            if data is None:
                return None
            if std:
                N = data.shape[1]
                return np.std(data, axis=1) / np.sqrt(N)
            else:
                return np.mean(data, axis=1)

        if not self.has_data():
            return None, None

        if self.has_background():
            ydata0, ydata1 = self._get_ydata_bg(last_n, normalize_n, complex_conv)
            return conv(ydata0), conv(ydata1)
        else:
            ydata0, ydata1 = self._get_ydata_nobg(last_n, normalize_n, complex_conv)
            return conv(ydata0), conv(ydata1)

    def get_image(self, complex_conv: str = "real") -> NDArray:
        return self.conv_complex(self.data, complex_conv)

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
        ydata = np.mean(self.conv_complex(self.data, self.fit_complex_conv)[:, -last_n:], axis=1)
        if normalize_n > 0:
            coeff = np.mean(np.sort(ydata)[-normalize_n:])
        else:
            coeff = np.mean(np.sort(ydata)[:-normalize_n])
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
