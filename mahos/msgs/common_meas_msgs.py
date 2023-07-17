#!/usr/bin/env python3

"""
Message Types for Common Meas Node.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import time
from collections import UserList

import numpy as np
import msgpack

from .common_msgs import Message, Request
from .data_msgs import Data


class BasicMeasData(Data):
    """Base Data class for Basic Meas.

    Basic Meas Node will produce Data which is 2D-plottable (x and y).
    BasicMeasData defines common interface to get x/y data and fitting data.

    :ivar running: True if measurement is running. initialized with False.
    :ivar start_time: initialized with time.time().
    :ivar finish_time: initialized with None.
    :ivar paused_periods: pairs of (paused_time, resumed_time). initialized with empty array.

    """

    def init_attrs(self):
        """initialize common attributes."""

        self.running: bool = False
        self.start_time: float = time.time()
        self.finish_time: float | None = None
        self.paused_periods: np.ndarray | None = None
        self._saved: bool = False

        self.clear_fit_data()
        self.init_axes()

    def finalize(self) -> float:
        """set attributes to finalize the measurement and data.

        - set `running` to False
        - set `finish_time`

        :returns: elapsed time (self.finish_time - self.start_time)

        """

        self.running = False
        self.finish_time = time.time()
        return self.finish_time - self.start_time

    def is_finalized(self):
        return not self.running and self.finish_time is not None

    def start(self):
        """set attributes to start the measurement.

        - set `running` to True

        """

        self.running = True

    def resume(self):
        """set attributes to resume the measurement.

        - set `running` to True
        - store the `paused_periods`
        - empty `finish_time`

        """

        self.running = True
        if self.paused_periods is None:
            self.paused_periods = np.array([(self.finish_time, time.time())])
        else:
            self.paused_periods = np.concatenate(
                (self.paused_periods, [(self.finish_time, time.time())])
            )
        self.finish_time = None

    def set_saved(self):
        self._saved = True

    def is_saved(self) -> bool:
        return self._saved

    def has_data(self) -> bool:
        """return True if data is ready and valid data could be read out."""

        raise NotImplementedError("has_data() is not implemented")

    def get_xdata(self):
        raise NotImplementedError("get_xdata() is not implemented")

    def get_ydata(self):
        raise NotImplementedError("get_xdata() is not implemented")

    def get_fit_xdata(self):
        return self.fit_xdata

    def get_fit_ydata(self):
        return self.fit_data

    def set_fit_data(self, x, y, params, result):
        self.fit_xdata = x
        self.fit_data = y
        self.fit_params = params
        self.fit_result = result

    def clear_fit_data(self):
        self.fit_xdata = self.fit_data = self.fit_params = self.fit_result = None

    def init_axes(self):
        self.xlabel: str = ""
        self.xunit: str = ""
        self.ylabel: str = ""
        self.yunit: str = ""
        self.xscale: str = "linear"
        self.yscale: str = "linear"

    def _h5_write_fit_result(self, val):
        d = val.copy()
        if "pcov" in d and d["pcov"] is not None:
            d["pcov"] = d["pcov"].tolist()
        return np.void(msgpack.dumps(d))

    def _h5_read_fit_result(self, val):
        d = msgpack.loads(np.array(val).tobytes())
        if "pcov" in d and d["pcov"] is not None:
            d["pcov"] = np.array(d["pcov"])
        return d

    def _h5_dataset_writers(self) -> dict:
        return {"fit_result": self._h5_write_fit_result}

    def _h5_readers(self) -> dict:
        return {"fit_result": self._h5_read_fit_result}


class Buffer(Message, UserList):
    def file_names(self) -> list[str]:
        return [n for n, data in self.data]

    def data_list(self) -> list[BasicMeasData]:
        return [data for n, data in self.data]

    def get_file_name(self, i) -> str:
        try:
            return self.data[i][0]
        except IndexError:
            return ""

    def get_data(self, i) -> BasicMeasData | None:
        try:
            return self.data[i][1]
        except IndexError:
            return None


class PopBufferReq(Request):
    """Generic Request to pop data buffer"""

    def __init__(self, index: int = -1):
        self.index = index


class ClearBufferReq(Request):
    """Generic Request to clear data buffer"""

    pass


class FitReq(Request):
    """Fit Measurement Result Request"""

    def __init__(self, params: dict, data_index=-1):
        self.params = params
        self.data_index = data_index


class ClearFitReq(Request):
    """Fit Measurement Result Clear Request"""

    def __init__(self, data_index=-1):
        self.data_index = data_index
