#!/usr/bin/env python3

from __future__ import annotations

import typing as T
import numpy as np
from numpy.typing import NDArray

from .common_meas_msgs import BasicMeasData


class IVData(BasicMeasData):
    """Measurement Data type for IV.

    :ivar data: (params["num"], number of sweeps) measurement results.
    :ivar params: parameter dict.
    :ivar running: measurement is running or stopped.

    """

    def __init__(self, data: T.Optional[NDArray[np.float64]], params: dict, running: bool):
        self.data = data
        self.params = params
        self.running = running

    def sweeps(self) -> int:
        """Get number of sweeps done."""

        if self.data is None:
            return 0
        return self.data.shape[1]

    def get_xdata(self):
        if self.params["logx"]:
            return np.logspace(
                np.log10(self.params["start"]), np.log10(self.params["stop"]), self.params["num"]
            )
        else:
            return np.linspace(self.params["start"], self.params["stop"], self.params["num"])

    def get_ydata(self):
        return np.mean(self.data, axis=1)


def update_data(data: IVData):
    return data
