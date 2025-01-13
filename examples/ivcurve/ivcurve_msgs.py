#!/usr/bin/env python3

import numpy as np
from numpy.typing import NDArray

from mahos.msgs.common_meas_msgs import BasicMeasData


class IVCurveData(BasicMeasData):
    """Measurement Data type for IVCurve.

    :ivar data: (params["num"], number of sweeps) measurement results.
    :ivar params: parameter dict.
    :ivar running: measurement is running or stopped.

    """

    def __init__(self, data: NDArray[np.float64] | None, params: dict, running: bool):
        self.data = data
        self.params = params
        self.running = running

    def sweeps(self) -> int:
        """Get number of sweeps done."""

        if self.data is None:
            return 0
        return self.data.shape[1]
