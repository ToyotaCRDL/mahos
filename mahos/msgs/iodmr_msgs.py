#!/usr/bin/env python3

"""
Message Types for Imaging ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import time

import numpy as np

from .data_msgs import Data


class IODMRData(Data):
    def __init__(self, params: dict | None = None):
        self.set_version(0)
        self.init_params(params)

        self.data_sum = None
        self.data_latest = None
        self.data_history = None
        self.sweeps = 0
        self.running = False
        self.start_time: float = time.time()
        self.finish_time: float | None = None
        self.paused_periods: np.ndarray | None = None

    def finalize(self) -> float:
        """set attributes to finalize the measurement and data.

        - set `running` to False
        - set `finish_time`

        :returns: elapsed time (self.finish_time - self.start_time)

        """

        self.running = False
        self.finish_time = time.time()
        return self.finish_time - self.start_time

    def start(self):
        """set attributes to start the measurement.

        - set `running` to True

        """

        self.running = True

    def is_finalized(self):
        return not self.running and self.finish_time is not None

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
