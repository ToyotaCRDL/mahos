#!/usr/bin/env python3

"""
Message Types for Time to Digital Converter (Time Digitizer) Instruments.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np

from .data_msgs import Data


class RawEvents(Data):
    def __init__(self, data: np.ndarray | None = None):
        self.data = data

    def set_data(self, data: np.ndarray):
        self.data = data
