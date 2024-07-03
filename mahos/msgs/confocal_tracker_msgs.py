#!/usr/bin/env python3

"""
Message Types for ConfocalTracker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import enum

from .common_msgs import Message, Status, Request, BinaryState


class OptMode(Message, enum.Enum):
    Disable = 0  # Disable
    POC = 1  # Phase Only Correlation
    Gauss2D = 2  # 2D Gaussian fitting
    Gauss1D_0 = 3  # 1D Gaussian along axis 0 (take average along axis 1)
    Gauss1D_1 = 4  # 1D Gaussian along axis 1 (take average along axis 0)
    Max2D = 5  # Maximum Pixel
    Max1D_0 = 6  # 1D Maximum along axis 0 (take average along axis 1)
    Max1D_1 = 7  # 1D Maximum along axis 1 (take average along axis 0)


class ConfocalTrackerStatus(Status):
    def __init__(self, state: BinaryState, tracking: bool):
        self.state = state
        self.tracking = tracking


class SaveParamsReq(Request):
    def __init__(self, params: dict, file_name=None):
        self.params = params
        self.file_name = file_name


class LoadParamsReq(Request):
    def __init__(self, file_name=None):
        self.file_name = file_name


class TrackNowReq(Request):
    pass
