#!/usr/bin/env python3

"""
Message Types for ConfocalTracker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import enum

from mahos.msgs.common_msgs import FileArtifact, Message, Status, Request, BinaryState


class OptMode(Message, enum.Enum):
    """Optimization mode selection for confocal tracker image processing."""

    Disable = 0  # Disable
    POC = 1  # Phase Only Correlation
    Gauss2D = 2  # 2D Gaussian fitting
    Gauss1D_0 = 3  # 1D Gaussian along axis 0 (take average along axis 1)
    Gauss1D_1 = 4  # 1D Gaussian along axis 1 (take average along axis 0)
    Max2D = 5  # Maximum Pixel
    Max1D_0 = 6  # 1D Maximum along axis 0 (take average along axis 1)
    Max1D_1 = 7  # 1D Maximum along axis 1 (take average along axis 0)


class ConfocalTrackerStatus(Status):
    """Runtime status of the confocal tracker node.

    :ivar state: Tracker state (IDLE/ACTIVE).
    :ivar tracking: True while a tracking cycle is currently running.

    """

    def __init__(self, state: BinaryState, tracking: bool):
        self.state = state
        self.tracking = tracking


class SaveParamsReq(Request):
    """Request to save tracker parameter dictionary to disk.

    :ivar params: Tracker parameter dictionary to persist.
    :ivar file_name: Optional output path; default path is used when omitted.

    """

    def __init__(self, params: dict, file_name=None):
        self.params = params
        self.file_name = file_name


class LoadParamsReq(Request):
    """Request to load tracker parameters from disk.

    :ivar file_name: Optional input path; default path is used when omitted.

    """

    def __init__(self, file_name=None):
        self.file_name = file_name


class SaveParamsArtifactReq(Request):
    """Request creation of a tracker-parameter artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.
    :ivar params: Tracker parameter dictionary to persist.

    """

    def __init__(self, file_name: str, params: dict):
        self.file_name = file_name
        self.params = params


class LoadParamsArtifactReq(Request):
    """Request loading tracker parameters from a shared artifact.

    :ivar artifact: Metadata identifying the staged parameter file.

    """

    def __init__(self, artifact: FileArtifact):
        self.artifact = artifact


class TrackNowReq(Request):
    """Request to force immediate tracker execution on the next node cycle."""

    pass
