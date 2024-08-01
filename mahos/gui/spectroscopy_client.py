#!/usr/bin/env python3

"""
Qt signal-based clients of Spectroscopy.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from .Qt import QtCore

from ..msgs.spectroscopy_msgs import SpectroscopyData, SpectroscopyStatus
from .client import QBasicMeasClient


class QSpectroscopyClient(QBasicMeasClient):
    """Qt-based client for Spectroscopy."""

    dataUpdated = QtCore.pyqtSignal(SpectroscopyData)
    stopped = QtCore.pyqtSignal(SpectroscopyData)
    statusUpdated = QtCore.pyqtSignal(SpectroscopyStatus)
