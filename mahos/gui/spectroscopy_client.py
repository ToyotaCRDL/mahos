#!/usr/bin/env python3

"""
Qt signal-based clients of Spectroscopy.

.. This file is a part of MAHOS project.

"""

from .Qt import QtCore

from ..msgs.spectroscopy_msgs import SpectroscopyData
from .client import QBasicMeasClient


class QSpectroscopyClient(QBasicMeasClient):
    """Qt-based client for Spectroscopy."""

    dataUpdated = QtCore.pyqtSignal(SpectroscopyData)
    stopped = QtCore.pyqtSignal(SpectroscopyData)
