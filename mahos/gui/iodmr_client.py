#!/usr/bin/env python3

"""
Qt signal-based clients of Imaging IODMR.

.. This file is a part of MAHOS project.

"""

from .Qt import QtCore

from ..msgs.iodmr_msgs import IODMRData
from .client import QBasicMeasClient


class QIODMRClient(QBasicMeasClient):
    """Qt-based client for IODMR."""

    dataUpdated = QtCore.pyqtSignal(IODMRData)
    stopped = QtCore.pyqtSignal(IODMRData)
