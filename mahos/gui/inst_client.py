#!/usr/bin/env python3

"""
Qt signal-based client of InstrumentServer.

.. This file is a part of MAHOS project.

"""

from .Qt import QtCore

from ..msgs.inst_server_msgs import ServerStatus
from .client import QStatusSubscriber


class QInstrumentSubscriber(QStatusSubscriber):
    """Subscribe to InstrumentServer's Status."""

    statusUpdated = QtCore.pyqtSignal(ServerStatus)
