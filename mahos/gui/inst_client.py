#!/usr/bin/env python3

"""
Qt signal-based client of InstrumentServer.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from .Qt import QtCore

from ..msgs.inst.server_msgs import ServerStatus
from .client import QStatusSubscriber


class QInstrumentSubscriber(QStatusSubscriber):
    """Subscribe to InstrumentServer's Status."""

    statusUpdated = QtCore.pyqtSignal(ServerStatus)
