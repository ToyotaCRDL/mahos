#!/usr/bin/env python3

"""
Qt signal-based clients of Camera stream.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from .Qt import QtCore

from ..msgs.camera_msgs import Image
from .client import QBasicMeasClient


class QCameraClient(QBasicMeasClient):
    """Qt-based client for Camera."""

    dataUpdated = QtCore.pyqtSignal(Image)
    stopped = QtCore.pyqtSignal(Image)
