#!/usr/bin/env python3

"""
Qt signal-based clients of ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from .Qt import QtCore

from ..msgs.odmr_msgs import ODMRData, ValidateReq
from .client import QBasicMeasClient


class QODMRClient(QBasicMeasClient):
    """Qt-based client for ODMR."""

    dataUpdated = QtCore.pyqtSignal(ODMRData)
    stopped = QtCore.pyqtSignal(ODMRData)

    def validate(self, params: dict) -> bool:
        resp = self.req.request(ValidateReq(params))
        return resp.success
