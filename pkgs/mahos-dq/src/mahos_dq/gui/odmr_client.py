#!/usr/bin/env python3

"""
Qt signal-based clients of ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.gui.Qt import QtCore

from mahos_dq.msgs.odmr_msgs import ODMRData, ValidateReq
from mahos.gui.client import QBasicMeasClient


class QODMRClient(QBasicMeasClient):
    """Qt-based client for ODMR."""

    dataUpdated = QtCore.pyqtSignal(ODMRData)
    stopped = QtCore.pyqtSignal(ODMRData)

    def validate(self, params: dict, label: str) -> tuple[bool, str, str]:
        """validate the ODMR measurement parameter.

        :returns: (success, error_message, pulse_summary)

        """

        rep = self.req.request(ValidateReq(params, label))
        return rep.success, rep.message, rep.ret
