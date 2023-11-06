#!/usr/bin/env python3

"""
Qt signal-based clients of Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from .Qt import QtCore

from ..msgs.podmr_msgs import PODMRData
from ..msgs.podmr_msgs import UpdatePlotParamsReq, ValidateReq, DiscardReq
from .client import QBasicMeasClient


class QPODMRClient(QBasicMeasClient):
    """Qt-based client for PODMR."""

    dataUpdated = QtCore.pyqtSignal(PODMRData)
    stopped = QtCore.pyqtSignal(PODMRData)

    def update_plot_params(self, params: dict) -> bool:
        rep = self.req.request(UpdatePlotParamsReq(params))
        return rep.success

    def validate(self, params: dict) -> bool:
        rep = self.req.request(ValidateReq(params))
        return rep.success

    def discard(self) -> bool:
        rep = self.req.request(DiscardReq())
        return rep.success
