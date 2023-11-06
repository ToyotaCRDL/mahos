#!/usr/bin/env python3

"""
Qt signal-based clients of Pulse ODMR with Slow detectors.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from .Qt import QtCore

from ..msgs.spodmr_msgs import SPODMRData
from ..msgs.spodmr_msgs import UpdatePlotParamsReq, ValidateReq
from .client import QBasicMeasClient


class QSPODMRClient(QBasicMeasClient):
    """Qt-based client for SPODMR."""

    dataUpdated = QtCore.pyqtSignal(SPODMRData)
    stopped = QtCore.pyqtSignal(SPODMRData)

    def update_plot_params(self, params: dict) -> bool:
        rep = self.req.request(UpdatePlotParamsReq(params))
        return rep.success

    def validate(self, params: dict) -> bool:
        rep = self.req.request(ValidateReq(params))
        return rep.success
