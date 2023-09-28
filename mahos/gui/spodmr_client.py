#!/usr/bin/env python3

"""
Qt signal-based clients of Pulse ODMR with Slow detectors.

.. This file is a part of MAHOS project.

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
        resp = self.req.request(UpdatePlotParamsReq(params))
        return resp.success

    def validate(self, params: dict) -> bool:
        resp = self.req.request(ValidateReq(params))
        return resp.success
