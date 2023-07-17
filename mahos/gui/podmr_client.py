#!/usr/bin/env python3

"""
Qt signal-based clients of Pulse ODMR.

.. This file is a part of MAHOS project.

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
        resp = self.req.request(UpdatePlotParamsReq(params))
        return resp.success

    def validate(self, params: dict) -> bool:
        resp = self.req.request(ValidateReq(params))
        return resp.success

    def discard(self) -> bool:
        resp = self.req.request(DiscardReq())
        return resp.success
