#!/usr/bin/env python3

"""
Qt signal-based clients of Analog-PD Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.gui.Qt import QtCore

from mahos_dq.msgs.apodmr_msgs import APODMRData
from mahos_dq.msgs.apodmr_msgs import (
    UpdatePlotParamsReq,
    ValidateReq,
    GetTimingInfoReq,
    TimingInfo,
)
from mahos.gui.client import QBasicMeasClient


class QAPODMRClient(QBasicMeasClient):
    """Qt-based client for APODMR."""

    dataUpdated = QtCore.pyqtSignal(APODMRData)
    stopped = QtCore.pyqtSignal(APODMRData)

    def update_plot_params(self, params: dict) -> bool:
        rep = self.req.request(UpdatePlotParamsReq(params))
        return rep.success

    def validate(self, params: dict, label: str) -> bool:
        rep = self.req.request(ValidateReq(params, label))
        return rep.success

    def get_timing_info(self, params: dict, label: str) -> TimingInfo | None:
        rep = self.req.request(GetTimingInfoReq(params, label))
        return rep.ret
