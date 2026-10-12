#!/usr/bin/env python3

"""
Qt signal-based clients of Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.gui.Qt import QtCore

from mahos_dq.msgs.podmr_msgs import (
    PODMRData,
    UpdatePlotParamsReq,
    ValidateReq,
    GetTimingInfoReq,
    TimingInfo,
    DiscardReq,
    FindLaserTimingReq,
    ClearLaserTimingReq,
)
from mahos.gui.client import QBasicMeasClient


class QPODMRClient(QBasicMeasClient):
    """Qt-based client for PODMR."""

    dataUpdated = QtCore.pyqtSignal(PODMRData)
    stopped = QtCore.pyqtSignal(PODMRData)

    def update_plot_params(self, params: dict) -> bool:
        rep = self.req.request(UpdatePlotParamsReq(params))
        return rep.success

    def validate(self, params: dict, label: str) -> bool:
        rep = self.req.request(ValidateReq(params, label))
        return rep.success

    def get_timing_info(self, params: dict, label: str) -> TimingInfo | None:
        rep = self.req.request(GetTimingInfoReq(params, label))
        return rep.ret

    def discard(self) -> bool:
        rep = self.req.request(DiscardReq())
        return rep.success

    def find_laser_timing(
        self,
        scope: tuple[float, float],
        smooth_window: int = 5,
        fraction: float = 0.5,
        monotonic: bool = True,
    ) -> bool:
        rep = self.req.request(FindLaserTimingReq(scope, smooth_window, fraction, monotonic))
        return rep.success

    def clear_laser_timing(self) -> bool:
        rep = self.req.request(ClearLaserTimingReq())
        return rep.success
