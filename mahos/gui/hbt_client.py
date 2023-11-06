#!/usr/bin/env python3

"""
Qt signal-based clients of HBT.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from .Qt import QtCore

from ..msgs.hbt_msgs import HBTData, UpdatePlotParamsReq
from .client import QBasicMeasClient


class QHBTClient(QBasicMeasClient):
    """Qt-based client for HBT."""

    dataUpdated = QtCore.pyqtSignal(HBTData)
    stopped = QtCore.pyqtSignal(HBTData)

    def update_plot_params(self, params: dict) -> bool:
        rep = self.req.request(UpdatePlotParamsReq(params))
        return rep.success
