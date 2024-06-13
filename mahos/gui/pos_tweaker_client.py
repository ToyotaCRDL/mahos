#!/usr/bin/env python3

"""
Qt signal-based client of PosTweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from .Qt import QtCore

from ..msgs.pos_tweaker_msgs import PosTweakerStatus, SetTargetReq
from ..msgs.pos_tweaker_msgs import HomeReq, HomeAllReq, StopReq, StopAllReq, LoadReq
from ..msgs.tweaker_msgs import SaveReq
from ..node.node import get_value
from .client import QStatusSubscriber


class QPosTweakerClient(QStatusSubscriber):
    """Qt-based client for PosTweaker."""

    statusUpdated = QtCore.pyqtSignal(PosTweakerStatus)

    def __init__(self, gconf: dict, name, context=None, parent=None, rep_endpoint="rep_endpoint"):
        QStatusSubscriber.__init__(self, gconf, name, context=context, parent=parent)

        self.req = self.ctx.add_req(
            self.conf[rep_endpoint],
            timeout_ms=get_value(gconf, self.conf, "req_timeout_ms"),
            logger=self.__class__.__name__,
        )

    def set_target(self, axis_pos: dict[str, float]) -> bool:
        rep = self.req.request(SetTargetReq(axis_pos))
        return rep.success

    def home(self, axis: str) -> bool:
        rep = self.req.request(HomeReq(axis))
        return rep.success

    def home_all(self) -> bool:
        rep = self.req.request(HomeAllReq())
        return rep.success

    def stop(self, axis: str) -> bool:
        rep = self.req.request(StopReq(axis))
        return rep.success

    def stop_all(self) -> bool:
        rep = self.req.request(StopAllReq())
        return rep.success

    def save(self, filename: str, group: str = "") -> bool:
        rep = self.req.request(SaveReq(filename, group))
        return rep.success

    def load(self, filename: str, group: str = "") -> bool:
        rep = self.req.request(LoadReq(filename, group))
        return rep.success
