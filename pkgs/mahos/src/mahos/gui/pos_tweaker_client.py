#!/usr/bin/env python3

"""
Qt signal-based client of PosTweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from mahos.gui.Qt import QtCore

from mahos.msgs.pos_tweaker_msgs import PosTweakerStatus, SetTargetReq
from mahos.msgs.pos_tweaker_msgs import HomeReq, HomeAllReq, StopReq, StopAllReq
from mahos.node.node import get_value
from mahos.gui.client import QStatusSubscriber
from mahos.meas.tweaker import TweakerFileReqMixin


class QPosTweakerClient(TweakerFileReqMixin, QStatusSubscriber):
    """Qt-based client for PosTweaker."""

    statusUpdated = QtCore.pyqtSignal(PosTweakerStatus)

    def __init__(
        self,
        gconf: dict,
        name,
        context=None,
        parent=None,
        rep_endpoint="rep_endpoint",
        file_transport_dir=None,
    ):
        QStatusSubscriber.__init__(self, gconf, name, context=context, parent=parent)

        self.req = self.ctx.add_req(
            self.conf[rep_endpoint],
            timeout_ms=get_value(gconf, self.conf, "req_timeout_ms"),
            logger=self.__class__.__name__,
        )
        self.init_file_transport(file_transport_dir)

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

    def save(self, file_name: str) -> bool:
        return self.save_file(file_name)

    def load(self, file_name: str, group: str = "") -> bool:
        rep = self.load_file(file_name, group)
        return rep.success
