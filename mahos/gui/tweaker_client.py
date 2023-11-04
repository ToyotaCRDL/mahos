#!/usr/bin/env python3

"""
Qt signal-based client of Tweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from .Qt import QtCore

from ..msgs import param_msgs as P
from ..msgs.tweaker_msgs import TweakerStatus, ReadReq, ReadAllReq, WriteReq, WriteAllReq
from ..msgs.tweaker_msgs import SaveReq, LoadReq
from ..node.node import get_value
from .client import QStatusSubscriber


class QTweakerClient(QStatusSubscriber):
    """Qt-based client for Tweaker."""

    statusUpdated = QtCore.pyqtSignal(TweakerStatus)

    def __init__(self, gconf: dict, name, context=None, parent=None, rep_endpoint="rep_endpoint"):
        QStatusSubscriber.__init__(self, gconf, name, context=context, parent=parent)

        self.req = self.ctx.add_req(
            self.conf[rep_endpoint],
            timeout_ms=get_value(gconf, self.conf, "req_timeout_ms"),
            logger=self.__class__.__name__,
        )

    def read_all(self) -> tuple[bool, dict[str, P.ParamDict[str, P.PDValue] | None]]:
        resp = self.req.request(ReadAllReq())
        return resp.success, resp.ret

    def read(self, param_dict_id: str) -> P.ParamDict[str, P.PDValue] | None:
        resp = self.req.request(ReadReq(param_dict_id))
        if resp.success:
            return resp.ret

    def write_all(self, param_dicts: dict[str, P.ParamDict[str, P.PDValue]]) -> bool:
        resp = self.req.request(WriteAllReq(param_dicts))
        return resp.success

    def write(self, param_dict_id: str, params: P.ParamDict[str, P.PDValue]) -> bool:
        resp = self.req.request(WriteReq(param_dict_id, params))
        return resp.success

    def save(self, filename: str, group: str = "") -> bool:
        resp = self.req.request(SaveReq(filename, group))
        return resp.success

    def load(
        self, filename: str, group: str = ""
    ) -> dict[str, P.ParamDict[str, P.PDValue] | None] | None:
        resp = self.req.request(LoadReq(filename, group))
        if resp.success:
            return resp.ret
