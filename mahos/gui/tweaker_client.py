#!/usr/bin/env python3

"""
Qt signal-based client of Tweaker.

.. This file is a part of MAHOS project.

"""

from .Qt import QtCore

from ..msgs import param_msgs as P
from ..msgs.tweaker_msgs import TweakerStatus, ReadReq, ReadAllReq, WriteReq, SaveReq, LoadReq
from ..node.node import get_value
from .client import QStatusSubscriber


class QTweakerClient(QStatusSubscriber):
    """Subscribe to StateManager."""

    statusUpdated = QtCore.pyqtSignal(TweakerStatus)

    def __init__(self, gconf: dict, name, context=None, parent=None, rep_endpoint="rep_endpoint"):
        QStatusSubscriber.__init__(self, gconf, name, context=context, parent=parent)

        self.req = self.ctx.add_req(
            self.conf[rep_endpoint],
            timeout_ms=get_value(gconf, self.conf, "req_timeout_ms"),
            logger=self.__class__.__name__,
        )

    def read_all(self) -> dict[str, P.ParamDict[str, P.PDValue] | None] | None:
        resp = self.req.request(ReadAllReq())
        if resp.success:
            return resp.ret

    def read(self, pd_name: str) -> P.ParamDict[str, P.PDValue] | None:
        resp = self.req.request(ReadReq(pd_name))
        if resp.success:
            return resp.ret

    def write(self, pd_name: str, params: P.ParamDict[str, P.PDValue]) -> bool:
        resp = self.req.request(WriteReq(pd_name, params))
        return resp.success

    def save(self, filename: str) -> bool:
        resp = self.req.request(SaveReq(filename))
        return resp.success

    def load(self, filename: str) -> bool:
        resp = self.req.request(LoadReq(filename))
        return resp.success
