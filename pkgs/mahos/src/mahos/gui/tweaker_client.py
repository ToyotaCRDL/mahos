#!/usr/bin/env python3

"""
Qt signal-based client of Tweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from mahos.gui.Qt import QtCore

from mahos.msgs import param_msgs as P
from mahos.msgs.tweaker_msgs import TweakerStatus, ReadReq, ReadAllReq, WriteReq, WriteAllReq
from mahos.msgs.tweaker_msgs import StartReq, StopReq, ResetReq
from mahos.node.node import get_value
from mahos.gui.client import QStatusSubscriber
from mahos.meas.tweaker import TweakerFileReqMixin


class QTweakerClient(TweakerFileReqMixin, QStatusSubscriber):
    """Qt-based client for Tweaker."""

    statusUpdated = QtCore.pyqtSignal(TweakerStatus)

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

    def read_all(self) -> tuple[bool, dict[str, P.ParamDict[str, P.PDValue] | None]]:
        rep = self.req.request(ReadAllReq())
        return rep.success, rep.ret

    def read(self, param_dict_id: str) -> P.ParamDict[str, P.PDValue] | None:
        rep = self.req.request(ReadReq(param_dict_id))
        if rep.success:
            return rep.ret

    def write_all(self, param_dicts: dict[str, P.ParamDict[str, P.PDValue]]) -> bool:
        rep = self.req.request(WriteAllReq(param_dicts))
        return rep.success

    def write(self, param_dict_id: str, params: P.ParamDict[str, P.PDValue]) -> bool:
        rep = self.req.request(WriteReq(param_dict_id, params))
        return rep.success

    def start(self, param_dict_id: str) -> bool:
        rep = self.req.request(StartReq(param_dict_id))
        return rep.success

    def stop(self, param_dict_id: str) -> bool:
        rep = self.req.request(StopReq(param_dict_id))
        return rep.success

    def reset(self, param_dict_id: str) -> bool:
        rep = self.req.request(ResetReq(param_dict_id))
        return rep.success

    def save(self, file_name: str) -> bool:
        return self.save_file(file_name)

    def load(
        self, file_name: str, group: str = ""
    ) -> dict[str, P.ParamDict[str, P.PDValue] | None] | None:
        rep = self.load_file(file_name, group)
        if rep.success:
            return rep.ret
