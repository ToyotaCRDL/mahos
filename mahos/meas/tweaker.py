#!/usr/bin/env python3

"""
Tweaker for manually-tuned instrument parameters.

.. This file is a part of MAHOS project.

"""

from ..msgs.common_msgs import Resp
from ..msgs import param_msgs as P
from ..msgs import tweaker_msgs
from ..msgs.tweaker_msgs import TweakerStatus, ReadReq, ReadAllReq, WriteReq, SaveReq, LoadReq
from ..node.node import Node
from ..node.client import StatusClient
from ..inst.server import MultiInstrumentClient


class TweakerClient(StatusClient):
    """Simple Tweaker Client."""

    M = tweaker_msgs

    def read_all(self) -> dict[str, P.ParamDict[str, P.PDValue] | None] | None:
        resp = self.req.request(ReadAllReq())
        if resp.success:
            return resp.ret

    def read(self, pd_name: str) -> P.ParamDict[str, P.PDValue] | None:
        resp = self.req.request(ReadReq(pd_name))
        if resp.success:
            return resp.ret

    def write(self, pd_name: str, params: P.ParamDict[str, P.PDValue]) -> bool:
        resp = self.req.request(WriteReq(pd_name))
        return resp.success

    def save(self, filename: str) -> bool:
        resp = self.req.request(SaveReq(filename))
        return resp.success

    def load(self, filename: str) -> dict[str, P.ParamDict[str, P.PDValue] | None] | None:
        resp = self.req.request(LoadReq(filename))
        if resp.success:
            return resp.ret


class Tweaker(Node):
    """Tweaker for manually-tuned instrument parameters."""

    CLIENT = TweakerClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.cli = MultiInstrumentClient(
            gconf, self.conf["target"]["servers"], context=self.ctx, prefix=self.joined_name()
        )

        self._pd_targets = self.conf["param_dicts"]
        self._param_dicts = {k: None for k in self._pd_targets}

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

    def wait(self):
        for inst_name in self.conf["target"]:
            self.cli.wait(inst_name)

    def read_all(self, msg: ReadAllReq) -> Resp:
        param_dicts = {pd: self._read(pd) for pd in self._pd_targets}
        return Resp(all([d is not None for d in param_dicts.values()]), ret=param_dicts)

    def read(self, msg: ReadReq) -> Resp:
        ret = self._read(msg.pd_name)
        return Resp(ret is not None, ret=ret)

    def _read(self, pd_name) -> P.ParamDict[str, P.PDValue] | None:
        if pd_name not in self._pd_targets:
            return self.fail_with(f"Unknown ParamDict name: {pd_name}")
        tgt = self._pd_targets[pd_name]
        d = self.cli.get_param_dict(tgt["inst_name"], tgt.get("pd_name", ""), tgt.get("group", ""))
        if d is None:
            self.logger.error(f"Failed to read ParamDict {pd_name}")
        return d

    def write(self, msg: WriteReq) -> Resp:
        if msg.pd_name not in self._pd_targets:
            return self.fail_with(f"Unknown ParamDict name: {msg.pd_name}")
        tgt = self._pd_targets[msg.pd_name]
        success = self.cli.configure(
            tgt["inst_name"], P.unwrap(msg.params), tgt.get("pd_name", ""), tgt.get("group", "")
        )
        if success:
            self._param_dicts[msg.pd_name] = msg.params
            return Resp(True)
        else:
            msg = f"Failed to read ParamDict {msg.pd_name}"
            self.logger.error(msg)
            return Resp(False, msg)

    def save(self, msg: SaveReq) -> Resp:
        return Resp(False, "not implemented")

    def load(self, msg: LoadReq) -> Resp:
        return Resp(False, "not implemented")

    def handle_req(self, msg):
        if isinstance(msg, ReadReq):
            return self.read(msg)
        elif isinstance(msg, ReadAllReq):
            return self.read_all(msg)
        elif isinstance(msg, WriteReq):
            return self.write(msg)
        elif isinstance(msg, SaveReq):
            return self.save(msg)
        elif isinstance(msg, LoadReq):
            return self.load(msg)
        else:
            return self.fail_with("Invalid message type")

    def _publish(self):
        s = TweakerStatus(param_dict_names=list(self._param_dicts.keys()))
        self.status_pub.publish(s)

    def main(self):
        self.poll()
        self._publish()
