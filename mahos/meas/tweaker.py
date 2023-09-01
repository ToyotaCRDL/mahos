#!/usr/bin/env python3

"""
Tweaker for manually-tuned instrument parameters.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import pickle

from ..msgs.common_msgs import Resp
from ..msgs import param_msgs as P
from ..msgs import tweaker_msgs
from ..msgs.tweaker_msgs import TweakerStatus, ReadReq, ReadAllReq, WriteReq, SaveReq, LoadReq
from ..node.node import Node
from ..node.client import StatusClient
from ..inst.server import MultiInstrumentClient


PARAM_DICT_ID_DELIM = "."


class TweakerClient(StatusClient):
    """Simple Tweaker Client."""

    M = tweaker_msgs

    def read_all(self) -> dict[str, P.ParamDict[str, P.PDValue] | None] | None:
        resp = self.req.request(ReadAllReq())
        if resp.success:
            return resp.ret

    def read(self, param_dict_id: str) -> P.ParamDict[str, P.PDValue] | None:
        resp = self.req.request(ReadReq(param_dict_id))
        if resp.success:
            return resp.ret

    def write(self, param_dict_id: str, params: P.ParamDict[str, P.PDValue]) -> bool:
        resp = self.req.request(WriteReq(param_dict_id, params))
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

        self._param_dict_ids = self.conf["param_dicts"]
        self._param_dicts = {pd: None for pd in self._param_dict_ids}

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

    def _parse_param_dict_id(self, param_dict_id: str) -> tuple[str, str, str]:
        ids = param_dict_id.split(PARAM_DICT_ID_DELIM)
        if len(ids) == 1:
            return (ids[0], "", "")
        elif len(ids) == 2:
            return (ids[0], ids[1], "")
        elif len(ids) == 3:
            return (ids[0], ids[2], ids[1])
        else:
            raise ValueError(f"Invalid param_dict_id: {param_dict_id}")

    def wait(self):
        for inst_name in self.conf["target"]["servers"]:
            self.cli.wait(inst_name)

    def read_all(self, msg: ReadAllReq) -> Resp:
        param_dicts = {pd: self._read(pd) for pd in self._param_dict_ids}
        return Resp(all([d is not None for d in param_dicts.values()]), ret=param_dicts)

    def read(self, msg: ReadReq) -> Resp:
        ret = self._read(msg.param_dict_id)
        return Resp(ret is not None, ret=ret)

    def _read(self, param_dict_id: str) -> P.ParamDict[str, P.PDValue] | None:
        if param_dict_id not in self._param_dict_ids:
            return self.fail_with(f"Unknown ParamDict id: {param_dict_id}")
        inst, name, group = self._parse_param_dict_id(param_dict_id)
        d = self.cli.get_param_dict(inst, name, group)
        if d is None:
            self.logger.error(f"Failed to read ParamDict {param_dict_id}")
        return d

    def write(self, msg: WriteReq) -> Resp:
        if msg.param_dict_id not in self._param_dict_ids:
            return self.fail_with(f"Unknown ParamDict name: {msg.param_dict_id}")
        inst, name, group = self._parse_param_dict_id(msg.param_dict_id)
        success = self.cli.configure(inst, P.unwrap(msg.params), name, group)
        if success:
            self._param_dicts[msg.param_dict_id] = msg.params
            return Resp(True)
        else:
            msg = f"Failed to read ParamDict {msg.param_dict_id}"
            self.logger.error(msg)
            return Resp(False, msg)

    def save(self, msg: SaveReq) -> Resp:
        """Tweaker state is saved using pickle, not h5.

        This is because the tweaker param_dicts is primarily a memo of experiment condition.

        """

        with open(msg.file_name, "wb") as f:
            pickle.dump(P.unwrap(self._param_dicts), f)

        return Resp(True)

    def load(self, msg: LoadReq) -> Resp:
        """Load the tweaker state (param_dicts).

        Load is done defensively, checking the existence and validity in current setup.

        """

        with open(msg.file_name, "rb") as f:
            param_dicts = pickle.load(f)
        for param_dict_id, pd in param_dicts.items():
            if (
                pd is None
                or param_dict_id not in self._param_dicts
                or self._param_dicts[param_dict_id] is None
            ):
                continue
            for key, param in self._param_dicts:
                if key not in pd:
                    continue
                if not param.set(pd[key]):
                    self.logger.error(f"Cannot set {param_dict_id}[{key}] to {pd[key]}")

        return Resp(True, ret=self._param_dicts)

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
