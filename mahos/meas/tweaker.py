#!/usr/bin/env python3

"""
Generic tweaker for manually-tunable Instrument's ParamDicts.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os

import h5py

from ..msgs.common_msgs import Reply
from ..msgs import param_msgs as P
from ..msgs import tweaker_msgs
from ..msgs.tweaker_msgs import TweakerStatus, ReadReq, ReadAllReq, WriteReq, WriteAllReq
from ..msgs.tweaker_msgs import StartReq, StopReq, ResetReq, SaveReq, LoadReq
from ..node.node import Node
from ..node.client import NodeClient, StatusClient
from ..inst.server import MultiInstrumentClient


class TweakerClient(StatusClient):
    """Simple Tweaker Client."""

    M = tweaker_msgs

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

    def save(self, filename: str, group: str = "") -> bool:
        rep = self.req.request(SaveReq(filename, group))
        return rep.success

    def load(
        self, filename: str, group: str = ""
    ) -> dict[str, P.ParamDict[str, P.PDValue] | None] | None:
        rep = self.req.request(LoadReq(filename, group))
        if rep.success:
            return rep.ret


class TweakSaver(NodeClient):
    """TweakerClient with limited capability, only save request."""

    M = tweaker_msgs

    def __init__(self, gconf: dict, name, context=None, prefix=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)
        self.req = self.add_req(gconf)

    def save(self, filename: str, group: str = "") -> bool:
        rep = self.req.request(SaveReq(filename, group))
        return rep.success


class Tweaker(Node):
    """Generic tweaker for manually-tunable Instrument's ParamDicts."""

    CLIENT = TweakerClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.cli = MultiInstrumentClient(
            gconf, self.conf["target"]["servers"], context=self.ctx, prefix=self.joined_name()
        )
        self.add_clients(self.cli)

        #: dict[param_dict_id, ParamDict | None]
        self._param_dicts = {pd: None for pd in self.conf["param_dicts"]}
        #: dict[param_dict_id, None | True | False]
        self._start_stop_states = {pd: None for pd in self.conf["param_dicts"]}

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

    def _parse_param_dict_id(self, pid: str) -> tuple[str, str]:
        """returns (inst, label)."""

        ids = P.split_label(pid)
        if len(ids) == 1:
            return (ids[0], "")
        elif len(ids) == 2:
            return ids
        else:
            raise ValueError(f"Invalid param_dict_id: {pid}")

    def wait(self):
        for inst_name in self.conf["target"]["servers"]:
            self.cli.wait(inst_name)

    def read_all(self, msg: ReadAllReq) -> Reply:
        success = True
        for pid in self._param_dicts:
            res = self._read(pid)
            if res is None:
                success = False
            else:
                self._param_dicts[pid] = res
        return Reply(success, ret=self._param_dicts)

    def read(self, msg: ReadReq) -> Reply:
        ret = self._read(msg.param_dict_id)
        if ret is None:
            return Reply(False)
        else:
            self._param_dicts[msg.param_dict_id] = ret
            return Reply(True, ret=ret)

    def _read(self, param_dict_id: str) -> P.ParamDict[str, P.PDValue] | None:
        if param_dict_id not in self._param_dicts:
            self.logger.error(f"Unknown ParamDict id: {param_dict_id}")
            return None
        inst, label = self._parse_param_dict_id(param_dict_id)
        d = self.cli.get_param_dict(inst, label)
        if d is None:
            self.logger.error(f"Failed to read ParamDict {param_dict_id}")
        return d

    def _write(self, param_dict_id: dict, params: P.ParamDict[str, P.PDValue]) -> Reply:
        if param_dict_id not in self._param_dicts:
            return self.fail_with(f"Unknown ParamDict id: {param_dict_id}")
        inst, label = self._parse_param_dict_id(param_dict_id)
        success = self.cli.configure(inst, P.unwrap(params), label)
        if success:
            self._param_dicts[param_dict_id] = params
            return Reply(True)
        else:
            return self.fail_with(f"Failed to write ParamDict {param_dict_id}")

    def write(self, msg: WriteReq) -> Reply:
        return self._write(msg.param_dict_id, msg.params)

    def write_all(self, msg: WriteAllReq) -> Reply:
        return Reply(
            all([self._write(pid, params).success for pid, params in msg.param_dicts.items()])
        )

    def start(self, msg: StartReq) -> Reply:
        inst, label = self._parse_param_dict_id(msg.param_dict_id)
        success = self.cli.start(inst, label)
        if success:
            self._start_stop_states[msg.param_dict_id] = True
        return Reply(success)

    def stop(self, msg: StopReq) -> Reply:
        inst, label = self._parse_param_dict_id(msg.param_dict_id)
        success = self.cli.stop(inst, label)
        if success:
            self._start_stop_states[msg.param_dict_id] = False
        return Reply(success)

    def reset(self, msg: StopReq) -> Reply:
        inst, label = self._parse_param_dict_id(msg.param_dict_id)
        success = self.cli.reset(inst, label)
        if success:
            self._start_stop_states[msg.param_dict_id] = False
        return Reply(success)

    def save(self, msg: SaveReq) -> Reply:
        """Save tweaker state (param_dicts and start_stop_state) to file using h5."""

        mode = "r+" if os.path.exists(msg.filename) else "w"
        with h5py.File(msg.filename, mode) as f:
            if msg.group:
                if msg.group in f:
                    g = f[msg.group]
                else:
                    g = f.create_group(msg.group)
            else:
                g = f
            for pid, params in self._param_dicts.items():
                if params is None:
                    continue
                group = g.create_group(pid)
                params.to_h5(group)

                state = self._start_stop_states[pid]
                #  Do not save unknown state (= None) because
                #  it might be just irrelevant for given instrument.
                if state is True:
                    group.attrs["__start_stop__"] = True
                elif state is False:
                    group.attrs["__start_stop__"] = False

        self.logger.info(f"Saved {msg.filename}.")
        return Reply(True)

    def load(self, msg: LoadReq) -> Reply:
        """Load the tweaker state (param_dicts).

        Load is done defensively, checking the existence and validity in current setup.

        """

        with h5py.File(msg.filename, "r") as f:
            if msg.group:
                if msg.group not in f:
                    self.logger.error(f"group {msg.group} doesn't exist in {msg.filename}")
                    return Reply(False)
                g = f[msg.group]
            else:
                g = f
            for pid, params in self._param_dicts.items():
                if pid not in g or params is None:
                    continue
                for key, lp in P.ParamDict.of_h5(g[pid]).items():
                    if (param := params.getf(key)) is not None:
                        if not param.set(lp):
                            self.logger.error(f"Cannot set {pid}[{key}] to {lp}")

        self.logger.info(f"Loaded {msg.filename}.")
        return Reply(True, ret=self._param_dicts)

    def handle_req(self, msg):
        if isinstance(msg, ReadReq):
            return self.read(msg)
        elif isinstance(msg, ReadAllReq):
            return self.read_all(msg)
        elif isinstance(msg, WriteReq):
            return self.write(msg)
        elif isinstance(msg, WriteAllReq):
            return self.write_all(msg)
        elif isinstance(msg, StartReq):
            return self.start(msg)
        elif isinstance(msg, StopReq):
            return self.stop(msg)
        elif isinstance(msg, ResetReq):
            return self.reset(msg)
        elif isinstance(msg, SaveReq):
            return self.save(msg)
        elif isinstance(msg, LoadReq):
            return self.load(msg)
        else:
            return self.fail_with("Invalid message type")

    def _publish(self):
        s = TweakerStatus(param_dict_ids=list(self._param_dicts.keys()))
        self.status_pub.publish(s)

    def main(self):
        self.poll()
        self._publish()
