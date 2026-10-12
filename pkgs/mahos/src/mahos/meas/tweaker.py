#!/usr/bin/env python3

"""
Generic tweaker for manually-tunable Instrument's ParamDicts.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import os

from mahos.msgs.common_msgs import CleanupArtifactReq, Reply
from mahos.msgs import param_msgs as P
from mahos.msgs import tweaker_msgs
from mahos.msgs.tweaker_msgs import TweakerStatus, ReadReq, ReadAllReq, WriteReq, WriteAllReq
from mahos.msgs.tweaker_msgs import StartReq, StopReq, ResetReq, SaveReq, LoadReq
from mahos.msgs.tweaker_msgs import GetStateReq, TweakerState, SaveArtifactReq, LoadArtifactReq
from mahos.msgs.pos_tweaker_msgs import PosTweakerState
from mahos.node.node import Node
from mahos.node.client import NodeClient, StatusClient
from mahos.inst.server import MultiInstrumentClient
from mahos.meas.file_transport import FileTransportClientMixin, FileTransportNodeMixin
from mahos.meas.tweaker_io import TweakerIO
from mahos.meas.pos_tweaker_io import PosTweakerIO


class TweakerFileReqMixin(FileTransportClientMixin):
    """Implement optional shared-file transport for Tweaker-like clients."""

    def save_file(self, file_name: str) -> bool:
        return self.request_output(
            SaveReq(file_name), SaveArtifactReq(os.path.basename(file_name)), file_name
        )

    def load_file(self, file_name: str, group: str = "") -> Reply:
        return self.request_input(
            LoadReq(file_name, group), lambda artifact: LoadArtifactReq(artifact, group), file_name
        )


class TweakerClient(TweakerFileReqMixin, StatusClient):
    """Simple Tweaker Client."""

    M = tweaker_msgs

    def __init__(
        self,
        gconf: dict,
        name,
        context=None,
        prefix=None,
        status_handler=None,
        file_transport_dir=None,
    ):
        StatusClient.__init__(
            self,
            gconf,
            name,
            context=context,
            prefix=prefix,
            status_handler=status_handler,
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


class TweakSaver(NodeClient):
    """Client that embeds remote Tweaker state into a local measurement file."""

    M = tweaker_msgs

    def __init__(self, gconf: dict, name, context=None, prefix=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)
        self.req = self.add_req(gconf)
        self.tweaker_io = TweakerIO(self.logger)
        self.pos_tweaker_io = PosTweakerIO(self.logger)

    def save(self, file_name: str, group: str = "") -> bool:
        rep = self.req.request(GetStateReq())
        if not rep.success:
            return False
        if isinstance(rep.ret, TweakerState):
            return self.tweaker_io.save_data(
                file_name, group, rep.ret.param_dicts, rep.ret.start_stop_states
            )
        elif isinstance(rep.ret, PosTweakerState):
            return self.pos_tweaker_io.save_data(file_name, group, rep.ret.axis_states)
        self.logger.error(f"Invalid Tweaker state type: {type(rep.ret)}")
        return False


class Tweaker(FileTransportNodeMixin, Node):
    """Generic tweaker for manually tunable Instrument ParamDicts.

    The instrument must provide a ParamDict-based interface, i.e.,
    ``get_param_dict_labels()``, ``get_param_dict()``, and ``configure()``.

    :param target.servers: InstrumentServer targets (instrument name, server full name).
        Required keys: instrument names referenced by ``param_dicts`` entries.
    :type target.servers: dict[str, str]
    :param target.log: The LogBroker target (broker full name).
    :type target.log: str
    :param param_dicts: The list of <instrument name>::<ParamDict label name>.
    :type param_dicts: list[str]
    :param file_transport_dir: Optional node-side directory for client-node file transport.
    :type file_transport_dir: str

    """

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

        self.io = TweakerIO(self.logger)
        self.init_file_transport(("save",))

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

    def get_state(self, msg: GetStateReq) -> Reply:
        """Return a serializable snapshot for local embedding by a measurement node."""

        return Reply(
            True,
            ret=TweakerState(self._param_dicts.copy(), self._start_stop_states.copy()),
        )

    def save(self, msg: SaveReq) -> Reply:
        """Save tweaker state (param_dicts and start_stop_state) to file using h5."""

        return Reply(
            self.io.save_data(msg.file_name, "", self._param_dicts, self._start_stop_states)
        )

    def load(self, msg: LoadReq) -> Reply:
        """Load the tweaker state into current param_dicts.

        Load is done defensively, checking the existence and validity in current setup.

        """

        pds = self.io.load_data(msg.file_name, msg.group)
        if not pds:
            return Reply(False)
        for pid, params in self._param_dicts.items():
            if pid not in pds:
                continue
            for key, lp in pds[pid].items():
                if (param := params.getf(key)) is not None:
                    if not param.read_only() and not param.set(lp):
                        self.logger.error(f"Cannot set {pid}[{key}] to {lp}")
        return Reply(True, ret=self._param_dicts)

    def save_artifact(self, msg: SaveArtifactReq) -> Reply:
        """Atomically create a Tweaker-state transport artifact."""

        return self.publish_artifact(
            msg.file_name,
            "save",
            lambda path: Reply(
                self.io.save_data(path, "", self._param_dicts, self._start_stop_states)
            ),
            "Failed to create Tweaker artifact",
        )

    def load_artifact(self, msg: LoadArtifactReq) -> Reply:
        """Load Tweaker state from a validated shared artifact."""

        return self.consume_artifact(
            msg.artifact,
            lambda path: self.load(LoadReq(path, msg.group)),
            "Failed to load Tweaker artifact",
        )

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
        elif isinstance(msg, GetStateReq):
            return self.get_state(msg)
        elif isinstance(msg, SaveReq):
            return self.save(msg)
        elif isinstance(msg, LoadReq):
            return self.load(msg)
        elif isinstance(msg, SaveArtifactReq):
            return self.save_artifact(msg)
        elif isinstance(msg, LoadArtifactReq):
            return self.load_artifact(msg)
        elif isinstance(msg, CleanupArtifactReq):
            return self.cleanup_artifact(msg)
        else:
            return self.fail_with("Invalid message type")

    def _publish(self):
        s = TweakerStatus(param_dict_ids=list(self._param_dicts.keys()))
        self.status_pub.publish(s)

    def main(self):
        self.poll()
        self._publish()
