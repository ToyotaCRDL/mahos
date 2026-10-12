#!/usr/bin/env python3

"""
Specialized tweaker for manually operated positioners.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from mahos.msgs.common_msgs import CleanupArtifactReq, Reply
from mahos.msgs import pos_tweaker_msgs
from mahos.msgs.pos_tweaker_msgs import PosTweakerState, PosTweakerStatus, SetTargetReq
from mahos.msgs.pos_tweaker_msgs import HomeReq, HomeAllReq, StopReq, StopAllReq, SaveReq, LoadReq
from mahos.msgs.tweaker_msgs import GetStateReq, SaveArtifactReq, LoadArtifactReq
from mahos.node.node import Node
from mahos.node.client import StatusClient
from mahos.inst.server import MultiInstrumentClient
from mahos.inst.positioner_interface import SinglePositionerInterface
from mahos.meas.file_transport import FileTransportNodeMixin
from mahos.meas.pos_tweaker_io import PosTweakerIO
from mahos.meas.tweaker import TweakerFileReqMixin


class PosTweakerClient(TweakerFileReqMixin, StatusClient):
    """Simple PosTweaker Client."""

    M = pos_tweaker_msgs

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

    # override for annotation
    def get_status(self) -> PosTweakerStatus:
        return self._get_status()

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


class PosTweaker(FileTransportNodeMixin, Node):
    """Specialized tweaker for manually operated positioners.

    The target instrument must provide
    :class:`SinglePositionerInterface <mahos.inst.positioner_interface.SinglePositionerInterface>`.

    :param target.servers: InstrumentServer targets (instrument name, server full name).
        Required keys: at least one axis key; each listed key is controlled as a positioner.
    :type target.servers: dict[str, str]
    :param target.log: The LogBroker target (broker full name).
    :type target.log: str
    :param file_transport_dir: Optional node-side directory for client-node file transport.
    :type file_transport_dir: str

    """

    CLIENT = PosTweakerClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.cli = MultiInstrumentClient(
            gconf, self.conf["target"]["servers"], context=self.ctx, prefix=self.joined_name()
        )
        self.add_clients(self.cli)

        #: dict[str, SinglePositionerInterface]
        self._axis_positioners = {
            ax: SinglePositionerInterface(self.cli, ax) for ax in self.conf["target"]["servers"]
        }
        #: dict[str, dict[str, [float, bool]] | None]
        self._axis_states = {ax: None for ax in self.conf["target"]["servers"]}

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

        self.io = PosTweakerIO(self.logger)
        self.init_file_transport(("save",))

    def wait(self):
        for inst_name in self.conf["target"]["servers"]:
            self.cli.wait(inst_name)

    def set_target(self, msg: SetTargetReq) -> Reply:
        for ax, pos in msg.axis_pos.items():
            if ax not in self._axis_positioners:
                return self.fail_with(f"Invalid axis {ax}")
            if not self._axis_positioners[ax].set_target(pos):
                return self.fail_with(f"Failed to set target of {ax}")
        return Reply(True)

    def home(self, msg: HomeReq) -> Reply:
        if msg.axis not in self._axis_positioners:
            return self.fail_with(f"Invalid axis {msg.axis}")

        if self._axis_positioners[msg.axis].reset():
            return Reply(True)
        else:
            return self.fail_with(f"Failed to home {msg.axis}")

    def home_all(self, msg: HomeAllReq) -> Reply:
        for ax, positioner in self._axis_positioners.items():
            if not positioner.reset():
                return self.fail_with(f"Failed to home {ax}")
        return Reply(True)

    def stop(self, msg: StopReq) -> Reply:
        if msg.axis not in self._axis_positioners:
            return self.fail_with(f"Invalid axis {msg.axis}")

        if self._axis_positioners[msg.axis].stop():
            return Reply(True)
        else:
            return self.fail_with(f"Failed to home {msg.axis}")

    def stop_all(self, msg: StopAllReq) -> Reply:
        success = True
        for ax, positioner in self._axis_positioners.items():
            if not positioner.stop():
                self.logger.error(f"Failed to stop {ax}")
                success = False
        return Reply(success)

    def get_state(self, msg: GetStateReq) -> Reply:
        """Return a serializable snapshot for local embedding by a measurement node."""

        return Reply(True, ret=PosTweakerState(self._axis_states.copy()))

    def save(self, msg: SaveReq) -> Reply:
        """Save tweaker state (pos, target, homed) to file using h5."""

        return Reply(self.io.save_data(msg.file_name, "", self._axis_states))

    def load(self, msg: LoadReq) -> Reply:
        """Load the tweaker state (target) and set the target."""

        ax_states = self.io.load_data(msg.file_name, msg.group)
        if not ax_states:
            return Reply(False)
        for ax, positioner in self._axis_positioners.items():
            if ax not in ax_states:
                self.logger.warn(f"axis {ax} is not in loaded state.")
                continue
            states = ax_states[ax]
            if "target" in states:
                positioner.set_target(states["target"])
        return Reply(True)

    def save_artifact(self, msg: SaveArtifactReq) -> Reply:
        """Atomically create a PosTweaker-state transport artifact."""

        return self.publish_artifact(
            msg.file_name,
            "save",
            lambda path: Reply(self.io.save_data(path, "", self._axis_states)),
            "Failed to create PosTweaker artifact",
        )

    def load_artifact(self, msg: LoadArtifactReq) -> Reply:
        """Load PosTweaker state from a validated shared artifact."""

        return self.consume_artifact(
            msg.artifact,
            lambda path: self.load(LoadReq(path, msg.group)),
            "Failed to load PosTweaker artifact",
        )

    def handle_req(self, msg):
        if isinstance(msg, SetTargetReq):
            return self.set_target(msg)
        elif isinstance(msg, HomeReq):
            return self.home(msg)
        elif isinstance(msg, HomeAllReq):
            return self.home_all(msg)
        elif isinstance(msg, StopReq):
            return self.stop(msg)
        elif isinstance(msg, StopAllReq):
            return self.stop_all(msg)
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

    def _update(self):
        for ax, positioner in self._axis_positioners.items():
            d = positioner.get_all()
            if d is None:
                continue
            if self._axis_states[ax] is None:
                self._axis_states[ax] = {}
            self._axis_states[ax].update(d)

    def _publish(self):
        s = PosTweakerStatus(axis_states=self._axis_states)
        self.status_pub.publish(s)

    def main(self):
        self.poll()
        self._update()
        self._publish()
