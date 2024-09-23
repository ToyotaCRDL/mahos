#!/usr/bin/env python3

"""
Specialized tweaker for manually operated positioners.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from ..msgs.common_msgs import Reply
from ..msgs import pos_tweaker_msgs
from ..msgs.pos_tweaker_msgs import PosTweakerStatus, SetTargetReq
from ..msgs.pos_tweaker_msgs import HomeReq, HomeAllReq, StopReq, StopAllReq, LoadReq
from ..msgs.tweaker_msgs import SaveReq
from ..node.node import Node
from ..node.client import StatusClient
from ..inst.server import MultiInstrumentClient
from ..inst.positioner_interface import SinglePositionerInterface
from .pos_tweaker_io import PosTweakerIO


class PosTweakerClient(StatusClient):
    """Simple PosTweaker Client.

    tweaker.TweakSaver can be used to use only save() function.

    """

    M = pos_tweaker_msgs

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

    def save(self, filename: str, group: str = "") -> bool:
        rep = self.req.request(SaveReq(filename, group))
        return rep.success

    def load(self, filename: str, group: str = "") -> bool:
        rep = self.req.request(LoadReq(filename, group))
        return rep.success


class PosTweaker(Node):
    """Specialized tweaker for manually operated positioners."""

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

    def save(self, msg: SaveReq) -> Reply:
        """Save tweaker state (pos, target, homed) to file using h5."""

        return Reply(self.io.save_data(msg.filename, msg.group, self._axis_states))

    def load(self, msg: LoadReq) -> Reply:
        """Load the tweaker state (target) and set the target."""

        ax_states = self.io.load_data(msg.filename, msg.group)
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
        elif isinstance(msg, SaveReq):
            return self.save(msg)
        elif isinstance(msg, LoadReq):
            return self.load(msg)
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
