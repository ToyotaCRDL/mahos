#!/usr/bin/env python3

"""
Specialized tweaker for manually operated positioners.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os

import h5py

from ..msgs.common_msgs import Reply
from ..msgs import pos_tweaker_msgs
from ..msgs.pos_tweaker_msgs import PosTweakerStatus, SetTargetReq
from ..msgs.pos_tweaker_msgs import HomeReq, HomeAllReq, LoadReq
from ..msgs.tweaker_msgs import SaveReq
from ..node.node import Node
from ..node.client import StatusClient
from ..inst.server import MultiInstrumentClient
from ..inst.positioner_interface import SinglePositionerInterface


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

    def save(self, msg: SaveReq) -> Reply:
        """Save tweaker state (pos, target, homed) to file using h5."""

        mode = "r+" if os.path.exists(msg.filename) else "w"
        with h5py.File(msg.filename, mode) as f:
            if msg.group:
                if msg.group in f:
                    g = f[msg.group]
                else:
                    g = f.create_group(msg.group)
            else:
                g = f
            for ax, state in self._axis_states.items():
                if state is None:
                    continue
                group = g.create_group(ax)
                for key in ("pos", "target", "homed"):
                    if key in state:
                        group.attrs[key] = state[key]

        self.logger.info(f"Saved {msg.filename}.")
        return Reply(True)

    def load(self, msg: LoadReq) -> Reply:
        """Load the tweaker state (target) and set the target."""

        with h5py.File(msg.filename, "r") as f:
            if msg.group:
                if msg.group not in f:
                    self.logger.error(f"group {msg.group} doesn't exist in {msg.filename}")
                    return Reply(False)
                g = f[msg.group]
            else:
                g = f
            for ax, positioner in self._axis_positioners.items():
                if ax not in g or "target" not in g[ax].attrs:
                    continue
                target = g[ax].attrs["target"]
                positioner.set_target(target)

        self.logger.info(f"Loaded {msg.filename}.")
        return Reply(True)

    def handle_req(self, msg):
        if isinstance(msg, SetTargetReq):
            return self.set_target(msg)
        elif isinstance(msg, HomeReq):
            return self.home(msg)
        elif isinstance(msg, HomeAllReq):
            return self.home_all(msg)
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
