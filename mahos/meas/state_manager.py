#!/usr/bin/env python3

"""
Measurement state manager.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import importlib

from ..msgs.common_msgs import Reply
from ..msgs import state_manager_msgs
from ..msgs.state_manager_msgs import ManagerStatus, RestoreReq, CommandReq
from ..node.node import Node
from ..node.client import StateClient, StatusClient


class StateManagerClient(StatusClient):
    """Simple StateManager Client."""

    M = state_manager_msgs

    def command(self, name) -> Reply:
        rep = self.req.request(CommandReq(name))
        return rep.success

    def restore(self, name) -> Reply:
        rep = self.req.request(RestoreReq(name))
        return rep.success


class StateManager(Node):
    """StateManager for meas nodes."""

    CLIENT = StateManagerClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self._clis = {}
        self._States = {}
        for node_name, (state_module, state_class) in self.conf["node"].items():
            cli = StateClient(gconf, node_name, context=self.ctx)
            self.add_clients(cli)
            self._clis[node_name] = cli
            sm = importlib.import_module(state_module)
            self._States[node_name] = getattr(sm, state_class)

        self._commands = {}
        self._last_states = {}
        if "command" in self.conf:
            for cmd_name, d in self.conf["command"].items():
                self._commands[cmd_name] = {}
                self._last_states[cmd_name] = None
                for node_name, state_name in d.items():
                    state = getattr(self._States[node_name], state_name)
                    self._commands[cmd_name][node_name] = state

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

    def wait(self):
        # waiting the clients is prone to deadlock??
        # (if this SM targets a client and that client is trying to use SM.)
        pass

    def store_states(self, cmd_name):
        d = {}
        for node_name in self._commands[cmd_name]:
            d[node_name] = self._clis[node_name].get_state()
        self._last_states[cmd_name] = d

    def command(self, msg: CommandReq) -> Reply:
        if msg.name not in self._commands:
            return self.fail_with(f"Unknown command {msg.name}.")

        self.store_states(msg.name)
        for node_name, state in self._commands[msg.name].items():
            last_state = self._last_states[msg.name][node_name]
            if not self._clis[node_name].is_up():
                self.logger.warn(f"Node {node_name} is not up. Skipping.")
                continue
            if state == last_state:
                continue
            self.logger.info(f"command[{msg.name}][{node_name}]: {last_state} to {state}")
            success = self._clis[node_name].change_state(state)
            if not success:
                return self.fail_with(f"Failed to change state of {node_name} to {state}")
        return Reply(True)

    def restore(self, msg: RestoreReq) -> Reply:
        if msg.name not in self._commands:
            return self.fail_with(f"Unknown command {msg.name}.")
        if self._last_states[msg.name] is None:
            return self.fail_with(f"Last state is not stored for command {msg.name}.")

        states = self.get_states()
        for node_name, state in self._last_states[msg.name].items():
            current_state = states[node_name]
            if state is None:
                self.logger.warn(f"Last state is not stored for {node_name}. Skipping.")
                continue
            if state == current_state:
                continue
            self.logger.info(f"restore[{msg.name}][{node_name}]: {current_state} to {state}")
            success = self._clis[node_name].change_state(state)
            if not success:
                return self.fail_with(f"Failed to change state of {node_name} to {state}")
        return Reply(True)

    def handle_req(self, msg):
        if isinstance(msg, CommandReq):
            return self.command(msg)
        elif isinstance(msg, RestoreReq):
            return self.restore(msg)
        else:
            return self.fail_with("Invalid message type")

    def get_states(self):
        s = {}
        for node_name, client in self._clis.items():
            s[node_name] = client.get_state()
        return s

    def _publish(self):
        s = ManagerStatus(states=self.get_states())
        self.status_pub.publish(s)

    def main(self):
        self.poll()
        self._publish()
