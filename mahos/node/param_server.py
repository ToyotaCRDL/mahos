#!/usr/bin/env python3

"""
Global parameter server.

.. This file is a part of MAHOS project.

"""

from ..msgs.common_msgs import Resp
from ..msgs import param_server_msgs
from ..msgs.param_server_msgs import ParamServerStatus, SetParamReq
from .node import Node
from .client import StatusClient


class ParamClient(StatusClient):
    """Simple ParamServer Client."""

    M = param_server_msgs

    def get_param(self, name: str, default=None):
        """Get a parameter. If `name` is not found, return `default`."""

        s = self.get_status()
        if s is None:
            return default
        else:
            return s.params.get(name, default)

    def set_param(self, name: str, value):
        """Set a parameter. Value can be any pickle-able Python object."""

        resp = self.req.request(SetParamReq(name, value))

        return resp.success


class ParamServer(Node):
    CLIENT = ParamClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

        self._param_dict = {}

    def set_param(self, msg: SetParamReq) -> Resp:
        self._param_dict[msg.name] = msg.value
        return Resp(True, "")

    def handle_req(self, msg):
        if isinstance(msg, SetParamReq):
            return self.set_param(msg)
        else:
            return Resp(False, "Invalid message type")

    def _publish(self):
        s = ParamServerStatus(params=self._param_dict)
        self.status_pub.publish(s)

    def main(self):
        self.poll()
        self._publish()
