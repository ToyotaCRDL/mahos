#!/usr/bin/env python3

"""
Global parameter dictionary.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from ..msgs.common_msgs import Resp
from ..msgs import global_params_msgs
from ..msgs.global_params_msgs import GlobalParamsStatus, SetParamReq
from .node import Node
from .client import StatusClient


class GlobalParamsClient(StatusClient):
    """Simple GlobalParams Client."""

    M = global_params_msgs

    def get_param(self, key: str, default=None):
        """Get a parameter. If `key` is not found, return `default`."""

        s = self.get_status()
        if s is None:
            return default
        else:
            return s.params.get(key, default)

    def set_param(self, key: str, value):
        """Set a parameter. Value can be any pickle-able Python object."""

        resp = self.req.request(SetParamReq(key, value))

        return resp.success


class GlobalParams(Node):
    """Node to handle the global parameter dictionary (gparams).

    A parameter is set by SetParamReq (set_param() method of the client).
    Whole dictionary is distributed inside GlobalParamsStatus ('status' topic).

    """

    CLIENT = GlobalParamsClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

        self._params = {}

    def set_param(self, msg: SetParamReq) -> Resp:
        self._params[msg.key] = msg.value
        return Resp(True, "")

    def handle_req(self, msg):
        if isinstance(msg, SetParamReq):
            return self.set_param(msg)
        else:
            return Resp(False, "Invalid message type")

    def _publish(self):
        s = GlobalParamsStatus(params=self._params)
        self.status_pub.publish(s)

    def main(self):
        self.poll()
        self._publish()
