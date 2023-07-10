#!/usr/bin/env python3

import typing as T

import datetime

from mahos.node.node import Node
from mahos.node.client import NodeClient
from mahos.util.timer import IntervalSleeper
from mahos.util.typing import NodeName


class ClockClient(NodeClient):
    def __init__(self, gconf: dict, name: NodeName, context=None, prefix=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self.get_time = self.add_sub(["time"])[0]


class Clock(Node):
    CLIENT = ClockClient

    def __init__(self, gconf: dict, name: NodeName, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.time_pub = self.add_pub("time")
        self.interval = IntervalSleeper(1.0)

    def main(self):
        self.interval.sleep()
        dt = datetime.datetime.now()
        self.time_pub.publish(dt.strftime("%F %T"))


class MultiplierClient(NodeClient):
    def __init__(self, gconf: dict, name: NodeName, context=None, prefix=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self.req = self.add_req(gconf)

    def multiply(self, a: int, b: int) -> int:
        return self.req.request((a, b))


class Multiplier(Node):
    CLIENT = MultiplierClient

    def __init__(self, gconf: dict, name: NodeName, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.add_rep(self.handle_multiply)

    def handle_multiply(self, req: T.Tuple[int, int]) -> int:
        a, b = req
        rep = a * b
        self.logger.info(f"{a} * {b} = {rep}")
        return rep


class AClient(NodeClient):
    def __init__(self, gconf: dict, name: NodeName, context=None, prefix=None, data_handler=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)
        self.get_data = self.add_sub([("data", data_handler)])[0]
        self.req = self.add_req(gconf)

    def set_data(self, data):
        return self.req.request(data)


class BClient(NodeClient):
    def __init__(self, gconf: dict, name: NodeName, context=None, prefix=None, data_handler=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)
        self.get_data = self.add_sub([("data", data_handler)])[0]
        self.req = self.add_req(gconf)

    def set_data(self, data):
        return self.req.request(data)


class A(Node):
    CLIENT = AClient

    def __init__(self, gconf: dict, name: NodeName, context=None):
        Node.__init__(self, gconf, name, context=context)
        self.dataA = self.dataB = 0
        self.B_cli = BClient(
            gconf, self.conf["target"]["B"], context=self.ctx, data_handler=self.set_dataB
        )
        self.add_client(self.B_cli)
        self.add_rep(self.set_dataA)
        self.data_pub = self.add_pub("data")
        self.both_pub = self.add_pub("both")
        self.interval = IntervalSleeper(0.5)

    def set_dataA(self, data):
        self.dataA = data

    def set_dataB(self, data):
        self.dataB = data

    def main(self):
        self.poll()
        self.interval.sleep()
        self.data_pub.publish(self.dataA)
        self.both_pub.publish(f"dataA: {self.dataA}, dataB: {self.dataB}")


class B(Node):
    CLIENT = BClient

    def __init__(self, gconf: dict, name: NodeName, context=None):
        Node.__init__(self, gconf, name, context=context)
        self.dataA = self.dataB = 100
        self.A_cli = AClient(
            gconf, self.conf["target"]["A"], context=self.ctx, data_handler=self.set_dataA
        )
        self.add_client(self.A_cli)
        self.add_rep(self.set_dataB)
        self.data_pub = self.add_pub("data")
        self.both_pub = self.add_pub("both")
        self.interval = IntervalSleeper(0.5)

    def set_dataA(self, data):
        self.dataA = data

    def set_dataB(self, data):
        self.dataB = data

    def main(self):
        self.poll()
        self.interval.sleep()
        self.data_pub.publish(self.dataB)
        self.both_pub.publish(f"dataA: {self.dataA}, dataB: {self.dataB}")
