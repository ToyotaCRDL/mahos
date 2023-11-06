#!/usr/bin/env python3

from __future__ import annotations
import datetime

from mahos.node.node import Node
from mahos.node.client import NodeClient
from mahos.util.timer import IntervalSleeper
from mahos.util.typing import NodeName

from msgs import DatetimeMP, CalcRepMP, CalcReqMP


class ClockClient(NodeClient):
    def __init__(self, gconf: dict, name: NodeName, context=None, prefix=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self.get_datetime = self.add_sub([("datetime", None, DatetimeMP)])[0]


class Clock(Node):
    CLIENT = ClockClient
    TOPIC_TYPES = {"datetime": DatetimeMP}

    def __init__(self, gconf: dict, name: NodeName, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.datetime_pub = self.add_pub("datetime")
        self.interval = IntervalSleeper(1.0)

    def main(self):
        self.interval.sleep()
        now = DatetimeMP(datetime.datetime.now())
        self.datetime_pub.publish(now)


class CalcClient(NodeClient):
    def __init__(self, gconf: dict, name: NodeName, context=None, prefix=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self.req = self.add_req(gconf, rep_type=CalcRepMP)

    def calc(self, a: int, b: int) -> int:
        return self.req.request(CalcReqMP(a, b))


class Calc(Node):
    CLIENT = CalcClient

    def __init__(self, gconf: dict, name: NodeName, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.add_rep(self.handle_calc, req_type=CalcReqMP)

    def handle_calc(self, req: CalcReqMP) -> CalcRepMP:
        a, b = req.a, req.b
        if b == 0:
            self.logger.error("b cannot be 0")
            return CalcRepMP(False, "b cannot be 0", a * b, None)

        product = a * b
        quotient = a / b
        self.logger.info(f"{a} * {b} = {product}, {a} / {b} = {quotient}")
        return CalcRepMP(True, "", product, quotient)
