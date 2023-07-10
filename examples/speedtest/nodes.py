#!/usr/bin/env python3

import numpy as np
import time

from mahos.node.node import Node, local_conf
from mahos.node.client import NodeClient


class Client(NodeClient):
    def __init__(self, gconf: dict, name, context=None, prefix=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)
        self.req = self.add_req(gconf)

    def request(self, msg):
        return self.req.request(msg)


class Server(Node):
    CLIENT = Client

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.data = np.random.default_rng(1).normal(size=self.conf.get("data_size", 100))
        self.add_rep(self.handler)

    def handler(self, msg):
        self._dummy = msg
        return self.data


class Tester(Node):
    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)
        self.cli = Client(gconf, self.conf["target"]["server"], context=context)
        server_conf = local_conf(gconf, self.conf["target"]["server"])

        self.data = np.random.default_rng(11).normal(size=server_conf.get("data_size", 100))
        self.data_bytes = self.data.size * self.data.itemsize
        self.count = -1
        print(f"Payload size: {self.data_bytes*1E-6} MB")

    def main(self):
        self._dummy = self.cli.request(self.data)

        self.count += 1
        if self.count == 0:
            self.start_time = time.perf_counter()
            return
        if self.count % 100:
            return
        elapsed = time.perf_counter() - self.start_time
        rate = self.count / elapsed
        rate_MB_s = 1e-6 * 2 * self.data_bytes * rate  # factor of 2 is request and reply
        print(
            f"{self.count:5d} requests in {elapsed:4.1f} sec.:"
            + f" {rate:5.2f} Hz, {rate_MB_s:7.2f} MB/s"
        )
