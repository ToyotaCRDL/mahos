#!/usr/bin/env python3

"""
mahos echo command.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import time
import argparse
import sys
import importlib

from ..node.node import join_name
from ..node.client import EchoSubscriber
from .util import init_gconf_host_node


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog="mahos echo", description="Echo published topic from running mahos node."
    )
    parser.add_argument(
        "-c", "--conf", type=str, default="conf.toml", help="config file name (default: conf.toml)"
    )
    parser.add_argument("-H", "--host", type=str, default="", help="host name")
    parser.add_argument(
        "-t", "--topic", type=str, default="status", help="topic name (default: status)"
    )
    parser.add_argument(
        "-r",
        "--rate",
        action="store_true",
        help="print rate statistics instead of message itself.",
    )
    parser.add_argument(
        "node", type=str, help="node name or full name ({})".format(join_name(("host", "node")))
    )

    args = parser.parse_args(args)

    return args


def main(args=None):
    args = parse_args(args)
    gconf, host, node = init_gconf_host_node(args.conf, args.host, args.node)
    joined_name = join_name((host, node))
    print("Subscribing to {} of {} (config file: {}).".format(args.topic, joined_name, args.conf))

    try:
        c = gconf[host][node]
    except KeyError:
        print(f"[ERROR] {joined_name} is not found in the config file.")
        sys.exit(1)
    module = importlib.import_module(c["module"])
    NodeClass = getattr(module, c["class"])
    if hasattr(NodeClass, "TOPIC_TYPES"):
        type_ = NodeClass.TOPIC_TYPES.get(args.topic)
    else:
        type_ = None

    sub = EchoSubscriber(gconf, joined_name, topic=args.topic, rate=args.rate, type_=type_)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exitting.")
    finally:
        sub.close()
