#!/usr/bin/env python3

import sys
import importlib
import argparse

from ..node.node import Node, join_name
from ..gui.gui_node import GUINode
from .util import init_gconf_host_node
from .threaded_nodes import ThreadedNodes


def parse_args(args):
    parser = argparse.ArgumentParser(prog="mahos run", description="Run a mahos node.")
    parser.add_argument(
        "-c", "--conf", type=str, default="conf.toml", help="config file name (default: conf.toml)"
    )
    parser.add_argument("-H", "--host", type=str, default="", help="host name")
    parser.add_argument("-t", "--thread", action="store_true", help="start threaded nodes")
    parser.add_argument(
        "node", type=str, help="node name or full name ({})".format(join_name(("host", "node")))
    )

    args = parser.parse_args(args)

    return args


def main_thread(gconf: dict, host: str, name: str):
    n = ThreadedNodes(gconf, host, name)
    return n.main_interrupt()


def main(args=None):
    args = parse_args(args)
    gconf, host, node = init_gconf_host_node(args.conf, args.host, args.node)

    if args.thread:
        return main_thread(gconf, host, node)

    joined_name = join_name((host, node))
    print("Starting {} (config file: {}).".format(joined_name, args.conf))

    try:
        c = gconf[host][node]
    except KeyError:
        print(f"[ERROR] {joined_name} is not found in the config file.")
        sys.exit(1)

    module = importlib.import_module(c["module"])
    NodeClass = getattr(module, c["class"])
    if issubclass(NodeClass, Node):
        n: Node = NodeClass(gconf, joined_name)
        return n.main_interrupt()
    elif issubclass(NodeClass, GUINode):
        n: GUINode = NodeClass(gconf, joined_name)
        return n.main()
    else:
        print("[ERROR] {} isn't a valid Node class: {}".format(joined_name, NodeClass.__name__))
