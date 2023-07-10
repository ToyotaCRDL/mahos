#!/usr/bin/env python3

import argparse

from ..node.node import join_name, local_conf, host_nodes, hosts, threaded_nodes
from .util import check_load_gconf


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog="mahos ls", description="List available node in a config file."
    )
    parser.add_argument(
        "-c", "--conf", type=str, default="conf.toml", help="config file name (default: conf.toml)"
    )

    args = parser.parse_args(args)

    return args


def main(args=None):
    args = parse_args(args)
    gconf = check_load_gconf(args.conf)
    for host in hosts(gconf):
        for tname, nodes in threaded_nodes(gconf, host).items():
            head = f"thread::{host}::{tname}"
            print("{:30s}: {}".format(head, ", ".join(nodes)))
    for host, node in host_nodes(gconf):
        joined_name = join_name((host, node))
        lconf = local_conf(gconf, joined_name)
        module = lconf.get("module", "<module unknown>")
        Class = lconf.get("class", "<class unknown>")
        print(f"{joined_name:30s}: {module}.{Class}")
