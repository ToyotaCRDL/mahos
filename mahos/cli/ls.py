#!/usr/bin/env python3

"""
mahos ls command.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

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
    parser.add_argument(
        "-i", "--inst", action="store_true", help="print list of Instruments for InstrumentServer"
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
        if args.inst and Class == "InstrumentServer":
            print("  Instruments")
            for inst, idict in lconf.get("instrument", {}).items():
                imodule = idict.get("module", "<module unknown>")
                iClass = idict.get("class", "<class unknown>")
                print(f"    {inst:26s}: {imodule}.{iClass}")
            if lconf.get("instrument_overlay"):
                print("  InstrumentOverlays")
            for lay, ldict in lconf.get("instrument_overlay", {}).items():
                lmodule = ldict.get("module", "<module unknown>")
                lClass = ldict.get("class", "<class unknown>")
                print(f"    {lay:26s}: {lmodule}.{lClass}")
