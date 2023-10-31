#!/usr/bin/env python3

"""
Common utilities for mahos cli.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import sys
from os import path
import platform

from ..node.node import load_gconf, infer_name


def check_load_gconf(fn):
    if not path.exists(fn):
        print("[ERROR] Config file doesn't exist.")
        sys.exit(1)
    return load_gconf(fn)


def infer_local_hostname(gconf: dict):
    hostname = platform.uname().node
    if hostname in gconf:
        return hostname
    else:
        return "localhost"


def host_is_local(gconf: dict, host: str) -> bool:
    return infer_local_hostname(gconf) == host


def init_gconf_host(gconf_fn, host):
    gconf = check_load_gconf(gconf_fn)
    if not host:
        host = infer_local_hostname(gconf)
    return gconf, host


def init_gconf_host_node(gconf_fn, host, node):
    gconf, host = init_gconf_host(gconf_fn, host)
    host, node = infer_name(node, host)
    return gconf, host, node


def init_gconf_host_nodes(gconf_fn, host, nodes):
    gconf, host = init_gconf_host(gconf_fn, host)
    new_nodes = []
    for node in nodes:
        # intentional: host can be overwritten here.
        host, nn = infer_name(node, host)
        new_nodes.append(nn)
    return gconf, host, new_nodes
