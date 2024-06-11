#!/usr/bin/env python3

"""
The mahos package

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import os

# top level exports for frequently used functions
from .node.node import load_gconf, local_conf

__all__ = ["cli", "gui", "inst", "meas", "msgs", "node", "util", "load_gconf", "local_conf"]

HOME_DIR = os.environ.get("MAHOS_HOME", os.path.expanduser(os.path.join("~", ".mahos")))
CONFIG_DIR = os.path.join(HOME_DIR, "config")
LOG_DIR = os.path.join(HOME_DIR, "log")

for d in (CONFIG_DIR, LOG_DIR):
    if not os.path.exists(d):
        os.makedirs(d)
        print("Directory {} is created.".format(d))
