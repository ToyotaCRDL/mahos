#!/usr/bin/env python3

"""
The mahos package

.. This file is a part of MAHOS project.

"""

__all__ = ["cli", "gui", "inst", "meas", "msgs", "node", "util"]

import os

config_dir = os.path.expanduser("~/.config/mahos")
cache_dir = os.path.expanduser("~/.cache/mahos")
log_dir = os.path.expanduser("~/.cache/mahos/log")

for d in (config_dir, cache_dir, log_dir):
    if not os.path.exists(d):
        os.makedirs(d)
        print("Directory {} is created.".format(d))
