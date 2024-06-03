#!/usr/bin/env python3

"""
File I/O for PosTweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from os import path

import h5py

from ..node.log import DummyLogger


class PosTweakerIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(self, filename: str, group: str, axis_states: dict) -> bool:
        """Save PosTweaker state (axis_states) to file using h5."""

        mode = "r+" if path.exists(filename) else "w"
        try:
            with h5py.File(filename, mode) as f:
                if group:
                    if group in f:
                        g = f[group]
                    else:
                        g = f.create_group(group)
                else:
                    g = f
                for ax, state in axis_states.items():
                    if state is None:
                        continue
                    gax = g.create_group(ax)
                    for key in ("pos", "target", "homed"):
                        if key in state:
                            gax.attrs[key] = state[key]
        except Exception:
            self.logger.exception(f"Error saving {filename}.")
            return False

        self.logger.info(f"Saved {filename}.")
        return True

    def load_data(self, filename: str, group: str, axes: list[str]) -> dict:
        """Load params from filename[group] and return as dict."""

        d = {}
        try:
            with h5py.File(filename, "r") as f:
                if group:
                    if group not in f:
                        self.logger.error(f"group {group} doesn't exist in {filename}")
                        return {}
                    g = f[group]
                else:
                    g = f
                for ax in axes:
                    d[ax] = {}
                    for key in ("pos", "target", "homed"):
                        if key in g[ax].attrs:
                            d[ax][key] = g[ax].attrs[key]
        except Exception:
            self.logger.exception(f"Error loading {filename}.")
            return {}

        self.logger.info(f"Loaded {filename}.")
        return d
