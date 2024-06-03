#!/usr/bin/env python3

"""
File I/O for Tweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from os import path

import h5py

from ..msgs import param_msgs as P
from ..node.log import DummyLogger


class TweakerIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(
        self, filename: str, group: str, param_dicts: dict, start_stop_states: dict
    ) -> bool:
        """Save tweaker state (param_dicts and start_stop_state) to file using h5."""

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
                for pid, params in param_dicts.items():
                    if params is None:
                        continue
                    group = g.create_group(pid)
                    params.to_h5(group)

                    state = start_stop_states[pid]
                    #  Do not save unknown state (= None) because
                    #  it might be just irrelevant for given instrument.
                    if state is True:
                        group.attrs["__start_stop__"] = True
                    elif state is False:
                        group.attrs["__start_stop__"] = False
        except Exception:
            self.logger.exception(f"Error saving {filename}.")
            return False

        self.logger.info(f"Saved {filename}.")
        return True

    def load_data(self, filename: str, group: str) -> dict:
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
                for pid in g:
                    d[pid] = P.ParamDict.of_h5(g[pid])
                # TODO: __start_stop__
        except Exception:
            self.logger.exception(f"Error loading {filename}.")
            return {}

        self.logger.info(f"Loaded {filename}.")
        return d
