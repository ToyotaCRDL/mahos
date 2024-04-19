#!/usr/bin/env python3

"""
Message Types for the Chrono.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from .common_meas_msgs import BasicMeasData


class ChronoData(BasicMeasData):
    def __init__(self, params: dict | None = None):
        self.set_version(0)
        self.init_params(params)
        self.init_attrs()

        self.data = {}
        self.units = {}
        self.xdata = []

    def set_units(self, units):
        self.units = units

    def append(self, x, y):
        self.xdata.append(x)
        for name, value in y.items():
            if name not in self.data:
                self.data[name] = []
            self.data[name].append(value)

    def init_axes(self):
        self.xlabel: str = "Time"
        self.xunit: str = "s"
        self.xscale: str = "linear"
        self.yscale: str = "linear"

    def has_data(self):
        return self.data and self.xdata
