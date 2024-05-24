#!/usr/bin/env python3

"""
Message Types for the Recorder.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np
import msgpack

from .common_meas_msgs import BasicMeasData


class RecorderData(BasicMeasData):
    def __init__(self, params: dict | None = None, label: str = ""):
        """Note that set_units() must be called before collecting actual data."""

        self.set_version(1)
        self.init_params(params, label)
        self.init_attrs()

        # not using dict for units and data here
        # because assoc. array is easier to save to / load from HDF5 file
        self.units: list[tuple[str, str]] = []
        self.data: list[list[float]] = []
        self.xdata: list[float] = []

    def set_units(self, units: list[tuple[str, str]]):
        """set units and initialize data.

        units is list of (instrument name, unit name).

        """

        self.units = units
        self.data = [[] for _ in units]

    def get_insts(self) -> list[str]:
        """Get list of recorded instrument names (data labels)."""

        return [u[0] for u in self.units]

    def get_units(self) -> list[str]:
        """Get list of units."""

        return [u[1] for u in self.units]

    def get_unit_to_insts(self) -> dict[str, list[str]]:
        """Get a map from unit to list of corresponding inst."""

        ret = {}
        for inst, unit in self.units:
            if unit not in ret:
                ret[unit] = [inst]
            else:
                ret[unit].append(inst)
        return ret

    def index(self, inst: str) -> int:
        """Get index of instrument name (data label)."""

        return self.get_insts().index(inst)

    def roll(self, data):
        max_len = self.params.get("max_len", 0)
        if max_len <= 0 or len(data) <= max_len:
            return data
        return data[len(data) - max_len :]

    def append(self, x: float, y: dict[str, float]):
        self.xdata.append(x)
        self.xdata = self.roll(self.xdata)
        for inst, value in y.items():
            i = self.index(inst)
            self.data[i].append(value)
            self.data[i] = self.roll(self.data[i])

    def get_xdata(self) -> np.ndarray:
        return np.array(self.xdata)

    def get_unit(self, inst: str) -> str:
        return self.unit[self.index(inst)][1]

    def get_ydata(self, inst: str) -> np.ndarray:
        return np.array(self.data[self.index(inst)])

    def init_axes(self):
        self.xlabel: str = "Time"
        self.xunit: str = "s"
        self.xscale: str = "linear"
        self.yscale: str = "linear"

    def has_data(self):
        return self.data and self.xdata

    # h5
    def _h5_write_units(self, val):
        d = val.copy()
        return np.void(msgpack.dumps(d))

    def _h5_write_data(self, val):
        return np.array(val)

    def _h5_read_units(self, val):
        return msgpack.loads(val.tobytes())

    def _h5_read_data(self, val):
        return np.array(val).tolist()

    def _h5_attr_writers(self) -> dict:
        return {"units": self._h5_write_units}

    def _h5_dataset_writers(self) -> dict:
        return {"xdata": self._h5_write_data, "data": self._h5_write_data}

    def _h5_readers(self) -> dict:
        return {
            "units": self._h5_read_units,
            "xdata": self._h5_read_data,
            "data": self._h5_read_data,
        }


def update_data(data: RecorderData):
    if data.version() <= 0:
        # version 0 to 1
        data.label = data.params["method"]
        del data.params["method"]
        data.set_version(1)

    return data
