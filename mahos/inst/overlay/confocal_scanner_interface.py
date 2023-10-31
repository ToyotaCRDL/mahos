#!/usr/bin/env python3

"""
Typed Interface for Confocal Scanner.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np

from ..interface import InstrumentInterface

from ...msgs.confocal_msgs import ScanMode, ScanDirection


class ConfocalScannerInterface(InstrumentInterface):
    """Interface for Confocal Scanner."""

    def get_line(self) -> np.ndarray | None:
        """Get single scan line. None is returned when scan is finished."""

        return self.get("line")

    def get_capability(self) -> list[ScanMode] | None:
        """Get scanner capablities."""

        return self.get("capability")

    def get_unit(self) -> str:
        return self.get("unit")

    def get_range(self):
        """Get travel range. range[i][0] (range[i][1]) is min (max) of i-th axis."""

        return self.get("range")

    def configure_scan(
        self,
        mode: ScanMode,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        xnum: int,
        ynum: int,
        z: float,
        direction: ScanDirection,
        time_window: float,
        line_timeout=None,
        dummy_samples=None,
    ) -> bool:
        params = {
            "mode": mode,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "xnum": xnum,
            "ynum": ynum,
            "z": z,
            "direction": direction,
            "time_window": time_window,
        }
        if line_timeout is not None:
            params["line_timeout"] = line_timeout
        if dummy_samples is not None:
            params["dummy_samples"] = dummy_samples

        return self.configure(params)
