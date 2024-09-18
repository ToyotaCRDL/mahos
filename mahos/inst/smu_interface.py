#!/usr/bin/env python3

"""
Typed Interface for Signal Generator.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import numpy as np
from .interface import InstrumentInterface


class SMUInterface(InstrumentInterface):
    """Interface for Signal Genetator."""

    def get_bounds(self) -> dict:
        """Get bounds.

        Returns:
            voltage (low, high): voltage bounds.
            current (low, high): current bounds.

        """

        return self.get("bounds")

    def get_opc(self, delay=None) -> bool:
        """Query OPC (operation complete) status."""

        return self.get("opc", delay)

    def get_data(self) -> np.ndarray:
        """Get data as 1D array."""
        return self.get("data")

    def configure_IV_curve(
        self, start: float, stop: float, num: int, delay: float, nplc: float, logx: bool = False
    ) -> bool:
        """Configure IV curve measurement.

        :param start: (V) start voltage.
        :param stop: (V) stop voltage.
        :param num: number of points.
        :param delay: (s) delay before measurement.
        :param nplc: measuerment time in nplc.
        :param logx: set True for log-space sweep.

        """

        return self.configure(
            {
                "start": start,
                "stop": stop,
                "point": num,
                "delay": delay,
                "nplc": nplc,
                "logx": logx,
            },
            label="iv_curve",
        )


def configure_VI_curve(
    self, start: float, stop: float, num: int, delay: float, nplc: float, logx: bool = False
) -> bool:
    """Configure IV curve measurement.

    :param start: (I) start current.
    :param stop: (I) stop current.
    :param num: number of points.
    :param delay: (s) delay before measurement.
    :param nplc: measuerment time in nplc.
    :param logx: set True for log-space sweep.

    """

    return self.configure(
        {
            "start": start,
            "stop": stop,
            "point": num,
            "delay": delay,
            "nplc": nplc,
            "logx": logx,
        },
        label="iv_curve",
    )
