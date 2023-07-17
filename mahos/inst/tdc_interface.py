#!/usr/bin/env python3

"""
Typed Interface for Time to Digital Converter.

.. This file is a part of MAHOS project.

"""

from typing import Optional

import numpy as np

from .interface import InstrumentInterface


class TDCInterface(InstrumentInterface):
    """Interface for Time to Digital Converter."""

    def configure_load_range_bin(self, fn: str, trange: float, tbin: float) -> bool:
        """Load control file and set range and binwidth according to trange and tbin in sec.

        Note that actual timebin maybe rounded.

        """

        params = {"file": fn, "range": trange, "bin": tbin}
        return self.configure(params)

    def configure_range_bin(self, trange: float, tbin: float) -> bool:
        """Set range and binwidth according to trange and tbin in sec.

        Note that actual timebin maybe rounded.

        """

        params = {"range": trange, "bin": tbin}
        return self.configure(params)

    def clear(self) -> bool:
        """Clear the data."""

        return self.set("clear")

    def set_sweeps(self, sweeps: int) -> bool:
        """set limit of sweeps. sweeps == 0 means unlimited."""

        return self.set("sweeps", sweeps)

    def get_range_bin(self) -> dict:
        """Get range and bin."""

        return self.get("range_bin")

    def get_timebin(self) -> float:
        """Get time bin in sec."""

        return self.get("bin")

    def get_data(self, ch: int) -> Optional[np.ndarray]:
        """Get data of channel `ch`."""

        return self.get("data", ch)

    def get_status(self, ch: int):
        """Get status of channel `ch`."""

        return self.get("status", ch)
