#!/usr/bin/env python3

"""
Typed Interface for Time to Digital Converter.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np

from .interface import InstrumentInterface

from ..msgs.inst_tdc_msgs import ChannelStatus, RawEvents


class TDCInterface(InstrumentInterface):
    """Interface for Time to Digital Converter."""

    def configure_raw_events(
        self, base_config: str, save_file: str, trange: float = 0.0, tbin: float = 0.0
    ) -> bool:
        """Configure for raw_events measurement.

        trange and tbin may not be required for some instruments.

        """

        params = {
            "base_config": base_config,
            "save_file": save_file,
            "range": trange,
            "bin": tbin,
        }
        return self.configure(params, label="raw_events")

    def configure_histogram(self, base_config: str, trange: float, tbin: float) -> bool:
        """Configure for histogram measurement.

        Note that actual timebin maybe rounded.
        Set tbin to 0.0 for minimum bin.

        """

        params = {"base_config": base_config, "range": trange, "bin": tbin}
        return self.configure(params, label="histogram")

    def configure_correlation(self, base_config: str, trange: float, tbin: float) -> bool:
        """Configure for correlation measurement.

        Note that actual timebin maybe rounded.
        Set tbin to 0.0 for minimum bin.

        """

        params = {"base_config": base_config, "range": trange, "bin": tbin}
        return self.configure(params, label="histogram")

    def clear(self) -> bool:
        """Clear the data."""

        return self.set("clear")

    def set_sweeps(self, sweeps: int) -> bool:
        """set limit of sweeps. sweeps == 0 means unlimited."""

        return self.set("sweeps", sweeps)

    def set_duration(self, duration: float) -> bool:
        """set limit of measurement duration. duration == 0.0 means unlimited."""

        return self.set("duration", duration)

    def set_save_file_name(self, name: str) -> bool:
        """set save file name."""

        return self.set("file_name", name)

    def get_range_bin(self) -> dict:
        """Get range and bin in sec."""

        return self.get("range_bin")

    def get_timebin(self) -> float:
        """Get time bin in sec."""

        return self.get("bin")

    def get_data(self, ch: int) -> np.ndarray | None:
        """Get data of channel `ch`."""

        return self.get("data", ch)

    def get_data_roi(self, ch: int, roi: list[tuple[int, int]]) -> list[np.ndarray] | None:
        """Get data of channel `ch` using ROI."""

        return self.get("data_roi", {"ch": ch, "roi": roi})

    def get_status(self, ch: int) -> ChannelStatus | None:
        """Get status of channel `ch`."""

        return self.get("status", ch)

    def get_raw_events(self) -> str | RawEvents | None:
        """Get raw events of last measurement.

        Because RawEvents can be huge (> GB), the TDC can choose
        to transfer data via file system, instead of mahos (ZeroMQ) message.

        :returns: str: the file name of saved RawEvents.
                  RawEvents: the RawEvents.
                  None: failure.

        """

        return self.get("raw_events")
