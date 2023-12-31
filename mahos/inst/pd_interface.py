#!/usr/bin/env python3

"""
Typed Interface for Photo Detectors.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np

from .daq_interface import BufferedReaderInterface


class PDInterface(BufferedReaderInterface):
    """Common interface for APDCounter and AnalogIn-based PD."""

    # override pop_*() methods because PD has one channel per class.

    def pop_block(self) -> np.ndarray:
        """Get data from buffer. If buffer is empty, this function blocks until data is ready."""

        return self.get("data", True)

    def pop_all_block(self) -> list[np.ndarray]:
        """Get all data from buffer as list.

        If buffer is empty, this function blocks until data is ready.

        """

        return self.get("all_data", True)

    def pop_opt(self) -> np.ndarray | None:
        """Get data from buffer. If buffer is empty, returns None."""

        return self.get("data", False)

    def pop_all_opt(self) -> list[np.ndarray] | None:
        """Get all data from buffer as list. If buffer is empty, returns None."""

        return self.get("all_data", False)


class APDCounterInterface(PDInterface):
    """Interface for APDCounter."""

    def correct_cps(self, raw_cps: list[float]) -> np.ndarray:
        """Correct the raw values in cps according to correction factors."""

        return self.get("correct", raw_cps)

    def get_correction_factor(self, xs_cps: list[float]) -> np.ndarray:
        """Get the correction factor for given cps values."""

        return self.get("correction_factor", xs_cps)
