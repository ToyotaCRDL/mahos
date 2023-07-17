#!/usr/bin/env python3

"""
Typed Interface for Imaging ODMR Sweeper.

.. This file is a part of MAHOS project.

"""

from typing import Optional

import numpy as np

from ..interface import InstrumentInterface


class IODMRSweeperInterface(InstrumentInterface):
    """Interface for Imaging ODMR Sweeper."""

    def get_frames(self) -> Optional[np.ndarray]:
        """Get single sweep frames."""

        return self.get("frames")

    def get_bounds(self) -> dict:
        """Get SG bounds.

        Returns:
            freq (low, high): frequency bounds.
            power (low, high): power bounds.

        """

        return self.get("bounds")

    def configure_sweep(
        self,
        start: float,
        stop: float,
        num: int,
        power: float,
        exposure_delay: float,
        exposure_time: float,
    ) -> bool:
        params = {
            "start": start,
            "stop": stop,
            "num": num,
            "power": power,
            "exposure_delay": exposure_delay,
            "exposure_time": exposure_time,
        }

        return self.configure(params)
