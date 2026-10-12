#!/usr/bin/env python3

"""
Typed Interface for ODMR Sweeper.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import numpy as np

from mahos.inst.interface import InstrumentInterface
from mahos.msgs.inst.pg_msgs import PulsePattern


class ODMRSweeperInterface(InstrumentInterface):
    """Interface for ODMR Sweeper."""

    def validate(self, params: dict, label: str) -> tuple[bool, str, str]:
        """Validate parameters using sweeper-specific constraints."""

        ret = self.get("validate", params, label)
        if ret is None:
            return False, "Failed to request validation from ODMR sweeper.", ""
        if not isinstance(ret, tuple) or len(ret) != 3:
            return False, "Invalid validation response from ODMR sweeper.", ""
        return ret

    def get_line(self) -> np.ndarray | None:
        """Get single sweep line."""

        return self.get("line")

    def get_point(self) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        """Get a point, paired with its raw traces when trace acquisition is enabled."""

        return self.get("point")

    def get_unit(self) -> str:
        """Get unit."""

        return self.get("unit")

    def get_bounds(self) -> dict:
        """Get SG bounds.

        Returns:
            freq (low, high): frequency bounds.
            power (low, high): power bounds.

        """

        return self.get("bounds")

    def get_capability(self) -> dict[str, bool]:
        """Get detector and pulse-generation capabilities."""

        return self.get("capability")

    def get_pulse_pattern(self) -> PulsePattern | None:
        """Get current pulse pattern."""

        return self.get("pulse_pattern")
