#!/usr/bin/env python3

"""
Typed Interface for Positioner.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from .interface import InstrumentInterface


class SinglePositionerInterface(InstrumentInterface):
    """Interface for Single Axis Positioner."""

    def set_target(self, value: float) -> bool:
        """Set the target position."""

        return self.set("target", value)

    def get_pos(self):
        """Get actual position."""

        return self.get("pos")

    def get_target(self):
        """Get target position."""

        return self.get("target")

    def get_status(self) -> dict[str, bool]:
        """Get status.

        :returns homed: True if device is homed.
        :returns moving: True if device is moving.

        """

        return self.get("status")

    def get_range(self):
        """Get travel range. range[0] (range[1]) is min (max)."""

        return self.get("range")

    def get_all(self) -> dict[str, [float, bool]]:
        """Get all important info about this device packed in a dict.

        :returns pos: current position.
        :returns target: target position.
        :returns range: travel range.
        :returns homed: True if device is homed.
        :returns moving: True if device is moving.

        """

        return self.get("all")
