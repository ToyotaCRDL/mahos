#!/usr/bin/env python3

"""
Typed Interface for Source Meter (Measure) Unit.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from .interface import InstrumentInterface


class SMUInterface(InstrumentInterface):
    """Interface for Source Meter (Measure) Unit."""

    def get_opc(self, delay=None) -> bool:
        """Query OPC (operation complete) status."""

        return self.get("opc", delay)

    def get_data(self) -> float:
        """Get measurement data.

        :returns: Depends on operation mode.
            - IV mode: current reading in float.

        """

        return self.get("data")
