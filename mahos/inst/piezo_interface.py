#!/usr/bin/env python3

"""
Typed Interface for Piezo Stage.

.. This file is a part of MAHOS project.

"""

from typing import Union, Sequence, Optional

from .interface import InstrumentInterface


class PiezoInterface(InstrumentInterface):
    """Interface for Piezo."""

    def set_target(self, value: Union[dict, Sequence[float]]) -> bool:
        """Set the piezo target position."""

        return self.set("target", value)

    def set_servo(self, on: bool) -> bool:
        """Set servo control."""

        return self.set("servo", on)

    def get_pos(self):
        """Get actual position."""

        return self.get("pos")

    def get_pos_ont(self):
        """Get actual position and on-target status."""

        return self.get("pos_ont")

    def get_target(self):
        """Get target position."""

        return self.get("target")

    def get_range(self):
        """Get position range."""

        return self.get("range")

    def configure_interactive(self) -> bool:
        """Configure piezo for interactive use."""

        return self.configure({})


class AOPiezoInterface(PiezoInterface):
    """Interface for Piezo with AnalogOut control."""

    def configure_clock(
        self,
        samples: int,
        rate: Optional[float] = None,
        clock_dir: bool = True,
        finite: bool = True,
        timeout_sec: float = 5.0,
    ) -> bool:
        """Configure for clock-sync mode."""

        params = {
            "clock_mode": True,
            "clock_dir": True,
            "finite": finite,
            "timeout_sec": timeout_sec,
        }
        if rate is not None:
            params["rate"] = rate
        return self.configure(params)
