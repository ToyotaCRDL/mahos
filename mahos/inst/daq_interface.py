#!/usr/bin/env python3

"""
Typed Interface for NI-DAQ.

.. This file is a part of Mahos project.

"""

from typing import Optional, List, Sequence

import numpy as np

from .interface import InstrumentInterface


class ClockSourceInterface(InstrumentInterface):
    """Interface for ClockSource."""

    def get_internal_output(self) -> str:
        """Get internal output name."""

        return self.get("internal_output")


class AnalogOutInterface(InstrumentInterface):
    """Interface for AnalogOut."""

    def set_output_once(self, volts: np.ndarray) -> bool:
        """Set output voltage once."""

        return self.set("voltage", volts)

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

    def configure_command(self, timeout_sec: float = 5.0) -> bool:
        """Configure for command-output mode."""

        params = {"clock_mode": False, "timeout_sec": timeout_sec}
        return self.configure(params)


class BufferedEdgeCounterInterface(InstrumentInterface):
    def pop_block(self) -> np.ndarray:
        """Get data from buffer. If buffer is empty, this function blocks until data is ready."""

        return self.get("data", True)

    def pop_all_block(self) -> List[np.ndarray]:
        """Get all data from buffer as list.

        If buffer is empty, this function blocks until data is ready.

        """

        return self.get("all_data", True)

    def pop_opt(self) -> Optional[np.ndarray]:
        """Get data from buffer. If buffer is empty, returns None."""

        return self.get("data", False)

    def pop_all_opt(self) -> Optional[List[np.ndarray]]:
        """Get all data from buffer as list. If buffer is empty, returns None."""

        return self.get("all_data", False)


class APDCounterInterface(BufferedEdgeCounterInterface):
    def correct_cps(self, raw_cps: Sequence[float]) -> np.ndarray:
        """Correct the raw values in cps according to correction factors."""

        return self.get("correct", raw_cps)

    def get_correction_factor(self, xs_cps: Sequence[float]) -> np.ndarray:
        """Get the correction factor for given cps values."""

        return self.get("correction_factor", xs_cps)


class DigitalOutInterface(InstrumentInterface):
    def set_output(self, data) -> bool:
        return self.set("out", data)

    def set_command(self, name: str) -> bool:
        return self.set("command", name)

    def set_output_low(self) -> bool:
        return self.set("low")

    def set_output_high(self) -> bool:
        return self.set("high")

    def set_output_pulse(self) -> bool:
        return self.set("pulse")
