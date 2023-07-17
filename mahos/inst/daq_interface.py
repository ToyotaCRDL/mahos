#!/usr/bin/env python3

"""
Typed Interface for NI-DAQ.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

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
        rate: float | None = None,
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


class BufferedReaderInterface(InstrumentInterface):
    """Common interface for AnalogIn and BufferedEdgeCounter."""

    def pop_block(
        self,
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]:
        """Get data from buffer.

        If buffer is empty, this function blocks until data is ready.

        """

        return self.get("data", True)

    def pop_all_block(
        self,
    ) -> list[np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]]:
        """Get all data from buffer as list.

        If buffer is empty, this function blocks until data is ready.

        """

        return self.get("all_data", True)

    def pop_opt(
        self,
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float] | None:
        """Get data from buffer.

        If buffer is empty, returns None.

        """

        return self.get("data", False)

    def pop_all_opt(
        self,
    ) -> list[np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]] | None:
        """Get all data from buffer as list.

        If buffer is empty, returns None.

        """

        return self.get("all_data", False)

    def get_unit(self) -> str:
        """Get reading / count unit in str."""

        return self.get("unit")


class AnalogInInterface(BufferedReaderInterface):
    def pop_block(
        self,
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]:
        """Get data from buffer.

        If buffer is empty, this function blocks until data is ready.

        see pop_block() for return value types.

        """

        return self.get("data", True)

    def pop_all_block(
        self,
    ) -> list[np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]]:
        """Get all data from buffer as list.

        If buffer is empty, this function blocks until data is ready.

        see pop_all_opt() for return value types.

        """

        return self.get("all_data", True)

    def pop_opt(
        self,
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float] | None:
        """Get data from buffer.

        :returns: np.ndarray if len(lines) == 1 and stamp == False.
                  list[np.ndarray] if len(lines) > 1 and stamp == False.
                  tuple[np.ndarray, float] if len(lines) == 1 and stamp == True.
                  tuple[list[np.ndarray], float] if len(lines) > 1 and stamp == True.
                  None if buffer is empty.

        """

        return self.get("data", False)

    def pop_all_opt(
        self,
    ) -> list[np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]] | None:
        """Get all data from buffer as list.

        :returns: list[np.ndarray] if len(lines) == 1 and stamp == False.
                  list[list[np.ndarray]] if len(lines) > 1 and stamp == False.
                  list[tuple[np.ndarray, float]] if len(lines) == 1 and stamp == True.
                  list[tuple[list[np.ndarray], float]] if len(lines) > 1 and stamp == True.
                  None if buffer is empty.

        """

        return self.get("all_data", False)


class BufferedEdgeCounterInterface(BufferedReaderInterface):
    # override pop_*() methods because BufferedEdgeCounter has one channel per class.

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
