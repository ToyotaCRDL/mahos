#!/usr/bin/env python3

"""
Typed Interface for Signal Generator.

.. This file is a part of MAHOS project.

"""

from typing import Optional

from .interface import InstrumentInterface


class SGInterface(InstrumentInterface):
    """Interface for Signal Genetator."""

    def get_bounds(self) -> dict:
        """Get bounds.

        Returns:
            freq (low, high): frequency bounds.
            power (low, high): power bounds.

        """

        return self.get("bounds")

    def get_opc(self, delay=None) -> bool:
        """Query OPC (operation complete) status."""

        return self.get("opc", delay)

    def set_output(self, on: bool) -> bool:
        """Set RF output switch."""

        return self.set("output", on)

    def set_dm_source(self, source: str) -> bool:
        """Set digital (IQ) modulation source."""

        return self.set("dm_source", source)

    def set_dm(self, on: bool) -> bool:
        """Set digital (IQ) modulation switch."""

        return self.set("dm", on)

    def set_modulation(self, on: bool) -> bool:
        """Set modulation switch."""

        return self.set("modulation", on)

    def set_init_cont(self, on: bool) -> bool:
        """Set continuous sweep initialization."""

        return self.set("init_cont", on)

    def set_abort(self) -> bool:
        """Abort the sweep operation."""

        return self.set("abort")

    def configure_cw(self, freq: float, power: float) -> bool:
        """Configure Continuous Wave output.

        :param freq: (Hz) frequnecy.
        :param power: (dBm) RF power.

        """

        return self.configure({"mode": "cw", "freq": freq, "power": power})

    def configure_point_trig_freq_sweep(
        self, start: float, stop: float, num: int, power: float, params: Optional[dict] = None
    ) -> bool:
        """Configure frequency sweep with point trigger.

        :param start: (Hz) start frequnecy.
        :param stop: (Hz) stop frequnecy.
        :param num: number of frequnecy points.
        :param power: (dBm) RF power.
        :param params: other parameters.

        """

        if params is None:
            params = {}
        params["mode"] = "point_trig_freq_sweep"
        params["start"] = start
        params["stop"] = stop
        params["num"] = num
        params["power"] = power

        return self.configure(params)
