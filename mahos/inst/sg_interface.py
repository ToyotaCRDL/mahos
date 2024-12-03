#!/usr/bin/env python3

"""
Typed Interface for Signal Generator.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from .interface import InstrumentInterface


class SGInterface(InstrumentInterface):
    """Interface for Signal Genetator."""

    def get_bounds(self, ch: int = 1) -> dict:
        """Get bounds of channel `ch`.

        Returns:
            freq (low, high): frequency bounds.
            power (low, high): power bounds.

        """

        l = "" if ch == 1 else f"{ch}"
        return self.get("bounds", label=l)

    def get_opc(self, delay=None) -> bool:
        """Query OPC (operation complete) status."""

        return self.get("opc", delay)

    def set_output(self, on: bool, ch: int = 1) -> bool:
        """Set RF output switch."""

        l = "" if ch == 1 else f"{ch}"
        return self.set("output", on, label=l)

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

    def configure_cw(self, freq: float, power: float, ch: int = 1, reset: bool = True) -> bool:
        """Configure Continuous Wave output.

        :param freq: (Hz) frequnecy.
        :param power: (dBm) RF power.

        """

        l = "cw" if ch == 1 else f"cw{ch}"
        return self.configure({"freq": freq, "power": power, "reset": reset}, label=l)

    def configure_ext_iq(self, freq: float, power: float, ch: int = 1, reset: bool = True) -> bool:
        """Configure external IQ modulation output.

        :param freq: (Hz) frequnecy.
        :param power: (dBm) RF power.

        """

        l = "ext_iq" if ch == 1 else f"ext_iq{ch}"
        return self.configure({"freq": freq, "power": power, "reset": reset}, label=l)

    def configure_point_trig_freq_sweep(
        self, start: float, stop: float, num: int, power: float, params: dict | None = None
    ) -> bool:
        """Configure frequency sweep with point trigger.

        :param start: (Hz) start frequnecy.
        :type start: float
        :param stop: (Hz) stop frequnecy.
        :type stop: float
        :param num: number of frequnecy points.
        :type num: int
        :param power: (dBm) RF power.
        :type power: float
        :param params: other parameters.
        :type params: dict[str, Any]

        """

        if params is None:
            params = {}
        params["start"] = start
        params["stop"] = stop
        params["num"] = num
        params["power"] = power

        return self.configure(params, label="point_trig_freq_sweep")
