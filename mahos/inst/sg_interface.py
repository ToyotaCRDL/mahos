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

    def set_init_cont(self, on: bool) -> bool:
        """Set continuous sweep initialization."""

        return self.set("init_cont", on)

    def set_abort(self) -> bool:
        """Abort the sweep operation."""

        return self.set("abort")

    def set_dm_source(self, source: str) -> bool:
        """[Deprecated] Use configure_iq_ext() etc. instead."""

        return self.set("dm_source", source)

    def set_dm(self, on: bool) -> bool:
        """[Deprecated] Use configure_iq_ext() etc. instead."""

        return self.set("dm", on)

    def set_modulation(self, on: bool) -> bool:
        """[Deprecated] Use configure_iq_ext() etc. instead."""

        return self.set("modulation", on)

    def configure_cw(self, freq: float, power: float, ch: int = 1, reset: bool = True) -> bool:
        """Configure Continuous Wave output.

        :param freq: (Hz) frequnecy.
        :param power: (dBm) RF power.

        """

        l = "cw" if ch == 1 else f"cw{ch}"
        return self.configure({"freq": freq, "power": power, "reset": reset}, label=l)

    def configure_iq_ext(self, ch: int = 1) -> bool:
        """Configure external IQ modulation."""

        l = "iq_ext" if ch == 1 else f"iq_ext{ch}"
        return self.configure({}, label=l)

    def configure_cw_iq_ext(
        self, freq: float, power: float, ch: int = 1, reset: bool = True
    ) -> bool:
        """Configure CW output and external IQ modulation."""

        return self.configure_cw(freq, power, ch=ch, reset=reset) and self.configure_iq_ext(ch)

    def configure_fm_ext(
        self, freq: float, power: float, deviation: float, ch: int = 1, reset: bool = True
    ) -> bool:
        """Configure external FM mode.

        :param freq: (Hz) frequnecy.
        :param power: (dBm) RF power.
        :param deviation: (Hz) FM deviation.

        """

        l = "fm_ext" if ch == 1 else f"fm_ext{ch}"
        return self.configure(
            {"freq": freq, "power": power, "deviation": deviation, "reset": reset}, label=l
        )

    def configure_am_ext(
        self,
        freq: float,
        power: float,
        depth: float,
        log: bool = False,
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure external AM mode.

        :param freq: (Hz) frequnecy.
        :param power: (dBm) RF power.
        :param depth: (dB | %) AM depth. If log is True (False), unit is dB (%).
        :param log: Set True (False) to set AM depth mode to log (linear) scale.

        """

        l = "am_ext" if ch == 1 else f"am_ext{ch}"
        return self.configure(
            {"freq": freq, "power": power, "depth": depth, "log": log, "reset": reset}, label=l
        )

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
