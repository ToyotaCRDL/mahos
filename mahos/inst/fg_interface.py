#!/usr/bin/env python3

"""
Typed Interface for Function Generator.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from .interface import InstrumentInterface


class FGInterface(InstrumentInterface):
    """Interface for Function Genetator."""

    def get_opc(self, delay=None) -> bool:
        """Query OPC (operation complete) status."""

        return self.get("opc", delay)

    def get_bounds(self, ch: int = 1) -> dict:
        """Get bounds.

        :returns: dict[str, tuple[float, float]] with following two keys.
            ampl (low, high): (V) voltage amplitude bounds in Vpp (peak-to-peak).
            freq (low, high): (Hz) frequency bounds.

        """

        return self.get("bounds", args=ch)

    def set_output(self, on: bool, ch: int = 1) -> bool:
        """Set output switch."""

        return self.set("output", {"on": on, "ch": ch})

    def configure_cw(
        self,
        wave: str,
        freq: float,
        ampl_Vpp: float,
        offset: float = 0.0,
        ch: int = 1,
        reset: bool = False,
    ) -> bool:
        """Configure Continuous Wave output.

        :param wave: wave form (function).
        :param freq: (Hz) frequnecy.
        :param ampl_Vpp: (V) amplitude in Vpp (peak-to-peak).
        :param offset: (V) offset in V.
        :param ch: channel.
        :param reset: If True, reset before configuration.

        """

        return self.configure(
            {"wave": wave, "freq": freq, "ampl": ampl_Vpp, "ch": ch, "reset": reset}, label="cw"
        )

    def configure_gate(
        self,
        wave: str,
        freq: float,
        ampl_Vpp: float,
        phase_deg: float,
        offset: float = 0.0,
        source: str = "",
        polarity: bool | None = None,
        idle_level: str = "",
        ch: int = 1,
        reset: bool = False,
    ) -> bool:
        """Configure Gated Burst output.

        :param wave: wave form (function).
        :param freq: (Hz) frequnecy.
        :param ampl_Vpp: (V) amplitude in Vpp (peak-to-peak).
        :param phase_deg: (Degrees) starting phase.
        :param offset: (V) offset in V.
        :param source: trigger source. empty to use default conf.
        :param polarity: trigger and gate polarity. True for positive, None to use default conf.
        :param idle_level: idle_level. empty to use default conf.
        :param ch: channel.
        :param reset: If True, reset before configuration.

        """

        params = {
            "wave": wave,
            "freq": freq,
            "ampl": ampl_Vpp,
            "offset": offset,
            "phase": phase_deg,
            "source": source,
            "polarity": polarity,
            "idle_level": idle_level,
            "ch": ch,
            "reset": reset,
        }

        return self.configure(params, label="gate")

    def configure_burst(
        self,
        wave: str,
        freq: float,
        ampl_Vpp: float,
        phase_deg: float,
        cycle: int,
        offset: float = 0.0,
        delay: float = 0.0,
        source: str = "",
        polarity: bool | None = None,
        idle_level: str = "",
        ch: int = 1,
        reset: bool = False,
    ) -> bool:
        """Configure Gated Burst output.

        :param wave: wave form (function).
        :param freq: (Hz) frequnecy.
        :param ampl_Vpp: (V) amplitude in Vpp (peak-to-peak).
        :param phase_deg: (Degrees) starting phase.
        :param cycle: Number of cycles for each burst.
        :param offset: (V) offset.
        :param delay: (s) trigger delay.
        :param source: trigger source. empty to use default conf.
        :param polarity: trigger polarity. True for positive, None to use default conf.
        :param idle_level: idle_level. empty to use default conf.
        :param ch: channel.
        :param reset: If True, reset before configuration.

        """

        params = {
            "wave": wave,
            "freq": freq,
            "ampl": ampl_Vpp,
            "phase": phase_deg,
            "cycle": cycle,
            "offset": offset,
            "delay": delay,
            "source": source,
            "polarity": polarity,
            "idle_level": idle_level,
            "ch": ch,
            "reset": reset,
        }

        return self.configure(params, label="burst")
