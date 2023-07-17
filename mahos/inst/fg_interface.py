#!/usr/bin/env python3

"""
Typed Interface for Function Generator.

.. This file is a part of MAHOS project.

"""

import typing as T

from .interface import InstrumentInterface


class FGInterface(InstrumentInterface):
    """Interface for Function Genetator."""

    def get_opc(self, delay=None) -> bool:
        """Query OPC (operation complete) status."""

        return self.get("opc", delay)

    def get_bounds(self) -> dict:
        """Get bounds.

        Returns:
            ampl (low, high): (V) voltage amplitude bounds in Vpp (peak-to-peak).
            freq (low, high): (Hz) frequency bounds.

        """

        return self.get("bounds")

    def set_output(self, on: bool, ch: int = 1) -> bool:
        """Set output switch."""

        return self.set("output", {"on": on, "ch": ch})

    def configure_cw(
        self, wave: str, freq: float, ampl_vpp: float, ch: int = 1, reset: bool = True
    ) -> bool:
        """Configure Continuous Wave output.

        :param wave: wave form (function).
        :param freq: (Hz) frequnecy.
        :param ampl_vpp: (V) amplitude in Vpp (peak-to-peak).
        :param ch: channel.
        :param reset: If True, reset before configuration.

        """

        return self.configure(
            {"mode": "cw", "wave": wave, "freq": freq, "ampl": ampl_vpp, "ch": ch, "reset": reset}
        )

    def configure_gate(
        self,
        wave: str,
        freq: float,
        ampl_vpp: float,
        phase_deg: float,
        source: str = "",
        polarity: T.Optional[bool] = None,
        idle_level: str = "",
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure Gated Burst output.

        :param wave: wave form (function).
        :param freq: (Hz) frequnecy.
        :param ampl_vpp: (V) amplitude in Vpp (peak-to-peak).
        :param phase_deg: (Degrees) starting phase.
        :param source: trigger source. empty to use default conf.
        :param polarity: trigger and gate polarity. True for positive, None to use default conf.
        :param idle_level: idle_level. empty to use default conf.
        :param ch: channel.
        :param reset: If True, reset before configuration.

        """

        params = {
            "mode": "gate",
            "wave": wave,
            "freq": freq,
            "ampl": ampl_vpp,
            "phase": phase_deg,
            "source": source,
            "polarity": polarity,
            "idle_level": idle_level,
            "ch": ch,
            "reset": reset,
        }

        return self.configure(params)
