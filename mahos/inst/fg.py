#!/usr/bin/env python3

"""
Function Generator module.

.. This file is a part of MAHOS project.

"""

import typing as T

from .visa_instrument import VisaInstrument


class DG2000(VisaInstrument):
    """Rigol DG2000 series Function Generator.

    :param ext_ref_clock: use external reference clock source.
    :type ext_ref_clock: bool
    :param gate.source: trigger source for gated burst. one of TRIG_SOURCE.
    :type gate.source: str
    :param gate.slope: trigger slope polarity. True for positive.
    :type gate.slope: bool
    :param gate.polarity: gate polarity. True for positive.
    :type gate.polarity: bool
    :param gate.idle_level: idle level. one of IDLE_LEVEL.
    :type gate.idle_level: str

    """

    TRIG_SOURCE = ("INT", "EXT", "MAN")
    IDLE_LEVEL = ("FPT", "TOP", "CENTER", "BOTTOM")

    def __init__(self, name, conf, prefix=None):
        VisaInstrument.__init__(self, name, conf, prefix=prefix)
        self.update_amplitude_bounds()
        self.update_freq_bounds()

        self.ext_ref_clock = self.conf.get("ext_ref_clock", False)
        if self.ext_ref_clock:
            self.logger.debug("Using external clock reference")
        c = self.conf.get("gate", {})
        self.gate_conf = {
            "source": c.get("source", "EXT"),
            "slope": c.get("slope", True),
            "polarity": c.get("polarity", True),
            "idle_level": c.get("idle_level", "CENTER"),
        }
        self.logger.debug(f"gate configuration: {self.gate_conf}")

    def update_amplitude_bounds(self):
        mn = float(self.inst.query(":VOLT? MIN"))
        mx = float(self.inst.query(":VOLT? MAX"))
        self._ampl_bounds = (mn, mx)

    def update_freq_bounds(self):
        mn = float(self.inst.query(":FREQ? MIN"))
        mx = float(self.inst.query(":FREQ? MAX"))
        self._freq_bounds = (mn, mx)

    def get_bounds(self):
        return {"ampl": self._ampl_bounds, "freq": self._freq_bounds}

    def _check_channel(self, ch: int) -> bool:
        if ch not in (1, 2):
            self.logger.error(f"Invalid channel {ch}")
            return False
        else:
            return True

    def set_output(self, on: bool, ch: int = 1) -> bool:
        if not self._check_channel(ch):
            return False

        on_off = "ON" if on else "OFF"
        self.inst.write(f":OUTP{ch}:STAT {on_off}")
        self.logger.info(f"Output{ch} {on_off}")
        return True

    def set_output_impedance(self, imp_ohm: T.Union[int, str], ch: int = 1) -> bool:
        """Set output impedance.

        :param imp_ohm: Impedance in Ohm. May be str that's one of (INF|MIN|MAX).
        :param ch: Output channel (1 or 2).

        """

        if not self._check_channel(ch):
            return False
        if isinstance(imp_ohm, str):
            imp_ohm = imp_ohm.upper()[:3]
            if imp_ohm not in ("INF", "MIN", "MAX"):
                self.logger.error(f"Invalid impedance string {imp_ohm}")
                return False
        elif isinstance(imp_ohm, int):
            if not (1 <= imp_ohm <= 10e3):
                self.logger.error(f"Impedance {imp_ohm} is out of bounds: [1, 10k]")
                return False

        self.inst.write(f":OUTP{ch}:IMP {imp_ohm}")
        self.logger.info(f"Output{ch} impedance: {imp_ohm}")

        # TODO: update amplitude bounds?

        return True

    def _fmt_freq(self, freq: T.Union[str, float, int]) -> str:
        if isinstance(freq, str):
            return freq
        elif isinstance(freq, (float, int)):
            return f"{freq:.12E}"
        else:
            raise TypeError("Invalid frequency value type.")

    def _fmt_period(self, t: T.Union[str, float, int]) -> str:
        if isinstance(t, str):
            return t
        elif isinstance(t, (float, int)):
            return f"{t:.11E}"
        else:
            raise TypeError("Invalid frequency value type.")

    def set_burst(self, on: bool, ch: int = 1) -> bool:
        if not self._check_channel(ch):
            return False

        on_off = "ON" if on else "OFF"
        self.inst.write(f":SOUR{ch}:BURS:STAT {on_off}")
        return True

    def set_burst_mode(self, mode: str, ch: int = 1) -> bool:
        if mode.upper() not in ("INF", "INFINITY", "TRIG", "TRIGGERED", "GAT", "GATED"):
            return False

        self.inst.write(f":SOUR{ch}:BURS:MODE {mode}")
        return True

    def trigger(self) -> bool:
        self.inst.write("*TRG")
        return True

    def set_burst_trig_source(self, source: str, ch: int = 1) -> bool:
        source = source.upper()[:3]
        if source not in self.TRIG_SOURCE:
            self.logger.error(f"Invalid Burst Trigger Source: {source}")
            return False

        self.inst.write(f":SOUR{ch}:BURS:TRIG:SOUR {source}")
        return True

    def set_burst_trig_slope(self, positive: bool, ch: int = 1) -> bool:
        slope = "POS" if positive else "NEG"
        self.inst.write(f":SOUR{ch}:BURS:TRIG:SLOP {slope}")
        return True

    def set_burst_gate_polarity(self, positive: bool, ch: int = 1) -> bool:
        pol = "NORM" if positive else "INV"
        self.inst.write(f":SOUR{ch}:BURS:GATE:POL {pol}")
        return True

    def set_burst_idle_level(self, level: str, ch: int = 1) -> bool:
        level = level.upper()
        if level not in self.IDLE_LEVEL:
            self.logger.error(f"Invalid Burst Idle Level: {level}")
            return False

        self.inst.write(f":SOUR{ch}:BURS:IDLE {level}")
        return True

    def set_function(self, func: str, ch: int = 1) -> bool:
        func = func.upper()
        self.inst.write(f":SOUR{ch}:FUNC {func}")
        return True

    def set_square_duty(self, duty_percent: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}FUNC:SQU:DCYC {duty_percent:.8E}")
        return True

    def set_freq(self, freq: float, ch: int = 1) -> bool:
        self.inst.write(":SOUR{}:FREQ {}".format(ch, self._fmt_freq(freq)))
        return True

    def set_period(self, t: float, ch: int = 1) -> bool:
        self.inst.write(":SOUR{}:FUNC:PULS:PER {}".format(ch, self._fmt_period(t)))
        return True

    def set_phase(self, phase_deg: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:PHAS {phase_deg:.3f}")
        return True

    def set_amplitude(self, ampl_vpp: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT {ampl_vpp:.5f}")
        return True

    def set_reference_clock(self, external: bool) -> bool:
        source = "EXT" if external else "INT"
        self.inst.write(f":ROSC:SOUR {source}")
        return True

    def configure_CW(
        self, wave: str, freq: float, ampl_vpp: float, ch: int = 1, reset: bool = True
    ) -> bool:
        """Configure Continuous Wave output."""

        success = (
            (self.rst_cls() if reset else True)
            and self.set_reference_clock(self.ext_ref_clock)
            and self.set_function(wave, ch=ch)
            and self.set_freq(freq, ch=ch)
            and self.set_amplitude(ampl_vpp, ch=ch)
        )

        self.logger.info("Configured for CW.")
        return success

    def configure_gate(
        self,
        wave: str,
        freq: float,
        ampl_vpp: float,
        phase_deg: float,
        source: str = "",
        slope: T.Optional[bool] = None,
        polarity: T.Optional[bool] = None,
        idle_level: str = "",
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure Gated Burst output."""

        success = (
            (self.rst_cls() if reset else True)
            and self.set_reference_clock(self.ext_ref_clock)
            and self.set_function(wave, ch=ch)
            and self.set_freq(freq, ch=ch)
            and self.set_amplitude(ampl_vpp, ch=ch)
            and self.set_phase(phase_deg, ch=ch)
            and self.set_burst(True, ch=ch)
            and self.set_burst_mode("GATED", ch=ch)
            and self.set_burst_trig_source(source or self.gate_conf["source"], ch=ch)
            and self.set_burst_trig_slope(
                slope if slope is not None else self.gate_conf["slope"], ch=ch
            )
            and self.set_burst_gate_polarity(
                polarity if polarity is not None else self.gate_conf["polarity"], ch=ch
            )
            and self.set_burst_idle_level(idle_level or self.gate_conf["idle_level"], ch=ch)
        )

        self.logger.info(
            "Configured for Gated Burst."
            + f" wave: {wave} ampl: {ampl_vpp:.3f} Vpp phase: {phase_deg:.1f} deg."
        )
        return success

    # Standard API

    def get(self, key: str, args=None):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "bounds":
            return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def set(self, key: str, value=None) -> bool:
        if key == "output":
            if isinstance(value, dict):
                if "on" in value and "ch" in value:
                    return self.set_output(value["on"], value["ch"])
                else:
                    return self.fail_with("set('output', dict): needs keys on and ch")
            elif isinstance(value, bool):
                return self.set_output(value)
            else:
                return self.fail_with(f"set('output', value): Ill-formed value {value}")
        elif key == "trigger":
            return self.trigger()
        else:
            return self.fail_with("Unknown set() key.")

    def configure(self, params: dict) -> bool:
        if not self.check_required_params(params, ("mode",)):
            return False

        mode = params["mode"].lower()
        if mode == "gate":
            if not self.check_required_params(params, ("wave", "freq", "ampl", "phase")):
                return False
            return self.configure_gate(
                params["wave"],
                params["freq"],
                params["ampl"],
                params["phase"],
                source=params.get("source", ""),
                slope=params.get("polarity"),  # this is intentional. slope = polarity.
                polarity=params.get("polarity"),
                idle_level=params.get("idle_level", ""),
                ch=params.get("ch", 1),
                reset=params.get("reset", True),
            )
        elif mode == "cw":
            if not self.check_required_params(params, ("wave", "freq", "ampl")):
                return False
            return self.configure_CW(
                params["wave"],
                params["freq"],
                params["ampl"],
                ch=params.get("ch", 1),
                reset=params.get("reset", True),
            )
        else:
            return self.fail_with("Unknown conf['mode']")
