#!/usr/bin/env python3

"""
Function Generator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import re
import enum
from functools import wraps

from .visa_instrument import VisaInstrument
from ..msgs import param_msgs as P


def ch_setter(channels):
    def _check_ch(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            self = args[0]
            ch = kwargs.get("ch", 1)
            if ch not in channels:
                self.logger.error(f"Invalid ch {ch} (available: {channels})")
                return False
            return f(*args, **kwargs)

        return wrapper

    return _check_ch


def ch_getter(channels):
    def _check_ch(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            self = args[0]
            ch = kwargs.get("ch", 1)
            if ch not in channels:
                self.logger.error(f"Invalid ch {ch} (available: {channels})")
                return None
            return f(*args, **kwargs)

        return wrapper

    return _check_ch


class RIGOL_DG2000(VisaInstrument):
    """RIGOL DG2000 series Function Generator.

    :param ext_ref_clock: use external reference clock source.
    :type ext_ref_clock: bool
    :param gate.source: trigger source for gated burst. one of TRIG_SOURCE.
    :type gate.source: str
    :param gate.polarity: gate polarity. True for positive.
    :type gate.polarity: bool
    :param gate.idle_level: idle level. one of IDLE_LEVEL.
    :type gate.idle_level: str
    :param burst.source: trigger source for cycle burst. one of TRIG_SOURCE.
    :type burst.source: str
    :param burst.polarity: burst trigger polarity. True for positive.
    :type burst.polarity: bool
    :param burst.idle_level: idle level. one of IDLE_LEVEL.
    :type burst.idle_level: str

    """

    TRIG_SOURCE = ("INT", "EXT", "MAN")
    IDLE_LEVEL = ("FPT", "TOP", "CENTER", "BOTTOM")
    OUTPUT_HighZ = 11_000

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\n"
        if "read_termination" not in conf:
            conf["read_termination"] = "\n"
        # set default timeout to a bit large value because
        # some RIGOL models show a little lagged (2, 3 seconds) response to queries.
        if "timeout" not in conf:
            conf["timeout"] = 6000
        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self._ampl_bounds = {}
        self._offs_bounds = {}
        self._freq_bounds = {}
        self.update_all_bounds()

        self._align_phase = True

        self.ext_ref_clock = bool(self.conf.get("ext_ref_clock", False))
        self.set_reference_clock(self.ext_ref_clock)

        c = self.conf.get("gate", {})
        self.gate_conf = {
            "source": c.get("source", "EXT"),
            "polarity": c.get("polarity", True),
            "idle_level": c.get("idle_level", "CENTER"),
        }
        self.logger.debug(f"gate configuration: {self.gate_conf}")

        c = self.conf.get("burst", {})
        self.burst_conf = {
            "source": c.get("source", "EXT"),
            "polarity": c.get("polarity", True),
            "idle_level": c.get("idle_level", "CENTER"),
        }
        self.logger.debug(f"burst configuration: {self.burst_conf}")

    def update_all_bounds(self):
        for ch in (1, 2):
            self.update_amplitude_bounds(ch)
            self.update_offset_bounds(ch)
            self.update_freq_bounds(ch)

    def update_amplitude_bounds(self, ch: int):
        mn = float(self.inst.query(f":SOUR{ch}:VOLT? MIN"))
        mx = float(self.inst.query(f":SOUR{ch}:VOLT? MAX"))
        self._ampl_bounds[ch] = (mn, mx)

    def update_offset_bounds(self, ch: int):
        mn = float(self.inst.query(f":SOUR{ch}:VOLT:OFFS? MIN"))
        mx = float(self.inst.query(f":SOUR{ch}:VOLT:OFFS? MAX"))
        self._offs_bounds[ch] = (mn, mx)

    def update_freq_bounds(self, ch: int):
        mn = float(self.inst.query(f":SOUR{ch}:FREQ? MIN"))
        mx = float(self.inst.query(f":SOUR{ch}:FREQ? MAX"))
        self._freq_bounds[ch] = (mn, mx)

    @ch_getter((1, 2))
    def get_bounds(self, ch: int = 1):
        return {
            "ampl": self._ampl_bounds[ch],
            "offs": self._offs_bounds[ch],
            "freq": self._freq_bounds[ch],
        }

    @ch_setter((1, 2))
    def set_output(self, on: bool, ch: int = 1) -> bool:
        on_off = "ON" if on else "OFF"
        self.inst.write(f":OUTP{ch}:STAT {on_off}")
        self.logger.info(f"Output{ch} {on_off}")
        return True

    @ch_getter((1, 2))
    def get_output(self, ch: int = 1) -> bool:
        return self.inst.query(f":OUTP{ch}:STAT?").strip() in ("ON", "1")

    @ch_setter((1, 2))
    def set_output_impedance(self, imp_ohm: int | str, ch: int = 1) -> bool:
        """Set output impedance.

        :param imp_ohm: Impedance in Ohm. May be str that's one of (INF|MIN|MAX).
            Value greater than 10e3 is considered HighZ (INF).
        :param ch: Output channel (1 or 2).

        """

        if isinstance(imp_ohm, str):
            imp_ohm = imp_ohm.upper()[:3]
            if imp_ohm not in ("INF", "MIN", "MAX"):
                return self.fail_with(f"Invalid impedance string {imp_ohm}")
        elif isinstance(imp_ohm, (int, float)):
            imp_ohm = int(round(imp_ohm))
            if imp_ohm < 1:
                return self.fail_with(f"Impedance {imp_ohm} must be greaer than 1")
            if imp_ohm > 10e3:
                imp_ohm = "INF"
        else:
            return self.fail_with(f"Impedance {imp_ohm} has invalid type {type(imp_ohm)}")

        self.inst.write(f":OUTP{ch}:LOAD {imp_ohm}")
        self.logger.info(f"Output{ch} impedance: {imp_ohm}")

        self.update_amplitude_bounds(ch)

        return True

    @ch_getter((1, 2))
    def get_output_impedance(self, ch: int = 1) -> int:
        """Get output impedance. OUTPUT_HighZ means highest impedance (INF)."""

        res = float(self.inst.query(f":OUTP{ch}:LOAD?"))
        # result in float repr, but should be treated as int
        if res >= 9e37:
            return self.OUTPUT_HighZ
        else:
            return int(round(res))

    def _fmt_freq(self, freq: str | float | int) -> str:
        if isinstance(freq, str):
            return freq
        elif isinstance(freq, (float, int)):
            return f"{freq:.12E}"
        else:
            raise TypeError("Invalid frequency value type.")

    def _fmt_period(self, t: str | float | int) -> str:
        if isinstance(t, str):
            return t
        elif isinstance(t, (float, int)):
            return f"{t:.11E}"
        else:
            raise TypeError("Invalid period value type.")

    @ch_setter((1, 2))
    def set_burst(self, on: bool, ch: int = 1) -> bool:
        on_off = "ON" if on else "OFF"
        self.inst.write(f":SOUR{ch}:BURS:STAT {on_off}")
        return True

    @ch_setter((1, 2))
    def set_burst_mode(self, mode: str, ch: int = 1) -> bool:
        if mode.upper() not in ("INF", "INFINITY", "TRIG", "TRIGGERED", "GAT", "GATED"):
            return False

        self.inst.write(f":SOUR{ch}:BURS:MODE {mode}")
        return True

    def trigger(self) -> bool:
        self.inst.write("*TRG")
        return True

    @ch_setter((1, 2))
    def set_burst_cycle(self, cycle: int | str, ch: int = 1) -> bool:
        if isinstance(cycle, str):
            cycle = cycle.upper()
            if cycle not in ("MIN", "MAX"):
                return self.fail_with("Invalid cycle in str. Valid values are MIN or MAX.")

        self.inst.write(f":SOUR{ch}:BURS:NCYC {cycle}")
        return True

    @ch_setter((1, 2))
    def set_burst_delay(self, delay: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:BURS:TDEL {delay:.8E}")
        return True

    @ch_setter((1, 2))
    def set_burst_trig_source(self, source: str, ch: int = 1) -> bool:
        source = source.upper()[:3]
        if source not in self.TRIG_SOURCE:
            self.logger.error(f"Invalid Burst Trigger Source: {source}")
            return False

        self.inst.write(f":SOUR{ch}:BURS:TRIG:SOUR {source}")
        return True

    @ch_setter((1, 2))
    def set_burst_trig_slope(self, positive: bool, ch: int = 1) -> bool:
        slope = "POS" if positive else "NEG"
        self.inst.write(f":SOUR{ch}:BURS:TRIG:SLOP {slope}")
        return True

    @ch_setter((1, 2))
    def set_burst_gate_polarity(self, positive: bool, ch: int = 1) -> bool:
        pol = "NORM" if positive else "INV"
        self.inst.write(f":SOUR{ch}:BURS:GATE:POL {pol}")
        return True

    @ch_setter((1, 2))
    def set_burst_idle_level(self, level: str, ch: int = 1) -> bool:
        level = level.upper()
        if level not in self.IDLE_LEVEL:
            self.logger.error(f"Invalid Burst Idle Level: {level}")
            return False

        self.inst.write(f":SOUR{ch}:BURS:IDLE {level}")
        return True

    @ch_setter((1, 2))
    def set_function(self, func: str, ch: int = 1) -> bool:
        func = func.upper()
        self.inst.write(f":SOUR{ch}:FUNC {func}")
        return True

    @ch_getter((1, 2))
    def get_function(self, ch: int = 1) -> str:
        return self.inst.query(f":SOUR{ch}:FUNC?")

    @ch_setter((1, 2))
    def set_square_duty(self, duty_percent: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}FUNC:SQU:DCYC {duty_percent:.8E}")
        return True

    @ch_setter((1, 2))
    def set_freq(self, freq: float, ch: int = 1) -> bool:
        self.inst.write(":SOUR{}:FREQ {}".format(ch, self._fmt_freq(freq)))
        return True

    @ch_getter((1, 2))
    def get_freq(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:FREQ?"))

    @ch_setter((1, 2))
    def set_period(self, t: float, ch: int = 1) -> bool:
        self.inst.write(":SOUR{}:FUNC:PULS:PER {}".format(ch, self._fmt_period(t)))
        return True

    @ch_setter((1, 2))
    def set_phase(self, phase_deg: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:PHAS {phase_deg:.3f}")
        return True

    @ch_getter((1, 2))
    def get_phase(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:PHAS?"))

    @ch_setter((1, 2))
    def set_amplitude(self, ampl_Vpp: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT {ampl_Vpp:.8E}")
        return True

    @ch_getter((1, 2))
    def get_amplitude(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:VOLT?"))

    @ch_setter((1, 2))
    def set_offset(self, offset: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT:OFFS {offset:.8E}")
        return True

    @ch_getter((1, 2))
    def get_offset(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:VOLT:OFFS?"))

    @ch_setter((1, 2))
    def set_high(self, volt: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT:HIGH {volt:.8E}")
        return True

    @ch_getter((1, 2))
    def get_high(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:VOLT:HIGH?"))

    @ch_setter((1, 2))
    def set_low(self, volt: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT:LOW {volt:.8E}")
        return True

    @ch_getter((1, 2))
    def get_low(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:VOLT:LOW?"))

    def set_reference_clock(self, external: bool) -> bool:
        source = "EXT" if external else "INT"
        self.inst.write(f":ROSC:SOUR {source}")
        self.logger.info(f"Reference clock: {source}")
        return True

    def align_phase(self, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:PHAS:SYNC")
        self.logger.info(f"Executed phase alignment at ch {ch}")
        return True

    def configure_cw(
        self,
        wave: str | None,
        freq: float | None,
        ampl_Vpp: float | None,
        offset: float = 0.0,
        phase_deg: float = 0.0,
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure Continuous Wave output."""

        success = True
        if reset:
            success &= self.reset()

        if wave is not None:
            success &= self.set_function(wave, ch=ch)
        if freq is not None:
            success &= self.set_freq(freq, ch=ch)
        if ampl_Vpp is not None:
            success &= self.set_amplitude(ampl_Vpp, ch=ch)
        success &= self.set_offset(offset, ch=ch)
        success &= self.set_phase(phase_deg, ch=ch)
        success &= self.check_error()

        if success:
            self.logger.info(f"Configured ch{ch} for CW.")
        else:
            self.logger.error(f"Failed to configure ch{ch} for CW.")
        return success

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
        reset: bool = True,
    ) -> bool:
        """Configure Gated Burst output."""

        success = (
            (self.reset() if reset else True)
            and self.set_function(wave, ch=ch)
            and self.set_freq(freq, ch=ch)
            and self.set_amplitude(ampl_Vpp, ch=ch)
            and self.set_offset(offset, ch=ch)
            and self.set_phase(phase_deg, ch=ch)
            and self.set_burst(True, ch=ch)
            and self.set_burst_mode("GATED", ch=ch)
            and self.set_burst_trig_source(source or self.gate_conf["source"], ch=ch)
            and self.set_burst_trig_slope(
                polarity if polarity is not None else self.gate_conf["polarity"], ch=ch
            )
            and self.set_burst_gate_polarity(
                polarity if polarity is not None else self.gate_conf["polarity"], ch=ch
            )
            and self.set_burst_idle_level(idle_level or self.gate_conf["idle_level"], ch=ch)
            and self.check_error()
        )

        if success:
            self.logger.info(
                f"Configured ch{ch} for Gated Burst."
                + f" wave: {wave} ampl: {ampl_Vpp:.3f} Vpp phase: {phase_deg:.1f} deg."
            )
        else:
            self.logger.error(f"Failed to configure ch{ch} for Gated Burst.")
        return success

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
        reset: bool = True,
    ) -> bool:
        """Configure Cycle Burst output."""

        success = (
            (self.reset() if reset else True)
            and self.set_function(wave, ch=ch)
            and self.set_freq(freq, ch=ch)
            and self.set_amplitude(ampl_Vpp, ch=ch)
            and self.set_offset(offset, ch=ch)
            and self.set_phase(phase_deg, ch=ch)
            and self.set_burst(True, ch=ch)
            and self.set_burst_mode("TRIG", ch=ch)
            and self.set_burst_trig_source(source or self.burst_conf["source"], ch=ch)
            and self.set_burst_trig_slope(
                polarity if polarity is not None else self.burst_conf["polarity"], ch=ch
            )
            and self.set_burst_idle_level(idle_level or self.burst_conf["idle_level"], ch=ch)
            and self.set_burst_cycle(cycle, ch=ch)
            and self.set_burst_delay(delay, ch=ch)
            and self.check_error()
        )

        if success:
            self.logger.info(
                f"Configured ch{ch} for Cycle Burst."
                + f" wave: {wave} ampl: {ampl_Vpp:.3f} Vpp phase: {phase_deg:.1f} deg."
            )
        else:
            self.logger.error(f"Failed to configure ch{ch} for Gated Burst.")
        return success

    def configure_output(self, params: dict):
        success = True
        if "ch1_imp" in params:
            success &= self.set_output_impedance(params["ch1_imp"], 1)
        if "ch2_imp" in params:
            success &= self.set_output_impedance(params["ch2_imp"], 2)
        if "align_phase" in params:
            self._align_phase = params["align_phase"]
            self.logger.info(f"Phase alignment: {self._align_phase}")
        success &= self.check_error()
        if success:
            self.logger.info("Configured output.")
        else:
            self.logger.error("Error configuring output.")
        return success

    # Standard API

    def reset(self, label: str = "") -> bool:
        success = self.rst_cls() and self.set_reference_clock(self.ext_ref_clock)
        self.update_all_bounds()
        return success

    def start(self, label: str = "") -> bool:
        if label.startswith("ch1"):
            ch = 1
            success = self.set_output(True, ch)
            both_on = self.get_output(2)
        elif label.startswith("ch2"):
            ch = 2
            success = self.set_output(True, ch)
            both_on = self.get_output(1)
        else:
            return self.fail_with(f"Unknown label {label} to start")

        if not success:
            return False

        # execute phase alignment if both channel has been turned on.
        if both_on and self._align_phase:
            return self.align_phase(ch=ch)
        else:
            return True

    def stop(self, label: str = "") -> bool:
        if label.startswith("ch1"):
            return self.set_output(False, 1)
        elif label.startswith("ch2"):
            return self.set_output(False, 2)
        else:
            return self.fail_with(f"Unknown label {label} to stop")

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "bounds":
            return self.get_bounds(args if args is not None else 1)
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def set(self, key: str, value=None, label: str = "") -> bool:
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

    def _cw_param_dict(self, ch: int):
        func = self.get_function(ch)
        functions = ("SIN", "SQU")
        if func not in functions:
            func = "SIN"

        return P.ParamDict(
            wave=P.StrChoiceParam(
                func,
                functions,
                doc="wave form",
            ),
            freq=P.FloatParam(
                self.get_freq(ch),
                self._freq_bounds[ch][0],
                self._freq_bounds[ch][1],
                unit="Hz",
                SI_prefix=True,
            ),
            ampl=P.FloatParam(
                self.get_amplitude(ch),
                self._ampl_bounds[ch][0],
                self._ampl_bounds[ch][1],
                unit="Vpp",
            ),
            offset=P.FloatParam(
                self.get_offset(ch),
                self._offs_bounds[ch][0],
                self._offs_bounds[ch][1],
                unit="V",
            ),
            phase=P.FloatParam(
                self.get_phase(ch),
                0.0,
                360.0,
                unit="deg",
            ),
        )

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        if label == "ch1_cw":
            return self._cw_param_dict(1)
        elif label == "ch2_cw":
            return self._cw_param_dict(2)
        elif label == "output":
            return P.ParamDict(
                ch1_imp=P.IntParam(
                    self.get_output_impedance(1),
                    1,
                    self.OUTPUT_HighZ,
                    unit="Ω",
                    doc=f"Output impedance. Set maximum ({self.OUTPUT_HighZ}) for HighZ",
                ),
                ch2_imp=P.IntParam(
                    self.get_output_impedance(2),
                    1,
                    self.OUTPUT_HighZ,
                    unit="Ω",
                    doc=f"Output impedance. Set maximum ({self.OUTPUT_HighZ}) for HighZ",
                ),
                align_phase=P.BoolParam(True),
            )
        else:
            self.logger.error(f"Invalid label {label}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        # "gate" and "cw" doesn't provide ParamDict
        return ["ch1_cw", "ch2_cw", "output"]

    def configure(self, params: dict, label: str = "") -> bool:
        params = P.unwrap(params)

        if label == "gate":
            if not self.check_required_params(params, ("wave", "freq", "ampl", "phase")):
                return False
            return self.configure_gate(
                params["wave"],
                params["freq"],
                params["ampl"],
                params["phase"],
                offset=params.get("offset", 0.0),
                source=params.get("source", ""),
                polarity=params.get("polarity"),
                idle_level=params.get("idle_level", ""),
                ch=params.get("ch", 1),
                reset=params.get("reset", False),
            )
        elif label == "burst":
            if not self.check_required_params(params, ("wave", "freq", "ampl", "phase", "cycle")):
                return False
            return self.configure_burst(
                params["wave"],
                params["freq"],
                params["ampl"],
                params["phase"],
                params["cycle"],
                offset=params.get("offset", 0.0),
                delay=params.get("delay", 0.0),
                source=params.get("source", ""),
                polarity=params.get("polarity"),
                idle_level=params.get("idle_level", ""),
                ch=params.get("ch", 1),
                reset=params.get("reset", False),
            )
        elif label == "cw":
            return self.configure_cw(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset=params.get("offset", 0.0),
                phase_deg=params.get("phase", 0.0),
                ch=params.get("ch", 1),
                reset=params.get("reset", False),
            )
        elif label == "ch1_cw":
            return self.configure_cw(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset=params.get("offset", 0.0),
                phase_deg=params.get("phase", 0.0),
                ch=1,
                reset=False,
            )
        elif label == "ch2_cw":
            return self.configure_cw(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset=params.get("offset", 0.0),
                phase_deg=params.get("phase", 0.0),
                ch=2,
                reset=False,
            )
        elif label == "output":
            return self.configure_output(params)
        else:
            return self.fail_with(f"Unknown label: {label}")


class RIGOL_DG800Pro(RIGOL_DG2000):
    """RIGOL DG800Pro series Function Generator.

    [WARNING/TODO] The gate and burst modes won't work properly.
    (Command sent, no error reported, but output is not as expected
    until we touch "local" key on instrument screen. I don't know how to fix this.)
    Avoid using this instrument if you need these modes.

    :param ext_ref_clock: use external reference clock source.
    :type ext_ref_clock: bool
    :param gate.source: trigger source for gated burst. one of TRIG_SOURCE.
    :type gate.source: str
    :param gate.polarity: gate polarity. True for positive.
    :type gate.polarity: bool
    :param gate.idle_level: idle level. one of IDLE_LEVEL.
    :type gate.idle_level: str
    :param burst.source: trigger source for cycle burst. one of TRIG_SOURCE.
    :type burst.source: str
    :param burst.polarity: burst trigger polarity. True for positive.
    :type burst.polarity: bool
    :param burst.idle_level: idle level. one of IDLE_LEVEL.
    :type burst.idle_level: str

    """

    TRIG_SOURCE = ("IMM", "EXT", "BUS", "TIM")

    @ch_setter((1, 2))
    def set_burst_mode(self, mode: str, ch: int = 1) -> bool:
        # Available modes are different from DG2000's.
        if mode.upper() not in ("TRIG", "TRIGGERED", "GAT", "GATED"):
            return False

        self.inst.write(f":SOUR{ch}:BURS:MODE {mode}")
        return True

    @ch_setter((1, 2))
    def set_trig_source(self, source: str, ch: int = 1) -> bool:
        source = source.upper()[:3]
        if source not in self.TRIG_SOURCE:
            self.logger.error(f"Invalid Trigger Source: {source}")
            return False

        self.inst.write(f":TRIG{ch}:SOUR {source}")
        return True

    @ch_setter((1, 2))
    def set_burst_trig_source(self, source: str, ch: int = 1) -> bool:
        # Command is unified to TRIG layer.
        return self.set_trig_source(source, ch=ch)

    @ch_setter((1, 2))
    def set_trig_slope(self, positive: bool, ch: int = 1) -> bool:
        slope = "POS" if positive else "NEG"
        self.inst.write(f":TRIG{ch}:SLOP {slope}")
        return True

    @ch_setter((1, 2))
    def set_burst_trig_slope(self, positive: bool, ch: int = 1) -> bool:
        # Command is unified to TRIG layer.
        return self.set_trig_slope(positive, ch=ch)

    @ch_setter((1, 2))
    def set_burst_idle_level(self, level: str, ch: int = 1) -> bool:
        level = level.upper()
        if level not in self.IDLE_LEVEL:
            self.logger.error(f"Invalid Burst Idle Level: {level}")
            return False

        # Command is different from DG2000's.
        self.inst.write(f":OUTP{ch}:IDLE {level}")
        return True

    @ch_setter((1, 2))
    def set_trig_delay(self, delay: float, ch: int = 1) -> bool:
        self.inst.write(f":TRIG{ch}:DEL {delay:.8E}")
        return True

    @ch_setter((1, 2))
    def set_burst_delay(self, delay: float, ch: int = 1) -> bool:
        # Command is unified to TRIG layer.
        return self.set_trig_delay(delay, ch=ch)

    def set_reference_clock(self, external: bool) -> bool:
        source = "EXT" if external else "INT"
        # Command is different from DG2000's.
        self.inst.write(f":SYST:ROSC:SOUR {source}")
        self.logger.info(f"Reference clock: {source}")
        return True

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
        reset: bool = True,
    ) -> bool:
        """Configure Gated Burst output."""

        msg = "This function won't work properly! Check if expected output is generated!"
        self.logger.warn(msg)

        return RIGOL_DG2000.configure_gate(
            self,
            wave,
            freq,
            ampl_Vpp,
            phase_deg,
            offset=offset,
            source=source,
            polarity=polarity,
            idle_level=idle_level,
            ch=ch,
            reset=reset,
        )

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
        reset: bool = True,
    ) -> bool:
        """Configure Cycle Burst output."""

        msg = "This function won't work properly! Check if expected output is generated!"
        self.logger.warn(msg)

        return RIGOL_DG2000.configure_burst(
            self,
            wave,
            freq,
            ampl_Vpp,
            phase_deg,
            cycle,
            offset=offset,
            delay=delay,
            source=source,
            polarity=polarity,
            idle_level=idle_level,
            ch=ch,
            reset=reset,
        )


class SIGLENT_SDG2000X(VisaInstrument):
    """SIGLENT SDG2000X series Function Generator.

    :param ampl_bounds: Amplitude bounds in Vpp (min, max).
    :type ampl_bounds: tuple[float, float]
    :param offs_bounds: Offset bounds in V (min, max).
    :type offs_bounds: tuple[float, float]
    :param freq_bounds: Frequency bounds in Hz (min, max).
    :type freq_bounds: tuple[float, float]
    :param ext_ref_clock: use external reference clock source.
    :type ext_ref_clock: bool
    :param gate.source: trigger source for gated burst. one of TRIG_SOURCE.
    :type gate.source: str
    :param gate.polarity: gate polarity. True for positive.
    :type gate.polarity: bool
    :param burst.source: trigger source for cycle burst. one of TRIG_SOURCE.
    :type burst.source: str
    :param burst.polarity: burst trigger polarity. True for positive.
    :type burst.polarity: bool

    """

    TRIG_SOURCE = ("INT", "EXT", "MAN")
    Load_HighZ = 110_000

    class Mode(enum.Enum):
        UNCONFIGURED = 0
        CW = 1
        BURST_GATE = 2
        BURST_CYCLE = 3

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\n"
        if "read_termination" not in conf:
            conf["read_termination"] = "\n"
        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self._ampl_bounds = self.conf.get("ampl_bounds", (0.0, 10.0))
        self._offs_bounds = self.conf.get("offs_bounds", (-10.0, 10.0))
        self._freq_bounds = self.conf.get("freq_bounds", (1e-6, 80e6))

        self.ext_ref_clock = bool(self.conf.get("ext_ref_clock", False))
        self.set_reference_clock(self.ext_ref_clock)

        c = self.conf.get("gate", {})
        self.gate_conf = {
            "source": c.get("source", "EXT"),
            "polarity": c.get("polarity", True),
        }
        self.logger.debug(f"gate configuration: {self.gate_conf}")

        c = self.conf.get("burst", {})
        self.burst_conf = {
            "source": c.get("source", "EXT"),
            "polarity": c.get("polarity", True),
        }
        self.logger.debug(f"burst configuration: {self.burst_conf}")

        self.reset_modes()

    def reset_modes(self):
        self._modes = {1: self.Mode.UNCONFIGURED, 2: self.Mode.UNCONFIGURED}

    def get_bounds(self):
        return {
            "ampl": self._ampl_bounds,
            "offs": self._offs_bounds,
            "freq": self._freq_bounds,
        }

    @ch_setter((1, 2))
    def set_output(self, on: bool, ch: int = 1) -> bool:
        on_off = "ON" if on else "OFF"
        self.inst.write(f"C{ch}:OUTP {on_off}")
        self.logger.info(f"Output{ch} {on_off}")
        return True

    @ch_getter((1, 2))
    def get_output(self, ch: int = 1) -> bool | None:
        res = self.inst.query(f"C{ch}:OUTP?")
        mo = re.search(r"OUTP (ON|OFF)\,", res)
        if mo is None:
            self.logger.error(f"Cannot parse C{ch}:OUTP? -> " + res)
            return None

        return mo.groups()[0] == "ON"

    @ch_setter((1, 2))
    def set_load_impedance(self, imp_ohm: int | str, ch: int = 1) -> bool:
        """Set load impedance.

        :param imp_ohm: Impedance in Ohm. May be str "INF".
            Value greater than 1e5 is considered HighZ (INF).
        :param ch: Output channel (1 or 2).

        """

        if isinstance(imp_ohm, str):
            imp_ohm = imp_ohm.upper()[:3]
            if imp_ohm != "INF":
                return self.fail_with(f"Invalid impedance string {imp_ohm}")
            imp_ohm = "HZ"
        elif isinstance(imp_ohm, (int, float)):
            imp_ohm = int(round(imp_ohm))
            if imp_ohm < 1:
                return self.fail_with(f"Impedance {imp_ohm} must be greaer than 1")
            if imp_ohm > 1e5:
                imp_ohm = "HZ"
        else:
            return self.fail_with(f"Impedance {imp_ohm} has invalid type {type(imp_ohm)}")

        self.inst.write(f"C{ch}:OUTP LOAD,{imp_ohm}")
        self.logger.info(f"Output{ch} impedance: {imp_ohm}")

        return True

    @ch_getter((1, 2))
    def get_output_impedance(self, ch: int = 1) -> int:
        """Get output impedance. OUTPUT_HighZ means highest impedance (INF)."""

        res = self.inst.query(f"C{ch}:OUTP?")
        mo = re.search(r"LOAD\,([0-9]*|HZ)\,", res)
        if mo is None:
            self.logger.error(f"Cannot parse C{ch}:OUTP? -> " + res)
            return 0

        value = mo.groups()[0]
        if value == "HZ":
            return self.Load_HighZ
        else:
            return int(value)

    def _fmt_freq(self, freq: str | float | int) -> str:
        if isinstance(freq, str):
            return freq
        elif isinstance(freq, (float, int)):
            return f"{freq:.12E}"
        else:
            raise TypeError("Invalid frequency value type.")

    # CW (BSWV: Basic Wave) settings

    @ch_setter((1, 2))
    def set_function(self, func: str, ch: int = 1) -> bool:
        func = func.upper()
        if func in ("SIN", "SINUSOID"):
            func = "SINE"
        if func not in ("SINE", "SQUARE", "RAMP", "PULSE", "NOISE", "ARB", "DC", "PRBS", "IQ"):
            return self.fail_with(f"Invalid function: {func}")
        self.inst.write(f"C{ch}:BSWV WVTP,{func}")
        return True

    @ch_getter((1, 2))
    def get_function(self, ch: int = 1) -> str:
        res = self.inst.query(f"C{ch}:BSWV?")
        mo = re.search(r"WVTP\,([A-Z]+)\,", res)
        if mo is None:
            self.logger.error(f"Cannot parse C{ch}:BSWV? -> " + res)
            return ""
        return mo.groups()[0]

    @ch_setter((1, 2))
    def set_freq(self, freq: float, ch: int = 1) -> bool:
        self.inst.write("C{}:BSWV FRQ,{}".format(ch, self._fmt_freq(freq)))
        return True

    @ch_getter((1, 2))
    def get_freq(self, ch: int = 1) -> float:
        res = self.inst.query(f"C{ch}:BSWV?")
        mo = re.search(r"FRQ\,([0-9.]+)HZ\,", res)
        if mo is None:
            self.logger.error(f"Cannot parse C{ch}:BSWV? -> " + res)
            return 0.0
        return float(mo.groups()[0])

    @ch_setter((1, 2))
    def set_phase(self, phase_deg: float, ch: int = 1) -> bool:
        self.inst.write(f"C{ch}:BSWV PHSE,{phase_deg:.3f}")
        return True

    @ch_getter((1, 2))
    def get_phase(self, ch: int = 1) -> float:
        res = self.inst.query(f"C{ch}:BSWV?")
        mo = re.search(r"PHSE\,([0-9.]+)", res)
        if mo is None:
            self.logger.error(f"Cannot parse C{ch}:BSWV? -> " + res)
            return 0.0
        return float(mo.groups()[0])

    @ch_setter((1, 2))
    def set_amplitude(self, ampl_Vpp: float, ch: int = 1) -> bool:
        self.inst.write(f"C{ch}:BSWV AMP,{ampl_Vpp:.8E}")
        return True

    @ch_getter((1, 2))
    def get_amplitude(self, ch: int = 1) -> float:
        res = self.inst.query(f"C{ch}:BSWV?")
        mo = re.search(r"AMP\,([0-9.]+)V\,", res)
        if mo is None:
            self.logger.error(f"Cannot parse C{ch}:BSWV? -> " + res)
            return 0.0
        return float(mo.groups()[0])

    @ch_setter((1, 2))
    def set_offset(self, offset: float, ch: int = 1) -> bool:
        self.inst.write(f"C{ch}:BSWV OFST,{offset:.8E}")
        return True

    @ch_getter((1, 2))
    def get_offset(self, ch: int = 1) -> float:
        res = self.inst.query(f"C{ch}:BSWV?")
        mo = re.search(r"OFST\,([0-9.]+)V\,", res)
        if mo is None:
            self.logger.error(f"Cannot parse C{ch}:BSWV? -> " + res)
            return 0.0
        return float(mo.groups()[0])

    # Burst (BTWV: Burst Wave) settings

    @ch_setter((1, 2))
    def set_burst(self, on: bool, ch: int = 1) -> bool:
        on_off = "ON" if on else "OFF"
        self.inst.write(f"C{ch}:BTWV STATE,{on_off}")
        return True

    @ch_setter((1, 2))
    def set_burst_mode(self, mode: str, ch: int = 1) -> bool:
        if mode.upper() not in ("GATE", "NCYC"):
            return False

        self.inst.write(f"C{ch}:BTWV GATE_NCYC,{mode}")
        return True

    def trigger(self, ch: int = 1) -> bool:
        """Execute manual (software) trigger.

        Only valid when trigger souce is MAN and configured as burst cycle mode.
        """

        if self._modes[ch] == self.Mode.BURST_CYCLE:
            self.inst.write(f"C{ch}:BTWV MTRIG")
            return True
        return self.fail_with("Improper mode to execute manual trigger.")

    @ch_setter((1, 2))
    def set_burst_trig_source(self, source: str, ch: int = 1) -> bool:
        source = source.upper()[:3]
        if source not in self.TRIG_SOURCE:
            self.logger.error(f"Invalid Burst Trigger Source: {source}")
            return False

        self.inst.write(f"C{ch}:BTWV TRSR,{source}")
        return True

    @ch_setter((1, 2))
    def set_burst_trig_slope(self, positive: bool, ch: int = 1) -> bool:
        slope = "RISE" if positive else "FALL"
        self.inst.write(f"C{ch}:BTWV EDGE,{slope}")
        return True

    @ch_setter((1, 2))
    def set_burst_gate_polarity(self, positive: bool, ch: int = 1) -> bool:
        pol = "POS" if positive else "NEG"
        self.inst.write(f"C{ch}:BTWV PLRT,{pol}")
        return True

    @ch_setter((1, 2))
    def set_burst_cycle(self, cycle: int | str, ch: int = 1) -> bool:
        if isinstance(cycle, str):
            cycle = cycle.upper()
            if cycle not in ("INF", "M"):
                return self.fail_with("Invalid cycle in str. Valid values are INF or M.")
        self.inst.write(f"C{ch}:BTWV TIME,{cycle}")
        return True

    @ch_setter((1, 2))
    def set_burst_function(self, func: str, ch: int = 1) -> bool:
        func = func.upper()
        if func in ("SIN", "SINUSOID"):
            func = "SINE"
        if func not in ("SINE", "SQUARE", "RAMP", "PULSE", "NOISE", "ARB"):
            return self.fail_with(f"Invalid function: {func}")
        self.inst.write(f"C{ch}:BTWV CARR,WVTP,{func}")
        return True

    @ch_setter((1, 2))
    def set_burst_freq(self, freq: float, ch: int = 1) -> bool:
        self.inst.write("C{}:BTWV CARR,FRQ,{}".format(ch, self._fmt_freq(freq)))
        return True

    @ch_setter((1, 2))
    def set_burst_phase(self, phase_deg: float, ch: int = 1) -> bool:
        self.inst.write(f"C{ch}:BTWV STPS,{phase_deg:.3f}")
        return True

    @ch_setter((1, 2))
    def set_burst_delay(self, delay_sec: float, ch: int = 1) -> bool:
        self.inst.write(f"C{ch}:BTWV DLAY,{delay_sec:.8E}")
        return True

    @ch_setter((1, 2))
    def set_burst_amplitude(self, ampl_Vpp: float, ch: int = 1) -> bool:
        self.inst.write(f"C{ch}:BTWV CARR,AMP,{ampl_Vpp:.8E}")
        return True

    @ch_setter((1, 2))
    def set_burst_offset(self, offset: float, ch: int = 1) -> bool:
        self.inst.write(f"C{ch}:BTWV CARR,OFST,{offset:.8E}")
        return True

    def set_reference_clock(self, external: bool) -> bool:
        source = "EXT" if external else "INT"
        self.inst.write(f"ROSC {source}")
        self.logger.info(f"Reference clock: {source}")
        return True

    def configure_cw(
        self,
        wave: str | None,
        freq: float | None,
        ampl_Vpp: float | None,
        offset: float = 0.0,
        phase_deg: float = 0.0,
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure Continuous Wave output."""

        success = True
        if reset:
            success &= self.reset()
            self.reset_modes()

        if wave is not None:
            success &= self.set_function(wave, ch=ch)
        if freq is not None:
            success &= self.set_freq(freq, ch=ch)
        if ampl_Vpp is not None:
            success &= self.set_amplitude(ampl_Vpp, ch=ch)
        success &= self.set_offset(offset, ch=ch)
        success &= self.set_phase(phase_deg, ch=ch)

        if success:
            self.logger.info(f"Configured ch{ch} for CW.")
            self._modes[ch] = self.Mode.CW
        else:
            self.logger.error(f"Failed to configure ch{ch} for CW.")

        return success

    def configure_gate(
        self,
        wave: str,
        freq: float,
        ampl_Vpp: float,
        phase_deg: float,
        offset: float = 0.0,
        source: str = "",
        polarity: bool | None = None,
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure Gated Burst output."""

        success = True
        if reset:
            success &= self.reset()
            self.reset_modes()

        success &= (
            self.set_burst(True, ch=ch)
            # Setting here mode=NCYC and delay=0.0 to fix questionable behaviour:
            # - Cannot set delay (via command) after entering GATE mode, as in the manual.
            # - But, if a delay is set before this, it is actually inserted in the signal.
            # NOTE that setting delay=0.0 doesn't mean exactly 0 delay,
            # but minimum possible delay (~ 600 ns) for the instrument.
            and self.set_burst_mode("NCYC", ch=ch)
            and self.set_burst_delay(0.0, ch=ch)
            # turn to GATE mode which we actually want to configure.
            and self.set_burst_mode("GATE", ch=ch)
            and self.set_burst_trig_source(source or self.gate_conf["source"], ch=ch)
            and self.set_burst_trig_slope(
                polarity if polarity is not None else self.gate_conf["polarity"], ch=ch
            )
            and self.set_burst_gate_polarity(
                polarity if polarity is not None else self.gate_conf["polarity"], ch=ch
            )
            and self.set_burst_function(wave, ch=ch)
            and self.set_burst_freq(freq, ch=ch)
            and self.set_burst_amplitude(ampl_Vpp, ch=ch)
            and self.set_burst_offset(offset, ch=ch)
            and self.set_burst_phase(phase_deg, ch=ch)
        )

        if success:
            self.logger.info(
                f"Configured ch{ch} for Gated Burst."
                + f" wave: {wave} ampl: {ampl_Vpp:.3f} Vpp phase: {phase_deg:.1f} deg."
            )
            self._modes[ch] = self.Mode.BURST_GATE
        else:
            self.logger.error(f"Failed to configure ch{ch} for Gated Burst.")

        return success

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
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure Cycle Burst output."""

        success = True
        if reset:
            success &= self.reset()
            self.reset_modes()

        success &= (
            self.set_burst(True, ch=ch)
            and self.set_burst_mode("NCYC", ch=ch)
            and self.set_burst_trig_source(source or self.burst_conf["source"], ch=ch)
            and self.set_burst_trig_slope(
                polarity if polarity is not None else self.burst_conf["polarity"], ch=ch
            )
            and self.set_burst_function(wave, ch=ch)
            and self.set_burst_freq(freq, ch=ch)
            and self.set_burst_amplitude(ampl_Vpp, ch=ch)
            and self.set_burst_offset(offset, ch=ch)
            and self.set_burst_phase(phase_deg, ch=ch)
            and self.set_burst_cycle(cycle, ch=ch)
            and self.set_burst_delay(delay, ch=ch)
        )

        if success:
            self.logger.info(
                f"Configured ch{ch} for Cycle Burst. cycle: {cycle}"
                + f" wave: {wave} ampl: {ampl_Vpp:.3f} Vpp phase: {phase_deg:.1f} deg."
            )
            self._modes[ch] = self.Mode.BURST_CYCLE
        else:
            self.logger.error(f"Failed to configure ch{ch} for Gated Burst.")

        return success

    def configure_output(self, params: dict):
        success = True
        if "ch1_imp" in params:
            success &= self.set_load_impedance(params["ch1_imp"], 1)
        if "ch2_imp" in params:
            success &= self.set_load_impedance(params["ch2_imp"], 2)
        return success

    # Standard API

    def start(self, label: str = "") -> bool:
        if label.startswith("ch1"):
            return self.set_output(True, 1)
        elif label.startswith("ch2"):
            return self.set_output(True, 2)
        else:
            return self.fail_with(f"Unknown label {label} to start")

    def stop(self, label: str = "") -> bool:
        if label.startswith("ch1"):
            return self.set_output(False, 1)
        elif label.startswith("ch2"):
            return self.set_output(False, 2)
        else:
            return self.fail_with(f"Unknown label {label} to stop")

    def reset(self, label: str = "") -> bool:
        return self.rst() and self.set_reference_clock(self.ext_ref_clock)

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "bounds":
            return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def set(self, key: str, value=None, label: str = "") -> bool:
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

    def _cw_param_dict(self, ch: int):
        func = self.get_function(ch)
        functions = ("SINE", "SQUARE")
        if func not in functions:
            func = "SINE"

        return P.ParamDict(
            wave=P.StrChoiceParam(
                func,
                functions,
                doc="wave form",
            ),
            freq=P.FloatParam(
                self.get_freq(ch),
                self._freq_bounds[0],
                self._freq_bounds[1],
                unit="Hz",
                SI_prefix=True,
            ),
            ampl=P.FloatParam(
                self.get_amplitude(ch),
                self._ampl_bounds[0],
                self._ampl_bounds[1],
                unit="Vpp",
            ),
            offset=P.FloatParam(
                self.get_offset(ch),
                self._offs_bounds[0],
                self._offs_bounds[1],
                unit="V",
            ),
            phase=P.FloatParam(
                self.get_phase(ch),
                0.0,
                360.0,
                unit="deg",
            ),
        )

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        if label == "ch1_cw":
            return self._cw_param_dict(1)
        elif label == "ch2_cw":
            return self._cw_param_dict(2)
        elif label == "output":
            return P.ParamDict(
                ch1_imp=P.IntParam(
                    self.get_output_impedance(1),
                    1,
                    self.Load_HighZ,
                    unit="Ω",
                    doc=f"Load impedance. Set maximum ({self.Load_HighZ}) for HighZ",
                ),
                ch2_imp=P.IntParam(
                    self.get_output_impedance(2),
                    1,
                    self.Load_HighZ,
                    unit="Ω",
                    doc=f"Load impedance. Set maximum ({self.Load_HighZ}) for HighZ",
                ),
            )
        else:
            self.logger.error(f"Invalid label {label}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        # "gate" and "cw" doesn't provide ParamDict
        return ["ch1_cw", "ch2_cw", "output"]

    def configure(self, params: dict, label: str = "") -> bool:
        params = P.unwrap(params)

        if label == "gate":
            if not self.check_required_params(params, ("wave", "freq", "ampl", "phase")):
                return False
            return self.configure_gate(
                params["wave"],
                params["freq"],
                params["ampl"],
                params["phase"],
                offset=params.get("offset", 0.0),
                source=params.get("source", ""),
                polarity=params.get("polarity"),
                ch=params.get("ch", 1),
                reset=params.get("reset", False),
            )
        elif label == "burst":
            if not self.check_required_params(params, ("wave", "freq", "ampl", "phase", "cycle")):
                return False
            return self.configure_burst(
                params["wave"],
                params["freq"],
                params["ampl"],
                params["phase"],
                params["cycle"],
                offset=params.get("offset", 0.0),
                delay=params.get("delay", 0.0),
                source=params.get("source", ""),
                polarity=params.get("polarity"),
                ch=params.get("ch", 1),
                reset=params.get("reset", False),
            )
        elif label == "cw":
            return self.configure_cw(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset=params.get("offset", 0.0),
                phase_deg=params.get("phase", 0.0),
                ch=params.get("ch", 1),
                reset=params.get("reset", False),
            )
        elif label == "ch1_cw":
            return self.configure_cw(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset=params.get("offset", 0.0),
                phase_deg=params.get("phase", 0.0),
                ch=1,
                reset=False,
            )
        elif label == "ch2_cw":
            return self.configure_cw(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset=params.get("offset", 0.0),
                phase_deg=params.get("phase", 0.0),
                ch=2,
                reset=False,
            )
        elif label == "output":
            return self.configure_output(params)
        else:
            return self.fail_with(f"Unknown label: {label}")
