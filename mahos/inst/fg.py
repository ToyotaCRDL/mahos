#!/usr/bin/env python3

"""
Function Generator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

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
    OUTPUT_HighZ = 11_000

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\n"
        if "read_termination" not in conf:
            conf["read_termination"] = "\n"
        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self._ampl_bounds = {}
        self._offs_bounds = {}
        self._freq_bounds = {}
        self.update_all_bounds()

        self.ext_ref_clock = self.conf.get("ext_ref_clock", False)
        self.set_reference_clock(bool(self.ext_ref_clock))

        c = self.conf.get("gate", {})
        self.gate_conf = {
            "source": c.get("source", "EXT"),
            "slope": c.get("slope", True),
            "polarity": c.get("polarity", True),
            "idle_level": c.get("idle_level", "CENTER"),
        }
        self.logger.debug(f"gate configuration: {self.gate_conf}")

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
        return self.inst.query(f":OUTP{ch}:STAT?") == "ON"

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

        self.inst.write(f":OUTP{ch}:IMP {imp_ohm}")
        self.logger.info(f"Output{ch} impedance: {imp_ohm}")

        self.update_amplitude_bounds(ch)

        return True

    @ch_getter((1, 2))
    def get_output_impedance(self, ch: int = 1) -> int:
        """Get output impedance. OUTPUT_HighZ means highest impedance (INF)."""

        res = float(self.inst.query(f":OUTP{ch}:IMP?"))
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
            raise TypeError("Invalid frequency value type.")

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
    def set_amplitude(self, ampl_vpp: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT {ampl_vpp:.5f}")
        return True

    @ch_getter((1, 2))
    def get_amplitude(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:VOLT?"))

    @ch_setter((1, 2))
    def set_offset(self, offset_volt: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT:OFFS {offset_volt:.5f}")
        return True

    @ch_getter((1, 2))
    def get_offset(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:VOLT:OFFS?"))

    @ch_setter((1, 2))
    def set_high(self, volt: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT:HIGH {volt:.5f}")
        return True

    @ch_getter((1, 2))
    def get_high(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:VOLT:HIGH?"))

    @ch_setter((1, 2))
    def set_low(self, volt: float, ch: int = 1) -> bool:
        self.inst.write(f":SOUR{ch}:VOLT:LOW {volt:.5f}")
        return True

    @ch_getter((1, 2))
    def get_low(self, ch: int = 1) -> float:
        return float(self.inst.query(f":SOUR{ch}:VOLT:LOW?"))

    def set_reference_clock(self, external: bool) -> bool:
        source = "EXT" if external else "INT"
        self.inst.write(f":ROSC:SOUR {source}")
        self.logger.info(f"Reference clock: {source}")
        return True

    def align_phase(self) -> bool:
        self.inst.write(":PHAS:SYNC")
        self.logger.info("Executed phase alignment")
        return True

    def configure_CW(
        self,
        wave: str | None,
        freq: float | None,
        ampl_vpp: float | None,
        offset_volt: float | None = 0.0,
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure Continuous Wave output."""

        success = True
        if reset:
            success &= self.reset()
            success &= self.set_reference_clock(self.ext_ref_clock)

        if wave is not None:
            success &= self.set_function(wave, ch=ch)
        if freq is not None:
            success &= self.set_freq(freq, ch=ch)
        if ampl_vpp is not None:
            success &= self.set_amplitude(ampl_vpp, ch=ch)
        if offset_volt is not None:
            success &= self.set_offset(offset_volt, ch=ch)

        self.logger.info("Configured for CW.")
        return success

    def configure_gate(
        self,
        wave: str,
        freq: float,
        ampl_vpp: float,
        phase_deg: float,
        offset_volt: float = 0.0,
        source: str = "",
        slope: bool | None = None,
        polarity: bool | None = None,
        idle_level: str = "",
        ch: int = 1,
        reset: bool = True,
    ) -> bool:
        """Configure Gated Burst output."""

        success = (
            (self.reset() if reset else True)
            and self.set_reference_clock(self.ext_ref_clock)
            and self.set_function(wave, ch=ch)
            and self.set_freq(freq, ch=ch)
            and self.set_amplitude(ampl_vpp, ch=ch)
            and self.set_offset(offset_volt, ch=ch)
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

    def configure_output(self, params: dict):
        success = True
        if "ch1_imp" in params:
            success &= self.set_output_impedance(params["ch1_imp"], 1)
        if "ch2_imp" in params:
            success &= self.set_output_impedance(params["ch2_imp"], 2)
        if "ch1" in params:
            success &= self.set_output(params["ch1"], 1)
        if "ch2" in params:
            success &= self.set_output(params["ch2"], 2)
        if all([params.get(k) for k in ["ch1", "ch2", "align_phase"]]):
            success &= self.align_phase()
        return success

    # Standard API

    def reset(self, label: str = "") -> bool:
        success = self.rst_cls()
        self.update_all_bounds()
        return success

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
        return P.ParamDict(
            wave=P.StrChoiceParam(
                self.get_function(ch),
                ("SIN", "SQU"),
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
                unit="Vpp",
            ),
        )

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        if label == "cw_ch1":
            return self._cw_param_dict(1)
        elif label == "cw_ch2":
            return self._cw_param_dict(2)
        elif label == "output":
            return P.ParamDict(
                ch1_imp=P.IntParam(
                    self.get_output_impedance(1),
                    1,
                    self.OUTPUT_HighZ,
                    unit="Ω",
                    doc=f"Set maximum ({self.OUTPUT_HighZ}) for HighZ",
                ),
                ch2_imp=P.IntParam(
                    self.get_output_impedance(2),
                    1,
                    self.OUTPUT_HighZ,
                    unit="Ω",
                    doc=f"Set maximum ({self.OUTPUT_HighZ}) for HighZ",
                ),
                ch1=P.BoolParam(self.get_output(1)),
                ch2=P.BoolParam(self.get_output(2)),
                align_phase=P.BoolParam(True),
            )

    def get_param_dict_labels(self) -> list[str]:
        # "gate" and "cw" doesn't provide ParamDict
        return ["cw_ch1", "cw_ch2", "output"]

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
                offset_volt=params.get("offset", 0.0),
                source=params.get("source", ""),
                slope=params.get("polarity"),  # this is intentional. slope = polarity.
                polarity=params.get("polarity"),
                idle_level=params.get("idle_level", ""),
                ch=params.get("ch", 1),
                reset=params.get("reset", False),
            )
        elif label == "cw":
            return self.configure_CW(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset_volt=params.get("offset", 0.0),
                ch=params.get("ch", 1),
                reset=params.get("reset", False),
            )
        elif label == "cw_ch1":
            return self.configure_CW(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset_volt=params.get("offset", 0.0),
                ch=1,
                reset=False,
            )
        elif label == "cw_ch2":
            return self.configure_CW(
                params.get("wave"),
                params.get("freq"),
                params.get("ampl"),
                offset_volt=params.get("offset", 0.0),
                ch=2,
                reset=False,
            )
        elif label == "output":
            return self.configure_output(params)
        else:
            return self.fail_with(f"Unknown label: {label}")
