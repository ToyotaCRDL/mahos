#!/usr/bin/env python3

"""
Lockin Amplifier module.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import enum

from .visa_instrument import VisaInstrument

from ..msgs import param_msgs as P


class RefSource(enum.Enum):
    EXT: 0
    INT: 1
    SIG: 2


class RefEdge(enum.Enum):
    SIN_POS: 0
    TTL_POS: 1
    TTL_NEG: 2


class InputSource(enum.Enum):
    A: 0
    A_B: 1
    I_6: 2
    I_8: 3


class LineFilter(enum.Enum):
    THRU: 0
    LINE: 1
    LINEx2: 2
    LINE_LINEx2: 3


class DynamicReserve(enum.Enum):
    HIGH: 0
    MEDIUM: 1
    LOW: 2


class VoltageSensitivity(enum.Enum):
    s2nV: 0
    s5nV: 1
    s10nV: 2
    s20nV: 3
    s50nV: 4
    s100nV: 5
    s200nV: 6
    s500nV: 7
    s1uV: 8
    s2uV: 9
    s5uV: 10
    s10uV: 11
    s20uV: 12
    s50uV: 13
    s100uV: 14
    s200uV: 15
    s500uV: 16
    s1mV: 17
    s2mV: 18
    s5mV: 19
    s10mV: 20
    s20mV: 21
    s50mV: 22
    s100mV: 23
    s200mV: 24
    s500mV: 25
    s1V: 26


class CurrentSensitivity(enum.Enum):
    s5fA: 1
    s10fA: 2
    s20fA: 3
    s50fA: 4
    s100fA: 5
    s200fA: 6
    s500fA: 7
    s1pA: 8
    s2pA: 9
    s5pA: 10
    s10pA: 11
    s20pA: 12
    s50pA: 13
    s100pA: 14
    s200pA: 15
    s500pA: 16
    s1nA: 17
    s2nA: 18
    s5nA: 19
    s10nA: 20
    s20nA: 21
    s50nA: 22
    s100nA: 23
    s200nA: 24
    s500nA: 25
    s1uA: 26


class TimeConstant(enum.Enum):
    c10us: 0
    c30us: 1
    c100us: 2
    c300us: 3
    c1ms: 4
    c3ms: 5
    c10ms: 6
    c30ms: 7
    c100ms: 8
    c300ms: 9
    c1s: 10
    c3s: 11
    c10s: 12
    c30s: 13
    c100s: 14
    c300s: 15
    c1ks: 16
    c3ks: 17
    c10ks: 18
    c30ks: 19


class Data1(enum.Enum):
    X: 0
    R: 1
    NOISE: 2
    AUX_IN1: 3


class Data2(enum.Enum):
    Y: 0
    Theta: 1
    AUX_IN1: 2
    AUX_IN2: 3


class DataNormalization(enum.Enum):
    OFF: 0
    dB: 1
    PERCENT: 2


class LI5640(VisaInstrument):
    """NF LI5640 Lockin Amplifier."""

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\n"
        if "read_termination" not in conf:
            conf["read_termination"] = "\n"
        VisaInstrument.__init__(self, name, conf, prefix=prefix)

    def get_phase_offset(self) -> float:
        """get current phase offset in degrees."""

        return float(self.inst.query("PHAS?"))

    def set_phase_offset(self, phase_deg: float) -> bool:
        """set phase offset in degrees."""

        while phase_deg > 180.0:
            phase_deg -= 360.0
        while phase_deg < -180.0:
            phase_deg += 360.0

        self.inst.write(f"PHAS {phase_deg:.4f}")
        return True

    def set_auto_phase_offset(self) -> bool:
        """set auto phase offset."""

        self.inst.write("APHS")
        return True

    def get_harmonics(self) -> int:
        """get current harmonics degree."""

        return int(self.inst.query("HARM?"))

    def set_harmonics(self, harmonics_deg: int) -> bool:
        """set harmonics degree."""

        if not (1 <= harmonics_deg <= 19999):
            return self.fail_with(f"harmonics out of range: {harmonics_deg}")

        self.inst.write(f"HARM {harmonics_deg:d}")
        return True

    def get_ref_source(self) -> RefSource | None:
        """get current reference source."""

        i = int(self.inst.query("RSRC?"))
        if i in (1, 2, 3):
            return RefSource(i)
        self.logger.error(f"Unexpected response to RSRC?: {i}")
        return None

    def set_ref_source(self, src: RefSource) -> bool:
        """set reference source."""

        self.inst.write(f"RSRC {src.value}")
        return True

    def get_ref_edge(self) -> RefEdge | None:
        """get current reference source rdge."""

        i = int(self.inst.query("REDG?"))
        if i in (1, 2, 3):
            return RefEdge(i)
        self.logger.error(f"Unexpected response to REDG?: {i}")
        return None

    def set_ref_edge(self, edge: RefEdge) -> bool:
        """set reference source edge."""

        self.inst.write(f"REDG {edge.value}")
        return True

    def get_input_source(self) -> InputSource | None:
        """get current signal input source."""

        i = int(self.inst.query("ISRC?"))
        if i in (0, 1, 2, 3):
            return InputSource(i)
        self.logger.error(f"Unexpected response to ISRC?: {i}")
        return None

    def set_input_source(self, src: InputSource) -> bool:
        """set signal input source."""

        self.inst.write(f"ISRC {src.value}")
        return True

    def get_coupling(self) -> bool:
        """get input coupling. True (False) if DC (AC) coupling."""

        return bool(self.inst.query("ICPL?"))

    def set_coupling(self, DC_coupling: bool) -> bool:
        """set input coupling."""

        self.inst.write(f"ICPL {bool(DC_coupling):d}")
        return True

    def get_ground(self) -> bool:
        """get input ground. True (False) if Grounded (Floating) input."""

        return bool(self.inst.query("IGND?"))

    def set_ground(self, ground: bool) -> bool:
        """set input ground."""

        self.inst.write(f"IGND {bool(ground):d}")
        return True

    def get_line_filter(self) -> LineFilter:
        """get line filter."""

        return LineFilter(self.inst.query("ILIN?"))

    def set_line_filter(self, filt: LineFilter) -> bool:
        """set line filter."""

        self.inst.write(f"ILIN {filt.value}")
        return True

    def get_line_freq(self) -> bool:
        """get line frequency. True (False) if 60 Hz (50 Hz)."""

        return bool(self.inst.query("IFRQ?"))

    def set_line_freq(self, is_60Hz: bool) -> bool:
        """set line frequency."""

        self.inst.write(f"IFRQ {bool(is_60Hz):d}")
        return True

    def get_LPF_through(self) -> bool:
        """get LPF pass through. True if LPF is passed through."""

        return bool(self.inst.query("ITHR?"))

    def set_LPF_through(self, pass_through: bool) -> bool:
        """set LPF pass through."""

        self.inst.write(f"ITHR {bool(pass_through):d}")
        return True

    def get_dynamic_reserve(self) -> DynamicReserve:
        """get dynamic reserve."""

        return DynamicReserve(self.inst.query("DRSV?"))

    def set_dynamic_reserve(self, reserve: DynamicReserve) -> bool:
        """set dynamic reserve."""

        self.inst.write(f"DRSV {reserve.value}")
        return True

    def get_volt_sensitivity(self) -> VoltageSensitivity:
        """get voltage sensitivity."""

        return VoltageSensitivity(self.inst.query("VSEN?"))

    def set_volt_sensitivity(self, sens: VoltageSensitivity) -> bool:
        """set voltage sensitivity."""

        self.inst.write(f"VSEN {sens.value}")
        return True

    def get_current_sensitivity(self) -> CurrentSensitivity:
        """get current sensitivity."""

        return CurrentSensitivity(self.inst.query("ISEN?"))

    def set_current_sensitivity(self, sens: CurrentSensitivity) -> bool:
        """set current sensitivity."""

        self.inst.write(f"ISEN {sens.value}")
        return True

    def get_time_constant(self) -> TimeConstant:
        """get time constant."""

        return TimeConstant(self.inst.query("TCON?"))

    def set_time_constant(self, cons: TimeConstant) -> bool:
        """set time constant."""

        self.inst.write(f"TCON {cons.value}")
        return True

    def get_sync_filter(self) -> bool:
        """get sync filter. True if sync filter is enabled."""

        return bool(self.inst.query("SYNC?"))

    def set_sync_filter(self, enable: bool) -> bool:
        """set sync filter."""

        self.inst.write(f"SYNC {bool(enable):d}")
        return True

    def get_slope(self) -> int:
        """get slope in dB/oct."""

        i = int(self.inst.query("SLOP?"))
        if i not in (0, 1, 2, 3):
            self.logger.error(f"Unexpected response to SLOP?: {i}")
            return 0
        return {0: 6, 1: 12, 2: 18, 3: 24}[i]

    def set_slope(self, slope_dB_oct: int) -> bool:
        """set slope in dB/oct."""

        if slope_dB_oct not in (6, 12, 18, 24):
            return self.fail_with(f"Unrecongizable slope: {slope_dB_oct}")
        i = {6: 0, 12: 1, 18: 2, 24: 3}[slope_dB_oct]
        self.inst.write(f"SLOP {i}")
        return True

    def get_data1(self) -> Data1:
        """get data1."""

        return Data1(self.inst.query("DDEF? 1"))

    def set_data1(self, data: Data1) -> bool:
        """set data1."""

        self.inst.write(f"DDEF 1, {data.value}")
        return True

    def get_data2(self) -> Data2:
        """get data2."""

        return Data2(self.inst.query("DDEF? 2"))

    def set_data2(self, data: Data2) -> bool:
        """set data2."""

        self.inst.write(f"DDEF 2, {data.value}")
        return True

    def get_data_normalization(self) -> DataNormalization:
        """get data normalization."""

        return DataNormalization(self.inst.query("NORM?"))

    def set_data_normalization(self, norm: DataNormalization) -> bool:
        """set data normalization."""

        self.inst.write(f"NORM {norm.value}")
        return True

    def get_data_normalization_voltage(self) -> float:
        """get data normalization std value for voltage."""

        return float(self.inst.query("VSTD?"))

    def set_data_normalization_voltage(self, volt: float) -> bool:
        """set data normalization std value for voltage."""

        self.inst.write(f"VSTD {volt:.6e}")
        return True

    def get_data_normalization_current(self) -> float:
        """get data normalization std value for current."""

        return float(self.inst.query("ISTD?"))

    def set_data_normalization_current(self, current: float) -> bool:
        """set data normalization std value for current."""

        self.inst.write(f"ISTD {current:.6e}")
        return True

    def get_noise_smooth(self) -> int:
        """get noise measurement smoothing factor."""

        i = int(self.inst.query("NOIS?"))
        if i not in (0, 1, 2, 3):
            self.logger.error(f"Unexpected response to NOIS?: {i}")
            return 0
        return {0: 1, 1: 4, 2: 16, 3: 64}[i]

    def set_noise_smooth(self, fac: int) -> bool:
        """set noise measurement smoothing factor."""

        if fac not in (1, 4, 16, 64):
            return self.fail_with(f"Unrecongizable factor: {fac}")
        i = {1: 0, 4: 1, 16: 2, 64: 3}[fac]
        self.inst.write(f"NOIS {i}")
        return True

    def get_xoffset_enable(self) -> bool:
        """get if offset for X is enabled."""

        return bool(self.inst.query("OFSO? 1"))

    def set_xoffset_enable(self, enable: bool) -> bool:
        """set if offset for X is enabled."""

        self.inst.write(f"OFSO 1, {bool(enable):d}")
        return True

    def get_yoffset_enable(self) -> bool:
        """get if offset for Y is enabled."""

        return bool(self.inst.query("OFSO? 2"))

    def set_yoffset_enable(self, enable: bool) -> bool:
        """set if offset for Y is enabled."""

        self.inst.write(f"OFSO 2, {bool(enable):d}")
        return True

    def get_xoffset(self) -> float:
        """get offset value for X."""

        return float(self.inst.query("OFFS? 1"))

    def set_xoffset(self, offset_percent: bool) -> bool:
        """set offset value for X."""

        self.inst.write(f"OFFS 1, {offset_percent:.2f}")
        return True

    def get_yoffset(self) -> float:
        """get offset value for Y."""

        return float(self.inst.query("OFFS? 2"))

    def set_yoffset(self, offset_percent: bool) -> bool:
        """set offset value for Y."""

        self.inst.write(f"OFFS 2, {offset_percent:.2f}")
        return True

    def get_data1_expand(self) -> int:
        """get output expand factor for data1."""

        i = int(self.inst.query("OEXP? 1"))
        if i not in (0, 1, 2):
            self.logger.error(f"Unexpected response to OEXP?: {i}")
            return 0
        return {0: 1, 1: 10, 2: 100}[i]

    def set_data1_expand(self, fac: int) -> bool:
        """set output expand factor for data1."""

        if fac not in (1, 10, 100):
            return self.fail_with(f"Unrecongizable factor: {fac}")
        i = {1: 0, 10: 1, 100: 2}[fac]
        self.inst.write(f"OEXP 1, {i}")
        return True

    def get_data2_expand(self) -> int:
        """get output expand factor for data2."""

        i = int(self.inst.query("OEXP? 2"))
        if i not in (0, 1, 2):
            self.logger.error(f"Unexpected response to OEXP?: {i}")
            return 0
        return {0: 1, 1: 10, 2: 100}[i]

    def set_data2_expand(self, fac: int) -> bool:
        """set output expand factor for data2."""

        if fac not in (1, 10, 100):
            return self.fail_with(f"Unrecongizable factor: {fac}")
        i = {1: 0, 10: 1, 100: 2}[fac]
        self.inst.write(f"OEXP 2, {i}")
        return True

    def get_ratio_mode(self) -> bool:
        """get if ratio mode is enabled."""

        return bool(self.inst.query("RAT?"))

    def set_ratio_mode(self, enable: bool) -> bool:
        """set if ratio mode is enabled."""

        self.inst.write(f"RAT {bool(enable):d}")
        return True

    def get_ratio_k(self) -> float:
        """get k-factor in ratio mode."""

        return float(self.inst.query("KFAC?"))

    def set_ratio_k(self, fac: bool) -> bool:
        """set k-factor in ratio mode."""

        self.inst.write(f"KFAC {fac:.4f}")
        return True

    def get_lamp(self) -> bool:
        """get (physical) lamp is enabled."""

        return bool(self.inst.query("LAMP?"))

    def set_lamp(self, enable: bool) -> bool:
        """set (physical) lamp is enabled."""

        self.inst.write(f"LAMP {bool(enable):d}")
        return True

    # Auto settings

    def set_initialize(self) -> bool:
        """execute setting initialization."""

        self.inst.write("INIT")
        return True

    def set_auto(self) -> bool:
        """execute auto setting."""

        self.inst.write("ASET")
        return True

    def set_auto_sensitivity(self) -> bool:
        """execute auto sensitivity setting."""

        self.inst.write("ASEN")
        return True

    def set_auto_time_constant(self) -> bool:
        """execute auto time constant setting."""

        self.inst.write("ATIM")
        return True

    def set_auto_offset(self) -> bool:
        """execute auto offset setting."""

        self.inst.write("AOFS")
        return True

    # Standard API

    def get(self, key: str, args=None):
        if key == "opc":
            return self.query_opc(delay=args)
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def set(self, key: str, value=None) -> bool:
        if key == "init":
            return self.set_initialize()
        elif key == "auto":
            return self.set_auto()
        elif key == "auto_sensitivity":
            return self.set_auto_sensitivity()
        elif key == "auto_time_constant":
            return self.set_auto_time_constant()
        elif key == "auto_phase_offset":
            return self.set_auto_phase_offset()
        else:
            return self.fail_with("Unknown set() key.")

    def get_param_dict_labels(self, group: str = "") -> list[str]:
        return [""]

    def get_param_dict(self, label: str = "", group: str = "") -> P.ParamDict[str, P.PDValue]:
        d = P.ParamDict(
            phase_offset=P.FloatParam(0.0, -180.0, 180.0, doc="phase offset in degrees"),
        )
        return d

    def configure(self, params: dict, label: str = "", group: str = "") -> bool:
        if "phase_offset" in params:
            self.set_phase_offset(params["phase_offset"])
        return True
