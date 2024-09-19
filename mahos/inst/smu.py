#!/usr/bin/env python3

"""
Source Meter (Measure) Unit module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import enum

import numpy as np

from ..msgs import param_msgs as P
from .visa_instrument import VisaInstrument


class Mode(enum.Enum):
    UNCONFIGURED = 0
    IV = 1
    IV_sweep = 2
    VI_sweep = 3


class Keithley_2450(VisaInstrument):
    """Keithley 2450 Source Meter Module."""

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 60_000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)
        self.volt_min, self.volt_max = self.conf.get("volt_bounds", (-200, 200))
        self.curr_min, self.curr_max = self.conf.get("curr_bounds", (-105e-3, 105e-3))

        self.auto_range = self.conf.get("auto_range", True)
        self._sweep_points = 0
        self._mode = Mode.UNCONFIGURED

    def get_VOLT_bounds(self):
        return self.volt_min, self.volt_max

    def get_CURR_bounds(self):
        return self.curr_min, self.curr_max

    def get_bounds(self):
        return {
            "voltage": self.get_VOLT_bounds(),
            "current": self.get_CURR_bounds(),
        }

    def configure_source_volt(self, compliance: float) -> bool:
        self.inst.write("SOUR:FUNC VOLT")  # source voltage mode
        self.inst.write(f"SOUR:VOLT:ILIMIT {compliance}")
        return self.check_error()

    def set_source_volt(self, volt: float) -> bool:
        self.inst.write(f"SOUR:VOLT {volt}")  # set voltage
        return self.check_error()

    def set_sweep_VOLT(
        self, start: float, stop: float, point: int, delay: float, log: bool = False
    ) -> bool:
        if log:
            self.inst.write(f"SOUR:SWE:VOLT:LOG {start}, {stop}, {point}, {delay}")
        else:
            self.inst.write(f"SOUR:SWE:VOLT:LIN {start}, {stop}, {point}, {delay}")
        return self.check_error()

    def configure_measure_current(self) -> bool:
        self.inst.write("SENS:FUNC 'CURR'")  # mode
        return self.check_error()

    def set_auto_range(self, on: bool) -> bool:
        if on:
            self.inst.write("SENS:CURR:RANG:AUTO ON")
        else:
            self.inst.write("SENS:CURR:RANG:AUTO OFF")
        return self.check_error()

    def set_current_nplc(self, nplc: float) -> bool:
        if not (0.01 <= nplc <= 10):
            return self.fail_with("NPLC must be between 0.01 and 10.")
        else:
            self.inst.write(f"SENS:CURR:NPLC {nplc}")
            return self.check_error()

    def set_output(self, on: bool) -> bool:
        if on:
            self.inst.write("OUTP ON")
        else:
            self.inst.write("OUTP OFF")
        return self.check_error()

    def configure_IV(self, compliance: float, nplc: float) -> bool:
        """Setup DC Voltage (on-demand) measurement."""

        success = (
            self.rst_cls()
            and self.configure_source_volt(compliance)
            and self.set_source_volt(0)
            and self.configure_measure_current()
            and self.set_auto_range(True)
            and self.set_current_nplc(nplc)
        )

        if success:
            self._mode = Mode.IV
            self.logger.info("Configured for DC Voltage supply.")
        else:
            self._mode = Mode.UNCONFIGURED
            self.logger.info("Failed to configure DC Voltage supply.")
        return success

    def configure_IV_sweep(
        self,
        compliance: float = 105e-3,
        delay: float = 0.1,
        start: float = 0,
        stop: float = 0.1,
        point: int = 2,
        nplc: float = 10.0,
        auto_range: bool | None = None,
        logx: bool = False,
    ) -> bool:
        """Setup IV curve measurement."""

        success = (
            self.rst_cls()
            and self.configure_source_volt(compliance)
            and self.configure_measure_current()
            and self.set_auto_range(auto_range if auto_range is not None else self.auto_range)
            and self.set_sweep_VOLT(start, stop, point, delay, logx)
            and self.set_current_nplc(nplc)
        )
        if success:
            self._mode = Mode.IV_sweep
            self._sweep_points = point
            self.logger.info("Configured for IV curve measurement.")
        else:
            self._mode = Mode.UNCONFIGURED
            self.logger.info("Failed to configure IV curve measurement.")
        return success

    def get_data_IV(self):
        val = self.inst.query("READ?")

        try:
            return float(val)
        except Exception:
            self.logger.error(f"Got invalid read {val}")
            return None

    def get_data_IV_sweep(self):
        val = (
            self.inst.write(":INIT")
            and self.inst.write("*WAI")
            and self.inst.query(f"TRAC:DATA? 1, {self._sweep_points}, 'defbuffer1', READ")
        )
        val_spl = val.split(",")
        try:
            voltages = np.array([float(v) for v in val_spl])
            # self.logger.info(" ".join([f"{v:.3f}" for v in voltages]))
            return voltages
        except Exception:
            self.logger.error(f"Got invalid read {val}")
            return None

    def get_data(self):
        if self._mode == Mode.IV:
            return self.get_data_IV()

        elif self._mode == Mode.IV_sweep:
            return self.get_data_IV_sweep()

        else:
            self.logger.error("get_data() is called but not configured.")
            return None

    def get_unit(self) -> str:
        if self._mode in (Mode.IV, Mode.IV_sweep):
            return "A"
        else:
            self.logger.error("get_unit() is called but not configured.")
            return ""

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "bounds":
            return self.get_bounds()
        elif key == "data":
            return self.get_data()
        elif key == "unit":
            return self.get_unit()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        return ["IV_source", "IV", "IV_sweep"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        if label == "IV_source":
            return P.ParamDict(
                volt=P.FloatParam(0.0, self.volt_min, self.volt_max, unit="V", SI_prefix=True),
            )

        elif label == "IV":
            return P.ParamDict(
                compliance=P.FloatParam(105e-3, -105e-3, 105e-3, unit="A", SI_prefix=True),
                nplc=P.FloatParam(10.0, 0.01, 10.0),
            )

        elif label == "IV_sweep":
            return P.ParamDict(
                start=P.FloatParam(0.0, self.volt_min, self.volt_max, unit="V", SI_prefix=True),
                stop=P.FloatParam(0.1, self.volt_min, self.volt_max, unit="V", SI_prefix=True),
                auto_range=P.BoolParam(self.auto_range),
                logx=P.BoolParam(False),
                point=P.IntParam(2),
                nplc=P.FloatParam(10.0, 0.01, 10.0),
                compliance=P.FloatParam(105e-3, -105e-3, 105e-3, unit="A", SI_prefix=True),
                delay=P.FloatParam(0.1, 0, 10),
            )

        else:
            return self.fail_with(f"unknown label {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label == "iv_source":
            return self.set_source_volt(volt=params.get("volt", 0.0))

        elif label == "iv":
            return self.configure_IV(
                compliance=params.get("compliance", 105e-3),
                nplc=params.get("nplc", 10.0),
            )

        elif label == "iv_sweep":
            return self.configure_IV_sweep(
                start=params.get("start", 0.0),
                stop=params.get("stop", 0.1),
                logx=params.get("logx", False),
                point=params.get("point", 2),
                nplc=params.get("nplc", 10.0),
                auto_range=params.get("auto_range", self.auto_range),
                delay=params.get("delay", 0.1),
                compliance=params.get("compliance", 105e-3),
            )

        else:
            self.logger.error(f"unknown label: {label}")
            return False

    def start(self, label: str = "") -> bool:
        if self._mode in (Mode.IV, Mode.IV_sweep):
            return self.set_output(True)
        else:
            return self.fail_with("start() is called but not configured.")

    def stop(self, label: str = "") -> bool:
        if self._mode in (Mode.IV, Mode.IV_sweep):
            return self.set_output(False)
        else:
            return self.fail_with("stop() is called but not configured.")


class Keithley_6430(VisaInstrument):
    """Keithley 6430 Sub-Femtoamp Remote SourceMeter module."""

    RANGE_SOURCE = (
        "BEST",
        "AUTO",
        "FIX",
    )

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 10_000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)
        self.volt_min, self.volt_max = self.conf.get("volt_bounds", (-210.0, 210.0))
        self.curr_min, self.curr_max = self.conf.get("curr_bounds", (-105e-3, 105e-3))

        self.auto_filter = self.conf.get("auto_filter", True)
        self.range = self.conf.get("range", "BEST")
        self._mode = Mode.UNCONFIGURED

    def get_VOLT_bounds(self):
        return self.volt_min, self.volt_max

    def get_CURR_bounds(self):
        return self.curr_min, self.curr_max

    def get_bounds(self):
        return {
            "voltage": self.get_VOLT_bounds(),
            "current": self.get_CURR_bounds(),
        }

    def configure_source_volt(self) -> bool:
        self.inst.write("SOUR:FUNC VOLT")  # source voltage mode
        self.inst.write("SOUR:VOLT:MODE FIX")  # fixed voltage source mode
        self.inst.write("SOUR:VOLT:RANG:AUTO 1")  # auto range enable
        self.inst.write("SOUR:VOLT:LEVEL 0.0")
        return self.check_error()

    def set_sweep_volt(self, delay: float, start: float, stop: float, point: int) -> bool:
        self.inst.write("SOUR:VOLT 0")  # set vias voltage 0V
        self.inst.write("SOUR:VOLT:MODE SWE")  # set sweep mode
        self.inst.write(f"SOUR:DEL {delay}")
        self.inst.write(f"SOUR:VOLT:STAR {start}")  # set start current
        self.inst.write(f"SOUR:VOLT:STOP {stop}")  # set stop current
        self.inst.write(f"SOUR:SWE:POIN {point}")  # set measure point

        return self.check_error()

    def set_sweep_current(self, delay: float, start: float, stop: float, point: int) -> bool:
        self.inst.write("SOUR:CURR 0")  # set vias current 0A
        self.inst.write("SOUR:CURR:MODE SWE")  # set sweep mode
        self.inst.write(f"SOUR:DEL {delay}")
        self.inst.write(f"SOUR:CURR:STAR {start}")  # set start current
        self.inst.write(f"SOUR:CURR:STOP {stop}")  # set stop current
        self.inst.write(f"SOUR:SWE:POIN {point}")  # set measure point
        return self.check_error()

    def set_range_source(self, range: str = "BEST") -> bool:
        range = range.upper()
        if range not in self.RANGE_SOURCE:
            self.logger.error("invalid scale source.")
            return False

        self.inst.write("SOUR:SWE:RANG " + range)
        return self.check_error()

    def set_source_volt(self, n: float) -> bool:
        self.inst.write(f"SOUR:VOLT:LEVEL {n}")
        return self.check_error()

    def set_scale_source(self, log: bool = False) -> bool:  # choose linear or log scale
        if log:
            self.inst.write("SOUR:SWE:SPAC LOG")
        else:
            self.inst.write("SOUR:SWE:SPAC LIN")
        return self.check_error()

    def configure_measure_current(self, compliance: float) -> bool:
        self.inst.write("SENS:FUNC 'CURR'")  # measure cuurent mode
        self.inst.write(f"SENS:CURR:PROT {compliance}")  # compliance limit
        self.inst.write("SENS:CURR:RANG:AUTO 1")  # auto range enable
        return self.check_error()

    def set_trigger(self, trigger: int) -> bool:
        self.inst.write(f"TRIG:COUN {trigger}")
        return self.check_error()

    def set_current_nplc(self, n: float) -> bool:
        if not (0.01 <= n <= 10):
            return self.fail_with("NPLC must be between 0.01 and 10.")
        else:
            self.inst.write(f"SENS:CURR:NPLC {n}")
            self.inst.write("DISP:DIG 7")
        return self.check_error()

    def set_volt_nplc(self, n: float) -> bool:
        if not (0.01 <= n <= 10):
            return self.fail_with("NPLC must be between 0.01 and 10.")
        else:
            self.inst.write(f"SENS:VOLT:NPLC {n}")
            self.inst.write("DISP:DIG 7")
        return self.check_error()

    def set_auto_filter(self, on: bool) -> bool:
        if on:
            self.inst.write("SENS:AVER:AUTO ON")
        else:
            self.inst.write("SENS:AVER:AUTO OFF")
        return self.check_error()

    def set_output(self, on: bool) -> bool:
        if on:
            self.inst.write("OUTP ON")
        else:
            self.inst.write("OUTP OFF")
        return self.check_error()

    def configure_IV(self, compliance: float, nplc: float = 10.0, filter: bool = True) -> bool:
        """Setup IV (V source I meter) measurement."""

        success = (
            self.rst_cls()
            and self.configure_source_volt()
            and self.configure_measure_current(compliance)
            and self.set_current_nplc(nplc)
            and self.set_auto_filter(filter)
        )
        if success:
            self._mode = Mode.IV
            self.logger.info("Configured for IV measurement.")
        else:
            self._mode = Mode.UNCONFIGURED
            self.logger.info("Failed to configure IV measurement.")
        return success

    def configure_IV_sweep(
        self,
        delay: float = 0.1,
        start: float = 0,
        stop: float = 0.1,
        point: int = 2,
        nplc: float = 10.0,
        auto_filter: bool | None = None,
        range: str = "",
        logx: bool = False,
    ) -> bool:
        """Setup IV curve measurement."""

        success = (
            self.rst_cls()
            and self.set_sweep_volt(delay, start, stop, point)
            and self.set_trigger(point)
            and self.set_current_nplc(nplc)
            and self.set_auto_filter(auto_filter if auto_filter is not None else self.auto_filter)
            and self.set_range_source(range or self.range)
            and self.set_scale_source(logx)
        )
        if success:
            self._mode = Mode.IV_sweep
            self.logger.info("Configured for IV curve measurement.")
        else:
            self._mode = Mode.UNCONFIGURED
            self.logger.info("Failed to configure IV curve measurement.")
        return success

    def configure_VI_sweep(
        self,
        delay: float = 0.1,
        start: float = 0,
        stop: float = 0.0,
        point: int = 2,
        nplc: float = 10.0,
        auto_filter: bool | None = None,
        range: str = "",
        logx: bool = False,
    ) -> bool:
        """Setup VI curve measurement."""

        success = (
            self.rst_cls()
            and self.set_sweep_current(delay, start, stop, point)
            and self.set_trigger(point)
            and self.set_volt_nplc(nplc)
            and self.set_auto_filter(auto_filter if auto_filter is not None else self.auto_filter)
            and self.set_range_source(range or self.range)
            and self.set_scale_source(logx)
        )
        if success:
            self._mode = Mode.VI_sweep
            self.logger.info("Configured for VI curve measurement.")
        else:
            self._mode = Mode.UNCONFIGURED
            self.logger.info("Failed to configure VI curve measurement.")
        return success

    def get_data_current(self):
        val = self.inst.query("READ?")
        val_spl = val.split(",")
        try:
            return float(val_spl[1])
        except Exception:
            self.logger.error(f"Got invalid read {val}")
            return None

    def get_data_sweep(self):
        val = self.inst.query(":READ?")
        val_spl = val.split(",")
        try:
            # voltages = np.array([float(v) for v in val_spl[::5]])
            # self.logger.info(" ".join([f"{v:.3f}" for v in voltages]))
            return np.array([float(v) for v in val_spl[1::5]])
        except Exception:
            self.logger.error(f"Got invalid read {val}")
            return None

    def get_data(self):
        if self._mode == Mode.IV:
            return self.get_data_current()

        elif self._mode in (Mode.IV_sweep, Mode.VI_sweep):
            return self.get_data_sweep()

        else:
            self.logger.error("get_data() is called but not configured.")
            return None

    def get_unit(self) -> str:
        if self._mode in (Mode.IV, Mode.IV_sweep):
            return "A"

        elif self._mode == Mode.VI_sweep:
            return "V"

        else:
            self.logger.error("get_unit() is called but not configured.")
            return ""

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "bounds":
            return self.get_bounds()
        elif key == "data":
            return self.get_data()
        elif key == "unit":
            return self.get_unit()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        return ["IV_source", "IV", "IV_sweep", "VI_sweep"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        if label == "IV_source":
            return P.ParamDict(
                volt=P.FloatParam(0.0, self.volt_min, self.volt_max, unit="V", SI_prefix=True),
            )

        elif label == "IV":
            return P.ParamDict(
                compliance=P.FloatParam(105e-3, -105e-3, 105e-3, unit="A", SI_prefix=True),
                nplc=P.FloatParam(10.0, 0.01, 10.0),
                filter=P.BoolParam(True),
            )

        elif label == "IV_sweep":
            return P.ParamDict(
                start=P.FloatParam(0.0, self.volt_min, self.volt_max, unit="V", SI_prefix=True),
                stop=P.FloatParam(0.0, self.volt_min, self.volt_max, unit="V", SI_prefix=True),
                range=P.StrChoiceParam(self.range, ["BEST", "AUTO", "FIX"]),
                logx=P.BoolParam(False),
                point=P.IntParam(21),
                nplc=P.FloatParam(10.0, 0.01, 10.0),
                auto_filter=P.BoolParam(self.auto_filter),
                delay=P.FloatParam(0.1, 0, 10),
            )

        elif label == "VI_sweep":
            return P.ParamDict(
                start=P.FloatParam(0.0, self.curr_min, self.curr_max, unit="A", SI_prefix=True),
                stop=P.FloatParam(0.0, self.curr_min, self.curr_max, unit="A", SI_prefix=True),
                range=P.StrChoiceParam(self.range, ["BEST", "AUTO", "FIX"]),
                logx=P.BoolParam(False),
                point=P.IntParam(21),
                nplc=P.FloatParam(10.0, 0.01, 10.0),
                auto_filter=P.BoolParam(self.auto_filter),
                delay=P.FloatParam(0.1, 0, 10),
            )

        else:
            return self.fail_with(f"unknown label {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label == "iv_source":
            return self.set_source_volt(volt=params.get("volt", 0.0))

        elif label == "iv":
            return self.configure_IV(
                compliance=params.get("compliance", 105e-3),
                nplc=params.get("nplc", 10.0),
                filter=params.get("filter", True),
            )

        elif label == "iv_sweep":
            return self.configure_IV_sweep(
                start=params.get("start", 0.0),
                stop=params.get("stop", 0.0),
                range=params.get("range", self.range),
                logx=params.get("logx", False),
                point=params.get("point", 21),
                nplc=params.get("nplc", 10.0),
                auto_filter=params.get("auto_filter", self.auto_filter),
                delay=params.get("delay", 0.1),
            )

        elif label == "vi_sweep":
            return self.configure_VI_sweep(
                start=params.get("start", 0.0),
                stop=params.get("stop", 0.0),
                range=params.get("range", self.range),
                logx=params.get("logx", False),
                point=params.get("point", 21),
                nplc=params.get("nplc", 10.0),
                auto_filter=params.get("auto_filter", self.auto_filter),
                delay=params.get("delay", 0.1),
            )
        else:
            self.logger.error(f"unknown label: {label}")
            return False

    def start(self, label: str = "") -> bool:
        if self._mode in (Mode.IV, Mode.IV_sweep, Mode.VI_sweep):
            return self.set_output(True)
        else:  # UNCONFIGURED
            return self.fail_with("start() is called but not configured.")

    def stop(self, label: str = "") -> bool:
        if self._mode in (Mode.IV, Mode.IV_sweep, Mode.VI_sweep):
            return self.set_output(False)
        else:  # UNCONFIGURED
            return self.fail_with("stop() is called but not configured.")
