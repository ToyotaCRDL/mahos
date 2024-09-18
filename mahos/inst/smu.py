#!/usr/bin/env python3

"""
Source Meter (Measure) Unit module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import enum

from ..msgs import param_msgs as P
from .visa_instrument import VisaInstrument


class Mode(enum.Enum):
    UNCONFIGURED = 0
    IV = 1


class Keithley_2450(VisaInstrument):
    """Keithley 2450 Source Meter Module."""

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 2_000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)
        self.volt_min, self.volt_max = self.conf.get("volt_bounds", (-200, 200))

    def configure_DCV_source(self, compliance: float) -> bool:
        self.inst.write("SOUR:FUNC VOLT")  # source voltage mode
        self.inst.write("SOUR:VOLT 0.0")
        self.inst.write(f"SOUR:VOLT:ILIMIT {compliance}")
        return self.check_error()

    def set_DCV_source(self, volt: float) -> bool:
        self.inst.write(f"SOUR:VOLT {volt}")  # set voltage
        return self.check_error()

    def configure_measure_current(self) -> bool:
        self.inst.write("SENS:FUNC 'CURR'")  # mode
        self.inst.write("SENS:CURR:RANG:AUTO ON")  # range AUTO
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
            and self.configure_DCV_source(compliance)
            and self.configure_measure_current()
            and self.set_current_nplc(nplc)
        )

        if success:
            self._mode = Mode.IV
            self.logger.info("Configured for DC Voltage supply.")
        else:
            self._mode = Mode.UNCONFIGURED
            self.logger.info("Failed to configure DC Voltage supply.")
        return success

    def get_data_CURR(self):
        val = self.inst.query("READ?")

        try:
            return float(val)
        except Exception:
            self.logger.error(f"Got invalid read {val}")
            return None

    def get_data(self):
        if self._mode == Mode.IV:
            return self.get_data_CURR()
        else:
            self.logger.error("get_data() is called but not configured.")
            return None

    def get_unit(self) -> str:
        if self._mode == Mode.IV:
            return "A"
        else:
            self.logger.error("get_unit() is called but not configured.")
            return ""

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "data":
            return self.get_data()
        elif key == "unit":
            return self.get_unit()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        return ["IV_source", "IV"]

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
        else:
            return self.fail_with(f"unknown label {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label == "iv_source":
            return self.set_DCV_source(volt=params.get("volt", 0.0))

        elif label == "iv":
            return self.configure_IV(
                compliance=params.get("compliance", 105e-3),
                nplc=params.get("nplc", 10.0),
            )

        else:
            self.logger.error(f"unknown label: {label}")
            return False

    def start(self, label: str = "") -> bool:
        if self._mode == Mode.IV:
            return self.set_output(True)
        else:
            return self.fail_with("start() is called but not configured.")

    def stop(self, label: str = "") -> bool:
        if self._mode == Mode.IV:
            return self.set_output(False)
        else:
            return self.fail_with("stop() is called but not configured.")


class Keithley_6430(VisaInstrument):
    """6430Sub-Femtoamp Remote SourceMeter module."""

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 2_000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)
        self.volt_min, self.volt_max = self.conf.get("volt_bounds", (-210.0, 210.0))

    def configure_source_VOLT(self) -> bool:
        self.inst.write("SOUR:FUNC VOLT")  # source voltage mode
        self.inst.write("SOUR:VOLT:MODE FIX")  # fixed voltage source mode
        self.inst.write("SOUR:VOLT:RANG:AUTO 1")  # auto range enable
        self.inst.write("SOUR:VOLT:LEVEL 0.0")
        return self.check_error()

    def set_source_VOLT(self, n: float) -> bool:
        self.inst.write(f"SOUR:VOLT:LEVEL {n}")
        return self.check_error()

    def configure_measure_CURR(self, compliance: float) -> bool:
        self.inst.write("SENS:FUNC 'CURR'")  # measure cuurent mode
        self.inst.write(f"SENS:CURR:PROT {compliance}")  # compliance limit
        self.inst.write("SENS:CURR:RANG:AUTO 1")  # auto range enable
        return self.check_error()

    def set_CURR_nplc(self, n: float) -> bool:
        if not (0.01 <= n <= 10):
            return self.fail_with("NPLC must be between 0.01 and 10.")
        else:
            self.inst.write(f"SENS:CURR:NPLC {n}")
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
            and self.configure_source_VOLT()
            and self.configure_measure_CURR(compliance)
            and self.set_CURR_nplc(nplc)
            and self.set_auto_filter(filter)
        )
        if success:
            self._mode = Mode.IV
            self.logger.info("Configured for IV measurement.")
        else:
            self._mode = Mode.UNCONFIGURED
            self.logger.info("Failed to configure IV measurement.")
        return success

    def get_data_CURR(self):
        val = self.inst.query("READ?")
        val_spl = val.split(",")
        try:
            return float(val_spl[1])
        except Exception:
            self.logger.error(f"Got invalid read {val}")
            return None

    def get_data(self):
        if self._mode == Mode.IV:
            return self.get_data_CURR()
        else:
            self.logger.error("get_data() is called but not configured.")
            return None

    def get_unit(self) -> str:
        if self._mode == Mode.IV:
            return "A"
        else:
            self.logger.error("get_unit() is called but not configured.")
            return ""

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "data":
            return self.get_data()
        elif key == "unit":
            return self.get_unit()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        return ["IV_source", "IV"]

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
        else:
            return self.fail_with(f"unknown label {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label == "iv_source":
            return self.set_source_VOLT(volt=params.get("volt", 0.0))

        elif label == "iv":
            return self.configure_IV(
                compliance=params.get("compliance", 105e-3),
                nplc=params.get("nplc", 10.0),
                filter=params.get("filter", True),
            )

        else:
            self.logger.error(f"unknown label: {label}")
            return False

    def start(self, label: str = "") -> bool:
        if self._mode == Mode.IV:
            return self.set_output(True)
        else:  # UNCONFIGURED
            return self.fail_with("start() is called but not configured.")

    def stop(self, label: str = "") -> bool:
        if self._mode == Mode.IV:
            return self.set_output(False)
        else:  # UNCONFIGURED
            return self.fail_with("stop() is called but not configured.")
