#!/usr/bin/env python3

"""
Digital MultiMeter module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import enum

from ..msgs import param_msgs as P
from .visa_instrument import VisaInstrument


class Mode(enum.Enum):
    UNCONFIGURED = 0
    DCV = 1


class Agilent34410A(VisaInstrument):
    """Agilent 34410A Digital MultiMeter module."""

    TRIG_SOURCE = (
        "IMM",
        "IMMEDIATE",
        "BUS",
        "EXT",
        "EXTERNAL",
    )

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 10_000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

    def set_trig_source(self, source: str) -> bool:
        """Set the trigger source."""

        source = source.upper()
        if source not in self.TRIG_SOURCE:
            self.logger.error("invalid trigger source.")
            return False

        self.inst.write("TRIG:SOUR " + source)
        return True

    def set_DCV_auto(self) -> bool:
        self.inst.write("SENS:FUNC 'VOLT:DC'")  # dcv mode
        self.inst.write("SENS:VOLT:DC:RANG:AUTO 1")  # auto range enable
        self.inst.write("SENS:VOLT:DC:ZERO:AUTO 1")  # auto zero enable
        return True

    def set_DCV_nplc(self, n: int) -> bool:
        if n not in (1, 2, 10, 100):
            return self.fail_with("NPLC must be one of 1, 2, 10, 100.")

        self.inst.write(f"SENS:VOLT:DC:NPLC {n}")
        return True

    def configure_DCV(self, nplc: int = 10, trigger: str = "IMM") -> bool:
        """Setup DC Voltage (on-demand) measurement."""

        success = (
            self.rst_cls()
            and self.set_DCV_auto()
            and self.set_DCV_nplc(nplc)
            and self.set_trig_source(trigger)
        )
        if success:
            self._mode = Mode.DCV
            self.logger.info("Configured for DC Voltage measurement.")
        else:
            self._mode = Mode.UNCONFIGURED
            self.logger.info("Failed to configure DC Voltage measurement.")
        return success

    def get_data_DCV(self):
        self.inst.write("INIT")
        val = self.inst.query("FETC?")
        try:
            return float(val)
        except Exception:
            self.logger.error(f"Got invalid read {val}")
            return None

    def get_data(self):
        if self._mode == Mode.DCV:
            return self.get_data_DCV()
        else:
            self.logger.error("get_data() is called but not configured.")
            return None

    def get_unit(self) -> str:
        if self._mode == Mode.DCV:
            return "V"
        else:
            self.logger.error("get_unit() is called but not configured.")
            return ""

    # Standard API

    def get(self, key: str, args=None):
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
        return ["dcv"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        if label == "dcv":
            return P.ParamDict(
                nplc=P.IntChoiceParam(10, [1, 2, 10, 100]),
                trigger=P.StrChoiceParam("IMM", ["IMM", "BUS", "EXT"]),
            )
        else:
            return self.fail_with(f"unknown label {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label == "dcv":
            return self.configure_DCV(
                nplc=params.get("nplc", 10), trigger=params.get("trigger", "IMM")
            )
        else:
            self.logger.error(f"unknown label: {label}")
            return False

    def start(self, label: str = "") -> bool:
        if self._mode == Mode.DCV:
            return True
        else:  # UNCONFIGURED
            return self.fail_with("start() is called but not configured.")

    def stop(self, label: str = "") -> bool:
        if self._mode == Mode.DCV:
            return True
        else:  # UNCONFIGURED
            return self.fail_with("stop() is called but not configured.")
