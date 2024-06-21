#!/usr/bin/env python3

"""
Digital MultiMeter module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import enum
import time

from ..msgs import param_msgs as P
from .visa_instrument import VisaInstrument


class Mode(enum.Enum):
    UNCONFIGURED = 0
    DCV = 1
    DCI = 2
    TEMP = 3


class ADC_7352E(VisaInstrument):
    """ADC 7352E Digital MultiMeter."""

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\r\n"
        conf["read_termination"] = "\r\n"
        if "timeout" not in conf:
            conf["timeout"] = 2_000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        # Continuous measurement will send us data without any query.
        # So reset and disable continuous measurement here.
        self.rst()
        time.sleep(1e-3)
        self.inst.write("INIC0")

        self._mode = {1: Mode.UNCONFIGURED, 2: Mode.UNCONFIGURED}

    def query_error(self):
        return self.inst.query("ERR?")

    def wait_opc(self) -> bool:
        for _ in range(25):
            if self.query_opc():
                return True
            time.sleep(0.2)

        return self.fail_with("OPC status cannot be reached.")

    def wait_opc_and_check_error(self) -> bool:
        # sending sequential query commands without wait can cause error.
        # sleep and wait for OPC to avoid this.

        time.sleep(20e-3)
        success = self.wait_opc()
        time.sleep(20e-3)
        success &= self.check_error()
        return success

    def set_DCV_auto(self, ch: int = 1) -> bool:
        code = 1 if ch == 1 else 12
        self.inst.write(f"DSP{ch},F{code},R0,AZ1")  # dcv, auto range enable, auto zero enable
        return self.wait_opc_and_check_error()

    def set_DCI_auto(self, ch: int = 1) -> bool:
        if ch == 1:
            self.inst.write("DSP1,F5,R0,AZ1")  # dci at chA, auto range enable, auto zero enable
        else:  # ch == 2
            self.inst.write("DSP2,F35,R8,AZ1")  # dci at chB, range is fixed, auto zero enable
        return self.wait_opc_and_check_error()

    def set_TEMP_auto(self, ch: int = 1) -> bool:
        self.inst.write(f"DSP{ch},F40,AZ1")  # temp, auto zero enable
        return self.wait_opc_and_check_error()

    def set_sampling_rate(self, sampling_rate: int) -> bool:
        if sampling_rate not in (1, 2, 3, 4):
            return self.fail_with("Sampling rate must be one of 1, 2, 3, 4.")

        self.inst.write(f"PR{sampling_rate}")
        return self.wait_opc_and_check_error()

    def configure_DCV(self, sampling_rate: int = 3, ch: int = 1) -> bool:
        """Setup DC Voltage (on-demand) measurement."""

        success = self.set_DCV_auto(ch) and self.set_sampling_rate(sampling_rate)

        if success:
            self._mode[ch] = Mode.DCV
            self.logger.info(f"Configured ch{ch} for DC Voltage measurement.")
        else:
            self._mode[ch] = Mode.UNCONFIGURED
            self.logger.info("Failed to configure DC Voltage measurement.")
        return success

    def configure_DCI(self, sampling_rate: int = 3, ch: int = 1) -> bool:
        """Setup DC Current (on-demand) measurement."""

        success = self.set_DCI_auto(ch) and self.set_sampling_rate(sampling_rate)

        if success:
            self._mode[ch] = Mode.DCI
            self.logger.info(f"Configured ch{ch} for DC Current measurement.")
        else:
            self._mode[ch] = Mode.UNCONFIGURED
            self.logger.info("Failed to configure DC Current measurement.")
        return success

    def configure_TEMP(self, sampling_rate: int = 3, ch: int = 1) -> bool:
        """Setup Temperature (on-demand) measurement."""

        success = self.set_TEMP_auto(ch) and self.set_sampling_rate(sampling_rate)

        if success:
            self._mode[ch] = Mode.TEMP
            self.logger.info(f"Configured ch{ch} for Temperature measurement.")
        else:
            self._mode[ch] = Mode.UNCONFIGURED
            self.logger.info("Failed to configure Temperature measurement.")
        return success

    def get_data(self, ch: int = 1):
        if self._mode[ch] == Mode.UNCONFIGURED:
            self.logger.error("get_data() is called but not configured.")
            return None

        val = self.inst.query("INI")
        val_spl = val.split(",")
        try:
            return float(val_spl[ch - 1][5:])
        except Exception:
            self.logger.error(f"Got invalid read {val}")
            return None

    def get_unit(self, ch: int = 1) -> str:
        if self._mode[ch] == Mode.DCV:
            return "V"

        if self._mode[ch] == Mode.DCI:
            return "A"

        if self._mode[ch] == Mode.TEMP:
            return "℃"

        else:
            self.logger.error("get_unit() is called but not configured.")
            return ""

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "data":
            return self.get_data(int(label[2]))
        elif key == "unit":
            return self.get_unit(int(label[2]))
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        return ["ch1_dcv", "ch1_dci", "ch1_temp", "ch2_dcv", "ch2_dci", "ch2_temp"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        if label in self.get_param_dict_labels():
            return P.ParamDict(
                sampling_rate=P.IntChoiceParam(
                    3, [1, 2, 3, 4], doc="small (large) value for fast (slow) sampling"
                ),
            )
        else:
            return self.fail_with(f"unknown label {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label in ["ch1_dcv", "ch2_dcv"]:
            return self.configure_DCV(
                sampling_rate=params.get("sampling_rate", 3), ch=int(label[2])
            )
        if label in ["ch1_dci", "ch2_dci"]:
            return self.configure_DCI(
                sampling_rate=params.get("sampling_rate", 3), ch=int(label[2])
            )
        if label in ["ch1_temp", "ch2_temp"]:
            return self.configure_TEMP(
                sampling_rate=params.get("sampling_rate", 3), ch=int(label[2])
            )
        else:
            self.logger.error(f"unknown label: {label}")
            return False

    def start(self, label: str = "") -> bool:
        if self._mode[int(label[2])] == Mode.UNCONFIGURED:
            return self.fail_with("start() is called but not configured.")
        else:
            return True

    def stop(self, label: str = "") -> bool:
        if self._mode[int(label[2])] == Mode.UNCONFIGURED:
            return self.fail_with("stop() is called but not configured.")
        else:
            return True


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
