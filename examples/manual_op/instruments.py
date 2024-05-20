#!/usr/bin/env python3

from __future__ import annotations
import enum

import numpy as np

from mahos.inst.instrument import Instrument
from mahos.msgs import param_msgs as P


class VoltageSource_mock(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(["resource"])
        resource = self.conf["resource"]
        self.logger.info(f"Open VoltageSource at {resource}.")

    def set_output(self, on: bool, ch: str) -> bool:
        self.logger.info("Set output " + ch + (" on" if on else " off"))
        return True

    def configure_voltage(self, volt: float, ch: str) -> bool:
        self.logger.debug(f"Dummy voltage at {ch} = {volt:.3f} V")
        return True

    # Standard API

    def start(self, label: str = "") -> bool:
        return self.set_output(True, label)

    def stop(self, label: str = "") -> bool:
        return self.set_output(False, label)

    def get_param_dict_labels(self) -> list[str]:
        return ["ch1", "ch2"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        if label in ("ch1", "ch2"):
            return P.ParamDict(
                voltage=P.FloatParam(
                    0.0, -10.0, 10.0, SI_prefix=True, unit="V", doc="voltage value"
                )
            )
        else:
            self.logger.error(f"invalid label {label}")
            return None

    def configure(self, params: dict, label: str = "") -> bool:
        if label in ("ch1", "ch2"):
            return self.configure_voltage(params.get("voltage", 0.0), label)
        else:
            self.logger.error(f"invalid label {label}")
            return False


class Multimeter_mock(Instrument):
    class Mode(enum.Enum):
        UNCONFIGURED = 0
        VOLTAGE = 1
        CURRENT = 2

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(["resource"])
        resource = self.conf["resource"]
        self.logger.info(f"Open Multimeter at {resource}.")
        self._mode = self.Mode.UNCONFIGURED

    def configure_current_meas(self, range_=0, navg=1) -> bool:
        self._mode = self.Mode.CURRENT
        # reset mock states
        self._rng = np.random.default_rng()

        self.logger.info(f"Dummy config for current. range: {range_:s} navg: {navg:d}")
        return True

    def configure_voltage_meas(self, range_=0, navg=1) -> bool:
        self._mode = self.Mode.VOLTAGE
        # reset mock states
        self._rng = np.random.default_rng()

        self.logger.info(f"Dummy config for voltage. range: {range_:s} navg: {navg:d}")
        return True

    def get_data(self) -> float:
        return self._rng.normal()

    def get_unit(self) -> str:
        if self._mode == self.Mode.CURRENT:
            return "A"
        elif self._mode == self.Mode.VOLTAGE:
            return "V"
        else:  # self.Mode.UNCONFIGURED
            return ""

    # Standard API

    def start(self, label: str = "") -> bool:
        self.logger.info("Started measurement.")
        return True

    def stop(self, label: str = "") -> bool:
        self.logger.info("Stopped measurement.")
        return True

    def get_param_dict_labels(self) -> list[str]:
        return ["voltage", "current"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        if label == "voltage":
            return P.ParamDict(
                range=P.StrChoiceParam(
                    "auto", ("auto", "1uV", "1mV", "1V"), doc="measurement range."
                ),
                navg=P.IntParam(1, 1, 1000, doc="number of samples for averaging."),
            )
        elif label == "current":
            return P.ParamDict(
                range=P.StrChoiceParam(
                    "auto", ("auto", "1uA", "1mA", "1A"), doc="measurement range."
                ),
                navg=P.IntParam(1, 1, 1000, doc="number of samples for averaging."),
            )
        else:
            self.logger.error(f"invalid label {label}")
            return None

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "current":
            return self.configure_current_meas(params.get("range", "auto"), params.get("navg", 1))
        elif label == "voltage":
            return self.configure_voltage_meas(params.get("range", "auto"), params.get("navg", 1))
        else:
            self.logger.error(f"Unknown label: {label}")
            return False

    def get(self, key: str, args=None, label: str = ""):
        if key == "data":
            return self.get_data()
        elif key == "unit":
            return self.get_unit()
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None
