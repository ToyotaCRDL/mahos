#!/usr/bin/env python3

import numpy as np

from mahos.inst.instrument import Instrument
from mahos.inst.interface import InstrumentInterface


class VoltageSource_mock(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(["resource"])
        resource = self.conf["resource"]
        self.logger.info(f"Open VoltageSource at {resource}.")

    def set_output(self, on: bool) -> bool:
        self.logger.info("Set output " + ("on" if on else "off"))
        return True

    def set_voltage(self, volt: float) -> bool:
        self.logger.debug(f"Dummy voltage {volt:.3f} V")
        return True

    # Standard API

    def start(self) -> bool:
        return self.set_output(True)

    def stop(self) -> bool:
        return self.set_output(False)

    def set(self, key: str, value=None) -> bool:
        if key == "volt":
            return self.set_voltage(value)
        else:
            self.logger.error(f"Unknown set() key: {key}")
            return False


class Multimeter_mock(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(["resource"])
        resource = self.conf["resource"]
        self.logger.info(f"Open Multimeter at {resource}.")

    def configure_current_meas(self, range_=0, navg=1) -> bool:
        # reset mock states
        self._rng = np.random.default_rng()

        self.logger.info(f"Dummy config range: {range_:d} navg: {navg:d}")
        return True

    def get_meas(self, volt: float = 0.0) -> float:
        R = 50.0
        I = volt / R
        return I + self._rng.normal(scale=1.0 / R)

    # Standard API

    def configure(self, params: dict, label: str = "") -> bool:
        mode = params.get("mode")
        if mode == "current":
            return self.configure_current_meas(params.get("range", 0), params.get("navg", 1))
        else:
            self.logger.error(f"Unknown mode: {mode}")
            return False

    def get(self, key: str, args=None):
        if key == "meas":
            return self.get_meas(args)
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None


class VoltageSourceInterface(InstrumentInterface):
    def set_voltage(self, volt: float) -> bool:
        """Set the output voltage."""

        return self.set("volt", volt)


class MultimeterInterface(InstrumentInterface):
    def configure_current_meas(self, range_: int = 0, navg: int = 1) -> bool:
        """Configure current measurement.

        :param range: (dummy parameter for illustration) measurement range.
        :param navg: (dummy parameter for illustration) number of averaging.

        """

        return self.configure({"mode": "current", "range": range_, "navg": navg})

    def get_meas(self, volt: float = 0.0) -> float:
        """Get measured value.

        volt is voltage for dummy value generation (will be removed for real instruments).

        """

        return self.get("meas", args=volt)
