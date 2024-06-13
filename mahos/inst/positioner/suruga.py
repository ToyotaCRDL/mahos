#!/usr/bin/env python3

"""
Suruga part of Positioner module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from ..visa_instrument import VisaInstrument
from ...msgs import param_msgs as P


class Suruga_DS102(VisaInstrument):
    """Instrument for Suruga DS102 stepping motor controller.

    :param axis: (default: 1) Axis identifier to control.
    :type axis: int
    :param homed: (default: False) Since the controller don't memorize whether homing (zeroing)
        has been performed, this class assumes non-homed initial state by default.
        By setting this True, already-homed state is assumed on start.
        Be careful setting this True because the position won't be reproduced if the controller
        hasn't actually performed zeroing yet.
    :type homed: bool
    :param range: (required) travel range.
        (lower, upper) bounds of the position.
    :type range: tuple[float, float]

    """

    def __init__(self, name, conf=None, prefix=None):
        conf["write_termination"] = "\r"
        conf["read_termination"] = "\r"
        conf["baud_rate"] = 38400
        if "timeout" not in conf:
            conf["timeout"] = 1_000.0
        VisaInstrument.__init__(self, name, conf, prefix=prefix)
        self.check_required_conf("range")
        self.range = self.conf["range"]
        self.logger.info(f"range: {self.range[0]:.3f} {self.range[1]:.3f}")
        self.axis = self.conf.get("axis", 1)
        self.homed = self.conf.get("homed", False)
        self.target = self.get_pos()

    def home(self) -> bool:
        self.inst.write(f"AXI{self.axis}:GO ORG")
        self.homed = True
        self.target = 0.0
        return True

    def move(self, pos: float) -> bool:
        if not self.is_homed():
            return self.fail_with("Cannot move because this device has not been homed yet.")
        if pos < self.range[0] or pos > self.range[1]:
            return self.fail_with(f"Target pos {pos:.7f} is out of range {self.range}.")
        self.inst.write(f"AXI{self.axis}:GOABSolute {pos:.7f}")
        self.target = pos
        return True

    def get_pos(self) -> float:
        return float(self.inst.query(f"AXI{self.axis}:POS?"))

    def get_target(self) -> float:
        return self.target

    def is_moving(self) -> bool:
        ret = self.inst.query(f"AXI{self.axis}:MOTION?")
        if ret == "1":
            return True
        elif ret == "0":
            return False
        self.logger.error(f"Unexpected answer AXI{self.axis}:MOTION? -> {ret}")
        return False

    def get_status(self) -> dict:
        return {
            "homed": self.is_homed(),
            "moving": self.is_moving(),
        }

    def get_all(self) -> dict[str, [float, bool]]:
        """Get all important info about this device packed in a dict.

        :returns pos: current position.
        :returns target: target position.
        :returns range: travel range.
        :returns homed: True if device is homed.
        :returns moving: True if device is moving.

        """

        d = self.get_status()
        d["pos"] = self.get_pos()
        d["target"] = self.get_target()
        d["range"] = self.get_range()

        return d

    def get_range(self) -> tuple[float, float]:
        return self.range

    def is_homed(self) -> bool:
        return self.homed

    # Standard API

    def stop(self, label: str = "") -> bool:
        """Stop motion of this device."""

        self.inst.write(f"AXI{self.axis}:STOP")
        return True

    def reset(self, label: str = "") -> bool:
        """Perform homing of this device."""

        return self.home()

    def get_param_dict_labels(self) -> list[str]:
        return ["pos"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        if label == "pos":
            return P.ParamDict(
                target=P.FloatParam(
                    self.get_target(), self.range[0], self.range[1], doc="target position"
                )
            )
        else:
            self.logger.error(f"Unknown label: {label}")
            return None

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "pos":
            return self.move(params["target"])
        else:
            return self.fail_with(f"Unknown label {label}")

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "target":
            return self.move(value)
        else:
            return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None, label: str = ""):
        if key == "all":
            return self.get_all()
        elif key == "pos":
            return self.get_pos()
        elif key == "target":
            return self.get_target()
        elif key == "status":
            return self.get_status()
        elif key == "range":
            return self.get_range()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
