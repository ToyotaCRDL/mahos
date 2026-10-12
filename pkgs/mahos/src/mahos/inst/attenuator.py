#!/usr/bin/env python3

"""
Programmable Attenuator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import sys
import os

from mahos.inst.instrument import Instrument
from mahos.msgs import param_msgs as P
from mahos.util.conf import ConfAccessorMixin


class MCL_USB_Attenuator(Instrument, ConfAccessorMixin):
    """Instrument for Mini-circuits USB-connected programmable attenuator.

    You need to install mcl_RUDAT_NET45.dll by Mini-circuits, pythonnet,
    and .NET framework 4.5 or later.
    Place the DLL in a location where pythonnet can find it, or specify its
    directory using ``dll_dir``.
    If Windows blocks the DLL because it was downloaded from the Internet,
    unblock it from the file's Properties dialog,
    or run ``Unblock-File path-to-the.dll`` in PowerShell.

    Current implementation supports single-channel devices only,
    and it is tested against RCDAT-6000-60.

    :param dll_dir: (default: "") The directory path containing the DLL.
    :type dll_dir: str
    :param serial: (default: "") Serial string to discriminate multiple devices.
        Blank is fine if only one device is connected.
    :type serial: str
    :param bounds: (default: (0.0, 60.0)) attenuation bounds.
        (lower, upper) bounds in dB.
    :type bounds: tuple[float, float]
    :param digit: (default: 5) Number of attenuation decimals for TweakerGUI use.
    :type digit: int
    :param step: (default: 0.25) Attenuation step in dB for TweakerGUI use.
    :type step: float

    """

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        dll_dir = self._conf_str("dll_dir", "")
        if dll_dir:
            sys.path.insert(0, os.path.expanduser(dll_dir))
        import clr

        clr.AddReference("mcl_RUDAT_NET45")
        from mcl_RUDAT_NET45 import USB_RUDAT

        self.lib = USB_RUDAT()
        devices = self.get_device_list()
        self.logger.debug("Available Devices: " + ", ".join(devices))

        if not devices:
            self.logger.error("No device detected.")
            raise ValueError("No device detected.")
        if len(devices) == 1:
            if "serial" in self.conf and self.conf["serial"] != devices[0]:
                self.logger.warn(
                    "Given serial {} looks wrong. Opening available one {} anyway.".format(
                        self.conf["serial"], devices[0]
                    )
                )
            self.serial = devices[0]
        else:
            if "serial" not in self.conf:
                msg = "Must specify conf['serial'] as multiple devices are detected."
                msg += "\nAvailable serials: " + ", ".join(devices)
                self.logger.error(msg)
                raise ValueError(msg)
            if self.conf["serial"] not in devices:
                msg = "Specified serial {} is not available. (not in ({}))".format(
                    self.conf["serial"], ", ".join(devices)
                )
                self.logger.error(msg)
                raise ValueError(msg)
            self.serial = self.conf["serial"]

        if not self._connect(self.serial):
            raise RuntimeError(f"Failed to connect to {self.serial}")
        name = self.get_model_name()
        self.logger.info(f"Connected to {name} ({self.serial})")
        self.bounds = self._conf_ascending_numbers("bounds", 2, (0.0, 60.0))
        self.digit = self._conf_nonneg_int("digit", 5)
        self.step = self._conf_pos_num("step", 0.25)

    def get_device_list(self) -> list[str]:
        """Get list of connected device serial numbers."""

        ret, val = self.lib.Get_Available_SN_List("")
        if ret == 1:
            return val.split(" ")
        self.logger.error(f"Failed Get_Available_SN_List: {ret}, {val}")
        return []

    def _connect(self, serial: str) -> bool:
        """Connect to the device using serial number."""

        ret, _ = self.lib.Connect(serial)
        if ret == 1:
            return True
        elif ret == 2:
            self.logger.warn(f"Already connected to {serial}")
            return True
        return self.fail_with(f"Failed to connect to {serial}. ret code: {ret}")

    def get_model_name(self) -> str:
        ret, val = self.lib.Read_ModelName("")
        if ret == 1:
            return str(val)
        self.logger.error(f"Failed Read_ModelName: {ret}, {val}")
        return ""

    def get_attenuation(self) -> float:
        ret, val = self.lib.Read_Att(0.0)
        if ret == 1:
            return float(val)
        self.logger.error(f"Failed Read_Att: {ret}, {val}")
        return float("nan")

    def set_attenuation(self, att_dB: float) -> bool:
        if not (self.bounds[0] <= att_dB <= self.bounds[1]):
            return self.fail_with(f"Attenuation {att_dB} dB is out of bounds {self.bounds}")
        ret = self.lib.SetAttenuation(att_dB)
        if ret == 1:
            return True
        self.logger.error(f"Failed SetAttenuation: {ret}")

    def close_resources(self):
        if hasattr(self, "lib"):
            self.lib.Disconnect()
            self.logger.info(f"Disconnected from {self.serial}")

    # Standard API

    def get_param_dict_labels(self) -> list[str]:
        return [""]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        return P.ParamDict(
            attenuation=P.FloatParam(
                self.get_attenuation(),
                self.bounds[0],
                self.bounds[1],
                unit="dB",
                digit=self.digit,
                step=self.step,
                doc="attenuation value",
            ),
        )

    def configure(self, params: dict, label: str = "") -> bool:
        params = P.unwrap(params)
        if "attenuation" not in params:
            return self.fail_with("must contain attenuation")
        return self.set_attenuation(params["attenuation"])

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "attenuation":
            return self.set_attenuation(value)
        else:
            return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None, label: str = ""):
        if key == "attenuation":
            return self.get_attenuation()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
