#!/usr/bin/env python3

"""
Photo Detector module.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import os
import ctypes

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .instrument import Instrument
from .daq import AnalogIn, BufferedEdgeCounter


class APDCounter(BufferedEdgeCounter):
    """BufferedEdgeCounter for counting APD output pulses.

    :param corr_x_kcps: x-data of correction factor.
    :type corr_x_kcps: np.ndarray
    :param corr_y: y-data of correction factor.
    :type corr_y: np.ndarray
    :param dark_cps: (default 0.0) dark count.
    :type dark_cps: float

    """

    def __init__(self, name, conf, prefix=None):
        BufferedEdgeCounter.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("corr_x_kcps", "corr_y"))

        self._spline = InterpolatedUnivariateSpline(
            1e3 * np.array(self.conf["corr_x_kcps"]), np.array(self.conf["corr_y"])
        )
        self._dark_count = self.conf.get("dark_cps", 0.0)

    def correction_factor(self, cnt):
        if cnt < 1e5:
            return 1.0
        else:
            return self._spline(cnt)

    def correct_cps(self, cps):
        if isinstance(cps, (np.ndarray, list, tuple)):
            cf = np.array([self.correction_factor(c) for c in cps])
        else:
            cf = self.correction_factor(cps)

        return cps * cf - self._dark_count

    def configure(self, params: dict) -> bool:
        if "time_window" not in params:
            self.logger.error("config must be given: time_window.")
            return False
        self.time_window = params["time_window"]
        self.average = params.get("average", False)

        return BufferedEdgeCounter.configure(self, params)

    def _append_data(self, data: np.ndarray):
        BufferedEdgeCounter._append_data(self, self._convert(data))

    def _convert(self, data: np.ndarray):
        """Convert count to cps value (divide by time window) and apply correction."""

        if self.average:  # TODO: What's the meaning of average mode?
            return self.correct_cps(data.mean() / self.time_window)
        else:
            return self.correct_cps(data.astype(np.float64) / self.time_window)

    # Standard API

    def get(self, key: str, args=None):
        if key == "correct":
            return np.array([self.correct_cps(x) for x in args])
        elif key == "correction_factor":
            return np.array([self.correction_factor(x) for x in args])
        elif key == "unit":
            return "cps"
        else:
            return BufferedEdgeCounter.get(self, key, args)


class LUCI10(Instrument):
    """Wrapper for FEMTO Messtechnik LUCI-10 DLL.

    You need following:
    1. Install LUCI-10 software (you can skip LabVIEW driver installation).
    2. Place C DLL (C:\\Program Files (x86)\\FEMTO\\LUCI-10\\Driver\\LUCI_10_x64.dll) somewhere.

    :param dll_path: The path to the directory containing DLL.
    :param index: (default: 1) Index of LUCI10 device.
    :type index: int

    """

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        if "dll_path" in self.conf:
            p = os.path.expanduser(self.conf["dll_path"])
            os.environ["PATH"] += os.pathsep + p
            os.add_dll_directory(p)

        self.index = self.conf.get("index", 1)

        self.dll = ctypes.cdll.LoadLibrary("LUCI_10_x64.dll")

        # We must get_devices() here for successful subsequent operation.
        devices = self.get_devices()
        self.logger.info(f"Opened LUCI10 {self.index} of {devices}")

    def _check_ret_value(self, ret: int) -> bool:
        if ret == 0:
            return True
        elif ret == -1:
            self.logger.error(f"LUCI10 {self.index} is not found.")
            return False
        elif ret == -2:
            self.logger.error(f"LUCI10 {self.index} is not responding.")
            return False

    def get_devices(self) -> int:
        return self.dll.EnumerateUsbDevices()

    def set_led(self, on: bool) -> bool:
        if on:
            ret = self.dll.LedOn(self.index)
        else:
            ret = self.dll.LedOff(self.index)

        return self._check_ret_value(ret)

    def get_id(self) -> int | None:
        id_ = ctypes.c_int()
        ret = self.dll.ReadAdapterID(self.index, ctypes.byref(id_))
        if self._check_ret_value(ret):
            return id_.value

    def get_pin(self, pin: int) -> bool | None:
        """Get pin status. True (False) is TTL High (Low).

        :param pin: pin number. one of 5, 6, 7.
        :type pin: int

        """

        s = ctypes.c_int()
        if pin == 5:
            ret = self.dll.GetStatusPin5(self.index, ctypes.byref(s))
        elif pin == 6:
            ret = self.dll.GetStatusPin6(self.index, ctypes.byref(s))
        elif pin == 7:
            ret = self.dll.GetStatusPin7(self.index, ctypes.byref(s))
        else:
            self.logger.error(f"invalid pin: {pin} (must be one of 5, 6, 7)")
            return None

        if self._check_ret_value(ret):
            return bool(s.value)

    def get_product_string(self) -> str:
        size = 64
        s = ctypes.create_string_buffer(size)
        ret = self.dll.GetProductString(self.index, ctypes.byref(s), size)
        if self._check_ret_value(ret):
            return s.value.decode()
        else:
            return ""

    def write_data(self, data: int) -> bool:
        """write data (output pin values).

        output data is 16bit corresponding to D-sub pins 10-25.
        (lower byte is D-sub pins 10-17, higher byte is D-sub pins 18-25.)

        """

        low = data & 0xFF
        high = (data & 0xFF00) >> 8
        ret = self.dll.WriteData(self.index, low, high)
        return self._check_ret_value(ret)

    # Standard API

    def set(self, key: str, value=None) -> bool:
        key = key.lower()
        if key == "led":
            return self.set_led(value)
        elif key == "data":
            return self.write_data(value)
        else:
            return self.fail_with(f"Unknown set() key: {key}.")

    def get(self, key: str, args=None):
        if key == "devices":
            return self.get_devices()
        elif key == "id":
            return self.get_id()
        elif key == "pin":
            return self.get_pin(args)
        elif key == "product":
            return self.get_product_string()
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None


class OE200(AnalogIn):
    """FEMTO Messtechnik OE-200 Variable Gain Photoreceiver.

    This class requires LUCI-10 installation (see LUCI10 class),
    and a NI-DAQ Analog Input port (see daq.AnalogIn class).

    :param line: DAQ's physical channel for AnalogIn.
    :type line: str

    :param dll_path: The path to the directory containing LUCI-10 DLL.
    :type dll_path: str
    :param index: (default: 1) Index of LUCI-10 device.
    :type index: int

    :param DC_coupling: (default: False) Init setting. True (False) for DC (AC) coupling.
    :type DC_coupling: bool
    :param low_noise: (default: True) Init setting. True (False) for low_noise (high_speed) mode.
    :type low_noise: bool
    :param gain_exponent: (default: 3) Init setting. The gain exponent.
    :type gain_exponent: int

    """

    GAINS = {
        True: {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6},
        False: {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6},
    }

    def __init__(self, name, conf, prefix=None):
        if "line" in conf and "lines" not in conf:
            conf["lines"] = [conf["line"]]
        AnalogIn.__init__(self, name, conf=conf, prefix=prefix)

        self.luci = LUCI10(name + "_luci", self.conf, prefix=prefix)

        self.DC_coupling = self.conf.get("DC_coupling", False)
        low_noise = self.conf.get("low_noise", True)
        gain_exponent = self.conf.get("gain_exponent", 3)
        if not self.set_gain(low_noise, gain_exponent):
            raise ValueError("Initial setting failed.")

    def _write_settings(self) -> bool:
        if self.DC_coupling:
            to_write = self.gain_value | 0b1000
        else:
            to_write = self.gain_value

        self.logger.debug(f"Writing LUCI-10 setting: 0b{to_write:b}")
        return self.luci.write_data(to_write)

    def _convert(self, data: np.ndarray | float) -> np.ndarray | float:
        """Convert raw reading (V) to power (W)."""

        return data / self.gain

    # override AnalogIn methods to convert readings.

    def _append_data(self, data: np.ndarray):
        AnalogIn._append_data(self, self._convert(data))

    def read_on_demand(self, oversample: int = 1) -> float | np.ndarray:
        return self._convert(AnalogIn.read_on_demand(self, oversample))

    def _update_gain(self, low_noise: bool, gain_exponent: int) -> bool:
        self.low_noise = low_noise
        self.gain_exponent = gain_exponent

        try:
            self.gain_value = self.GAINS[low_noise][gain_exponent]
        except KeyError:
            return self.fail_with(f"Invalid arguments: {low_noise}, {gain_exponent}")

        if low_noise:
            self.gain_value |= 0b1_0000

        self.gain = 10**gain_exponent
        return True

    def set_gain(self, low_noise: bool, gain_exponent: int) -> bool:
        return self._update_gain(low_noise, gain_exponent) and self._write_settings()

    def set_coupling(self, DC_coupling: bool) -> bool:
        self.DC_coupling = DC_coupling
        return self._write_settings()

    def set_gain_coupling(self, low_noise: bool, gain_exponent: int, DC_coupling: bool) -> bool:
        self.DC_coupling = DC_coupling
        return self._update_gain(low_noise, gain_exponent) and self._write_settings()

    def get_gain_coupling(self) -> dict:
        return {
            "low_noise": self.low_noise,
            "gain_exponent": self.gain_exponent,
            "DC_coupling": self.DC_coupling,
        }

    def set(self, key: str, value=None) -> bool:
        key = key.lower()
        if key == "led":
            return self.luci.set_led(value)
        elif key == "gain":
            try:
                return self.set_gain(value["low_noise"], value["gain_exponent"])
            except (KeyError, TypeError):
                return self.fail_with("value must be a dict with low_noise and gain_exponent.")
        elif key == "coupling":
            return self.set_coupling(bool(value))
        elif key == "gain_coupling":
            try:
                return self.set_gain_coupling(
                    value["low_noise"], value["gain_exponent"], value["DC_coupling"]
                )
            except (KeyError, TypeError):
                msg = "value must be a dict with low_noise, gain_exponent, and DC_coupling."
                return self.fail_with(msg)
        else:
            return self.fail_with(f"Unknown set() key: {key}.")

    def get(self, key: str, args=None):
        if key in ("devices", "id", "pin", "product"):
            return self.luci.get(key, args)
        if key in ("data", "all_data"):
            return AnalogIn.get(self, key, args)
        if key == "unit":
            return "W"
        elif key == "gain_coupling":
            return self.get_gain_coupling()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
