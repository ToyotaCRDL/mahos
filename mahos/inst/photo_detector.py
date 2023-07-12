#!/usr/bin/env python3

"""
Photo Detector module.

.. This file is a part of Mahos project.

"""

from __future__ import annotations
import os
import ctypes

from .instrument import Instrument


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
