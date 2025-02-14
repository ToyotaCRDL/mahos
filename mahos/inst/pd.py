#!/usr/bin/env python3

"""
Photo Detector module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os
import ctypes

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .instrument import Instrument
from .daq import AnalogIn, BufferedEdgeCounter
from ..msgs import param_msgs as P


class APDCounter(BufferedEdgeCounter):
    """BufferedEdgeCounter for counting APD output pulses.

    :param counter: The device name for counter (like /Dev1/Ctr0).
    :type counter: str
    :param source: The pin name for counter source (like /Dev1/PFI0).
    :type source: str
    :param source_dir: (default: True) Source direction. True (False) for rising (falling) edge.
    :type source_dir: bool
    :param queue_size: (default: 10000) Software buffer (queue) size.
    :type queue_size: int

    :param corr_x_kcps: x-data of correction factor.
    :type corr_x_kcps: np.ndarray
    :param corr_y: y-data of correction factor.
    :type corr_y: np.ndarray
    :param dark_cps: (default 0.0) Module dark count rate.
    :type dark_cps: float
    :param dead_time_ns: (default 0.0) Module dead time in ns.
        If given, correction factor is calculated using this.
        And then corr_x_kcps and corr_y are ignored.
    :type dead_time_ns: float

    """

    def __init__(self, name, conf, prefix=None):
        BufferedEdgeCounter.__init__(self, name, conf=conf, prefix=prefix)

        if "dead_time_ns" in self.conf:
            self._dead_time = self.conf.get("dead_time_ns", 0.0) * 1e-9

            if "corr_x_kcps" in self.conf or "corr_y" in self.conf:
                self.logger.warn("corr_x_kcps or corr_y is ignored when dead_time_ns is given.")
            self._spline = None
        else:
            self.check_required_conf(("corr_x_kcps", "corr_y"))
            self._spline = InterpolatedUnivariateSpline(
                1e3 * np.array(self.conf["corr_x_kcps"]), np.array(self.conf["corr_y"])
            )

            if "dead_time_ns" in self.conf:
                self.logger.warn("dead_time_ns is ignored when corr_x_kcps and corr_y are given.")
            self._dead_time = None

        self._dark_count = self.conf.get("dark_cps", 0.0)

    def _correction_factor_dead_time(self, cps):
        return 1.0 / (1.0 - cps * self._dead_time)

    def _correction_factor_spline(self, cps):
        if cps < 1e5:
            return 1.0
        else:
            return self._spline(cps)

    def correction_factor(self, cps):
        if self._dead_time is not None:
            return self._correction_factor_dead_time(cps)
        else:
            return self._correction_factor_spline(cps)

    def correct_cps(self, cps):
        if isinstance(cps, (np.ndarray, list, tuple)):
            cf = np.array([self.correction_factor(c) for c in cps])
        else:
            cf = self.correction_factor(cps)

        return cps * cf - self._dark_count

    def configure(self, params: dict, label: str = "") -> bool:
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

    def get(self, key: str, args=None, label: str = ""):
        if key == "correct":
            return np.array([self.correct_cps(x) for x in args])
        elif key == "correction_factor":
            return np.array([self.correction_factor(x) for x in args])
        elif key == "unit":
            return "cps"
        else:
            return BufferedEdgeCounter.get(self, key, args)


class AnalogPD(AnalogIn):
    """Generic Photo Detector based on DAQ AnalogIn with fixed amplifier gain and unit.

    :param line: DAQ's physical channel for AnalogIn.
    :type line: str
    :param buffer_size: (default: 10000) Software buffer (queue) size.
    :type buffer_size: int

    :param unit: (default: V) unit after conversion.
    :type unit: str
    :param gain: (default: 1.0) the fixed gain in [unit] / V.
        Example) when a transimpedance amp with 1000 V / A is used for a photo diode and
        the unit is 'A', gain should be set 1000.
    :type gain: float

    """

    def __init__(self, name, conf, prefix=None):
        if "line" in conf and "lines" not in conf:
            conf["lines"] = [conf["line"]]
        AnalogIn.__init__(self, name, conf=conf, prefix=prefix)

        self.gain = self.conf.get("gain", 1.0)
        self.unit = self.conf.get("unit", "V")

    def _convert(
        self, data: np.ndarray | list[np.ndarray] | float
    ) -> np.ndarray | list[np.ndarray] | float:
        """Convert raw reading (V) to self.unit."""

        if isinstance(data, list):
            return [d / self.gain for d in data]
        return data / self.gain

    # override AnalogIn methods to convert readings.

    def _append_data(self, data: np.ndarray | list[np.ndarray]):
        AnalogIn._append_data(self, self._convert(data))

    def read_on_demand(self, oversample: int = 1) -> float | np.ndarray:
        return self._convert(AnalogIn.read_on_demand(self, oversample))

    def set(self, key: str, value=None, label: str = "") -> bool:
        key = key.lower()
        if key == "gain":
            if isinstance(value, (float, int, np.floating, np.integer)):
                self.gain = float(value)
                return True
            else:
                return self.fail_with("gain must be a number (float or int)")
        elif key == "unit":
            self.unit = str(value)
            return True
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get(self, key: str, args=None, label: str = ""):
        if key in ("data", "all_data"):
            return AnalogIn.get(self, key, args)
        elif key == "unit":
            return self.unit
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class LockinAnalogPD(AnalogIn):
    """Generic Photo Detector with Lockin Amp read by DAQ AnalogIn.

    :param lines: two DAQ's physical channels (X, Y) for AnalogIn.
    :type lines: list[str]
    :param buffer_size: (default: 10000) Software buffer (queue) size.
    :type buffer_size: int

    :param unit: (default: V) unit after conversion.
    :type unit: str
    :param gain: (default: 1.0) the fixed gain in [unit] / V.
        Example) when a transimpedance amp with 1000 V / A is used for a photo diode and
        the unit is 'A', gain should be set 1000.
    :type gain: float

    """

    def __init__(self, name, conf, prefix=None):
        AnalogIn.__init__(self, name, conf=conf, prefix=prefix)
        if len(self.conf["lines"]) != 2:
            raise ValueError("len(lines) must be 2.")

        self.gain = self.conf.get("gain", 1.0)
        self.unit = self.conf.get("unit", "V")

    def _convert(self, data: list[np.ndarray] | np.ndarray) -> np.ndarray | np.cdouble:
        """Convert raw reading (V, double) to self.unit, and merge as cdouble."""

        if len(data) != 2:
            raise ValueError(f"data has unexpected length {len(data)}: {data}")

        if isinstance(data, list):
            # buffered read (append_data) returns list of two double arrays
            out = np.empty(len(data[0]), dtype=np.cdouble)
            out.real = data[0] / self.gain
            out.imag = data[1] / self.gain
            return out
        else:
            # read_on_demand returns double array of length two
            return np.cdouble(data[0] + 1.0j * data[1])

    # override AnalogIn methods to convert readings.

    def _append_data(self, data: list[np.ndarray]):
        AnalogIn._append_data(self, self._convert(data))

    def read_on_demand(self, oversample: int = 1) -> np.cdouble:
        return self._convert(AnalogIn.read_on_demand(self, oversample))

    def set(self, key: str, value=None, label: str = "") -> bool:
        key = key.lower()
        if key == "gain":
            if isinstance(value, (float, int, np.floating, np.integer)):
                self.gain = float(value)
                return True
            else:
                return self.fail_with("gain must be a number (float or int)")
        elif key == "unit":
            self.unit = str(value)
            return True
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get(self, key: str, args=None, label: str = ""):
        if key in ("data", "all_data"):
            return AnalogIn.get(self, key, args)
        elif key == "unit":
            return self.unit
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class LUCI10(Instrument):
    """Wrapper for FEMTO Messtechnik LUCI-10 DLL.

    You need following:
    1. Install LUCI-10 software (you can skip LabVIEW driver installation).
    2. Place C DLL (C:\\Program Files (x86)\\FEMTO\\LUCI-10\\Driver\\LUCI_10_x64.dll) somewhere.

    :param dll_dir: The directory path containing DLL.
    :type dll_dir: str
    :param index: (default: 1) Index of LUCI-10 device.
    :type index: int
    :param id: (default: -1) ID of LUCI-10 device.
        If valid value (0 <= id <= 255) is given, access the device via id instead of index.
        Because the id is persistent (it's written in device's EEPROM) and index is not,
        it would be better to use id instead of index when you have more than 2 LUCI-10s.
    :type id: int

    """

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        if "dll_dir" in self.conf:
            p = os.path.expanduser(self.conf["dll_dir"])
            os.environ["PATH"] += os.pathsep + p
            os.add_dll_directory(p)

        id_ = self.conf.get("id", -1)

        self.dll = ctypes.cdll.LoadLibrary("LUCI_10_x64.dll")

        # We must get_devices() here for successful subsequent operation.
        devices = self.get_devices()

        if id_ >= 0:
            id_to_index = {self.get_id(i + 1): i + 1 for i in range(devices)}
            if id_ not in id_to_index:
                raise ValueError("Given device id is not found in " + str(id_to_index.keys()))
            self.index = id_to_index[id_]
        else:
            self.index = self.conf.get("index", 1)

        self.logger.info(f"Opened LUCI10 index {self.index} of {devices} (ID {self.get_my_id()})")

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

    def get_id(self, index: int) -> int | None:
        id_ = ctypes.c_int()
        ret = self.dll.ReadAdapterID(index, ctypes.byref(id_))
        if self._check_ret_value(ret):
            return id_.value

    def get_my_id(self) -> int | None:
        return self.get_id(self.index)

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
        ret = self.dll.GetProductString(self.index, s, size)
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

    def set(self, key: str, value=None, label: str = "") -> bool:
        key = key.lower()
        if key == "led":
            return self.set_led(value)
        elif key == "data":
            return self.write_data(value)
        else:
            return self.fail_with(f"Unknown set() key: {key}.")

    def get(self, key: str, args=None, label: str = ""):
        if key == "devices":
            return self.get_devices()
        elif key == "id":
            return self.get_my_id()
        elif key == "pin":
            return self.get_pin(args)
        elif key == "product":
            return self.get_product_string()
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None


class LUCI_OE200(LUCI10):
    """LUCI-10 for FEMTO Messtechnik OE-200 Variable Gain Photoreceiver.

    This class requires LUCI-10 installation (see LUCI10 class).

    :param dll_dir: The directory path containing LUCI-10 DLL.
    :type dll_dir: str
    :param index: (default: 1) Index of LUCI-10 device.
    :type index: int
    :param id: (default: -1) ID of LUCI-10 device.
        If valid value (0 <= id <= 255) is given, access the device via id instead of index.
        Because the id is persistent (it's written in device's EEPROM) and index is not,
        it would be better to use id instead of index when you have more than 2 LUCI-10s.
    :type id: int

    :param DC_coupling: (default: True) Init setting. True (False) for DC (AC) coupling.
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
        LUCI10.__init__(self, name, conf=conf, prefix=prefix)

        self.DC_coupling = self.conf.get("DC_coupling", True)
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
        return self.write_data(to_write)

    def _update_gain(self, low_noise: bool, gain_exponent: int) -> bool:
        try:
            self.gain_value = self.GAINS[low_noise][gain_exponent]
        except KeyError:
            return self.fail_with(f"Invalid arguments: {low_noise}, {gain_exponent}")

        self.low_noise = low_noise
        self.gain_exponent = gain_exponent

        if low_noise:
            self.gain_value |= 0b1_0000

        self.gain = 10**gain_exponent
        return True

    def set_gain(self, low_noise: bool, gain_exponent: int) -> bool:
        return self._update_gain(low_noise, gain_exponent) and self._write_settings()

    def set_coupling(self, DC_coupling: bool) -> bool:
        self.DC_coupling = DC_coupling
        return self._write_settings()

    def configure_gain_coupling(
        self, low_noise: bool, gain_exponent: int, DC_coupling: bool
    ) -> bool:
        self.DC_coupling = DC_coupling
        return self._update_gain(low_noise, gain_exponent) and self._write_settings()

    def set(self, key: str, value=None, label: str = "") -> bool:
        key = key.lower()
        if key == "led":
            return self.set_led(value)
        elif key == "gain":
            try:
                return self.set_gain(value["low_noise"], value["gain_exponent"])
            except (KeyError, TypeError):
                return self.fail_with("value must be a dict with low_noise and gain_exponent.")
        elif key == "coupling":
            return self.set_coupling(bool(value))
        else:
            return self.fail_with(f"Unknown set() key: {key}.")

    def get(self, key: str, args=None, label: str = ""):
        if key in ("devices", "id", "pin", "product"):
            return LUCI10.get(self, key, args)
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        return [""]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        return P.ParamDict(
            low_noise=P.BoolParam(
                self.low_noise, doc="True for Low-Noise, False for High-Speed mode"
            ),
            gain_exponent=P.IntChoiceParam(
                self.gain_exponent,
                list(range(3, 12)),
                doc="3-9 for Low-Noise, 5-11 for High-Speed",
            ),
            DC_coupling=P.BoolParam(self.DC_coupling, doc="True for DC, False for AC coupling"),
        )

    def configure(self, params: dict, label: str = "") -> bool:
        return self.configure_gain_coupling(
            params["low_noise"], params["gain_exponent"], params["DC_coupling"]
        )
