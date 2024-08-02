#!/usr/bin/env python3

"""
Andor Spectrometer module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np

from ..instrument import Instrument
from ...msgs.inst.spectrometer_msgs import Temperature


# imports for Andor
try:
    from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors
    from pyAndorSpectrograph.spectrograph import ATSpectrograph

    codes = atmcd_codes
    SDK_OK = atmcd_errors.Error_Codes.DRV_SUCCESS
    SPC_OK = ATSpectrograph.ATSPECTROGRAPH_SUCCESS

except ImportError:
    print("mahos.inst.spectrometer: failed to import pyAndor modules")


class Andor_Spectrometer(Instrument):
    """Andor Spectrometer.

    :param base_configs: The base configurations.
    :type base_configs: dict
    :param detector_temperature: Set value of detector temperature.
    :type detector_temperature: float
    :param invert_wavelength: (default: False) Set True to invert wavelength (x-axis).
        This is relevant only if the detector is attached in inverted position.
    :type invert_wavelength: bool
    :param device_id: (default: 0) Device ID for spectrometer.
    :type device_id: int

    """

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf("base_configs")
        self._base_configs = self.conf["base_configs"]
        self._base_config = self.conf.get("base_config", list(self._base_configs.keys())[0])
        self._invert_wavelength = self.conf.get("invert_wavelength", False)
        self._device = self.conf.get("device_id", 0)
        self._target_T = self.conf.get("detector_temperature", -60)

        self.sdk = atmcd()
        self.spc = ATSpectrograph()

        if self.spc.Initialize("") == SPC_OK:
            self.logger.info("Initialized ATSpectrograph (Shamrock).")
        else:
            raise RuntimeError("Failed to initialize ATSpectrograph (Shamrock).")

        if self.sdk.Initialize("") == SDK_OK:
            self.logger.info("Initialized atmcd (SDK).")
        else:
            raise RuntimeError("Failed to initialize atmcd (SDK).")

        _, serial = self.sdk.GetCameraSerialNumber()
        self.logger.info(f"Camera Serial: {serial}")

        if self.check_sdk(self.sdk.SetTemperature(self._target_T)) and self.check_sdk(
            self.sdk.CoolerON()
        ):
            self.logger.info("Cooler ON")
        else:
            self.logger.error("Failed to Cooler ON")

    def close_resources(self):
        if hasattr(self, "sdk"):
            self.sdk.ShutDown()
        if hasattr(self, "spc"):
            self.spc.Close()

    def check_sdk(self, ret: int) -> bool:
        if ret == SDK_OK:
            return True
        else:
            self.logger.error(f"SDK function returned code {ret}.")
            return False

    def check_spc(self, ret: int) -> bool:
        if ret == SPC_OK:
            return True
        else:
            self.logger.error(f"SPC function returned code {ret}.")
            return False

    def set_exposure_time(self, time_ms: float) -> bool:
        return self.check_sdk(self.sdk.SetExposureTime(time_ms * 1e-3))

    def set_exposures_per_frame(self, exposures: int) -> bool:
        if exposures <= 0:
            return self.fail_with("Exposures must be a positive integer.")

        return self.check_sdk(self.sdk.SetNumberAccumulations(exposures))

    def set_grating_center_wavelength(self, wavelength_nm: float) -> bool:
        if wavelength_nm <= 0.0:
            self.logger.error("wavelength_nm must be a positive float.")
            return False
        self.spc.SetWavelength(self._device, wavelength_nm)
        return True

    def get_grating_center_wavelength(self) -> float:
        ret, wavelength = self.spc.GetWavelength(self._device)
        if ret == SPC_OK:
            return wavelength
        else:
            self.logger.error("spc.GetWavelength() failed.")
            return 0.0

    def get_temperature(self) -> Temperature:
        _, current = self.sdk.GetTemperature()
        return Temperature(current, self._target_T)

    def configure_base_config(self, params: dict) -> bool:
        base_config = params["base_config"]
        if base_config not in self._base_configs:
            return self.fail_with(f"base_config {base_config} is unknown.")
        c = self._base_configs[base_config]

        # For now, we don't expect the grating in params.
        grating = c.get("grating", 1)

        if "center_wavelength" not in params and "center_wavelength" not in c:
            return self.fail_with("center_wavelength must be given in params or base_config.")
        if "center_wavelength" in params:
            wavelength = params["center_wavelength"]
        else:
            wavelength = c["center_wavelength"]
        exposure_time_ms = params.get("exposure_time", c.get("exposure_time", 1.0))
        exposures = params.get("exposures", c.get("exposures", 1))
        # cycle_time is period of multiple exposures. convert ms to sec here.
        # in our usage, putting 0 is fine (minimum possible cycle_time is automatically set).
        cycle_time_sec = params.get("cycle_time", c.get("cycle_time", 0.0) * 1e-3)

        if not (
            self.check_sdk(self.sdk.SetAcquisitionMode(codes.Acquisition_Mode.ACCUMULATE))
            and self.check_sdk(self.sdk.SetReadMode(codes.Read_Mode.FULL_VERTICAL_BINNING))
            and self.check_sdk(self.sdk.SetTriggerMode(codes.Trigger_Mode.INTERNAL))
            and self.check_sdk(self.sdk.SetNumberAccumulations(exposures))
            and self.check_sdk(self.sdk.SetAccumulationCycleTime(cycle_time_sec))
        ):
            return False
        (ret, self.xpixels, self.ypixels) = self.sdk.GetDetector()
        if not (
            self.check_sdk(ret)
            and self.check_sdk(self.sdk.SetImage(1, 1, 1, self.xpixels, 1, self.ypixels))
            and self.set_exposure_time(exposure_time_ms)
            and self.sdk.PrepareAcquisition()
        ):
            return False

        if not (
            self.check_spc(self.spc.SetGrating(self._device, grating))
            and self.set_grating_center_wavelength(wavelength)
        ):
            return False
        self._base_config = base_config
        return True

    def get_base_config(self) -> str:
        return self._base_config

    def get_base_configs(self) -> list[str]:
        return list(self._base_configs.keys())

    def capture(self) -> np.ndarray | None:
        if not (
            self.check_sdk(self.sdk.StartAcquisition())
            and self.check_sdk(self.sdk.WaitForAcquisition())
        ):
            self.logger.error("Failed to start / wait for acquisition.")
            return None

        (ret1, y, validfirst, validlast) = self.sdk.GetImages(1, 1, self.xpixels)
        (ret2, xsize, ysize) = self.sdk.GetPixelSize()
        if not (self.check_sdk(ret1) and self.check_sdk(ret2)):
            self.logger.error("Failed to get image / pixel size.")
            return None

        if not (
            self.check_spc(self.spc.SetNumberPixels(0, self.xpixels))
            and self.check_spc(self.spc.SetPixelWidth(0, xsize))
        ):
            self.logger.error("Failed to set number of pixels.")
            return None
        (ret, x) = self.spc.GetCalibration(0, self.xpixels)
        if not self.check_spc(ret):
            self.logger.error("Failed to get calibration.")
            return None

        if len(x) != len(y):
            self.logger.error(
                f"X {len(x)} and Y {len(y)} lengths are mismatching. Review settings."
            )
            return None

        if self._invert_wavelength:
            y = y[::-1]
        return np.vstack((x, y))

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "data":
            return self.capture()
        elif key == "base_config":
            return self.get_base_config()
        elif key == "base_configs":
            return self.get_base_configs()
        elif key == "temperature":
            return self.get_temperature()
        else:
            self.logger.error(f"Unknown get() key: {key}.")
            return None

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "exposure_time":
            return self.set_exposure_time(value)
        elif key == "exposures":
            return self.set_exposures_per_frame(value)
        elif key == "center_wavelength":
            return self.set_grating_center_wavelength(value)
        else:
            return self.fail_with(f"Unknown set() key: {key}.")

    def configure(self, params: dict, label: str = "") -> bool:
        if not self.check_required_params(params, "base_config"):
            return False
        return self.configure_base_config(params)
