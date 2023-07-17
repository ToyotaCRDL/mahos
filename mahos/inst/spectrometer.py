#!/usr/bin/env python3

"""
Spectrometer module.

.. This file is a part of MAHOS project.

"""

import sys
import os
import typing as T

import numpy as np
import clr

from .instrument import Instrument


# paths and imports for LightField
from System import String
from System import Int64
from System import Double
from System.Collections.Generic import List

sys.path.append(os.environ["LIGHTFIELD_ROOT"])
sys.path.append(os.environ["LIGHTFIELD_ROOT"] + "\\AddInViews")
clr.AddReference("PrincetonInstruments.LightFieldViewV5")
clr.AddReference("PrincetonInstruments.LightField.AutomationV5")
clr.AddReference("PrincetonInstruments.LightFieldAddInSupportServices")

from PrincetonInstruments.LightField.Automation import Automation  # noqa: E402
from PrincetonInstruments.LightField.AddIns import ExperimentSettings  # noqa: E402
from PrincetonInstruments.LightField.AddIns import CameraSettings  # noqa: E402
from PrincetonInstruments.LightField.AddIns import SpectrometerSettings  # noqa: E402
from PrincetonInstruments.LightField.AddIns import FrameCombinationMethod  # noqa: E402

# from PrincetonInstruments.LightField.AddIns import DeviceType


class LightField(Instrument):
    """Wrapper for LightField Software from Princeton Instrument."""

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        # pass empty command line option (experiment name could be given here)
        opts = List[String]()
        self.auto = Automation(True, opts)
        self.expt = self.auto.LightFieldApplication.Experiment

        if self.conf.get("base_config"):
            self.set_base_config(self.conf["base_config"])
        else:
            self._current_base_config = None

    def close(self):
        if hasattr(self, "auto"):
            # This will close the LightField window.
            self.auto.Dispose()

    def set_base_config(self, name: str) -> bool:
        """set and load a base config (saved experiment)."""

        if name in self.expt.GetSavedExperiments():
            self.expt.Load(name)
            self._current_base_config = name
            return True
        else:
            self.logger.error(f"{name} is not a valid base_config.")
            return False

    def get_base_config(self) -> T.Optional[str]:
        """get current base config."""

        return self._current_base_config

    def get_base_configs(self) -> T.List[str]:
        """get list of available base configs."""

        return list(self.expt.GetSavedExperiments())

    def set_exposure_time(self, time_ms: float) -> bool:
        if time_ms <= 0.0:
            self.logger.error("time_ms must be a positive float.")
            return False
        self.expt.SetValue(CameraSettings.ShutterTimingExposureTime, Double(time_ms))
        return True

    def get_exposure_time(self) -> float:
        return self.expt.GetValue(CameraSettings.ShutterTimingExposureTime)

    def set_exposures_per_frame(self, exposures: int) -> bool:
        if exposures <= 0:
            self.logger.error("Exposures must be a positive integer.")
            return False

        self.expt.SetValue(
            ExperimentSettings.OnlineProcessingFrameCombinationFramesCombined, Int64(exposures)
        )
        return True

    def get_exposures_per_frame(self) -> int:
        return self.expt.GetValue(
            ExperimentSettings.OnlineProcessingFrameCombinationFramesCombined
        )

    def set_frame_combination_method(self, average: bool = True) -> bool:
        if average:
            m = FrameCombinationMethod.Average
        else:
            m = FrameCombinationMethod.Sum

        self.expt.SetValue(ExperimentSettings.OnlineProcessingFrameCombinationMethod, m)
        return True

    def get_frame_combination_method(self) -> bool:
        m = self.expt.GetValue(ExperimentSettings.OnlineProcessingFrameCombinationMethod)
        is_average = m == FrameCombinationMethod.Average
        return is_average

    def set_grating_center_wavelength(self, wavelength_nm: float) -> bool:
        if wavelength_nm <= 0.0:
            self.logger.error("wavelength_nm must be a positive float.")
            return False
        self.expt.SetValue(SpectrometerSettings.GratingCenterWavelength, Double(wavelength_nm))
        return True

    def get_grating_center_wavelength(self) -> float:
        return self.expt.GetValue(SpectrometerSettings.GratingCenterWavelength)

    def capture(self) -> T.Optional[np.ndarray]:
        nframes = 1
        frames = self.expt.Capture(nframes)
        if frames.Regions.Length != 1:
            self.logger.error("There are multiple ROIs. Review experiment settings.")
            return None

        if frames.Frames != nframes:
            self.logger.error("Number of captured frames is broken.")
            return None

        x = np.array(self.expt.SystemColumnCalibration)
        frame = frames.GetFrame(0, 0)  # regionIndex, frameIndex
        y = np.array(frame.GetData())

        if x.shape != y.shape:
            self.logger.error(
                f"X {x.shape} and Y {y.shape} shapes are mismatching. Review ROI settings."
            )
            return None

        return np.vstack((x, y))

    # Standard API

    def get(self, key: str, args=None):
        if key == "data":
            return self.capture()
        elif key == "config":
            return {
                "base_config": self.get_base_config(),
                "exposure_time": self.get_exposure_time(),
                "exposures": self.get_exposures_per_frame(),
                "center_wavelength": self.get_grating_center_wavelength(),
            }
        elif key == "base_config":
            return self.get_base_config()
        elif key == "base_configs":
            return self.get_base_configs()
        elif key == "exposure_time":
            return self.get_exposure_time()
        elif key == "exposures":
            return self.get_exposures_per_frame()
        elif key == "center_wavelength":
            return self.get_grating_center_wavelength()
        else:
            self.logger.error(f"Unknown get() key: {key}.")
            return None

    def set(self, key: str, value=None) -> bool:
        if key == "base_config":
            return self.set_base_config(value)
        elif key == "exposure_time":
            return self.set_exposure_time(value)
        elif key == "exposures":
            return self.set_exposures_per_frame(value)
        elif key == "center_wavelength":
            return self.set_grating_center_wavelength(value)
        else:
            return self.fail_with(f"Unknown set() key: {key}.")

    def configure(self, params: dict) -> bool:
        success = True
        if params.get("base_config"):
            success &= self.set_base_config(params["base_config"])
        if params.get("exposure_time"):
            success &= self.set_exposure_time(params["exposure_time"])
        if params.get("exposures"):
            success &= self.set_exposures_per_frame(params["exposures"])
        if params.get("center_wavelength"):
            success &= self.set_grating_center_wavelength(params["center_wavelength"])

        if not success:
            self.fail_with("Failed to configure.")

        return success
