#!/usr/bin/env python3

"""
Typed Interface for Spectrometer.

.. This file is a part of MAHOS project.

"""

import typing as T

import numpy as np

from .interface import InstrumentInterface


class SpectrometerInterface(InstrumentInterface):
    """Interface for Spectrometer."""

    def get_data(self) -> T.Optional[np.ndarray]:
        """Start one-time acquisition and return data.

        Shape of returned array is (2, number of pixels).
        array[0] contains x-axis data (wavelength in nm).
        array[1] contains y-axis data (intensity in counts).

        This function blocks during acquisition.

        """

        return self.get("data")

    def get_config(self) -> dict:
        """Get config dict including following parameters.

        Returns:
            base_config (str): loaded base config.
            exposure_time (float): exposure time in ms.
            exposures (int): number of repeated exposures.
            center_wavelength (float): center wavelength in nm.

        """

        return self.get("config")

    def get_base_config(self) -> T.Optional[str]:
        """get current base config (saved experiment)."""

        return self.get("base_config")

    def get_base_configs(self) -> T.List[str]:
        """get list of available base configs."""

        return self.get("base_configs")

    def set_base_config(self, name: str) -> bool:
        """set and load a base config (saved experiment)."""

        return self.set("base_config", name)

    def get_exposure_time(self) -> float:
        """Get exposure time in ms."""

        return self.get("exposure_time")

    def set_exposure_time(self, time_ms: float) -> bool:
        """Set exposure time."""

        return self.set("exposure_time", time_ms)

    def get_exposures(self) -> int:
        """Get repeated exposures per acquisitions."""

        return self.get("exposures")

    def set_exposures(self, exposures: int) -> bool:
        """Set repeated exposures per acquisitions."""

        return self.set("exposures", exposures)

    def get_center_wavelength(self) -> float:
        """Get center wavelength in nm."""

        return self.get("center_wavelength")

    def set_center_wavelength(self, wavelength_nm: float) -> bool:
        """Set center wavelength."""

        return self.set("center_wavelength", wavelength_nm)

    def configure_acquisition(
        self,
        base_config: T.Optional[str] = None,
        exposure_time_ms: T.Optional[float] = None,
        exposures: T.Optional[int] = None,
        center_wavelength_nm: T.Optional[float] = None,
    ) -> bool:
        """Configure acquisition.

        :param base_config: name of predefined base configuration.
        :param exposure_time_ms: (ms) exposure time in ms.
        :param exposures: number of repeated exposures.
        :param center_wavelength_nm: (nm) center wavelength in nm.

        """

        params = {}
        if base_config:
            params["base_config"] = base_config
        if exposure_time_ms:
            params["exposure_time"] = exposure_time_ms
        if exposures:
            params["exposures"] = exposures
        if center_wavelength_nm:
            params["center_wavelength"] = center_wavelength_nm
        return self.configure(params)
