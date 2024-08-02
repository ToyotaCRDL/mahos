#!/usr/bin/env python3

"""
Typed Interface for Spectrometer.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np

from .interface import InstrumentInterface
from ..msgs.inst.spectrometer_msgs import Temperature


class SpectrometerInterface(InstrumentInterface):
    """Interface for Spectrometer."""

    def get_data(self) -> np.ndarray | None:
        """Start one-time acquisition and return data.

        Shape of returned array is (2, number of pixels).
        array[0] contains x-axis data (wavelength in nm).
        array[1] contains y-axis data (intensity in counts).

        This function blocks during acquisition.

        """

        return self.get("data")

    def get_base_config(self) -> str | None:
        """get current base config name."""

        return self.get("base_config")

    def get_base_configs(self) -> list[str]:
        """get list of available base configs."""

        return self.get("base_configs")

    def get_temperature(self) -> Temperature:
        """Get detector temperature information."""

        return self.get("temperature")

    def set_exposure_time(self, time_ms: float) -> bool:
        """Set exposure time."""

        return self.set("exposure_time", time_ms)

    def set_exposures(self, exposures: int) -> bool:
        """Set repeated exposures per acquisitions."""

        return self.set("exposures", exposures)

    def set_center_wavelength(self, wavelength_nm: float) -> bool:
        """Set center wavelength."""

        return self.set("center_wavelength", wavelength_nm)

    def configure_acquisition(
        self,
        base_config: str | None = None,
        exposure_time_ms: float | None = None,
        exposures: int | None = None,
        center_wavelength_nm: float | None = None,
    ) -> bool:
        """Configure acquisition.

        :param base_config: name of predefined base configuration.
        :type base_config: str
        :param exposure_time_ms: (ms) exposure time in ms.
        :type exposure_time_ms: float
        :param exposures: number of repeated exposures.
        :type exposures: int
        :param center_wavelength_nm: (nm) center wavelength in nm.
        :type center_wavelength_nm: float

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
