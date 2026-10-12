#!/usr/bin/env python3

"""
Typed Interface for Photo Detectors.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np

from mahos.inst.daq_interface import BufferedReaderInterface


class PDInterface(BufferedReaderInterface):
    """Common interface for SinglePhotonCounter (BufferedEdgeCounter) and AnalogIn-based PD."""

    def configure_triggered(
        self,
        trigger_source: str,
        cb_samples: int,
        samples: int,
        rate: float,
        *,
        buffer_size: int = 0,
        bounds=(-10.0, 10.0),
        finite: bool = False,
        every: bool = False,
        drop_first: int = 0,
        oversample: int = 1,
        block_samples: int = 1,
        block_reduce_factor: int = 1,
        block_reduce_op: str = "mean",
        reduce_factor: int = 1,
        reduce_op: str = "mean",
        data_transfer: str | None = None,
        trigger_dir: bool = True,
        hardware_average: bool = True,
    ) -> bool:
        """Configure triggered fixed-length acquisition.

        This is only for Analog PD now (not for SinglePhotonCounter/BufferedEdgeCounter).

        ``trigger_source`` is the DAQ sample clock for conventional AnalogIn and the hardware
        trigger input for Spectrum digitizers.

        """

        params = {
            "cb_samples": cb_samples,
            "samples": samples,
            "rate": rate,
            "bounds": bounds,
            "finite": finite,
            "buffer_size": buffer_size,
            "drop_first": drop_first,
            "oversample": oversample,
            "block_samples": block_samples,
            "block_reduce_factor": block_reduce_factor,
            "block_reduce_op": block_reduce_op,
            "reduce_factor": reduce_factor,
            "reduce_op": reduce_op,
            # DAQ (AnalogIn) parameters
            "clock": trigger_source,
            "clock_mode": True,
            "clock_dir": trigger_dir,
            "data_transfer": data_transfer,
            "every": every,
            # Spectrum_AnalogIn parameters
            "trigger_source": trigger_source,
            "trigger_dir": trigger_dir,
            "hardware_average": hardware_average,
        }
        # label is referred by Spectrum_AnalogIn only
        return self.configure(params, "triggered")

    def configure_stream(
        self,
        cb_samples: int,
        buffer_size: int,
        rate: float,
        time_window: float,
        *,
        bounds=(-10.0, 10.0),
        oversample: int = 1,
        clock: str = "",
        stamp: bool = True,
        data_transfer: str | None = None,
    ) -> bool:
        """Configure continuously running infinite (stream) acquisition."""

        params = {
            "cb_samples": cb_samples,
            "buffer_size": buffer_size,
            "rate": rate,
            "stamp": stamp,
            # BufferedEdgeCounter parameters
            "time_window": time_window,
            # Analog (AnalogIn and Spectrum_AnalogIn) parameters
            "oversample": oversample,
            "bounds": bounds,
            # DAQ (BufferedEdgeCounter and AnalogIn) parameters
            "samples": buffer_size,
            "clock": clock,
            "finite": False,
            "every": False,
            # AnalogIn parameters
            "clock_mode": True,
            "data_transfer": data_transfer,
        }
        # label is referred by Spectrum_AnalogIn only
        return self.configure(params, "stream")


class SinglePhotonCounterInterface(PDInterface):
    """Interface for SinglePhotonCounter."""

    def correct_cps(self, raw_cps: list[float]) -> np.ndarray:
        """Correct the raw values in cps according to correction factors."""

        return self.get("correct", raw_cps)

    def get_correction_factor(self, xs_cps: list[float]) -> np.ndarray:
        """Get the correction factor for given cps values."""

        return self.get("correction_factor", xs_cps)


# Backward-compatible alias for existing code.
APDCounterInterface = SinglePhotonCounterInterface
