#!/usr/bin/env python3

"""
Message Types for Analog-PD Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np

# classes other than PODMRData are just for re-export
from mahos_dq.msgs.podmr_msgs import (  # noqa: F401
    PODMRData,
    MWMode,
    ValidateReq,
    GetTimingInfoReq,
    TimingInfo,
    UpdatePlotParamsReq,
)


class APODMRData(PODMRData):
    """Analog-PD pulse ODMR data container.

    :ivar raw_data: Retained raw trace records with shape ``(record, trace, sample)``.
    :ivar raw_data_sum: Sum of all captured raw trace records with shape ``(trace, sample)``.
    :ivar raw_xdata: Trace-local time axis for a single aggregated trace.
    :ivar records: Number of raw trace records captured so far.
    :ivar signal_history: Individual signal window means as a float64 array with shape
        ``(record, trace)``, or ``None`` when empty.
    :ivar reference_history: Individual reference window means with the same shape and type.
    :ivar history_start_record: Zero-based absolute record index of the first history row.
        For empty history, this equals ``records``.
    :ivar trace_laser_timing: Laser-start timing in trace-local time, measured from trigger.
    :ivar marker_indices: Marker indices with shape ``(4,)`` as
        ``(sig_head, sig_tail, ref_head, ref_tail)`` shared by all traces.
    :ivar trigger_timing: Trigger timings for each trace in sequence time.
    :ivar laser_timing: Laser-start timings in sequence time (same semantics as
        :class:`mahos_dq.msgs.podmr_msgs.PODMRData`), kept for compatibility;
        APODMR analysis uses ``trace_laser_timing`` instead. When
        ``burst_num > 1``, each entry marks only the first shot of a repeated point.

    ``data0`` .. ``data3`` and ``data0ref`` .. ``data3ref`` retain the same
    analyzed-data semantics as :class:`mahos_dq.msgs.podmr_msgs.PODMRData`.

    History preserves raw trace ordering and is enabled by ``save_history`` (default: True).

    """

    def __init__(self, params: dict | None = None, label: str = ""):
        super().__init__(params, label)
        self.set_version(2)

        self.tdc_status = None
        self.raw_data_sum = None
        self.records = 0
        self.signal_history = None
        self.reference_history = None
        self.history_start_record = 0
        self.trace_laser_timing = None
        self.trigger_timing = None

    def clear_history(self):
        """Clear analyzed history and advance its start to the current record count."""

        self.signal_history = None
        self.reference_history = None
        self.history_start_record = self.records

    def can_resume(self, params: dict | None, label: str) -> bool:
        """Check resume compatibility while allowing history collection to be toggled."""

        if params is None:
            return False
        params = params.copy()
        if self.has_params() and "save_history" in self.params:
            params["save_history"] = self.params["save_history"]
        else:
            params.pop("save_history", None)
        return super().can_resume(params, label)

    def _h5_attr_writers(self) -> dict:
        d = super()._h5_attr_writers()
        d.pop("tdc_status", None)
        return d

    def _h5_readers(self) -> dict:
        d = super()._h5_readers()
        d.pop("tdc_status", None)
        return d

    def get_raw_xdata(self) -> np.ndarray | None:
        """Get trace-local x-axis of raw data."""

        if self.raw_xdata is not None:
            return self.raw_xdata

        sample_period = self.get_bin()
        samples_per_trace = self.get_samples_per_trace()
        if sample_period is None or samples_per_trace is None:
            return None

        self.raw_xdata = np.arange(samples_per_trace) * sample_period
        return self.raw_xdata

    def retained_records(self) -> int:
        if self.raw_data is None:
            return 0
        return int(self.raw_data.shape[0])

    def get_samples_per_trace(self) -> int | None:
        if self.raw_data is not None and np.size(self.raw_data) > 0:
            return int(self.raw_data.shape[-1])
        try:
            samples = int(self.params["instrument"]["samples_per_trace"])
            if samples > 0:
                return samples
        except (KeyError, TypeError, ValueError):
            pass
        try:
            trange = float(self.params["instrument"]["trange"])
            tbin = float(self.params["instrument"]["tbin"])
            if tbin > 0.0 and trange >= 0.0:
                return max(1, int(round(trange / tbin)))
        except (KeyError, TypeError, ValueError):
            pass
        return None

    def get_sweeps_per_record(self) -> int:
        if not self.has_params():
            return 1
        try:
            return max(1, int(self.params.get("sweeps_per_record", 1)))
        except (TypeError, ValueError):
            return 1

    def sweeps(self) -> int:
        return self.records * self.get_sweeps_per_record()

    def get_sampling_interval(self) -> float:
        """Get the nominal analyzed-record sampling interval in seconds.

        Computed as ``sweeps_per_record * instrument.length / instrument.pg_freq``.
        Assumes uninterrupted acquisition without dropped records; resume gaps are not
        represented. Use ``sweeps_per_record=1`` for sweep resolution.
        Returns 0.0 when timing metadata is unavailable or the frequency is nonpositive.

        """

        if not self.has_params():
            return 0.0
        try:
            freq = float(self.params["instrument"]["pg_freq"])
            length = float(self.params["instrument"]["length"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if freq <= 0.0:
            return 0.0
        return self.get_sweeps_per_record() * length / freq

    def measurement_time(self) -> float:
        return self.records * self.get_sampling_interval()

    def has_raw_data(self) -> bool:
        return self.raw_data is not None and np.size(self.raw_data) > 0

    def has_raw_data_sum(self) -> bool:
        return self.raw_data_sum is not None and np.size(self.raw_data_sum) > 0

    def _get_xdata_head(self, xdata):
        raise ValueError("taumode 'head' is unsupported for APODMR")


def update_data(data: APODMRData):
    """Update APODMR data to the latest schema."""

    if data.version() <= 0:
        # version 0 to 1
        if data.has_params() and "shots_per_point" in data.params:
            data.params["burst_num"] = data.params.pop("shots_per_point")
        data.set_version(1)

    if data.version() <= 1:
        data.clear_history()
        data.set_version(2)

    return data
