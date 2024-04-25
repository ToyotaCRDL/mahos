#!/usr/bin/env python3

"""
Swabian Instruments Time Tagger module

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os
import time

import numpy as np
import TimeTagger as tt

from ..instrument import Instrument
from ...msgs.inst_tdc_msgs import ChannelStatus, RawEvents
from ...util.io import save_h5


class TimeTagger(Instrument):
    """Swabian Instruments Time Tagger

    :param base_configs: Mapping from base config name to channels and levels definitions.
        Used in configure_histogram() etc.
    :type base_configs: dict[str, str]
    :param ext_ref_clock: (default: 0) external reference clock source.
        0: internal clock, 10: 10 MHz ext clock, 500: 500 MHz ext clock.
    :type ext_ref_clock: int
    :param raw_events_dir: (default: user home) The directory to save RawEvents data.
    :type raw_events_dir: str

    """

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        tt.setLogger(self.log_from_time_tagger)
        self.tagger = tt.createTimeTagger(self.conf.get("serial", ""))
        self.logger.info(f"Opened TimeTagger {self.tagger.getSerial()}")

        self._base_configs = self.conf.get("base_configs", {})
        self._raw_events_dir = os.path.expanduser(self.conf.get("raw_events_dir", "~"))
        self._remove_ttbin = self.conf.get("remove_ttbin", False)
        self.logger.debug(f"available base configs: {self._base_configs}")

        # always use SynchronizedMeasurements to avoid autostart
        # on initialization of measurement class.
        self.sync: tt.SynchronizedMeasurements | None = None
        self.meas: tt.Correlation | tt.FileWriter | list[tt.Histogram] | None = None
        self.counter: tt.Counter | None = None
        self._duration_ps: int = 0

        self._save_file_name = None
        self._tstart = time.time()
        self.trange = self.tbin = 0.0

        # self.set_reference_clock(conf.get("ext_ref_clock", 0))

    def log_from_time_tagger(self, level: int, msg: str):
        self.logger.debug("[TimeTagger lib] " + msg)

    def set_reference_clock(self, clock_MHz: int) -> bool:
        if clock_MHz == 10:
            self.tagger.xtra_setClockSource(1)
            self.logger.info("External clock at 10 MHz.")
        elif clock_MHz == 500:
            self.tagger.xtra_setClockSource(2)
            self.logger.info("External clock at 500 MHz.")
        else:
            if clock_MHz != 0:
                self.logger.warn(f"External clock {clock_MHz} is invalid.")
            self.tagger.xtra_setClockSource(0)
            self.logger.info("Internal clock.")

    def configure_histogram(self, base_config: str, trange: float, tbin: float) -> bool:
        """Configure histogram measurement.

        If channels is more than 2, the first channel is considered common start,
        and the other channels are considered stops.

        """

        if base_config not in self._base_configs:
            return self.fail_with("Unknown base config name")

        conf = self._base_configs[base_config]
        if not all(key in conf for key in ("channels", "levels")):
            return self.fail_with(
                f"'channels' / 'levels' is not defined in base_configs['{base_config}']"
            )

        for channel, level in zip(conf["channels"], conf["levels"]):
            self.tagger.setTriggerLevel(channel, level)

        tbin_ps, n_bins = self.set_range_bin(trange, tbin)
        self.sync = tt.SynchronizedMeasurements(self.tagger)
        self.meas = []
        for ch in conf["channels"][1:]:
            self.meas.append(
                tt.Histogram(
                    self.sync.getTagger(),
                    ch,
                    start_channel=conf["channels"][0],
                    binwidth=tbin_ps,
                    n_bins=n_bins,
                )
            )
        self.counter = tt.Counter(self.sync.getTagger(), conf["channels"])
        self._duration_ps = 0
        return True

    def configure_correlation(self, base_config: str, trange: float, tbin: float) -> bool:
        if base_config not in self._base_configs:
            return self.fail_with("Unknown base config name")

        conf = self._base_configs[base_config]
        if not all(key in conf for key in ("channels", "levels")):
            return self.fail_with(
                f"'channels' / 'levels' is not defined in base_configs['{base_config}']"
            )

        for i in range(2):
            self.tagger.setTriggerLevel(conf["channels"][i], conf["levels"][i])

        tbin_ps, n_bins = self.set_range_bin(trange, tbin)
        self.sync = tt.SynchronizedMeasurements(self.tagger)
        self.meas = tt.Correlation(
            self.sync.getTagger(),
            channel_1=conf["channels"][0],
            channel_2=conf["channels"][1],
            binwidth=tbin_ps,
            n_bins=n_bins,
        )
        self.counter = tt.Counter(self.sync.getTagger(), conf["channels"])
        self._duration_ps = 0
        return True

    def configure_raw_events(self, base_config: str, save_file: str) -> bool:
        if base_config not in self._base_configs:
            return self.fail_with("Unknown base config name")

        conf = self._base_configs[base_config]
        if not all(key in conf for key in ("channels", "levels")):
            return self.fail_with(
                f"'channels' / 'levels' is not defined in base_configs['{base_config}']"
            )

        for channel, level in zip(conf["channels"], conf["levels"]):
            self.tagger.setTriggerLevel(channel, level)

        self.sync = tt.SynchronizedMeasurements(self.tagger)
        self.meas = tt.FileWriter(
            self.sync.getTagger(),
            filename=save_file,
            channels=conf["channels"],
        )
        self.counter = tt.Counter(self.sync.getTagger(), conf["channels"])
        self._duration_ps = 0

        self._save_file_name = save_file
        # unit of raw event (or tbin) is always ps
        # trange has no meaning.
        self.tbin = 1e-12
        self.trange = 0.0

        return True

    def set_range_bin(self, trange: float, tbin: float) -> tuple[int, int]:
        """round range and timebin in sec."""

        if tbin == 0.0:
            # set minimum bin, 1 ps
            tbin_ps = 1
            n_bins = round(trange * 1e12)
        else:
            tbin_ps = round(tbin * 1e12)
            n_bins = round(trange / tbin)

        # store actual value of tbin and trange for query of get_range_bin().
        self.tbin = tbin_ps * 1e-12
        self.trange = n_bins * tbin_ps * 1e-12

        return tbin_ps, n_bins

    def get_range_bin(self) -> dict:
        """Get range and time bin in sec."""

        return {"range": self.trange, "bin": self.tbin}

    def get_timebin(self) -> float:
        """Get time bin in sec."""

        return self.tbin

    def clear(self) -> bool:
        if self.sync is None:
            return self.fail_with("Measurement is not configured.")
        self.sync.clear()
        return True

    def set_duration(self, duration: float) -> bool:
        self._duration_ps = round(duration * 1e12)

    def get_data(self, ch: int = 0) -> np.ndarray | None:
        if isinstance(self.meas, list):
            try:
                return self.meas[ch].getData()
            except IndexError:
                self.logger.error(f"channel {ch} is out of bounds.")
                return None

        return self.meas.getData()

    def get_data_normalized(self, ch: int = 0) -> np.ndarray | None:
        if isinstance(self.meas, list):
            try:
                return self.meas[ch].getDataNormalized()
            except IndexError:
                self.logger.error(f"channel {ch} is out of bounds.")
                return None
            except AttributeError:
                self.logger.error("current measurement doesn't support data_normalized.")
                return None

        try:
            return self.meas.getDataNormalized()
        except AttributeError:
            self.logger.error("current measurement doesn't support data_normalized.")
            return None

    def get_data_roi(self, ch: int, roi: list[tuple[int, int]]) -> list[np.ndarray] | None:
        data = self.get_data(ch)
        if data is None:
            return None
        data_roi = []
        for start, stop in roi:
            # fill up out-of-bounds in ROI with zeros
            if start < 0:
                d = np.concatenate((np.zeros(abs(start), dtype=data.dtype), data[:stop]))
            else:
                d = data[start:stop]
            if stop > len(data):
                d = np.concatenate((d, np.zeros(stop - len(data), dtype=data.dtype)))
            data_roi.append(d)
        return data_roi

    def get_status(self, ch: int) -> ChannelStatus | None:
        runtime = time.time() - self._tstart
        counts = self.counter.getDataTotalCounts()
        running = self.sync.isRunning()
        if isinstance(self.meas, list):
            # Histogram measurement
            starts = counts[0]
            try:
                total = counts[1 + ch]
            except IndexError:
                self.logger.error(f"ch {ch} is out of bounds.")
                return None
            return ChannelStatus(running, runtime, total, starts)
        elif isinstance(self.meas, (tt.Correlation, tt.FileWriter)):
            # Correlation / Raw events measurement
            starts = 0
            try:
                total = counts[ch]
            except IndexError:
                self.logger.error(f"ch {ch} is out of bounds.")
                return None
            return ChannelStatus(running, runtime, total, starts)
        else:
            self.logger.error("Measurement is not configured")
            return None

    def get_raw_events(self) -> str | None:
        if not self._save_file_name:
            self.logger.error("save file name has not been set.")
            return None

        ttbin_name = os.path.splitext(self._save_file_name)[0] + ".ttbin"
        ttbin_path = os.path.join(self._raw_events_dir, ttbin_name)

        # assuming all data can be load on memory
        reader = tt.FileReader(ttbin_path)
        events = []
        while reader.hasData():
            events.append(np.array(reader.getData(1_000_000).getTimestamps(), dtype=np.int64))

        self.logger.debug("Start sorting raw events")
        data = np.concatenate(events)
        # in-place sort to reduce memory consumption? (effect not confirmed)
        data.sort()
        self.logger.debug("Finished sorting raw events")

        h5_name = os.path.splitext(self._save_file_name)[0] + ".h5"
        h5_path = os.path.join(self._raw_events_dir, h5_name)

        self.logger.info(f"Saving converted raw events to {h5_path}")
        success = save_h5(h5_path, RawEvents(data), RawEvents, self.logger, compression="lzf")
        if success:
            return h5_name
        else:
            return None

    # Standard API

    def close(self):
        tt.freeTimeTagger(self.tagger)

    def start(self) -> bool:
        """Clear and start a new measurement."""

        if self.sync is None:
            return self.fail_with("Measurement is not configured.")
        if self._duration_ps:
            self.sync.startFor(self._duration_ps, clear=True)
        else:
            self.sync.clear()
            self.sync.start()
        self._tstart = time.time()
        return True

    def stop(self) -> bool:
        """Stop measurement if running."""

        if self.sync is None:
            return self.fail_with("Measurement is not configured.")
        self.sync.stop()
        return True

    def resume(self) -> bool:
        """Resume measurement."""

        if self._duration_ps:
            self.sync.startFor(self._duration_ps, clear=False)
        else:
            self.sync.start()
        return True

    def configure(self, params: dict, label: str = "", group: str = "") -> bool:
        if label == "histogram":
            if all([k in params for k in ("base_config", "range", "bin")]):
                return self.configure_histogram(
                    params["base_config"], params["range"], params["bin"]
                )
        elif label == "correlation":
            if all([k in params for k in ("base_config", "range", "bin")]):
                return self.configure_correlation(
                    params["base_config"], params["range"], params["bin"]
                )
        elif label == "raw_events":
            if all([k in params for k in ("base_config", "save_file")]):
                return self.configure_raw_events(params["base_config"], params["save_file"])
        else:
            return self.fail_with(f"invalid label {label}")

        return self.fail_with("invalid configure() params keys.")

    def set(self, key: str, value=None) -> bool:
        if key == "clear":
            return self.clear()
        elif key == "sweeps":
            if value == 0:
                return True
            else:
                return self.fail_with("Cannot set sweeps limit")
        elif key == "duration":
            return self.set_duration(value)
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key == "range_bin":
            return self.get_range_bin()
        elif key == "bin":
            return self.get_timebin()
        elif key == "data":
            return self.get_data(args)
        elif key == "data_normalized":
            return self.get_data_normalized(args)
        elif key == "data_roi":
            return self.get_data_roi(args["ch"], args["roi"])
        elif key == "status":
            return self.get_status(args)
        elif key == "raw_events":
            return self.get_raw_events()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
