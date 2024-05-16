#!/usr/bin/env python3

"""
Swabian Instruments Time Tagger module

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os
import time
import glob

import numpy as np
import TimeTagger as tt

from ..instrument import Instrument
from ...msgs.inst.tdc_msgs import ChannelStatus, RawEvents
from ...util.io import save_h5


class TimeTagger(Instrument):
    """Swabian Instruments Time Tagger.

    :param base_configs: Mapping from base config name to channels and levels definitions.
        Used in configure_histogram() etc.
    :type base_configs: dict[str, str]
    :param raw_events_dir: (default: user home) The directory to save RawEvents data.
    :type raw_events_dir: str
    :param ext_ref_clock: (default: 0) channel to use as external reference clock.
        Because it uses "software clock" of Time Tagger, clock must be supplied to
        one of input channel, not "CLK IN" connector.
        Putting 0 disables the software clock feature.
    :type ext_ref_clock: int
    :param ext_ref_clock_level: (default: 0.0) threshold level in volts for
        ext_ref_clock channel.
    :type ext_ref_clock_level: float
    :param ext_ref_clock_freq: (default: 10e6) frequency of ext_ref_clock.
    :type ext_ref_clock_freq: float
    :param remove_ttbin: (default: True) Remove raw events (.ttbin) file after load.
    :type remove_ttbin: bool
    :param serial: (default: "") Serial string to discriminate multiple TimeTaggers.
        Blank is fine if only one TimeTagger is connected.
    :type serial: str

    """

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        tt.setLogger(self.log_from_time_tagger)
        self.tagger = tt.createTimeTagger(self.conf.get("serial", ""))
        self.logger.info(f"Opened TimeTagger {self.tagger.getSerial()}")

        self._base_configs = self.conf.get("base_configs", {})
        self._raw_events_dir = os.path.expanduser(self.conf.get("raw_events_dir", "~"))
        self._remove_ttbin = self.conf.get("remove_ttbin", True)
        self.logger.debug(f"available base configs: {self._base_configs}")

        if not os.path.exists(self._raw_events_dir):
            raise FileNotFoundError(f"raw_events_dir {self._raw_events_dir} doesn't exist")

        # always use SynchronizedMeasurements to avoid autostart
        # on initialization of measurement class.
        self.sync: tt.SynchronizedMeasurements | None = None
        self.meas: tt.Correlation | tt.FileWriter | list[tt.Histogram] | None = None
        self.counter: tt.Counter | None = None
        self._duration_ps: int = 0

        self._raw_events_start_ch = None
        self._save_file_name = None
        self._tstart = time.time()
        self.trange = self.tbin = 0.0

        self.clock_ch = self.conf.get("ext_ref_clock", 0)
        self.clock_level = self.conf.get("ext_ref_clock_level", 0.0)
        self.clock_freq = self.conf.get("ext_ref_clock_freq", 10e6)
        self.set_reference_clock(self.clock_ch, self.clock_level, self.clock_freq)

    def log_from_time_tagger(self, level: int, msg: str):
        # TODO: want to log according to given level.
        # But I don't know the definition of levels.
        self.logger.info(f"[TimeTagger lib] level={level:d} {msg}")

    def set_reference_clock(self, ch: int, level: float = 0.0, freq: float = 10e6) -> bool:
        if ch:
            self.tagger.setTriggerLevel(ch, level)
            self.tagger.setSoftwareClock(ch, freq)
            self.logger.info(f"Software reference clock at ch={ch} freq={freq*1e-6:.1f} MHz.")
        else:
            self.tagger.disableSoftwareClock()
            self.logger.info("Disabled software reference clock.")
        return True

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
        save_file_path = os.path.join(self._raw_events_dir, save_file)

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
            filename=save_file_path,
            channels=conf["channels"],
        )
        self.counter = tt.Counter(self.sync.getTagger(), conf["channels"])
        self._duration_ps = 0

        self._raw_events_start_ch = conf.get("start_channel")
        self._save_file_name = save_file
        # unit of raw event (or tbin) is always ps
        # trange has no meaning.
        self.tbin = 1e-12
        self.trange = 0.0
        msg = f"Configured raw_events mode. start_ch: {self._raw_events_start_ch}"
        msg += f" file: {save_file_path}"
        self.logger.info(msg)

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
        if self._duration_ps:
            self.logger.info(f"Set duration limit of {self._duration_ps} ps")
        else:
            self.logger.info("Unset duration limit.")
        return True

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

        ttbin_name = self._save_file_name + ".ttbin"
        ttbin_path = os.path.join(self._raw_events_dir, ttbin_name)

        # assuming all data can be load on memory
        reader = tt.FileReader(ttbin_path)
        events = []
        start_stamp = None
        while reader.hasData():
            data = reader.getData(1_000_000)
            # stamps is np.int64 array
            stamps = data.getTimestamps()
            if self._raw_events_start_ch is not None:
                if start_stamp is None:
                    # find the first start event
                    # data.getChannels() is np.int32 array
                    indices = np.where(data.getChannels() == self._raw_events_start_ch)[0]
                    if len(indices) == 0:
                        self.logger.warn("Start stamp not found in this chunk")
                        continue
                    start_stamp = stamps[indices[0]]
                    self.logger.info(f"Found Start stamp: {start_stamp}")

                stamps = stamps[data.getChannels() != self._raw_events_start_ch]
            events.append(stamps)

        data = np.concatenate(events)
        if self._raw_events_start_ch is not None:
            if start_stamp is None:
                return self.fail_with("Cannot find time stamp of start channel.")
            data = data - start_stamp
            data = data[data >= 0]

        self.logger.debug("Start sorting raw events")
        # in-place sort to reduce memory consumption? (effect not confirmed)
        data.sort()
        self.logger.debug("Finished sorting raw events")

        h5_name = self._save_file_name + ".h5"
        h5_path = os.path.join(self._raw_events_dir, h5_name)

        self.logger.info(f"Saving converted raw events to {h5_path}")
        success = save_h5(h5_path, RawEvents(data), RawEvents, self.logger, compression="lzf")

        if self._remove_ttbin:
            head = os.path.join(self._raw_events_dir, self._save_file_name)
            for fn in glob.glob(head + "*.ttbin"):
                os.remove(fn)

        if success:
            return h5_name
        else:
            return None

    # Standard API

    def close_resources(self):
        tt.freeTimeTagger(self.tagger)

    def start(self, label: str = "") -> bool:
        """Clear and start a new measurement."""

        if self.sync is None:
            return self.fail_with("Measurement is not configured.")
        if self._duration_ps:
            self.sync.startFor(self._duration_ps, clear=True)
        else:
            self.sync.clear()
            self.sync.start()
        self._tstart = time.time()
        self.logger.info("Started measurement.")
        return True

    def stop(self, label: str = "") -> bool:
        """Stop measurement if running."""

        if self.sync is None:
            return self.fail_with("Measurement is not configured.")
        self.sync.stop()
        self.logger.info("Stopped measurement.")
        return True

    def resume(self, label: str = "") -> bool:
        """Resume measurement."""

        if self._duration_ps:
            self.sync.startFor(self._duration_ps, clear=False)
        else:
            self.sync.start()
        self.logger.info("Resumed measurement.")
        return True

    def configure(self, params: dict, label: str = "") -> bool:
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
