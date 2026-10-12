#!/usr/bin/env python3

"""
Workers for Confocal.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import copy

import numpy as np

from mahos.util.timer import IntervalTimer
from mahos_dq.msgs.confocal_msgs import (
    PiezoPos,
    Image,
    Trace,
    Axis,
    ScanDirection,
    ScanMode,
    LineMode,
)
from mahos.msgs import param_msgs as P
from mahos.inst.piezo_interface import PiezoInterface
from mahos.inst.daq_interface import ClockSourceInterface
from mahos.inst.pd_interface import PDInterface
from mahos_dq.inst.overlay.confocal_scanner_interface import ConfocalScannerInterface
from mahos.meas.common_worker import Worker

DEFAULT_TRACER_SIZE = 500


class Piezo(Worker):
    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger)
        self.piezo = PiezoInterface(cli, "piezo")
        self.add_instruments(self.piezo)

        self.interval_sec = conf.get("interval_sec", 0.5)
        self._range_trials = conf.get("range_trials", 5)

        self.pos = PiezoPos()

        self.running = False
        self.timer = None

    def get_range(self) -> bool:
        if self.pos.has_range():
            return True

        ret = self.piezo.get_range()
        if ret is not None:
            self.logger.info("Got piezo range")
            self.pos.x_range, self.pos.y_range, self.pos.z_range = ret
            return True
        else:
            if self._range_trials >= 0:
                self._range_trials -= 1
                self.logger.error("Couldn't get piezo range")
            return False

    def get_pos(self) -> PiezoPos:
        # range is queried only once.
        # try here because the query on init may fail if server is not ready.
        # target is queried here because move is allowed
        # even when Confocal is not in Interact state.
        # don't query actual position here because it may take longer time for query
        # due to A/D converter etc. Thus we poll at update_pos() (usually at slower rate).

        self.get_range()
        self.get_target()
        return self.pos

    def get_target(self):
        tgt = self.piezo.get_target()
        if tgt is None:
            # self.logger.error("Failed to get piezo target pos.")
            return None
        self.pos.x_tgt, self.pos.y_tgt, self.pos.z_tgt = tgt
        return tgt

    def start(self) -> bool:
        success = self.piezo.lock() and self.piezo.configure_interactive() and self.piezo.start()
        if not success:
            return self.fail_with_release("Error starting piezo polling.")

        self.running = True
        self.timer = IntervalTimer(self.interval_sec)
        self.logger.info("Started polling piezo.")
        return True

    def stop(self) -> bool:
        # avoid double-stop
        if not self.running:
            return False

        success = self.piezo.stop() and self.piezo.release()
        if success:
            self.timer = None
            self.running = False
            self.logger.info("Stopped polling piezo.")
        else:
            self.logger.error("Error releasing lock for piezo.")
        return success

    def work(self):
        if not self.running:
            return

        if self.timer.check():
            self.update_pos()

    def update_pos(self):
        """Update actual position."""

        pos_ont = self.piezo.get_pos_ont()
        if pos_ont is None:
            self.logger.error("Failed to get piezo pos_ont.")
            return
        pos, ont = pos_ont

        self.pos.x, self.pos.y, self.pos.z = pos
        self.pos.x_ont, self.pos.y_ont, self.pos.z_ont = ont

    def move(self, ax: Axis | list[Axis], val: float | list[float]):
        success = self.piezo.set_target({"ax": ax, "pos": val})
        if not success:
            self.logger.error(f"Failed to move {ax}.")
        return success


class Tracer(Worker):
    """Tracer.

    Apart from main confocal state, this class has local state
    which determines if the Trace message should be updated.
    This state is operated through pause_msg() and resume_msg().
    By confocal state change (start()), the pause is automatically canceled.

    """

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger, conf)
        self._pd_spectrum = self.conf.get("pd_spectrum", False)
        self.clock = None if self._pd_spectrum else ClockSourceInterface(cli, "clock")
        self.pd_names = conf.get("pd_names", ["pd0", "pd1"])
        self.pds = [PDInterface(cli, n) for n in self.pd_names]
        self.pd_channels = conf.get("pd_channels", [1] * len(self.pds))
        if isinstance(self.pd_channels, int):
            if len(self.pds) != 1:
                raise ValueError("pd_channels as int is only valid when len(pd_names) == 1.")
            self.pd_channels = [self.pd_channels]
        if len(self.pd_channels) != len(self.pds):
            raise ValueError("len(pd_channels) must match len(pd_names).")
        if any(ch < 1 for ch in self.pd_channels):
            raise ValueError("all pd_channels values must be positive.")
        self.add_instruments(self.clock, *self.pds)

        self.interval_sec = self.conf.get("interval_sec", 0.5)
        self.size = self.conf.get("size", DEFAULT_TRACER_SIZE)
        self.cb_samples = self.conf.get("samples", 5)
        self.time_window_sec = self.conf.get("time_window_sec", 0.01)
        self.pd_bounds = self.conf.get("pd_bounds", (-10.0, 10.0))
        self._pd_data_transfer = self.conf.get("pd_data_transfer")

        if "pd_rate" in self.conf:
            self.pd_rate = self.conf["pd_rate"]
            if "oversample" in self.conf:
                msg = "pd_rate and oversample are both defined in conf. oversample is disregarded."
                self.logger.warn(msg)
            self.oversample = None
        else:
            self.pd_rate = None
            self.oversample = self.conf.get("oversample", 1)

        self.trace = Trace(
            size=self.size, channels=sum(self.pd_channels), _complex=conf.get("complex", False)
        )
        self.paused_trace: Trace | None = None

        self.running = False
        self.timer = None

    def start(self) -> bool:
        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        self.trace.yunit = self.pds[0].get_unit()

        if self.pd_rate is not None:
            freq = self.pd_rate
            oversample = int(round(freq * self.time_window_sec))
        else:
            freq = 1.0 / self.time_window_sec * self.oversample
            oversample = self.oversample
        buffer_size = self.cb_samples * self.conf.get("buffer_size_coeff", 200)
        params_clock = {"freq": freq, "samples": buffer_size, "finite": False}
        if self.clock is not None:
            if not self.clock.configure(params_clock):
                return self.fail_with_release("Failed to configure clock.")
            clock_pd = self.clock.get_internal_output()
        else:
            clock_pd = ""
        if not all(
            [
                pd.configure_stream(
                    self.cb_samples,
                    buffer_size,
                    freq,
                    self.time_window_sec,
                    bounds=self.pd_bounds,
                    oversample=oversample,
                    clock=clock_pd,
                    stamp=True,
                    data_transfer=self._pd_data_transfer,
                )
                for pd in self.pds
            ]
        ):
            return self.fail_with_release("Failed to configure PDs.")
        if not all([pd.start() for pd in self.pds]):
            for pd in self.pds:
                pd.stop()
            return self.fail_with_release("Failed to start PDs.")
        if self.clock is not None and not self.clock.start():
            self.clock.stop()
            for pd in self.pds:
                pd.stop()
            return self.fail_with_release("Failed to start clock.")

        self.running = True
        self.timer = IntervalTimer(self.interval_sec)
        self.paused_trace = None
        self.logger.info("Started tracer.")

        return True

    def stop(self) -> bool:
        # avoid double-stop
        if not self.running:
            return False

        success = all([pd.stop() for pd in self.pds]) and all([pd.release() for pd in self.pds])
        if self.clock is not None:
            success &= self.clock.stop() and self.clock.release()
        if success:
            self.timer = None
            self.running = False
            self.logger.info("Stopped tracer.")
        else:
            self.logger.error("Error releasing lock for PDs.")
        return success

    def clear_buf(self) -> bool:
        self.trace.clear()
        self.logger.info("Cleared trace buffer.")

    def pause_msg(self):
        self.paused_trace = copy.deepcopy(self.trace)
        self.logger.info("Paused message update.")

    def resume_msg(self):
        self.paused_trace = None
        self.logger.info("Resumed message update.")

    def work(self):
        if not self.running:
            return

        if self.timer.check():
            self.get_data()

    def _interp_stamps(self, stamps: tuple[int], last_stamp: int):
        """Interpolate block timestamps using integer nanoseconds."""

        if not last_stamp:
            last_stamp = stamps[0] - self.cb_samples * round(self.time_window_sec * 1e9)
        # Exclude the previous block's endpoint and include the current one.
        # Python integers avoid both float precision loss and intermediate int64 overflow.
        return np.fromiter(
            (
                int(t0) + (int(t1) - int(t0)) * i // self.cb_samples
                for t0, t1 in zip((last_stamp,) + stamps[:-1], stamps)
                for i in range(1, self.cb_samples + 1)
            ),
            dtype=np.int64,
            count=len(stamps) * self.cb_samples,
        ).view("datetime64[ns]")

    def _split_pd_data(self, data) -> list[np.ndarray]:
        """Split one PD queue item payload into trace-channel arrays."""

        if isinstance(data, list):
            return data
        else:
            return [data]

    def _append_trace_data(self, ch: int, new_data: np.ndarray, new_stamps: np.ndarray):
        if len(new_data) > self.size:
            new_data = new_data[: self.size]
            new_stamps = new_stamps[: self.size]
        # shift data. see pyqtgraph/examples/scrollingPlots.py
        s = len(new_data)
        self.trace.traces[ch][:-s] = self.trace.traces[ch][s:]
        self.trace.traces[ch][-s:] = new_data
        self.trace.stamps[ch][:-s] = self.trace.stamps[ch][s:]
        self.trace.stamps[ch][-s:] = new_stamps

    def get_data(self):
        trace_ch = 0
        for i, pd in enumerate(self.pds):
            data = pd.pop_all_opt()
            if data is not None:
                chunks, stamps = zip(*data)
                split_chunks = [self._split_pd_data(chunk) for chunk in chunks]
                channels = len(split_chunks[0])
                if channels != self.pd_channels[i]:
                    self.logger.error(
                        f"PD {self.pd_names[i]} returned {channels} channels, "
                        f"expected {self.pd_channels[i]}."
                    )
                    trace_ch += self.pd_channels[i]
                    continue
                if any(len(chunk) != channels for chunk in split_chunks):
                    self.logger.error(f"PD {self.pd_names[i]} returned inconsistent channels.")
                    trace_ch += self.pd_channels[i]
                    continue
                for j in range(channels):
                    new_data = np.hstack([chunk[j] for chunk in split_chunks])
                    last_stamp = self.trace.stamps[trace_ch + j][-1].view(np.int64)
                    new_stamps = self._interp_stamps(stamps, last_stamp)
                    self._append_trace_data(trace_ch + j, new_data, new_stamps)
            trace_ch += self.pd_channels[i]

    def is_paused(self):
        return self.paused_trace is not None

    def trace_msg(self) -> Trace:
        if self.is_paused():
            return self.paused_trace
        else:
            return self.trace


class Scanner(Worker):
    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger)
        self.scanner = ConfocalScannerInterface(cli, "scanner")
        self.add_instruments(self.scanner)
        self._conf = conf

        self.image = Image()

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        capability = self.scanner.get_capability()
        range_ = self.scanner.get_range()
        if capability is None or range_ is None:
            self.logger.error("Failed to get scanner capability or range.")
            return None

        d = P.ParamDict(
            xnum=P.IntParam(self._conf.get("xnum", 51), 1, 1001),
            ynum=P.IntParam(self._conf.get("ynum", 51), 1, 1001),
            delay=P.FloatParam(self._conf.get("delay", 0.0), 0.0, 100e-3),
            direction=P.EnumParam(ScanDirection, ScanDirection.XY),
            time_window=P.FloatParam(self._conf.get("time_window", 10e-3), 0.1e-3, 1.0),
            ident=P.UUIDParam(),
            mode=P.EnumParam(ScanMode, capability[0], capability),
            line_mode=P.EnumParam(LineMode, LineMode.ASCEND),
        )

        if ScanMode.ANALOG in capability:
            d["dummy_samples"] = P.IntParam(self._conf.get("dummy_samples", 10), 1, 1000)
            d["oversample"] = P.IntParam(self._conf.get("oversample", 1), 1, 10_000_000)

        if self._conf.get("pd_analog", False):
            lb, ub = self._conf.get("pd_bounds", (-10.0, 10.0))
            d["pd_bounds"] = [
                P.FloatParam(lb, -10.0, 10.0, unit="V"),
                P.FloatParam(ub, -10.0, 10.0, unit="V"),
            ]

        if ScanMode.COM_DIPOLL in capability:
            d["poll_samples"] = P.IntParam(self._conf.get("poll_samples", 1), 1, 1000)

        if label.startswith("xy"):
            xrange, yrange, zrange = range_[0], range_[1], range_[2]
            d["direction"].set(ScanDirection.XY)
        elif label.startswith("xz"):
            xrange, yrange, zrange = range_[0], range_[2], range_[1]
            d["direction"].set(ScanDirection.XZ)
        elif label.startswith("yz"):
            xrange, yrange, zrange = range_[1], range_[2], range_[0]
            d["direction"].set(ScanDirection.YZ)
        else:
            return d

        d["xmin"] = P.FloatParam(xrange[0], xrange[0], xrange[1], unit="um")
        d["xmax"] = P.FloatParam(xrange[1], xrange[0], xrange[1], unit="um")
        d["ymin"] = P.FloatParam(yrange[0], yrange[0], yrange[1], unit="um")
        d["ymax"] = P.FloatParam(yrange[1], yrange[0], yrange[1], unit="um")
        d["z"] = P.FloatParam((zrange[0] + zrange[1]) / 2.0, zrange[0], zrange[1], unit="um")

        return d

    def start(self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        params = P.unwrap(params)

        self.image = Image(params)
        if not self.scanner.lock():
            return self.fail_with_release("Error acquiring scanner lock.")
        self.image.cunit = self.scanner.get_unit()

        success = self.scanner.configure(params) and self.scanner.start()

        if not success:
            return self.fail_with_release("Error starting scanner.")

        self.logger.info("Started scanner.")
        self.image.running = True
        return True

    def append_line(self, line):
        if self.image.image is None:
            self.image.image = np.array(line, ndmin=2).T
        else:
            self.image.image = np.append(self.image.image, np.array(line, ndmin=2).T, axis=1)

    def work(self) -> bool:
        if not self.image.running:
            return False

        line = self.scanner.get_line()
        if line is None:
            return self.stop(aborting=False)
        else:
            self.append_line(line)
            return False

    def stop(self, aborting) -> bool:
        # avoid double-stop (abort status can be broken)
        # double-stop is not failure because this measurement will stop by itself.
        if not self.image.running:
            return True

        success = self.scanner.stop() and self.scanner.release()

        if success:
            self.logger.info(
                "Stopped scanner in {:.2f} sec.".format(self.image.finalize(aborting))
            )
        else:
            self.logger.error("Error stopping scanner.")
        return success

    def image_msg(self) -> Image:
        return self.image

    def running(self) -> bool:
        return self.image.running
