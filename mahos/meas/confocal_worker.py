#!/usr/bin/env python3

"""
Workers for Confocal.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import copy

import numpy as np

from ..util.timer import IntervalTimer
from ..msgs.confocal_msgs import PiezoPos, Image, Trace, Axis, ScanDirection, ScanMode, LineMode
from ..msgs import param_msgs as P
from ..inst.piezo_interface import PiezoInterface
from ..inst.daq_interface import ClockSourceInterface
from ..inst.pd_interface import PDInterface
from ..inst.overlay.confocal_scanner_interface import ConfocalScannerInterface
from .common_worker import Worker

DEFAULT_TRACER_SIZE = 500


class Piezo(Worker):
    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger)
        self.piezo = PiezoInterface(cli, "piezo")
        self.add_instrument(self.piezo)

        self.interval_sec = conf.get("interval_sec", 0.5)

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
            self.logger.error("Failed to get piezo target pos.")
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
        Worker.__init__(self, cli, logger)
        self.clock = ClockSourceInterface(cli, "clock")
        self.pds = [PDInterface(cli, n) for n in conf.get("pds", ["pd0", "pd1"])]
        self.add_instruments(self.clock, *self.pds)

        self.interval_sec = conf.get("interval_sec", 0.5)
        self.size = conf.get("size", DEFAULT_TRACER_SIZE)
        self.cb_samples = conf.get("samples", 5)
        self.oversample = conf.get("oversample", 1)
        self.time_window_sec = conf.get("time_window_sec", 0.01)

        self.trace = Trace(size=self.size, channels=len(self.pds))
        self.paused_trace: Trace | None = None

        self.running = False
        self.timer = None

    def start(self) -> bool:
        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        self.trace.yunit = self.pds[0].get_unit()

        freq = 1.0 / self.time_window_sec * self.oversample
        samples = self.cb_samples * 10  # large samples to assure enough buffer size
        params_clock = {"freq": freq, "samples": samples, "finite": False}
        params_pd = {
            "cb_samples": self.cb_samples,
            "samples": samples,
            "rate": freq,
            "finite": False,
            "every": False,
            "stamp": True,
            "clock": self.clock.get_internal_output(),
            "time_window": self.time_window_sec,  # only for APDCounter
            "clock_mode": True,  # only for AnalogIn
            "oversample": self.oversample,  # only for AnalogIn
        }
        success = (
            self.clock.configure(params_clock)
            and all([pd.configure(params_pd) for pd in self.pds])
            and all([pd.start() for pd in self.pds])
            and self.clock.start()
        )

        if not success:
            return self.fail_with_release("Error starting tracer.")

        self.running = True
        self.timer = IntervalTimer(self.interval_sec)
        self.paused_trace = None
        self.logger.info("Started tracer.")

        return True

    def stop(self) -> bool:
        # avoid double-stop
        if not self.running:
            return False

        success = (
            self.clock.stop()
            and all([pd.stop() for pd in self.pds])
            and self.clock.release()
            and all([pd.release() for pd in self.pds])
        )
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
        if not last_stamp:
            last_stamp = stamps[0] - self.cb_samples * round(self.time_window_sec * 1e9)
        return np.hstack(
            [
                np.linspace(t1, t0, num=self.cb_samples, endpoint=False, dtype="datetime64[ns]")[
                    ::-1
                ]
                for t0, t1 in zip((last_stamp,) + stamps[:-1], stamps)
            ]
        )

    def get_data(self):
        for i, pd in enumerate(self.pds):
            data = pd.pop_all_opt()
            if data is not None:
                d, t = zip(*data)
                new_data = np.hstack(d)
                new_stamps = self._interp_stamps(t, self.trace.stamps[i][-1].view(np.int64))
                if len(new_data) > self.size:
                    new_data = new_data[: self.size]
                    new_stamps = new_stamps[: self.size]
                # shift data. see pyqtgraph/examples/scrollingPlots.py
                s = len(new_data)
                self.trace.traces[i][:-s] = self.trace.traces[i][s:]
                self.trace.traces[i][-s:] = new_data
                self.trace.stamps[i][:-s] = self.trace.stamps[i][s:]
                self.trace.stamps[i][-s:] = new_stamps

    def is_paused(self):
        return self.paused_trace is not None

    def trace_msg(self) -> Trace:
        if self.is_paused():
            return self.paused_trace
        else:
            return self.trace


class Scanner(Worker):
    def __init__(self, cli, logger):
        Worker.__init__(self, cli, logger)
        self.scanner = ConfocalScannerInterface(cli, "scanner")
        self.add_instrument(self.scanner)

        self.image = Image()

    def get_param_dict(self) -> P.ParamDict[str, P.PDValue] | None:
        capability = self.scanner.get_capability()
        if capability is None:
            self.logger.error("Failed to get scanner capability.")
            return None

        d = P.ParamDict(  # TODO: accept direction and give ranges ?
            # "xmin": 10.0, "xmax": 20.0,
            # "ymin": 5.0, "ymax": 25.0,
            # "xnum": 10, "ynum": 20,
            # "z": 50.0,
            delay=P.FloatParam(0.0, 0.0, 100e-3),
            direction=P.EnumParam(ScanDirection, ScanDirection.XY),
            time_window=P.FloatParam(10e-3, 0.1e-3, 1.0),
            ident=P.UUIDParam(),
            mode=P.EnumParam(ScanMode, capability[0], capability),
            line_mode=P.EnumParam(LineMode, LineMode.ASCEND),
        )
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
