#!/usr/bin/env python3

import numpy as np
import time
import threading

from ...util import conv
from .overlay import InstrumentOverlay
from ...msgs.confocal_msgs import ScanMode, ScanDirection, LineMode
from ..daq import BufferedEdgeCounter, AnalogIn


class ConfocalScannerMixin(object):
    def move_piezo_to_initial(self) -> bool:
        self.logger.info("Moving piezo to scan starting point.")
        return self.piezo.set_target_pos(self.scan_array[0, :])

    def _set_attrs(self, params: dict):
        self.lines_sent = 0
        self.xmin, self.xmax = params["xmin"], params["xmax"]
        self.ymin, self.ymax = params["ymin"], params["ymax"]
        self.xnum, self.ynum = params["xnum"], params["ynum"]
        self.xstep = conv.num_to_step(self.xmin, self.xmax, self.xnum)
        self.ystep = conv.num_to_step(self.ymin, self.ymax, self.ynum)
        self.z = params["z"]
        self.direction: ScanDirection = params["direction"]
        self.line_mode: LineMode = params["line_mode"]
        self.time_window = params["time_window"]
        self.delay = params.get("delay", 0.0)
        self.oversample = params.get("oversample", 1)

    def _make_scan_array(self, ndummy):
        def xar_zigzag(x, ylen):
            # x[:1] and x[-1:] are dummy (repeated) point at the start of each line
            dummy_a = np.repeat(x[:1], ndummy)
            dummy_d = np.repeat(x[-1:], ndummy)
            xar = np.concatenate((dummy_a, x, dummy_d, x[::-1]))
            xar = np.tile(xar, ylen // 2)
            if ylen % 2 == 1:
                xar = np.concatenate((xar, dummy_a, x))
            return xar

        def xar_ascend(x, ylen):
            dummy = np.repeat(x[:1], ndummy)
            xar = np.concatenate((dummy, x))
            xar = np.tile(xar, ylen)
            return xar

        def xar_descend(x, ylen):
            dummy = np.repeat(x[-1:], ndummy)
            xar = np.concatenate((dummy, x[::-1]))
            xar = np.tile(xar, ylen)
            return xar

        xs = np.linspace(self.xmin, self.xmax, self.xnum)
        ys = np.linspace(self.ymin, self.ymax, self.ynum)
        # add dummy (repeated) sampling points at the start of each line
        self.xlen = len(xs) + ndummy
        ylen = len(ys)
        yar = np.repeat(ys, self.xlen)
        if self.line_mode == LineMode.ZIGZAG:
            xar = xar_zigzag(xs, ylen)
        elif self.line_mode == LineMode.ASCEND:
            xar = xar_ascend(xs, ylen)
        else:  # LineMode.DESCEND
            xar = xar_descend(xs, ylen)
        a = np.column_stack((xar, yar))

        z = self.z * np.ones(shape=(a.shape[0], 1))
        if self.direction == ScanDirection.XY:
            self.scan_array = np.concatenate((a, z), axis=1)
        elif self.direction == ScanDirection.XZ:
            self.scan_array = np.concatenate((a[:, :1], z, a[:, 1:]), axis=1)
        elif self.direction == ScanDirection.YZ:
            self.scan_array = np.concatenate((z, a), axis=1)


class ConfocalScannerAnalog(InstrumentOverlay, ConfocalScannerMixin):
    """ConfocalScannerAnalog provides primitive operations for confocal scanning.

    This class performs the scanning with DAQ analog outputs.

    """

    def __init__(self, name, conf, prefix=None):
        InstrumentOverlay.__init__(self, name, conf=conf, prefix=prefix)
        self.clock = self.conf.get("clock")
        self.divider = self.conf.get("divider")
        self.piezo = self.conf.get("piezo")
        self.pds = [self.conf.get(n) for n in self.conf.get("pds", ["pd0", "pd1"])]
        self.add_instruments(self.clock, self.piezo, *self.pds)

        self.line_timeout = None
        self.loop_stop_ev = self.loop_thread = None
        self.running = False

    def get_capability(self):
        return (ScanMode.ANALOG,)

    def line_loop(self, ev: threading.Event):
        for i in range(self.ynum):
            self.piezo.join(self.line_timeout)
            self.clock.join(self.line_timeout)
            self.logger.info(f"Scanned line #{i}")
            self.piezo.stop()
            self.clock.stop()
            if i >= self.ynum - 1:
                self.logger.info("Scan finished.")
                return
            if ev.is_set():
                self.logger.info("Quitting line loop.")
                return
            self.clock.configure(self.params_clock)
            self.piezo.configure(self.params_piezo)
            self.piezo.start()
            self.piezo.write_scan_array(
                self.scan_array[(i + 1) * self.xlen : (i + 2) * self.xlen, :]
            )
            self.clock.start()

    def get_line(self):
        if self.lines_sent >= self.ynum:
            return None

        # discard dummy samples at the start.
        lines = [pd.pop_block()[self.dummy_samples :] for pd in self.pds]
        data = np.sum(lines, axis=0)
        if self.line_mode == LineMode.DESCEND or (
            self.line_mode == LineMode.ZIGZAG and self.lines_sent % 2 == 1
        ):
            data = data[::-1]
        self.lines_sent += 1
        return data

    # Standard API

    def close(self):
        self.stop()

    def configure(self, params: dict) -> bool:
        if not self.check_required_params(
            params,
            (
                "mode",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "xnum",
                "ynum",
                "z",
                "direction",
                "time_window",
            ),
        ):
            return False

        self.line_timeout = params.get("line_timeout", 20.0)
        if not isinstance(params["direction"], ScanDirection):
            return self.fail_with("direction must be ScanDirection.")
        if params["mode"] != ScanMode.ANALOG:
            return self.fail_with("Unsupported mode: {}".format(params["mode"]))
        self.dummy_samples = params.get("dummy_samples", 1)
        if not (isinstance(self.dummy_samples, int) and self.dummy_samples > 0):
            return self.fail_with("dummy_samples must be positive integer")

        self._set_attrs(params)
        self._make_scan_array(self.dummy_samples)

        if (
            self.divider is None or isinstance(self.pds[0], BufferedEdgeCounter)
        ) and self.oversample != 1:
            return self.fail_with(f"oversample == {self.oversample} is not supported.")
        if isinstance(self.pds[0], AnalogIn) and self.oversample < 1:
            return self.fail_with("oversample must be positive integer.")
        if not self._check_pd_rate():
            return False

        self.loop_stop_ev = self.loop_thread = None

        self.logger.info(f"Configured with dummy_samples == {self.dummy_samples}")

        return True

    def _check_pd_rate(self) -> bool:
        freq_pd = self.oversample / self.time_window
        for pd in self.pds:
            rate = pd.get_max_rate()
            if freq_pd > rate:
                msg = f"PD freq ({freq_pd:.2f} Hz) exceeds max rate ({rate:.2f}) Hz"
                return self.fail_with(msg)
        return True

    def _init_piezo(self) -> bool:
        success = self.piezo.stop() and self.piezo.configure({}) and self.piezo.start()
        if not success:
            return self.fail_with("Failed to initialize piezo.")

        size = self.piezo.get_scan_buffer_size()
        if self.xlen > size:
            return self.fail_with(f"xlen {self.xlen} is greater than buffer size {size}")

        success = self.move_piezo_to_initial() and self.piezo.stop()

        if not success:
            return self.fail_with("Failed to move piezo to initial.")
        return True

    def _start_piezo_pd(self) -> bool:
        freq_piezo = 1.0 / self.time_window
        freq_pd = freq_piezo * self.oversample
        total_samples = len(self.scan_array)

        if self.oversample == 1:
            self.params_clock = {"freq": freq_piezo, "samples": self.xlen, "finite": True}
            if not self.clock.configure(self.params_clock):
                return self.fail_with("failed to configure clock.")
            clock_pd = clock_piezo = self.clock.get_internal_output()
        else:
            self.params_clock = {
                "freq": freq_pd,
                "samples": self.xlen * self.oversample,
                "finite": True,
            }
            if not self.clock.configure(self.params_clock):
                return self.fail_with("failed to configure clock.")
            clock_pd = self.clock.get_internal_output()
            params_divider = {
                "ratio": self.oversample,
                "samples": total_samples,
                "source": clock_pd,
                "finite": True,
            }
            if not self.divider.configure(params_divider):
                return self.fail_with("failed to configure divider.")
            clock_piezo = self.divider.get_internal_output()

        self.params_piezo = {
            "clock_mode": True,
            "clock": clock_piezo,
            "samples": self.xlen,
            "rate": freq_piezo,
        }
        params_pd = {
            "cb_samples": self.xlen,
            "samples": total_samples,
            "rate": freq_pd,
            "finite": True,
            "every": False,
            "clock": clock_pd,
            "time_window": self.time_window,  # only for APDCounter
            "clock_mode": True,  # only for AnalogIn
            "oversample": self.oversample,  # only for AnalogIn
        }

        success = (
            self.piezo.configure(self.params_piezo)
            and self.piezo.start()
            and self.piezo.write_scan_array(self.scan_array[: self.xlen, :])
        )
        if not success:
            return self.fail_with("failed to configure and start piezo.")

        if not (
            all([pd.configure(params_pd) for pd in self.pds])
            and all([pd.start() for pd in self.pds])
        ):
            return self.fail_with("failed to configure and start PD.")

        return True

    def start(self) -> bool:
        if self.running:
            self.logger.warn("start() is called while running.")
            return True

        if not self._init_piezo():
            return False
        if not self._start_piezo_pd():
            return False

        if self.oversample != 1:
            if not self.divider.start():
                return self.fail_with("failed to start divider.")
        if not self.clock.start():
            return self.fail_with("failed to start clock.")

        self.loop_stop_ev = threading.Event()
        self.loop_thread = threading.Thread(target=self.line_loop, args=(self.loop_stop_ev,))
        self.loop_thread.start()

        self.running = True

        return True

    def stop(self) -> bool:
        if not self.running:
            return True
        self.running = False

        self.logger.info("Stopping scanner.")

        self.loop_stop_ev.set()
        self.loop_thread.join()
        success = self.clock.stop() and all([pd.stop() for pd in self.pds]) and self.piezo.stop()
        if self.oversample != 1:
            success &= self.divider.stop()

        if not success:
            self.logger.error("failed to stop piezo, PD, or clock.")
            return False

        return True

    def get(self, key: str, args=None):
        if key == "line":
            return self.get_line()
        elif key == "capability":
            return self.get_capability()
        elif key == "unit":
            return self.pds[0].get("unit")
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class ConfocalScannerCommand(InstrumentOverlay, ConfocalScannerMixin):
    """ConfocalScannerCommand provides primitive operations for confocal scanning.

    This class performs the scanning with piezo commands.

    TODO: This class has not been tested using real hardware.

    """

    def __init__(self, name, conf, prefix=None):
        InstrumentOverlay.__init__(self, name, conf=conf, prefix=prefix)
        self.pulser = self.conf.get("pulser")
        self.piezo = self.conf.get("piezo")
        self.pds = [self.conf.get(n) for n in self.conf.get("pds", ["pd0", "pd1"])]
        self.add_instruments(self.pulser, self.piezo, *self.pds)

        # TODO: remove this message after testing.
        self.logger.warn("Use with care! This class has not been tested using real hardware!")

    def get_capability(self):
        # ScanMode.COM_DIPOLL is not implemented yet.
        return (ScanMode.COM_NOWAIT, ScanMode.COM_COMMAND)

    def get_line(self):
        if self.lines_sent >= self.ynum:
            return None

        lines = [pd.pop_block() for pd in self.pds]
        data = np.sum(lines, axis=0)
        if self.lines_sent % 2 == 1:
            data = data[::-1]
        self.lines_sent += 1
        return data

    def init_inst(self):
        self.move_piezo_to_initial()
        p0 = {
            "cb_samples": self.xnum,
            "samples": len(self.scan_array),
            "time_window": self.time_window,
            "finite": True,
            "every": True,
            "every1_handler": self.move_wait_piezo,
        }
        self.pds[0].configure(p0)
        p1 = {
            "cb_samples": self.xnum,
            "samples": len(self.scan_array),
            "time_window": self.time_window,
            "finite": True,
            "every": False,
        }
        for pd in self.pds[1:]:
            pd.configure(p1)

        for pd in self.pds:
            pd.start()

        self.pulser.out_low()  # just for safety

    def move_piezo(self):
        if self.scan_array[self.i, 0] != self.scan_array[self.i - 1, 0]:
            self.piezo.set_target_X(self.scan_array[self.i, 0])
        if self.scan_array[self.i, 1] != self.scan_array[self.i - 1, 1]:
            self.piezo.set_target_Y(self.scan_array[self.i, 1])
        if self.scan_array[self.i, 2] != self.scan_array[self.i - 1, 2]:
            self.piezo.set_target_Z(self.scan_array[self.i, 2])

    def wait_piezo_command(self):
        for i in range(100000):
            if all(self.piezo.query_on_target()):
                return i
        return -1

    def move_wait_piezo(self):
        if self.i >= len(self.scan_array):
            return  # there is no next for the last point

        t = time.perf_counter()
        if self.i > 0:
            self.move_piezo()
        t1 = time.perf_counter()
        self.logger.debug("move piezo in ", t1 - t)

        n = self.wait_piezo()
        if n < 0:
            self.logger.error("Piezo on-target timed out!")
            self.stop()
        t2 = time.perf_counter()
        self.logger.debug("Waited piezo for {}".format(t2 - t1))

        if self.delay:
            self.logger.debug("Delay for {:2f} ms".format(self.delay * 1e3))
            time.sleep(self.delay)

        self.logger.debug("{:d} th move compl. in {}".format(self.i, t2 - self.tim))
        self.tim = t2
        self.i += 1

        # trig
        self.pulser.pulse()

    # Standard API

    def configure(self, params: dict) -> bool:
        if not self.check_required_params(
            params,
            (
                "mode",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "xnum",
                "ynum",
                "z",
                "direction",
                "time_window",
            ),
        ):
            return False

        if not isinstance(params["direction"], ScanDirection):
            self.logger.error("direction must be ScanDirection.")
            return False

        if params["mode"] == ScanMode.COM_NOWAIT:
            self.wait_piezo = lambda: 0
        elif params["mode"] == ScanMode.COM_COMMAND:
            self.wait_piezo = self.wait_piezo_command
        else:
            self.logger.error("Unsupported mode: {}".format(params["mode"]))
            return False

        self.i = 0
        self._set_attrs(params)
        self._make_scan_array(0)

        return True

    def start(self) -> bool:
        self.init_inst()

        self.tim = time.perf_counter()
        self.move_wait_piezo()

        return True

    def stop(self) -> bool:
        self.logger.info("Stopping scaner.")
        for pd in self.pds:
            pd.stop()
        return True

    def get(self, key: str, args=None):
        if key == "line":
            return self.get_line()
        elif key == "capability":
            return self.get_capability()
        elif key == "unit":
            return self.pds[0].get("unit")
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
