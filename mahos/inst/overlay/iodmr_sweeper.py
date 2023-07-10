#!/usr/bin/env python3

import time
import threading

import numpy as np

from .overlay import InstrumentOverlay
from ...util.locked_queue import LockedQueue


class IODMRSweeperCommand(InstrumentOverlay):
    """IODMRSweeperCommand provides primitive operations for Imaging ODMR sweep.

    This class performs the sweep by issuing commands every step.
    Thus, the performance (sweep speed) might not be very good.

    """

    def __init__(self, name, conf, prefix=None):
        InstrumentOverlay.__init__(self, name, conf=conf, prefix=prefix)
        self.sg = self.conf.get("sg")
        self.camera = self.conf.get("camera")
        self.add_instruments(self.sg, self.camera)

        self._queue_size = self.conf.get("queue_size", 8)
        self._queue = LockedQueue(self._queue_size)
        self._stop_ev = self._thread = None
        self.running = False

    def _set_attrs(self, params):
        self.start_f, self.stop_f = params["start"], params["stop"]
        self.num = params["num"]
        self.freqs = np.linspace(self.start_f, self.stop_f, self.num)
        self.power = params["power"]
        self.exposure_delay = params["exposure_delay"]
        self.exposure_time = params["exposure_time"]
        self.burst_num = params.get("burst_num", 1)
        self.binning = params.get("binning", 1)
        self.roi = params.get("roi")

    def get_frames(self):
        return self._queue.pop_block()

    def sweep_loop(self, ev: threading.Event):
        while True:
            frames = []
            for f in self.freqs:
                self.sg.set_freq_CW(f)
                # self.sg.flush_writebuf()
                time.sleep(self.exposure_delay)
                res = self.camera.get_frame()
                if res.is_empty():
                    self.logger.error("Failed to capture frame. Quitting sweep loop.")
                    return
                if ev.is_set():
                    self.logger.info("Quitting sweep loop.")
                    return
                frames.append(res.frame)
            self._queue.append(np.array(frames))

    # Standard API

    def configure(self, params: dict) -> bool:
        if not self.check_required_params(
            params, ("start", "stop", "num", "power", "exposure_delay", "exposure_time")
        ):
            return False

        self._set_attrs(params)
        self._queue = LockedQueue(self._queue_size)

        if not (
            self.sg.configure_CW(self.start_f, self.power)
            and self.camera.configure_soft_trigger(
                self.exposure_time, burst_num=self.burst_num, binning=self.binning, roi=self.roi
            )
        ):
            return self.fail_with("failed to configure sg or camera.")

        return True

    def start(self) -> bool:
        if self.running:
            self.logger.warn("start() is called while running.")
            return True

        if not self.sg.set_output(True):
            return self.fail_with("Failed to start sg.")
        if not self.camera.start():
            return self.fail_with("Failed to start camera.")

        self._stop_ev = threading.Event()
        self._thread = threading.Thread(target=self.sweep_loop, args=(self._stop_ev,))
        self._thread.start()

        self.running = True

        return True

    def stop(self) -> bool:
        if not self.running:
            return True
        self.running = False

        self.logger.info("Stopping sweeper.")

        self._stop_ev.set()
        self._thread.join()
        success = self.sg.set_output(False) and self.camera.stop()
        # self.sg.flush_writebuf()

        if not success:
            return self.fail_with("failed to stop sg or camera.")

        return True

    def get(self, key: str, args=None):
        if key == "frames":
            return self.get_frames()
        elif key == "bounds":
            return self.sg.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
