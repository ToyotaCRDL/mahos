#!/usr/bin/env python3

import threading
import time

import numpy as np

from .iodmr_sweeper import IODMRSweeperCommand


def lorentz(x, x0, gamma):
    """lorentz function which takes peak value 1 at x=x0."""
    return gamma**2 / ((x - x0) ** 2 + gamma**2)


class IODMRSweeperCommand_mock(IODMRSweeperCommand):
    """Mock actually calls sg and camera."""

    def sweep_loop(self, ev: threading.Event):
        while True:
            frames = []
            for f, c in zip(self.freqs, self._contrasts):
                self.sg.set_freq_CW(f)
                time.sleep(self.exposure_delay)
                res = self.camera.get_frame()
                if res.is_empty():
                    self.logger.error("Failed to capture frame. Quitting sweep loop.")
                    return
                if ev.is_set():
                    self.logger.info("Quitting sweep loop.")
                    return
                frames.append(res.frame * c)  # add artificial ODMR contrast
            self._queue.append(np.array(frames))

    def configure(self, params: dict) -> bool:
        success = IODMRSweeperCommand.configure(self, params)

        x = np.linspace(0, 1.0, self.num)
        c = 0.5
        self._contrasts = np.ones(self.num) - c * (
            lorentz(x, x0=0.25, gamma=0.03) + lorentz(x, x0=0.75, gamma=0.03)
        )

        return success
