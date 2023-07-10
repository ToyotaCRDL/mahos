#!/usr/bin/env python3

import typing as T
import time

import numpy as np
from numpy.typing import NDArray

from mahos.inst.overlay.overlay import InstrumentOverlay
from mahos.inst.interface import InstrumentInterface


class IVSweeper(InstrumentOverlay):
    def __init__(self, name, conf, prefix=None):
        InstrumentOverlay.__init__(self, name, conf=conf, prefix=prefix)
        self.source = self.conf.get("source")
        self.meter = self.conf.get("meter")
        self.add_instruments(self.source, self.meter)

        self._voltages = None
        self._delay_sec = 0.0
        self._running = False

    def sweep_once(self) -> T.Optional[NDArray]:
        if self._voltages is None or not self._running:
            self.logger.error("Sweep has not been configured / started yet.")
            return None

        currents = []
        for v in self._voltages:
            self.source.set_voltage(v)
            time.sleep(self._delay_sec)
            currents.append(self.meter.get_meas(v))

        return np.array(currents)

    # Standard API

    def configure(self, params: dict) -> bool:
        if not self.check_required_params(params, ("start", "stop", "num")):
            return False
        self._voltages = np.linspace(params["start"], params["stop"], params["num"])
        self._delay_sec = params.get("delay_sec", 0.0)
        return self.meter.configure_current_meas(
            range_=params.get("range", 0), navg=params.get("navg", 1)
        )

    def start(self) -> bool:
        self._running = True
        return self.source.start()

    def stop(self) -> bool:
        self._running = False
        return self.source.set_voltage(0.0) and self.source.stop()

    def get(self, key: str, args=None):
        if key == "sweep":
            return self.sweep_once()
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None


class IVSweeperInterface(InstrumentInterface):
    def sweep_once(self) -> T.Optional[NDArray]:
        return self.get("sweep")
