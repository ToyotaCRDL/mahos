#!/usr/bin/env python3

import numpy as np

from ...util import conv, fit
from .overlay import InstrumentOverlay
from ...msgs.confocal_msgs import ScanMode


DUMMY_CAPABILITY = (
    ScanMode.COM_NOWAIT,
    ScanMode.COM_COMMAND,
    ScanMode.COM_DIPOLL,
    ScanMode.ANALOG,
)


class ConfocalScanner_mock(InstrumentOverlay):
    """ConfocalScanner provides primitive operations for confocal scanning."""

    def __init__(self, name, conf, prefix):
        InstrumentOverlay.__init__(self, name, conf, prefix)
        self.piezo = self.conf["piezo"]
        self.pds = [self.conf.get(n) for n in self.conf.get("pds", ["pd0", "pd1"])]
        self.add_instruments(self.piezo, *self.pds)

    def get_capability(self):
        return DUMMY_CAPABILITY

    def get_line(self):
        if self.i >= self.ynum:
            ar = None
        else:
            ar = self.debug_data[:, self.i]
        self.i += 1
        return ar

    # Standard API

    def configure(self, params: dict) -> bool:
        self.params = params

        req_keys = (
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "xnum",
            "ynum",
            "z",
            "direction",
            "time_window",
        )
        if not self.check_required_params(params, req_keys):
            return False

        self.i = 0
        self.xmin, self.xmax = params["xmin"], params["xmax"]
        self.ymin, self.ymax = params["ymin"], params["ymax"]
        self.xnum, self.ynum = params["xnum"], params["ynum"]
        self.xstep = conv.num_to_step(self.xmin, self.xmax, self.xnum)
        self.ystep = conv.num_to_step(self.ymin, self.ymax, self.ynum)
        self.z = params["z"]
        self.direction = params["direction"]
        self.time_window = params["time_window"]

        # Create dummy scan data
        Xin, Yin = np.mgrid[0 : self.xnum, 0 : self.ynum]
        cx, cy = self.xnum / 2.0 - 0.5, self.ynum / 2.0 - 0.5
        wx, wy = self.xnum / 4.0, self.ynum / 4.0
        bg = 20.0  # background
        self.debug_data = (
            fit.gaussian2d(100.0, cx, cy, wx, wy)(Xin, Yin)
            + np.random.normal(scale=10.0, size=Xin.shape)
            + bg
        )

        return True

    def get(self, key: str, args=None):
        if key == "line":
            return self.get_line()
        elif key == "capability":
            return self.get_capability()
        elif key == "unit":
            return "cps"
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def start(self) -> bool:
        self.logger.info("Starting mock scanner.")
        return True

    def stop(self) -> bool:
        self.logger.info("Stopping mock scaner.")
        return True
