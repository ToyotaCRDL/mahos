#!/usr/bin/env python3

"""
Worker for Camera stream.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np

from ..msgs.camera_msgs import Image
from ..msgs import param_msgs as P
from ..inst.camera_interface import CameraInterface
from ..msgs.inst_camera_msgs import FrameResult
from .common_worker import Worker


class Poller(Worker):
    """Worker for Camera streaming."""

    def __init__(self, cli, logger, conf: dict):
        Worker.__init__(self, cli, logger)
        self.camera = CameraInterface(cli, "camera")
        self.add_instrument(self.camera)

        self._transform_conf = conf

        self.image = Image()
        self.result = FrameResult()

    def get_param_dict(self, name: str) -> P.ParamDict[str, P.PDValue] | None:
        # TODO query bounds from camera ?
        d = P.ParamDict(
            exposure_time=P.FloatParam(10e-3, 1e-6, 10.0, unit="s", SI_prefix=True),
            roi=P.ParamDict(
                width=P.IntParam(100, minimum=0, optional=True, enable=False),
                height=P.IntParam(100, minimum=0, optional=True, enable=False),
                woffset=P.IntParam(0, minimum=0, optional=True, enable=False),
                hoffset=P.IntParam(0, minimum=0, optional=True, enable=False),
            ),
            binning=P.IntParam(1, 0, 4),
        )
        return d

    def start(self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is None:
            params = {}
        else:
            params = P.unwrap(params)
        if not self.camera.lock():
            return self.fail_with_release("Error acquiring camera's lock")

        if not self.camera.configure_continuous(
            params.get("exposure_time", 10e-3),
            frame_rate_Hz=params.get("frame_rate"),
            binning=params.get("binning", 0),
            roi=params.get("roi"),
        ):
            return self.fail_with_release("Error configuring camera.")

        if not self.camera.start():
            return self.fail_with_release("Error starting camera.")

        self.image = Image(params)
        self.image.running = True
        self.result = FrameResult()
        self.logger.info("Started poller.")

        return True

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.image.running:
            return False

        success = self.camera.stop() and self.camera.release()

        self.image.running = False
        if success:
            self.logger.info("Stopped poller.")
        else:
            self.logger.error("Error stopping poller.")
        return success

    def transform(self, image: np.ndarray):
        if self._transform_conf.get("swap", False):
            image = image.T
        # after swap, 0-th axis should be row (Height), 1-st axis should be column (Width)
        if self._transform_conf.get("flipH", False):
            image = np.flip(image, axis=0)
        if self._transform_conf.get("flipW", False):
            image = np.flip(image, axis=1)
        return image

    def work(self) -> bool:
        if not self.image.running:
            return False

        res = self.camera.get_frame()
        if not res.is_empty():
            try:
                self.result = res
                self.result.frame = self.transform(self.result.frame)
                self.image.update(self.result)
                return True
            except Exception:
                self.logger.exception("Ill-formed data received.")
                return False

    def image_msg(self) -> Image | None:
        if self.result.is_empty():
            return None
        return self.image
