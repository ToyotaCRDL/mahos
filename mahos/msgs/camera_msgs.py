#!/usr/bin/env python3

"""
Message Types for Camera stream.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from .data_msgs import Data
from .inst_camera_msgs import FrameResult


class Image(Data):
    def __init__(self, params: dict | None = None):
        self.set_version(0)
        self.init_params(params)

        self.image = None
        self.running: bool = False
        self.time: float = 0.0
        self.count: int = 0

    def update(self, result: FrameResult):
        self.image = result.frame
        self.time = result.time
        self.count = result.count
