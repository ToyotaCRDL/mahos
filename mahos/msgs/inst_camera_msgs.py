#!/usr/bin/env python3

"""
Message Types for Camera Instruments.

.. This file is a part of MAHOS project.

"""

import typing as T

import numpy as np

from .inst_common_msgs import GetResult


class FrameResult(GetResult):
    """Result of get_frame().

    :param frame: None indicates timeout or invalid call (not running etc.).
    :param invalid: True indicates fatal error (buffer overflow etc.),
                    where measurement should be considered invalid.

    """

    def __init__(
        self,
        frame: T.Optional[np.ndarray] = None,
        count: int = 0,
        time: float = 0.0,
        invalid: bool = False,
    ):
        self.frame = frame
        self.count = count
        self.time = time
        self.invalid = invalid

    def is_empty(self) -> bool:
        return self.frame is None

    def is_invalid(self) -> bool:
        return self.invalid
