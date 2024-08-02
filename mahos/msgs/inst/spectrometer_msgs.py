#!/usr/bin/env python3

"""
Message Types for Spectrometer Instruments.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from ..data_msgs import Message


class Temperature(Message):
    """Spectrometer's detector temperature.

    :ivar current: Current temperature in degC.
    :ivar target: Target temperature in degC.

    """

    def __init__(self, current: float, target: float):
        self.current = current
        self.target = target
