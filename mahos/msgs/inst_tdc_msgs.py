#!/usr/bin/env python3

"""
Message Types for Time to Digital Converter (Time Digitizer) Instruments.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import numpy as np

from .data_msgs import Message, Data


class ChannelStatus(Message):
    """Status of a (stop) channel.

    :ivar running: Measurement is running.
    :ivar runtime: Measurement runtime in sec.
    :ivar total: Total count of (stop) events of this channel.
    :ivar starts: Number of start events (0 if irrelevant in current measurement mode).

    """

    def __init__(self, running: bool, runtime: float, total: int, starts: int = 0):
        self.running = running
        self.runtime = runtime
        self.total = total
        self.starts = starts


class RawEvents(Data):
    """The raw events data.

    :ivar data: The events data. Should be sorted before storing.

    """

    def __init__(self, data: np.ndarray | None = None):
        self.data = data

    def set_data(self, data: np.ndarray):
        self.data = data
