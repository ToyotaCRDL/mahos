#!/usr/bin/env python3

"""
Message Types for TDC Instruments.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from numpy.typing import NDArray

from .common_msgs import Message


class RawEvents(Message):
    """Raw events data from TDC.

    - *.lst file of Fast ComTec MCS6 / MCS8.

    """

    def __init__(self, header: str, format_info: dict, data: NDArray):
        #: raw header to hint the data format.
        self.header: str = header

        #: format info extracted from the header.
        self.format_info: dict = format_info

        #: data format is instrument-dependent.
        self.data: NDArray = NDArray
