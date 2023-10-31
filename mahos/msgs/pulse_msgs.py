#!/usr/bin/env python3

"""
Message Types for the pulse pattern for visualization and debug.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import uuid

from .inst_pg_msgs import Block, Blocks, BlockSeq
from .common_msgs import Message


class PulsePattern(Message):
    """Pulse Pattern message for visualization / debug."""

    def __init__(
        self,
        blocks: Blocks[Block] | BlockSeq,
        freq: float,
        markers: list[int] | None = None,
        ident: uuid.UUID | None = None,
    ):
        self.blocks = blocks
        self.freq = freq
        self.markers = markers
        if ident is None:
            self.ident = uuid.uuid4()
        else:
            self.ident = ident
