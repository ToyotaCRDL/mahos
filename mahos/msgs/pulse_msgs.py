#!/usr/bin/env python3

from __future__ import annotations
import typing as T
import uuid

from .inst_pg_msgs import Block, Blocks
from .common_msgs import Message


class PulsePattern(Message):
    """Pulse Pattern message for visualization / debug."""

    def __init__(
        self,
        blocks: Blocks[Block],
        freq: float,
        markers: T.Optional[T.List[int]] = None,
        ident: T.Optional[uuid.UUID] = None,
    ):
        self.blocks = blocks
        self.freq = freq
        self.markers = markers
        if ident is None:
            self.ident = uuid.uuid4()
        else:
            self.ident = ident
