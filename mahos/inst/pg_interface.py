#!/usr/bin/env python3

"""
Typed Interface for Pulse Generator.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import typing as T

from .interface import InstrumentInterface

from ..msgs.inst_pg_msgs import TriggerType, Block, Blocks


class PGInterface(InstrumentInterface):
    """Interface for Pulse Genetator."""

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_type: T.Optional[TriggerType] = None,
        n_runs: T.Optional[int] = None,
    ) -> bool:
        """Configure the PG using blocks representation.

        :param blocks: block representation of pulse pattern.
        :param freq: frequency in Hz.
        :param trigger_type: the trigger type. if None, instrument's default is used.
        :param n_runs: repetition number. if None, runs infinitely.

        """

        return self.configure(
            {"blocks": blocks, "freq": freq, "trigger_type": trigger_type, "n_runs": n_runs}
        )

    def clear(self) -> bool:
        """Clear status."""

        return self.set("clear")

    def trigger(self):
        """Issue software trigger."""

        return self.set("trigger")

    def get_opc(self, delay=None) -> bool:
        """Get OPC (operation complete) status."""

        return self.get("opc", delay)

    def get_length(self) -> int:
        """Get total block length of last configure_blocks() call."""

        return self.get("length")

    def get_offsets(self, blocks=None, freq=None):
        """Get additional offsets of given blocks.

        If blocks and freq are None, get offsets of last configure_blocks() call.

        """

        if blocks is None and freq is None:
            args = None
        else:
            args = {"blocks": blocks, "freq": freq}
        return self.get("offsets", args)

    def validate_blocks(self, blocks, freq):
        """Validate the blocks with freq."""

        args = {"blocks": blocks, "freq": freq}
        return self.get("validate", args)
