#!/usr/bin/env python3

"""
Typed Interface for Pulse Generator.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from .interface import InstrumentInterface

from ..msgs.inst_pg_msgs import TriggerType, Block, Blocks, BlockSeq


class PGInterface(InstrumentInterface):
    """Interface for Pulse Genetator."""

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Configure the PG using blocks (Blocks[Block]) representation.

        :param blocks: block representation of pulse pattern.
        :param freq: frequency in Hz.
        :param trigger_type: the trigger type. if None, instrument's default is used.
        :param n_runs: repetition number. if None, runs infinitely.

        """

        return self.configure(
            {"blocks": blocks, "freq": freq, "trigger_type": trigger_type, "n_runs": n_runs}
        )

    def configure_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Configure the PG using BlockSeq representation.

        :param blockseq: BlockSeq representation of pulse pattern.
        :param freq: frequency in Hz.
        :param trigger_type: the trigger type. if None, instrument's default is used.
        :param n_runs: repetition number. if None, runs infinitely.

        """

        return self.configure(
            {"blockseq": blockseq, "freq": freq, "trigger_type": trigger_type, "n_runs": n_runs}
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

    def validate_blocks(self, blocks: Block[Blocks], freq: float):
        """Validate the blocks with freq."""

        args = {"blocks": blocks, "freq": freq}
        return self.get("validate", args)

    def validate_blockseq(self, blockseq: BlockSeq, freq: float):
        """Validate the blocks with freq."""

        args = {"blockseq": blockseq, "freq": freq}
        return self.get("validate", args)
