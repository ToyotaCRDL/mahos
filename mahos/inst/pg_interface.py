#!/usr/bin/env python3

"""
Typed Interface for Pulse Generator.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from .interface import InstrumentInterface

from ..msgs.inst.pg_msgs import TriggerType, Block, Blocks, BlockSeq


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

        How pulse patterns are output depends on n_runs and existence of triggered Blocks.
        Following two cases are standardized, however, the other case may be instrument-dependent.
        - n_runs = None, no triggered blocks: Infinite loop mode. The output starts immediately
            on start() after this configure(). And it's repeated infinitely until stop().
        - n_runs = 1, (only) first block is triggered: Triggered one-shot mode.
            On start() after this configure(), output is not started but trigger is armed.
            The pattern is output once on each trigger event
            (software trigger() or hardware trigger signal).

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
        """Get total block length of last configure() call."""

        return self.get("length")

    def get_offsets(self) -> list[int]:
        """Get additional offsets of last configure() call."""

        return self.get("offsets")

    def validate_blocks(self, blocks: Block[Blocks], freq: float) -> list[int] | None:
        """Validate the blocks with freq.

        :returns: list of additional offsets if valid. None if invalid.

        """

        args = {"blocks": blocks, "freq": freq}
        return self.get("validate", args)

    def validate_blockseq(self, blockseq: BlockSeq, freq: float) -> list[int] | None:
        """Validate the blocks with freq.

        :returns: list of additional offsets if valid. None if invalid.

        """

        args = {"blockseq": blockseq, "freq": freq}
        return self.get("validate", args)
