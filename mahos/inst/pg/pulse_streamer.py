#!/usr/bin/env python3

"""
PulseStreamer part of Pulse Generator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import pulsestreamer

from ..instrument import Instrument
from ...msgs.inst_pg_msgs import TriggerType, Block, Blocks, BlockSeq


class PulseStreamer(Instrument):
    """Wrapper for Swabian Instrument Pulse Streamer 8/2.

    :param channels: mapping from channel names to indices.
    :type channels: dict[str | bytes, int]
    :param analog.channels: list of channel names to represent analog output
    :type analog.channels: list[str | bytes]
    :param analog.values: mapping of analog channel patterns to analog output values
        {"00": [0.5, 0.5], "01": [-0.5, 0.0]} reads as follows.
        When analog.channels[0] is L and analog.channels[1] is L, output A0 = 0.5 V, A1 = 0.5 V.
        When analog.channels[0] is L and analog.channels[1] is H, output A0 = -0.5 V, A1 = 0.0 V.
        It's also possible to use more than two channels.
    :type analog.values: dict[str, tuple[float, float]]
    :param ext_ref_clock: external reference clock source.
        Set 10 for 10 MHz or 125 for 125 MHz.
        All the other values are considered `disable` (internal clock source is used).
    :type ext_ref_clock: int
    :param strict: (default: True) If True, check generated data strictly.
        Since offset will not be allowed in strict mode, the pulse pattern data must have
        total length of integer multiple of 8 ns.
    :type strict: bool

    """

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(("resource",))

        self.CHANNELS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        if "channels" in conf:
            self.CHANNELS.update(conf["channels"])
        if "analog" in conf:
            self.analog_channels = conf["analog"]["channels"]
            self.analog_values = conf["analog"]["values"]
        else:
            self.analog_channels = []
            self.analog_values = {}
        self._strict = conf.get("strict", True)

        self.ps = pulsestreamer.PulseStreamer(self.conf["resource"])

        self.sequence = None
        self.n_runs = None
        # last total block length and offsets.
        self.length = 0
        self.offsets = None
        self._trigger_type = None

        self.logger.info(
            "Opened PulseStreamer at {}. Serial: {}".format(
                self.conf["resource"], self.ps.getSerial()
            )
        )

        clk = conf.get("ext_ref_clock", 0)
        if clk == 10:
            self.ps.selectClock(pulsestreamer.ClockSource.EXT_10MHZ)
            self.logger.info("External clock at 10 MHz.")
        elif clk == 125:
            self.ps.selectClock(pulsestreamer.ClockSource.EXT_125MHZ)
            self.logger.info("External clock at 125 MHz.")
        else:
            self.ps.selectClock(pulsestreamer.ClockSource.INTERNAL)
            self.logger.info("Internal clock.")

    def channels_to_ints(self, channels) -> list[int]:
        """NOTE: if channels is a container, channels in analog_channels are excluded."""

        def parse(c):
            if isinstance(c, (bytes, str)):
                return self.CHANNELS[c]
            elif isinstance(c, int):
                return c
            else:
                raise TypeError("Invalid type for channel. Use bytes, str or int.")

        if channels is None:
            return []
        elif isinstance(channels, (bytes, str, int)):
            return [parse(channels)]
        else:  # container (list or tuple) of (bytes, str, int)
            return [parse(c) for c in channels if c not in self.analog_channels]

    def _included_channels_blocks(self, blocks: Blocks[Block]) -> list[int]:
        """create sorted list of channels in blocks converted to ints."""

        # take set() here because channels_to_ints() may contain non-unique integers
        # (CHANNELS is not always injective), though it's quite rare usecase.
        return sorted(list(set(self.channels_to_ints(blocks.channels()))))

    def _extract_trigger_blocks(self, blocks: Blocks[Block]) -> bool | None:
        trigger = False
        for i, block in enumerate(blocks):
            if block.trigger:
                if i:
                    self.logger.error("Can set trigger for first block only.")
                    return None
                trigger = True
        return trigger

    def _generate_seq_blocks(self, blocks: Blocks[Block]) -> pulsestreamer.Sequence:
        """Generate sequence from blocks."""

        rle_patterns = [(ch, []) for ch in self._included_channels_blocks(blocks)]
        a0_patterns = []
        a1_patterns = []

        for i, block in enumerate(blocks):
            for channels, length in block.total_pattern():
                high_channels = self.channels_to_ints(channels)
                for ch, pat in rle_patterns:
                    pat.append((length, 1 if ch in high_channels else 0))

                if self.analog_channels:
                    label = "".join(
                        ("1" if ch in channels else "0" for ch in self.analog_channels)
                    )
                    a0, a1 = self.analog_values[label]
                    a0_patterns.append((length, a0))
                    a1_patterns.append((length, a1))

        seq = self.ps.createSequence()
        for ch, pat in rle_patterns:
            seq.setDigital(ch, pat)
        if self.analog_channels:
            seq.setAnalog(0, a0_patterns)
            seq.setAnalog(1, a1_patterns)

        return seq

    def _scale_blocks(self, blocks: Blocks[Block], freq: float) -> Blocks[Block] | None:
        """Return scaled blocks according to freq, or None if any failure."""

        freq = round(freq)
        base_freq = round(1.0e9)
        if freq > base_freq:
            self.logger.error("PulseStreamer's max freq is 1 GHz.")
            return None
        if base_freq % freq:
            self.logger.error("freq must be a divisor of 1 GHz.")
            return None
        scale = base_freq // freq
        if scale > 1:
            self.logger.info(f"Scaling the blocks by {scale}")
            return blocks.scale(scale)
        else:
            return blocks

    def _adjust_blocks(self, blocks: Blocks[Block]) -> list[int] | None:
        """(mutating) adjust block length so that total_length becomes N * 8 (ns).

        :returns: list of offset values, or None if any failure.

        """

        remainder = blocks.total_length() % 8
        if self._strict and remainder:
            self.logger.error("blocks' total_length is not integer multiple of 8.")
            return None
        if not remainder:
            return [0] * len(blocks)

        lb = blocks[-1]
        if lb.Nrep != 1:
            msg = "blocks' total_length is not integer multiple of 8 and"
            msg += f" last block's Nrep is not 1 ({lb.Nrep})."
            self.logger.error(msg)
            return None
        dur = lb.pattern[-1].duration
        offset = 8 - remainder
        lb.update_duration(-1, dur + offset)
        self.logger.warn(f"Adjusted last block by offset {offset}.")
        return [0] * (len(blocks) - 1) + [offset]

    def validate_blocks(self, blocks: Blocks[Block], freq: float) -> list[int] | None:
        blocks = self._scale_blocks(blocks, freq)
        if blocks is None:
            return None
        return self._adjust_blocks(blocks)

    def validate_blockseq(self, blockseq: BlockSeq[Block], freq: float) -> list[int] | None:
        # Since PulseStreamer doesn't have loop control mechanism,
        # given BlockSeq is equivalent to collapsed one.
        return self.validate_blocks(Blocks([blockseq.collapse()]), freq)

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Make sequence from blocks."""

        blocks = self._scale_blocks(blocks, freq)
        if blocks is None:
            return False

        trigger = self._extract_trigger_blocks(blocks)
        if trigger is None:
            return False

        if trigger:
            if trigger_type == TriggerType.HARDWARE_RISING:
                self.ps.setTrigger(pulsestreamer.TriggerStart.HARDWARE_RISING)
            elif trigger_type == TriggerType.HARDWARE_FALLING:
                self.ps.setTrigger(pulsestreamer.TriggerStart.HARDWARE_FALLING)
            elif trigger_type == TriggerType.SOFTWARE:
                self.ps.setTrigger(pulsestreamer.TriggerStart.SOFTWARE)
            else:  # TriggerType.IMMEDIATE or None
                return self.fail_with(
                    "trigger is in the blocks but trigger_type is not specified."
                )
            self._trigger_type = trigger_type
        else:
            self.ps.setTrigger(pulsestreamer.TriggerStart.IMMEDIATE)
            self._trigger_type = TriggerType.IMMEDIATE

        if n_runs is None:
            self.n_runs = pulsestreamer.PulseStreamer.REPEAT_INFINITELY
        else:
            self.n_runs = int(n_runs)

        self.offsets = self._adjust_blocks(blocks)
        if self.offsets is None:
            return False
        self.length = blocks.total_length()

        try:
            self.sequence = self._generate_seq_blocks(blocks)
        except ValueError:
            self.logger.exception("Failed to generate sequence. Check channel name settings.")
            return False

        # sanity check
        if self.sequence.getDuration() != self.length:
            return self.fail_with("length is not correct. Debug _generate_seq_blocks().")

        msg = f"Configured sequence. length: {self.length} offset: {self.offsets[-1]}"
        msg += f" trigger: {trigger}"
        self.logger.info(msg)
        return True

    def configure_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Make sequence from blockseq."""

        # Since PulseStreamer doesn't have loop control mechanism,
        # given BlockSeq is equivalent to collapsed one.
        return self.configure_blocks(Blocks([blockseq.collapse()]), freq, trigger_type, n_runs)

    def trigger(self) -> bool:
        """issue a software trigger.

        The software triggering is active if trigger mode is IMMEDIATE or SOFTWARE.

        """

        if self._trigger_type in (TriggerType.IMMEDIATE, TriggerType.SOFTWARE):
            self.ps.startNow()
            return True
        else:
            msg = "Cannot issue software trigger in hardware trigger mode."
            msg += "\nConsider using PulseStreamerDAQTrigger instread."
            self.logger.error(msg)
            return False

    # Standard API

    def reset(self) -> bool:
        self.ps.reset()
        return True

    def start(self) -> bool:
        if self.sequence is None or self.n_runs is None:
            return self.fail_with("No sequence defined. configure() first.")
        self.ps.stream(self.sequence, self.n_runs)
        self.logger.info("Start streaming.")
        return True

    def stop(self) -> bool:
        self.ps.constant()
        self.logger.info("Stop streaming and outputting constant all-zero.")
        return True

    def configure(self, params: dict, label: str = "") -> bool:
        if "blocks" in params and "freq" in params:
            return self.configure_blocks(
                params["blocks"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
                n_runs=params.get("n_runs"),
            )
        elif "blockseq" in params and "freq" in params:
            return self.configure_blockseq(
                params["blockseq"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
                n_runs=params.get("n_runs"),
            )
        else:
            return self.fail_with("These params must be given: 'blocks' | 'blockseq' and 'freq'")

    def set(self, key: str, value=None) -> bool:
        if key == "trigger":
            return self.trigger()
        elif key == "clear":  # for API compatibility
            return True
        else:
            return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None):
        if key == "length":
            return self.length  # length of last configure_blocks
        elif key == "offsets":  # for API compatibility
            if args is None:
                return self.offsets
            elif "blocks" in args:
                return [0] * len(args["blocks"])
            elif "blockseq" in args:
                return [0]
        elif key == "opc":  # for API compatibility
            return True
        elif key == "validate":  # for API compatibility
            if "blocks" in args and "freq" in args:
                return self.validate_blocks(args["blocks"], args["freq"])
            elif "blockseq" in args and "freq" in args:
                return self.validate_blockseq(args["blockseq"], args["freq"])
            else:
                self.logger.error(f"Invalid args for get(validate): {args}")
                return None
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class PulseStreamerDAQTrigger(PulseStreamer):
    """PulseStreamer with DAQ trigger extension.

    This class enables software trigger (trigger() method) even when
    trigger_type is set TriggerType.HARDWARE_RISING or TriggerType.HARDWARE_FALLING.
    This is achieved by a little hardware modification.
    You have to prepare logic 2-input AND / OR / NAND / NOR gate
    (something like 74LS08, 74LS32 etc.), and connect as follows:

    - Input1: DAQ Digital I/O line (specified by conf["do_line"])
    - Input2: Trigger source of a hardware.
    - Output: "Trigger In" connector of PulseStreamer.

    Since the objective of this modification is to make logical or,
    the recommended selection of gate circuit and conf["idle_logic"]
    depends on the logic polarity of your trigger source (Input2).
    The selection would be as follows.

    If trigger source (Input2) is positive logic (active high):

    - logic gate: OR / NOR.
    - conf["idle_logic"]: False.

    If trigger source (Input2) is negative logic (active low):

    - logic gate: AND / NAND.
    - conf["idle_logic"]: True.

    :param do_line: DAQ line for DigitalOut
    :type do_line: str
    :param idle_logic: Idle state for DigitalOut. True (False) for High (Low).
    :type idle_logic: bool

    """

    def __init__(self, name, conf, prefix=None):
        PulseStreamer.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(("do_line",))

        self._idle_logic = self.conf.get("idle_logic", False)

        do_name = name + "_do"
        do_conf = {"lines": [self.conf["do_line"]]}

        # import here to make PulseStreamer usable on PC without DAQ
        from ..daq import DigitalOut

        self._do = DigitalOut(do_name, do_conf, prefix=prefix)
        self._set_idle_logic()

    def _set_idle_logic(self) -> bool:
        if self._idle_logic:
            return self._do.set_output_high()
        else:
            return self._do.set_output_low()

    def close_do(self):
        if hasattr(self, "_do"):
            self._do.close_once()

    def close(self):
        self.close_do()

    def trigger(self) -> bool:
        """issue a software trigger or hardware trigger from DAQ DO.

        The software triggering is active if trigger mode is IMMEDIATE or SOFTWARE.

        """

        if self._trigger_type in (TriggerType.IMMEDIATE, TriggerType.SOFTWARE):
            self.ps.startNow()
            return True

        elif self._trigger_type in (TriggerType.HARDWARE_RISING, TriggerType.HARDWARE_FALLING):
            if self._idle_logic:
                # High (idle) -> Low -> High
                return self._do.set_output_pulse_neg()
            else:
                # Low (idle) -> High -> Low
                return self._do.set_output_pulse()

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Make sequence from blocks."""

        if trigger_type in (TriggerType.HARDWARE_RISING, TriggerType.HARDWARE_FALLING):
            if not self._set_idle_logic():
                return self.fail_with("Failed to output idle logic")

        return PulseStreamer.configure_blocks(
            self,
            blocks,
            freq,
            trigger_type=trigger_type,
            n_runs=n_runs,
        )
