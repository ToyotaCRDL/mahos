#!/usr/bin/env python3

"""
Pulse Generator module.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import typing as T

import pulsestreamer

from .instrument import Instrument
from ..msgs.inst_pg_msgs import TriggerType, Block, Blocks


class PulseStreamer(Instrument):
    """Wrapper for Swabian Instrument Pulse Streamer 8/2."""

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(("resource",))

        self.CHANNELS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        if "channels" in conf:
            self.CHANNELS.update(conf["channels"])

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

    def channels_to_ints(self, channels) -> T.List[int]:
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
            return [parse(c) for c in channels]

    def _included_channels_blocks(self, blocks: Blocks[Block]) -> T.List[int]:
        s = set()
        for block in blocks:
            for channels, length in block.pattern:
                s.update(self.channels_to_ints(channels))
        return sorted(list(s))

    def _extract_trigger_blocks(self, blocks: Blocks[Block]) -> T.Optional[bool]:
        trigger = False
        for i, block in enumerate(blocks):
            if block.trigger:
                if i:
                    self.logger.error("Can set trigger for first block only.")
                    return None
                trigger = True
        return trigger

    def _generate_seq_blocks(self, blocks: Blocks[Block]) -> pulsestreamer.Sequence:
        """Generate sequence from DTG-compatible 'blocks'."""

        rle_patterns = [(ch, []) for ch in self._included_channels_blocks(blocks)]
        for i, block in enumerate(blocks):
            for channels, length in block.total_pattern():
                high_channels = self.channels_to_ints(channels)
                for ch, pat in rle_patterns:
                    pat.append((length, 1 if ch in high_channels else 0))

        seq = self.ps.createSequence()
        for ch, pat in rle_patterns:
            seq.setDigital(ch, pat)

        return seq

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Make sequence from DTG-compatible 'blocks'."""

        freq = round(freq)
        base_freq = round(1.0e9)
        if freq > base_freq:
            return self.fail_with("PulseStreamer's max freq is 1 GHz.")
        if base_freq % freq:
            return self.fail_with("freq must be a divisor of 1 GHz.")
        scale = base_freq // freq
        if scale > 1:
            self.logger.info(f"Scaling the blocks by {scale}")
            blocks = blocks.scale(scale)

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

        try:
            self.sequence = self._generate_seq_blocks(blocks)
        except ValueError:
            self.logger.exception("Failed to generate sequence. Check channel name settings.")
            return False

        self.length = blocks.total_length()
        # offsets is for DTG-compatibility.
        self.offsets = [0] * len(blocks)

        # sanity check
        if self.sequence.getDuration() != self.length:
            return self.fail_with("length is not correct. Debug _generate_seq_blocks().")

        self.logger.info(f"Configured sequence. length: {self.length} trigger: {trigger}")
        return True

    def trigger(self) -> bool:
        """issue a software trigger.

        The software triggering is active if trigger mode is IMMEDIATE or SOFTWARE.

        """

        self.ps.startNow()
        return True

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

    def configure(self, params: dict) -> bool:
        if "blocks" in params and "freq" in params:
            return self.configure_blocks(
                params["blocks"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
                n_runs=params.get("n_runs"),
            )
        else:  # TODO: maybe other API (not DTG-compatible) ?
            return self.fail_with("These params must be given: 'blocks' and 'freq'")

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
            else:
                return [0] * len(args["blocks"])
        elif key == "opc":  # for API compatibility
            return True
        elif key == "validate":  # for API compatibility
            return True
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class PulseStreamerDAQTrigger(PulseStreamer):
    """PulseStreamer with DAQ trigger extension."""

    def __init__(self, name, conf, prefix=None):
        PulseStreamer.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(("do_line",))

        self._idle_logic = self.conf.get("idle_logic", False)

        do_name = name + "_do"
        do_conf = {"lines": [self.conf["do_line"]]}

        # import here to make PulseStreamer usable on PC without DAQ
        from .daq import DigitalOut

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
        self, blocks, freq, trigger_type: TriggerType | None = None, n_runs: int | None = None
    ) -> bool:
        """Make sequence from DTG-compatible 'blocks'."""

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
