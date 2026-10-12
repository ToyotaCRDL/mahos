#!/usr/bin/env python3

"""
PG sequence for ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import math

from mahos.msgs.inst.pg_msgs import Block, Blocks, TriggerType, PulsePattern
from mahos.util.param import ParamAccessor, ParamError
from mahos.util.unit import SI_format
from mahos_dq.meas.podmr_generator import generator_kernel as K
from mahos_dq.meas.odmr_pd import trace_samples, validate_trace_params


class ODMRPGMixin(object):
    """PG configuration for ODMR.

    Required attributes:
    - conf
        - pg_freq_cw
    - pg
    - _pd_analog
    - _pd_spectrum
    - _pd_trace
    - _pd_chop
    - _block_base
    - _channel_remap
    - _minimum_block_length

    """

    def _adjust_block(self, block: Block, index: int):
        """Mutate block so that block's total_length is integer multiple of block base."""

        duration = block.total_length()
        if M := duration % self._block_base:
            ch, d = block.pattern[index].channels, block.pattern[index].duration
            block.pattern[index] = (ch, d + self._block_base - M)

    def _finalize_blocks(self, blocks: Blocks, mw_offset: int = 0) -> Blocks:
        """Apply MW offset and channel remapping, then simplify blocks."""

        if mw_offset:
            blocks = K.apply_mw_offset(blocks, mw_offset)
        if self._channel_remap is not None:
            blocks = blocks.replace(self._channel_remap)
        return blocks.simplify()

    def _configure_blocks(self, blocks: Blocks, freq: float, trigger_type: TriggerType) -> bool:
        """Configure PG blocks and store the resulting pulse pattern."""

        success = self.pg.configure_blocks(blocks, freq, trigger_type=trigger_type, n_runs=1)
        if success:
            self.pulse_pattern = PulsePattern(blocks, freq)
        return success

    def apd_time_window(self, params: dict, label: str) -> float:
        """Return the effective APD (SinglePhotonCounter) integration time."""

        timing = params["timing"]
        if label != "pulse":
            return timing["time_window"]
        width = timing["chop_width"] if self._pd_chop else timing["laser_width"]
        return timing["burst_num"] * width

    def _add_chop(self, block: Block, laser_start: int, chop_delay: int, chop_width: int) -> Block:
        """Overlay a chop pulse on one raw pulse-sequence unit."""

        repetitions = block.Nrep
        unit = block.copy()
        unit.Nrep = 1
        chop_start = laser_start + chop_delay
        chop_tail = unit.raw_length() - chop_start - chop_width
        pattern = []
        if chop_start:
            pattern.append((None, chop_start))
        pattern.append(("chop", chop_width))
        if chop_tail:
            pattern.append((None, chop_tail))
        united = unit.union(Block(block.name + "-CHOP", pattern))
        united.Nrep = repetitions
        return united

    def configure_pg_CW_analog(self, params: dict, trigger_type: TriggerType) -> bool:
        freq = self.conf["pg_freq_cw"]
        # gate / trigger pulse width
        unit = round(freq * 1.0e-6)
        window = round(freq * params["timing"]["time_window"])
        gate_delay = round(freq * params["timing"].get("gate_delay", 0.0))
        post_gate_delay = round(freq * params["timing"].get("post_gate_delay", 0.0))
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        final_delay = round(freq * params.get("final_delay", 0.0))

        if params.get("background", False):
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    (("laser", "mw"), gate_delay),
                    (("laser", "mw", "gate"), unit),
                    # As measurement window is defined by AnalogIn sampling side,
                    # here we give long enough laser / mw pulse width (no "window - unit" below).
                    (("laser", "mw"), window + post_gate_delay),
                    (None, max(unit, bg_delay)),
                    ("laser", gate_delay),
                    (("laser", "gate"), unit),
                    ("laser", window + post_gate_delay),
                    (None, final_delay),
                    ("trigger", unit),
                ],
                trigger=True,
            )
        else:  # no background
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    (("laser", "mw"), gate_delay),
                    (("laser", "mw", "gate"), unit),
                    (("laser", "mw"), window + post_gate_delay),
                    (None, final_delay),
                    ("trigger", unit),
                ],
                trigger=True,
            )

        self._adjust_block(b, 0)
        blocks = self._finalize_blocks(Blocks([b]))
        return self._configure_blocks(blocks, freq, trigger_type)

    def configure_pg_pulse_analog(self, params: dict, trigger_type: TriggerType) -> bool:
        freq = self.conf["pg_freq_pulse"]
        min_len = self._minimum_block_length
        # gate / trigger pulse width
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        final_delay = round(freq * params.get("final_delay", 0.0))
        background = params.get("background", False)

        laser_delay, laser_width, mw_delay, mw_width, trigger_width = [
            round(params["timing"][k] * freq)
            for k in ("laser_delay", "laser_width", "mw_delay", "mw_width", "trigger_width")
        ]
        mw_offset = round(freq * params["timing"].get("mw_offset", 0.0))

        if trigger_width < 1:
            self.logger.error("trigger_width must be at least one PG tick.")
            return False
        if mw_delay < trigger_width:
            self.logger.error("mw_delay >= trigger_width must be satisfied.")
            return False

        seq_length = sum(
            params["timing"][k] for k in ("laser_delay", "laser_width", "mw_delay", "mw_width")
        )

        # convert time_window and gate_delay to number of sequence repetitions
        # round-up here because total sequence a bit shorter than time_window can be a problem,
        # but a bit longer sequence is not.
        # post_gate_delay is just added to burst_num, but this is not included in PD's
        # integration window. So this can be used to ensure the main pulse sequence is
        # longer than integration window.
        burst_num = math.ceil(params["timing"]["time_window"] / seq_length)
        gate_delay_num = math.ceil(params["timing"].get("gate_delay", 0.0) / seq_length)
        post_gate_delay_num = math.ceil(params["timing"].get("post_gate_delay", 0.0) / seq_length)
        self.logger.info(
            f"Burst num: {burst_num} + {post_gate_delay_num} Gate delay num: {gate_delay_num}"
        )

        blocks = []
        if delay:
            blk_delay = Block(
                "Delay",
                [
                    (None, max(delay, min_len)),
                ],
                trigger=True,
            )
            self._adjust_block(blk_delay, 0)
            blocks.append(blk_delay)

        if gate_delay_num:
            blk_gate_delay = Block(
                "Gate-Delay",
                [
                    ("laser", laser_width),
                    (None, mw_delay),
                    ("mw", mw_width),
                    (None, laser_delay),
                ],
                Nrep=gate_delay_num,
                trigger=not bool(delay),
            )
            self._adjust_block(blk_gate_delay, 0)
            blocks.append(blk_gate_delay)

        blk_gate = Block(
            "Gate",
            [
                ("laser", laser_width),
                ("gate", trigger_width),
                (None, mw_delay - trigger_width),
            ],
            trigger=not (bool(delay) or bool(gate_delay_num)),
        )
        self._adjust_block(blk_gate, 0)
        blocks.append(blk_gate)

        blk_main = Block(
            "Main",
            [
                ("mw", mw_width),
                (None, laser_delay),
                ("laser", laser_width),
                (None, mw_delay),
            ],
            Nrep=burst_num + post_gate_delay_num,
        )
        self._adjust_block(blk_main, 0)
        blocks.append(blk_main)

        if background:
            if bg_delay:
                blk_bg_delay = Block(
                    "BG-Delay",
                    [
                        (None, max(bg_delay, min_len)),
                    ],
                )
                self._adjust_block(blk_bg_delay, 0)
                blocks.append(blk_bg_delay)

            if gate_delay_num:
                blk_bg_gate_delay = Block(
                    "BG-Gate-Delay",
                    [
                        ("laser", laser_width),
                        (None, mw_delay + mw_width + laser_delay),
                    ],
                    Nrep=gate_delay_num,
                )
                self._adjust_block(blk_bg_gate_delay, 0)
                blocks.append(blk_bg_gate_delay)

            blk_bg_gate = Block(
                "BG-Gate",
                [
                    ("laser", laser_width),
                    ("gate", trigger_width),
                    (None, mw_delay - trigger_width),
                ],
            )
            self._adjust_block(blk_bg_gate, 0)
            blocks.append(blk_bg_gate)

            blk_bg_main = Block(
                "BG-Main",
                [
                    (None, mw_width + laser_delay),
                    ("laser", laser_width),
                    (None, mw_delay),
                ],
                Nrep=burst_num + post_gate_delay_num,
            )
            self._adjust_block(blk_bg_main, 0)
            blocks.append(blk_bg_main)

        blk_trigger = Block(
            "Trigger",
            [
                (None, max(final_delay, min_len - trigger_width)),
                ("trigger", trigger_width),
            ],
        )
        self._adjust_block(blk_trigger, 0)
        blocks.append(blk_trigger)

        blocks = self._finalize_blocks(Blocks(blocks), mw_offset)
        return self._configure_blocks(blocks, freq, trigger_type)

    def _make_block_pulse_trace_main(
        self,
        name,
        laser_delay,
        laser_width,
        mw_delay,
        mw_width,
        trigger_width,
        roi_head,
        burst_num,
        background=False,
    ):
        unit_length = mw_delay + mw_width + laser_delay + laser_width
        if residual := unit_length % self._block_base:
            mw_delay += self._block_base - residual
        mw_channels = None if background else "mw"
        op = Block(
            name + "-OP",
            [
                (None, mw_delay),
                (mw_channels, mw_width),
                (None, laser_delay),
            ],
        )
        op = K.inject_trigger(op, roi_head, trigger_width).replace({"trigger": "gate"})
        main = op.concatenate(Block(name + "-READ", [("laser", laser_width)]))
        main.name = name
        main.Nrep = burst_num
        return main

    def _make_blocks_pulse_trace_nobg(
        self,
        delay,
        final_delay,
        laser_delay,
        laser_width,
        mw_delay,
        mw_width,
        trigger_width,
        roi_head,
        burst_num,
        mw_offset,
    ):
        min_len = self._minimum_block_length
        init = Block(
            "INIT",
            [
                (None, max(delay, min_len - laser_width)),
                ("laser", laser_width),
            ],
            trigger=True,
        )
        main = self._make_block_pulse_trace_main(
            "MAIN",
            laser_delay,
            laser_width,
            mw_delay,
            mw_width,
            trigger_width,
            roi_head,
            burst_num,
        )
        final = Block(
            "FINAL",
            [
                (None, max(final_delay, min_len - trigger_width)),
                ("trigger", trigger_width),
            ],
        )
        for block in (init, final):
            self._adjust_block(block, 0)

        return self._finalize_blocks(Blocks([init, main, final]), mw_offset)

    def _make_blocks_pulse_trace_bg(
        self,
        delay,
        bg_delay,
        final_delay,
        laser_delay,
        laser_width,
        mw_delay,
        mw_width,
        trigger_width,
        roi_head,
        burst_num,
        mw_offset,
    ):
        min_len = self._minimum_block_length
        init = Block(
            "INIT",
            [
                (None, max(delay, min_len - laser_width)),
                ("laser", laser_width),
            ],
            trigger=True,
        )
        main = self._make_block_pulse_trace_main(
            "MAIN",
            laser_delay,
            laser_width,
            mw_delay,
            mw_width,
            trigger_width,
            roi_head,
            burst_num,
        )
        init_bg = Block(
            "INIT-BG",
            [
                (None, max(bg_delay, min_len - laser_width)),
                ("laser", laser_width),
            ],
        )
        main_bg = self._make_block_pulse_trace_main(
            "MAIN-BG",
            laser_delay,
            laser_width,
            mw_delay,
            mw_width,
            trigger_width,
            roi_head,
            burst_num,
            background=True,
        )
        final = Block(
            "FINAL",
            [
                (None, max(final_delay, min_len - trigger_width)),
                ("trigger", trigger_width),
            ],
        )
        for block in (init, init_bg, final):
            self._adjust_block(block, 0)

        return self._finalize_blocks(Blocks([init, main, init_bg, main_bg, final]), mw_offset)

    def validate_pulse_params(self, params: dict) -> tuple[bool, str, str]:
        """Validate params for the pulse mode."""

        try:
            p = ParamAccessor(params)
            for key in ("delay", "background_delay", "final_delay"):
                p.nonneg_num(key, 0.0)
            p.bool("background", False)

            timing = p.child("timing")
            laser_delay = timing.nonneg_num("laser_delay")
            laser_width = timing.nonneg_num("laser_width")
            mw_delay = timing.nonneg_num("mw_delay")
            mw_width = timing.nonneg_num("mw_width")
            trigger_width = timing.pos_num("trigger_width")
            timing.num("mw_offset", 0.0)

            window = laser_delay + laser_width + mw_delay + mw_width
            if window <= 0.0:
                raise ParamError("Length of laser and MW pulse sequence must be positive.")

            freq = self.conf["pg_freq_pulse"]
            trigger_width_ticks = round(trigger_width * freq)
            if trigger_width_ticks < 1:
                raise ParamError("timing.trigger_width must be at least one PG tick.")

            time_window = None
            if self._pd_analog and not self._pd_trace:
                time_window = timing.pos_num("time_window")
                timing.nonneg_num("gate_delay", 0.0)
                timing.nonneg_num("post_gate_delay", 0.0)
            else:
                timing.pos_int("burst_num")

            if self._pd_chop:
                chop_delay = timing.nonneg_num("chop_delay")
                chop_width = timing.pos_num("chop_width")
                laser_width_ticks = round(laser_width * freq)
                mw_delay_ticks = round(mw_delay * freq)
                chop_delay_ticks = round(chop_delay * freq)
                chop_width_ticks = round(chop_width * freq)
                if chop_width_ticks < 1:
                    raise ParamError("timing.chop_width must be at least one PG tick.")
                if chop_delay_ticks + chop_width_ticks > laser_width_ticks + mw_delay_ticks:
                    raise ParamError(
                        "timing.chop_delay + timing.chop_width must not exceed "
                        "timing.laser_width + timing.mw_delay after PG rounding."
                    )

            if not self._pd_trace:
                if laser_width <= 0.0:
                    raise ParamError("timing.laser_width must be positive.")
                if round(mw_delay * freq) < trigger_width_ticks:
                    raise ParamError("timing.mw_delay >= timing.trigger_width must be satisfied.")

            if self._pd_analog:
                pd = p.child("pd")
                rate = pd.pos_num("rate")
                pd.pos_int("buffer_size_coeff", self.conf.get("buffer_size_coeff", 20))
                pd.bool("hardware_average", True)
                pd.ascending_numbers("bounds", 2, (-10.0, 10.0))
                if time_window is not None and round(time_window * rate) < 1:
                    raise ParamError("timing.time_window must contain at least one PD sample.")

            if self._pd_trace:
                p.child("pd").nonneg_num("eos_deadtime", 0.0)
                for key in (
                    "roi_head",
                    "roi_tail",
                    "sig_delay",
                    "sig_width",
                    "ref_delay",
                    "ref_width",
                ):
                    timing.nonneg_num(key)
                timing.str("refmode")
                success, msg = self.validate_pg_pulse_trace(params)
                if not success:
                    return False, msg, ""

            return True, "", self.make_pulse_summary(params, window)

        except (KeyError, TypeError, ValueError, OverflowError) as e:
            self.logger.error(f"Invalid pulse parameters: {e}")
            return False, str(e), ""

    def make_pulse_summary(self, params: dict, window: float) -> str:
        timing = params["timing"]
        if "burst_num" in timing:
            total = SI_format(window * timing["burst_num"], suffix="s")
            single = SI_format(window, suffix="s")
            rate = SI_format(1 / window, suffix="Hz")
            return f"Total window: {total}, Single window: {single} (Rate: {rate})"
        else:  # slow AnalogPD (pd_analog = True, pd_trace = False) configuration.
            time_window = timing["time_window"]
            burst_num = math.ceil(time_window / window)
            s = f"Burst num: {burst_num}"
            if "post_gate_delay" in timing:
                post_gate_delay_num = math.ceil(timing["post_gate_delay"] / window)
                s += f" (+ Post gate delay num: {post_gate_delay_num})"
            if "gate_delay" in timing:
                gate_delay_num = math.ceil(timing["gate_delay"] / window)
                s += f", Gate delay num: {gate_delay_num}"
            return s

    def validate_pg_pulse_trace(self, params: dict) -> tuple[bool, str]:
        """Validate trace parameters using the configured detector and PG constraints."""

        try:
            validate_trace_params(params, self.conf, self._pd_spectrum)
            freq = self.conf["pg_freq_pulse"]
            laser_delay, mw_delay, mw_width, trigger_width, roi_head = [
                round(params["timing"][key] * freq)
                for key in ("laser_delay", "mw_delay", "mw_width", "trigger_width", "roi_head")
            ]
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Invalid trace parameters: {e}")
            return False, str(e)

        if trigger_width < 1:
            msg = "trigger_width must be at least one PG tick."
            self.logger.error(msg)
            return False, msg
        if trigger_width > roi_head:
            msg = "trigger_width <= roi_head must be satisfied."
            self.logger.error(msg)
            return False, msg
        if roi_head > mw_delay + mw_width + laser_delay:
            msg = "roi_head must fit inside the complete pre-laser sequence."
            self.logger.error(msg)
            return False, msg
        return True, ""

    def configure_pg_pulse_trace(self, params: dict, trigger_type: TriggerType) -> bool:
        """Configure the laser-resolved AnalogPD trace pulse sequence."""

        success, _ = self.validate_pg_pulse_trace(params)
        if not success:
            return False

        if params.get("background", False):
            samples = trace_samples(params, self.conf, self._pd_spectrum)
            timing = params["timing"]
            minimum_bg_delay = max(
                0.0,
                samples.realized / params["pd"]["rate"]
                - timing["roi_head"]
                - timing["laser_width"],
            )
            bg_delay = params.get("background_delay", 0.0)
            if bg_delay < minimum_bg_delay:
                self.logger.warning(
                    f"background_delay {bg_delay * 1e9:.1f} ns is shorter than the recommended "
                    f"minimum {minimum_bg_delay * 1e9:.1f} ns; the background initialization "
                    "laser may start before the final foreground trace finishes",
                )

        freq = self.conf["pg_freq_pulse"]
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        final_delay = round(freq * params.get("final_delay", 0.0))
        laser_delay, laser_width, mw_delay, mw_width, trigger_width, roi_head = [
            round(params["timing"][key] * freq)
            for key in (
                "laser_delay",
                "laser_width",
                "mw_delay",
                "mw_width",
                "trigger_width",
                "roi_head",
            )
        ]
        burst_num = params["timing"]["burst_num"]
        mw_offset = round(freq * params["timing"].get("mw_offset", 0.0))

        if params.get("background", False):
            blocks = self._make_blocks_pulse_trace_bg(
                delay,
                bg_delay,
                final_delay,
                laser_delay,
                laser_width,
                mw_delay,
                mw_width,
                trigger_width,
                roi_head,
                burst_num,
                mw_offset,
            )
        else:
            blocks = self._make_blocks_pulse_trace_nobg(
                delay,
                final_delay,
                laser_delay,
                laser_width,
                mw_delay,
                mw_width,
                trigger_width,
                roi_head,
                burst_num,
                mw_offset,
            )

        return self._configure_blocks(blocks, freq, trigger_type)

    def configure_pg_CW_apd(self, params: dict, trigger_type: TriggerType) -> bool:
        freq = self.conf["pg_freq_cw"]
        time_window = params["timing"]["time_window"]
        if time_window < 1.0e-6:
            self.logger.error("time_window must be at least 1 us for CW APD measurement.")
            return False

        # gate / trigger pulse width
        unit = round(freq * 1.0e-6)
        window = round(freq * time_window)
        gate_delay = round(freq * params["timing"].get("gate_delay", 0.0))
        post_gate_delay = round(freq * params["timing"].get("post_gate_delay", 0.0))
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        final_delay = round(freq * params.get("final_delay", 0.0))

        if params.get("background", False):
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    (("laser", "mw"), gate_delay),
                    # here we define measurement window using interval between gate pulses
                    (("laser", "mw", "gate"), unit),
                    (("laser", "mw"), window - unit),
                    (("laser", "mw", "gate"), unit),
                    (("laser", "mw"), post_gate_delay),
                    (None, max(unit, bg_delay)),
                    ("laser", gate_delay),
                    (("laser", "gate"), unit),
                    ("laser", window - unit),
                    (("laser", "gate"), unit),
                    ("laser", post_gate_delay),
                    (None, final_delay),
                    ("trigger", unit),
                ],
                trigger=True,
            )
        else:  # no background
            b = Block(
                "CW-ODMR",
                [
                    (None, max(unit, delay)),
                    (("laser", "mw"), gate_delay),
                    (("laser", "mw", "gate"), unit),
                    (("laser", "mw"), window - unit),
                    (("laser", "mw", "gate"), unit),
                    (("laser", "mw"), post_gate_delay),
                    (None, final_delay),
                    ("trigger", unit),
                ],
                trigger=True,
            )

        self._adjust_block(b, 0)
        if self._pd_chop:
            b = b.union(Block("CW-ODMR-CHOP", [("chop", b.total_length())]))
        blocks = self._finalize_blocks(Blocks([b]))
        return self._configure_blocks(blocks, freq, trigger_type)

    def _make_blocks_pulse_apd_nobg(
        self,
        delay,
        final_delay,
        laser_delay,
        laser_width,
        mw_delay,
        mw_width,
        trigger_width,
        burst_num,
        mw_offset,
        chop_delay=0,
        chop_width=0,
    ):
        min_len = self._minimum_block_length

        init = Block(
            "INIT",
            [
                (None, max(delay, min_len - laser_width - mw_delay)),
                ("laser", laser_width),
                ("gate", trigger_width),
                (None, mw_delay - trigger_width),
            ],
            trigger=True,
        )

        main = Block(
            "MAIN",
            [
                ("mw", mw_width),
                (None, laser_delay),
                ("laser", laser_width),
                (None, mw_delay),
            ],
            Nrep=burst_num,
        )

        final = Block(
            "FINAL",
            [
                ("gate", trigger_width),
                (None, max(final_delay, min_len - 2 * trigger_width)),
                ("trigger", trigger_width),
            ],
        )

        for b in (init, final):
            self._adjust_block(b, 0)
        self._adjust_block(main, -1)

        if self._pd_chop:
            main = self._add_chop(main, mw_width + laser_delay, chop_delay, chop_width)

        return self._finalize_blocks(Blocks([init, main, final]), mw_offset)

    def _make_blocks_pulse_apd_bg(
        self,
        delay,
        bg_delay,
        final_delay,
        laser_delay,
        laser_width,
        mw_delay,
        mw_width,
        trigger_width,
        burst_num,
        mw_offset,
        chop_delay=0,
        chop_width=0,
    ):
        min_len = self._minimum_block_length

        init = Block(
            "INIT",
            [
                (None, max(delay, min_len - laser_width - mw_delay)),
                ("laser", laser_width),
                ("gate", trigger_width),
                (None, mw_delay - trigger_width),
            ],
            trigger=True,
        )

        main = Block(
            "MAIN",
            [
                ("mw", mw_width),
                (None, laser_delay),
                ("laser", laser_width),
                (None, mw_delay),
            ],
            Nrep=burst_num,
        )

        final = Block(
            "FINAL",
            [
                ("gate", trigger_width),
                (None, max(0, min_len - trigger_width)),
            ],
        )

        init_bg = Block(
            "INIT-BG",
            [
                (None, max(bg_delay, min_len - laser_width - mw_delay)),
                ("laser", laser_width),
                ("gate", trigger_width),
                (None, mw_delay - trigger_width),
            ],
        )

        main_bg = Block(
            "MAIN-BG",
            [
                (None, mw_width),
                (None, laser_delay),
                ("laser", laser_width),
                (None, mw_delay),
            ],
            Nrep=burst_num,
        )

        final_bg = Block(
            "FINAL-BG",
            [
                ("gate", trigger_width),
                (None, max(final_delay, min_len - 2 * trigger_width)),
                ("trigger", trigger_width),
            ],
        )

        for b in (init, final, init_bg, final_bg):
            self._adjust_block(b, 0)
        for b in (main, main_bg):
            self._adjust_block(b, -1)

        if self._pd_chop:
            laser_start = mw_width + laser_delay
            main = self._add_chop(main, laser_start, chop_delay, chop_width)
            main_bg = self._add_chop(main_bg, laser_start, chop_delay, chop_width)

        return self._finalize_blocks(
            Blocks([init, main, final, init_bg, main_bg, final_bg]), mw_offset
        )

    def configure_pg_pulse_apd(self, params: dict, trigger_type: TriggerType) -> bool:
        freq = self.conf["pg_freq_pulse"]
        delay = round(freq * params.get("delay", 0.0))
        bg_delay = round(freq * params.get("background_delay", 0.0))
        final_delay = round(freq * params.get("final_delay", 0.0))
        laser_delay, laser_width, mw_delay, mw_width, trigger_width = [
            round(params["timing"][k] * freq)
            for k in ("laser_delay", "laser_width", "mw_delay", "mw_width", "trigger_width")
        ]
        burst_num = params["timing"]["burst_num"]
        mw_offset = round(freq * params["timing"].get("mw_offset", 0.0))
        if self._pd_chop:
            chop_delay = round(freq * params["timing"]["chop_delay"])
            chop_width = round(freq * params["timing"]["chop_width"])
        else:
            chop_delay = chop_width = 0

        if trigger_width < 1:
            self.logger.error("trigger_width must be at least one PG tick.")
            return False
        if mw_delay < trigger_width:
            self.logger.error("mw_delay >= trigger_width must be satisfied.")
            return False

        if params.get("background", False):
            blocks = self._make_blocks_pulse_apd_bg(
                delay,
                bg_delay,
                final_delay,
                laser_delay,
                laser_width,
                mw_delay,
                mw_width,
                trigger_width,
                burst_num,
                mw_offset,
                chop_delay,
                chop_width,
            )
        else:
            blocks = self._make_blocks_pulse_apd_nobg(
                delay,
                final_delay,
                laser_delay,
                laser_width,
                mw_delay,
                mw_width,
                trigger_width,
                burst_num,
                mw_offset,
                chop_delay,
                chop_width,
            )

        return self._configure_blocks(blocks, freq, trigger_type)

    def configure_pg(self, params: dict, label: str, trigger_type: TriggerType) -> bool:
        if not (self.pg.stop() and self.pg.clear()):
            return False
        if self._pd_analog:
            if label != "pulse":
                return self.configure_pg_CW_analog(params, trigger_type)
            elif self._pd_trace:
                return self.configure_pg_pulse_trace(params, trigger_type)
            else:
                return self.configure_pg_pulse_analog(params, trigger_type)
        else:
            if label != "pulse":
                return self.configure_pg_CW_apd(params, trigger_type)
            else:
                return self.configure_pg_pulse_apd(params, trigger_type)
