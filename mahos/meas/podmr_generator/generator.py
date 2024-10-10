#!/usr/bin/env python3

"""
Pattern generators for Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import re

import numpy as np
from . import generator_kernel as K

from ...msgs import param_msgs as P
from ...msgs.inst.pg_msgs import Channels, AnalogChannel


mw_x = K.mw_x
mw_y = K.mw_y
mw_x_inv = K.mw_x_inv
mw_y_inv = K.mw_y_inv

mw1_x = AnalogChannel("mw1_phase", 0)
mw1_y = AnalogChannel("mw1_phase", 90)
mw1_x_inv = AnalogChannel("mw1_phase", 180)
mw1_y_inv = AnalogChannel("mw1_phase", 270)


def mw_inverted(phase) -> bool:
    return phase in (mw_x_inv, mw_y_inv)


def mw_uninvert(phase) -> bool:
    if phase == mw_x_inv:
        return mw_x
    elif phase == mw_y_inv:
        return mw_y
    return phase


class PatternGenerator(object):
    def __init__(
        self,
        freq: float = 2.0e9,
        reduce_start_divisor: int = 2,
        split_fraction: int = 4,
        minimum_block_length: int = 1000,
        block_base: int = 4,
        mw_modes: tuple[int] = (0,),
        iq_amplitude: float = 0.0,
        channel_remap: dict | None = None,
        print_fn=print,
        method: str = "",
    ):
        self.freq = freq
        self.reduce_start_divisor = reduce_start_divisor
        self.split_fraction = split_fraction
        self.minimum_block_length = minimum_block_length
        self.block_base = block_base
        self.mw_modes = tuple(mw_modes)
        self.iq_amplitude = iq_amplitude
        self.channel_remap = channel_remap
        self.print_fn = print_fn
        self.method = method
        self._params = None

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        """Return ParamDict of additional pulse params."""

        return P.ParamDict()

    def num_mw(self) -> int:
        """Return number of required MW channels."""

        return 1

    def mode(self, ch=None) -> int:
        """Return MW mode at given ch or currently-active ch."""

        if ch is not None:
            return self.mw_modes[ch]

        if self.num_mw() == 1:
            # infer currently-active channel for single channel sequence.
            # when mw0 is disabled and mw1 is enabled, use mw1.
            # otherwise, default to mw0.
            nomw = self._params.get("nomw", False)
            nomw1 = "nomw1" not in self._params or self._params["nomw1"]
            if nomw and not nomw1:
                return self.mw_modes[1]
            else:
                return self.mw_modes[0]

        # Won't reach here: ch will be given for multi channel sequence.
        return self.mw_modes[0]

    def get_common_pulses(self, params: dict):
        keys = [
            "base_width",
            "laser_delay",
            "laser_width",
            "mw_delay",
            "trigger_width",
            "init_delay",
            "final_delay",
        ]
        return [params[k] for k in keys]

    def is_sweepN(self) -> bool:
        return False

    def get_pulse_params(self, params: dict) -> dict:
        pp = params.get("pulse", {})
        return {k: pp[k] for k in self.pulse_params()}

    def generate(self, xdata, params: dict):
        """Generate the blocks."""

        if params.get("enable_reduce", False):
            reduce_start_divisor = self.reduce_start_divisor
        else:
            reduce_start_divisor = 0

        self._params = params
        pulse_params = self.get_pulse_params(params)
        blocks, freq, common_pulses = self._generate(
            xdata,
            self.get_common_pulses(params),
            pulse_params,
            params.get("partial", -1),
            reduce_start_divisor,
            params.get("fix_base_width"),
        )

        blocks, laser_timing = K.build_blocks(
            blocks,
            common_pulses,
            params,
            divide=params.get("divide_block", False),
            invertY=pulse_params.get("invertY", False),
            minimum_block_length=self.minimum_block_length,
            block_base=self.block_base,
            mw_modes=self.mw_modes,
            num_mw=self.num_mw(),
            iq_amplitude=self.iq_amplitude,
            channel_remap=self.channel_remap,
        )
        return blocks, freq, laser_timing

    def generate_raw_blocks(self, xdata, params: dict):
        """Generate raw blocks without finishing it with build_blocks. Debug purpose."""

        if params.get("enable_reduce", False):
            reduce_start_divisor = self.reduce_start_divisor
        else:
            reduce_start_divisor = 0
        self._params = params

        blocks, freq, common_pulses = self._generate(
            xdata,
            self.get_common_pulses(params),
            self.get_pulse_params(params),
            params.get("partial", -1),
            reduce_start_divisor,
            params.get("fix_base_width"),
        )
        return blocks, freq, common_pulses

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        raise NotImplementedError("_generate() is not implemented.")


class RabiGenerator(PatternGenerator):
    """Generate Pulse Pattern for Rabi nutation measurement.

    pattern0 => Rabi (mw pulse with tau duration)
    pattern1 => no operation (T1 limited)

    """

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, [], [], reduce_start_divisor, self.print_fn
        )
        p0 = [True]
        p1 = [False]

        def gen(v, operate):
            if operate:
                return [((mw_x, "mw"), v)]
            else:
                return [((mw_x,), v)]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=mw_x,
                read_phase1=mw_x,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class T1Generator(PatternGenerator):
    """Generate Pulse Pattern for T1 (spin-lattice relaxation time) measurement.

    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param flip_head: If True (False), 180pulse is applied at start (end) of relaxation period.
    :type flip_head: bool

    pattern0 => (no 180 pulse) - tau
    pattern1 => 180 (pi) pulse - tau or tau - 180 (pi) pulse

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["180pulse"] = P.FloatParam(
            10e-9, 1e-9, 1e-6, unit="s", SI_prefix=True, step=1e-9, doc="180 deg (pi) pulse width."
        )
        pd["flip_head"] = P.BoolParam(False)
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p180, flip_head = pulse_params["180pulse"], pulse_params["flip_head"]

        p0 = [p180]
        p1 = [p180]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [False, flip_head]
        p1 = p1 + [True, flip_head]

        def gen(v, p180, operate, flip_head):
            if operate:
                if flip_head:
                    return [((mw_x, "mw"), p180), ((mw_x,), v)]
                else:
                    return [((mw_x,), v), ((mw_x, "mw"), p180)]
            else:
                return [((mw_x,), p180 + v)]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=mw_x,
                read_phase1=mw_x,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class FIDGenerator(PatternGenerator):
    """Generate Pulse Pattern for FID measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float

    pattern0 => FID (pi/2 - tau - pi/2)
    pattern1 => FID (pi/2 - tau - pi/2_inv)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90 = pulse_params["90pulse"]

        p0 = [p90]
        p1 = [p90]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [mw_x]
        p1 = p1 + [mw_x_inv]

        def gen(v, p90, read_phase):
            if self.mode() in (0, 2):
                v_f, v_l = K.split_int(v, self.split_fraction)
                return [
                    ((mw_x, "mw"), p90),
                    ((mw_x,), v_f),
                    ((read_phase,), v_l),
                    ((read_phase, "mw"), p90),
                ]
            elif self.mode() == 1:
                p = [
                    ((mw_x, "mw"), p90),
                    ((mw_x,), v),
                    ((mw_x, "mw"), p90),
                ]
                # TODO maybe we'd better to add parameter p180
                if mw_inverted(read_phase):
                    # 270 deg pulse
                    p.append(((mw_x, "mw"), p90 * 2))
                else:
                    # same delay time to match length of pattern0 and pattern1.
                    p.append(((mw_x,), p90 * 2))
                return p
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

        if self.mode() in (0, 2):
            read_phase1 = mw_x_inv
        elif self.mode() == 1:
            read_phase1 = mw_x
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=mw_x,
                read_phase1=read_phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class SpinEchoGenerator(PatternGenerator):
    """Generate Pulse Pattern for Spin Echo measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param readY: If True, readout by Y projection.
    :type readY: bool

    pattern0 => SpinEcho (pi/2 - tau - pi - tau - pi/2)
    pattern1 => SpinEcho (pi/2 - tau - pi - tau - pi/2_inv)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["readY"] = P.BoolParam(False, doc="readout (apply pi/2 pulse) with phase Y.")
        pd["invertY"] = P.BoolParam(False, doc="invert Y phase.")
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, readY = [pulse_params[k] for k in ["90pulse", "180pulse", "readY"]]

        read_phase0 = mw_y if readY else mw_x
        read_phase1 = mw_y_inv if readY else mw_x_inv
        p0 = [p90, p180]
        p1 = [p90, p180]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [read_phase0]
        p1 = p1 + [read_phase1]

        def gen(v, p90, p180, read_phase):
            v_f, v_l = K.split_int(v, self.split_fraction)
            if self.mode() in (0, 2):
                return [
                    ((mw_x, "mw"), p90),
                    ((mw_x,), v),
                    ((mw_x, "mw"), p180),
                    ((mw_x,), v_f),
                    ((read_phase,), v_l),
                    ((read_phase, "mw"), p90),
                ]
            elif self.mode() == 1:
                phase = mw_uninvert(read_phase)
                p = [
                    ((mw_x, "mw"), p90),
                    ((mw_x,), v),
                    ((mw_x, "mw"), p180),
                    ((mw_x,), v_f),
                    ((phase,), v_l),
                    ((phase, "mw"), p90),
                ]
                if mw_inverted(read_phase):
                    p.append(((phase, "mw"), p180))
                else:
                    p.append(((phase,), p180))
                return p
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

        if self.mode() in (0, 2):
            phase0 = read_phase0
            phase1 = read_phase1
        elif self.mode() == 1:
            # Don't use _inv phase.
            phase0 = phase1 = read_phase0
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=phase0,
                read_phase1=phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]

        return blocks, freq, common_pulses


class TRSEGenerator(PatternGenerator):
    """Generate Pulse Pattern for Time-resolved Spin Echo measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param tauconst: duration of first free evolution period
    :type tauconst: float

    pattern0 => SpinEcho (pi/2 - tauconst - pi - tau - pi/2)
    pattern1 => SpinEcho (pi/2 - tauconst - pi - tau - pi/2_inv)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["tauconst"] = P.FloatParam(
            1e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, step=1e-9, doc="first inter-pulse time."
        )
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, tauconst = [pulse_params[k] for k in ["90pulse", "180pulse", "tauconst"]]

        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [mw_x]
        p1 = p1 + [mw_x_inv]

        def gen(v, p90, p180, tauconst, read_phase):
            if self.mode() in (0, 2):
                v_f, v_l = K.split_int(v, self.split_fraction)
                return [
                    ((mw_x, "mw"), p90),
                    ((mw_x,), tauconst),
                    ((mw_x, "mw"), p180),
                    ((mw_x,), v_f),
                    ((read_phase,), v_l),
                    ((read_phase, "mw"), p90),
                ]
            elif self.mode() == 1:
                p = [
                    ((mw_x, "mw"), p90),
                    ((mw_x,), tauconst),
                    ((mw_x, "mw"), p180),
                    ((mw_x,), v),
                    ((mw_x, "mw"), p90),
                ]
                if mw_inverted(read_phase):
                    # 270 deg pulse
                    p.append(((mw_x, "mw"), p180))
                else:
                    # same delay time to match length of pattern0 and pattern1.
                    p.append(((mw_x,), p180))
                return p
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

        if self.mode() in (0, 2):
            read_phase1 = mw_x_inv
        elif self.mode() == 1:
            read_phase1 = mw_x
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=mw_x,
                read_phase1=read_phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]

        return blocks, freq, common_pulses


class DDGenerator(PatternGenerator):
    """Generate Pulse Pattern for Dynamical Decoupling measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param Nconst: Number of pulse repeats
    :type Nconst: int
    :param readY: If True, readout by Y projection.
    :type readY: bool
    :param supersample: Constant for supersampling (only for xy8 and xy16)
    :type supersample: int

    pattern0 => invert read (readout |0>)
    pattern1 => normal read (readout |1>)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["Nconst"] = P.IntParam(4, 1, 10000, doc="Number of pulse train repetitions.")
        pd["readY"] = P.BoolParam(False, doc="readout (apply pi/2 pulse) with phase Y.")
        pd["invertY"] = P.BoolParam(False, doc="invert Y phase.")
        if self.method in ("xy8", "xy16"):
            pd["supersample"] = P.IntParam(1, 1, 1000, doc="coefficient for supersamling.")
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        if self.method in ("xy8", "xy16"):
            supersample = pulse_params["supersample"]
        else:
            supersample = 1

        if supersample == 1:
            return self._generate_normal(
                xdata,
                common_pulses,
                pulse_params,
                partial,
                reduce_start_divisor,
                fix_base_width,
            )
        else:
            return self._generate_supersample(
                xdata,
                common_pulses,
                pulse_params,
                partial,
                reduce_start_divisor,
                fix_base_width,
            )

    def _generate_normal(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, Nconst, readY = [
            pulse_params[k] for k in ["90pulse", "180pulse", "Nconst", "readY"]
        ]
        read_phase0 = mw_y_inv if readY else mw_x_inv
        read_phase1 = mw_y if readY else mw_x
        p0 = [p90, p180]
        p1 = [p90, p180]

        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, read_phase0, self.method]
        p1 = p1 + [Nconst, read_phase1, self.method]

        def gen(tau, p90, p180, Nconst, read_phase, method):
            tau_f, tau_l = K.split_int(tau, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tau * 2, self.split_fraction)
            init_ptn = [((mw_x, "mw"), p90), ((mw_x,), tau_f)]
            if self.mode() in (0, 2):
                read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), p90)]
            elif self.mode() == 1:
                phase = mw_uninvert(read_phase)
                read_ptn = [((phase,), tau_l), ((phase, "mw"), p90)]
                if mw_inverted(read_phase):
                    read_ptn.append(((phase, "mw"), p180))
                else:
                    read_ptn.append(((phase,), p180))
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

            pattern = []
            px = [((mw_x,), tau2_l), ((mw_x, "mw"), p180), ((mw_x,), tau2_f)]
            py = [((mw_y,), tau2_l), ((mw_y, "mw"), p180), ((mw_y,), tau2_f)]
            ix = [((mw_x_inv,), tau2_l), ((mw_x_inv, "mw"), p180), ((mw_x_inv,), tau2_f)]
            iy = [((mw_y_inv,), tau2_l), ((mw_y_inv, "mw"), p180), ((mw_y_inv,), tau2_f)]

            if method == "cp":
                pattern = px
            elif method == "cpmg":
                pattern = py
            elif method == "xy4":
                pattern = px + py + px + py
            elif method == "xy8":
                pattern = px + py + px + py + py + px + py + px
            elif method == "xy16":
                pattern = (
                    px + py + px + py + py + px + py + px + ix + iy + ix + iy + iy + ix + iy + ix
                )
            else:
                raise ValueError(f"Unknown method {method}")

            pattern *= Nconst
            pattern[0] = (pattern[0][0], tau_l)
            pattern[-1] = (pattern[-1][0], tau_f)

            return init_ptn + pattern + read_ptn

        if self.mode() in (0, 2):
            read_phase0_ = read_phase0
        elif self.mode() == 1:
            read_phase0_ = mw_uninvert(read_phase0)
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=read_phase0_,
                read_phase1=read_phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses

    def _generate_supersample(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, Nconst, readY, supersample = [
            pulse_params[k] for k in ["90pulse", "180pulse", "Nconst", "readY", "supersample"]
        ]
        read_phase0 = mw_y_inv if readY else mw_x_inv
        read_phase1 = mw_y if readY else mw_x
        p0 = [p90, p180]
        p1 = [p90, p180]

        self.print_fn(f"Quantum Interpolation: {supersample}")

        freq, tau, common_pulses, p0, p1 = K.round_pulses(
            self.freq,
            xdata[::supersample],
            common_pulses,
            p0,
            p1,
            reduce_start_divisor,
            self.print_fn,
        )

        if self.method == "xy8":
            pulse_num = Nconst * 8
            phase_list = [mw_x, mw_y, mw_x, mw_y, mw_y, mw_x, mw_y, mw_x] * Nconst
        elif self.method == "xy16":
            pulse_num = Nconst * 16
            phase_list = [
                mw_x,
                mw_y,
                mw_x,
                mw_y,
                mw_y,
                mw_x,
                mw_y,
                mw_x,
                mw_x_inv,
                mw_y_inv,
                mw_x_inv,
                mw_y_inv,
                mw_y_inv,
                mw_x_inv,
                mw_y_inv,
                mw_x_inv,
            ] * Nconst
        else:
            raise ValueError(f"Unknown method {self.method}")

        sample = (np.linspace(0, 1, num=supersample + 1) * pulse_num).astype(np.int64)[
            :-1
        ] / pulse_num
        dt = abs(tau[0] - tau[1])
        p0 = p0 + [dt, read_phase0, pulse_num, phase_list]
        p1 = p1 + [dt, read_phase1, pulse_num, phase_list]

        def gen_QI(v, p90, p180, dt, read_phase, pulse_num, phase_list):
            tau, sample = v

            U_0 = [tau, 2 * tau, tau]
            U_1 = [tau + dt, 2 * (tau + dt), tau + dt]
            tau_list = [0]
            m = 0
            for i in range(4 * pulse_num // 8):
                m += sample
                if abs(m) <= 1 / 2:
                    tau_list[-1] = tau_list[-1] + U_0[0]
                    tau_list = tau_list + U_0[1:]
                else:
                    tau_list[-1] = tau_list[-1] + U_1[0]
                    tau_list = tau_list + U_1[1:]
                    m -= 1

            init_ptn = [
                ((mw_x, "mw"), p90),
                ((mw_x,), K.split_int(tau_list[0], self.split_fraction)[0]),
            ]
            if self.mode() in (0, 2):
                read_ptn = [
                    ((read_phase,), K.split_int(tau_list[-1], self.split_fraction)[1]),
                    ((read_phase, "mw"), p90),
                ]
            elif self.mode() == 1:
                phase = mw_uninvert(read_phase)
                read_ptn = [
                    ((phase,), K.split_int(tau_list[-1], self.split_fraction)[1]),
                    ((phase, "mw"), p90),
                ]
                if mw_inverted(read_phase):
                    read_ptn.append(((phase, "mw"), p180))
                else:
                    read_ptn.append(((phase,), p180))
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

            pattern = []

            def mw_pattern(phase_before, tau_before, phase_next, tau_next):
                return [
                    ((phase_before,), K.split_int(tau_before, self.split_fraction)[1]),
                    ((phase_before, "mw"), p180),
                    ((phase_next,), K.split_int(tau_next, self.split_fraction)[0]),
                ]

            phase_list_ = phase_list + [read_phase]
            for i in range(len(phase_list)):
                pattern += mw_pattern(
                    phase_list_[i], tau_list[i], phase_list_[i + 1], tau_list[i + 1]
                )

            return init_ptn + pattern + read_ptn

        if self.mode() in (0, 2):
            read_phase0_ = read_phase0
        elif self.mode() == 1:
            read_phase0_ = mw_uninvert(read_phase0)
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        ind = np.arange(len(tau) * len(sample)).reshape((len(tau), len(sample)))
        blocks = [
            K.generate_blocks(
                i,
                (tau[j], sample[k]),
                common_pulses,
                gen_QI,
                p0,
                p1,
                read_phase0=read_phase0_,
                read_phase1=read_phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for (j, k), i in np.ndenumerate(ind)
        ]
        return blocks, freq, common_pulses


class RDDGenerator(PatternGenerator):
    """Generate Pulse Pattern for Randomized Dynamical Decoupling measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param Nconst: Number of pulse repeats
    :type Nconst: int
    :param readY: If True, readout by Y projection.
    :type readY: bool
    :param seed: Random seed
    :type seed: int
    :param randomize_each: if "tau", generate randomized phases for each tau.
        if "pattern", generate randomized phases for each pattern.
    :type randomize_each: str

    pattern0 => invert read (readout |0>)
    pattern1 => normal read (readout |1>)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["Nconst"] = P.IntParam(4, 1, 10000, doc="Number of pulse train repetitions.")
        pd["readY"] = P.BoolParam(False, doc="readout (apply pi/2 pulse) with phase Y.")
        pd["invertY"] = P.BoolParam(False, doc="invert Y phase.")
        pd["seed"] = P.IntParam(0, doc="random seed.")
        pd["randomize_each"] = P.StrChoiceParam(
            "no",
            ["no", "tau", "pattern"],
            doc="generate randomized phases for each tau or pattern generation.",
        )
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        if self.mode() in (0, 1):
            raise ValueError("MW mode must be 2.")

        p90, p180, Nconst, readY = [
            pulse_params[k] for k in ["90pulse", "180pulse", "Nconst", "readY"]
        ]
        read_phase0 = mw_y_inv if readY else mw_x_inv
        read_phase1 = mw_y if readY else mw_x
        p0 = [p90, p180]
        p1 = [p90, p180]

        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, read_phase0, self.method]
        p1 = p1 + [Nconst, read_phase1, self.method]
        rng = np.random.default_rng(pulse_params["seed"])
        each = pulse_params["randomize_each"]
        global_phases = rng.uniform(0.0, 360.0, size=Nconst)

        def rotate(unit_ptn, global_phase):
            ret = []
            for channels, duration in unit_ptn:
                ach = channels[0]
                rot = AnalogChannel(ach.name(), ach.value() + global_phase)
                ret.append(((rot,) + channels[1:], duration))
            return ret

        def gen(tau, p90, p180, Nconst, read_phase, method):
            tau_f, tau_l = K.split_int(tau, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tau * 2, self.split_fraction)
            init_ptn = [((mw_x, "mw"), p90), ((mw_x,), tau_f)]
            read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), p90)]

            px = [((mw_x,), tau2_l), ((mw_x, "mw"), p180), ((mw_x,), tau2_f)]
            py = [((mw_y,), tau2_l), ((mw_y, "mw"), p180), ((mw_y,), tau2_f)]
            ix = [((mw_x_inv,), tau2_l), ((mw_x_inv, "mw"), p180), ((mw_x_inv,), tau2_f)]
            iy = [((mw_y_inv,), tau2_l), ((mw_y_inv, "mw"), p180), ((mw_y_inv,), tau2_f)]

            if method == "cp":
                unit = px
            # cpmg is equivalent to cp when randomized
            # elif method == "cpmg":
            #     pattern = py
            elif method == "xy4":
                unit = px + py + px + py
            elif method == "xy8":
                unit = px + py + px + py + py + px + py + px
            elif method == "xy16":
                unit = (
                    px + py + px + py + py + px + py + px + ix + iy + ix + iy + iy + ix + iy + ix
                )
            else:
                raise ValueError(f"Unknown method {method}")

            nonlocal global_phases
            if each == "pattern":
                global_phases = rng.uniform(0.0, 360.0, size=Nconst)
            elif each == "tau" and read_phase == read_phase0:
                # if each == "tau", regenerate at p0 only to make phase patterns of p0 and p1
                # identical for given tau.
                global_phases = rng.uniform(0.0, 360.0, size=Nconst)
            pattern = []
            for phi in global_phases:
                pattern.extend(rotate(unit, phi))

            pattern[0] = (pattern[0][0], tau_l)
            pattern[-1] = (pattern[-1][0], tau_f)

            return init_ptn + pattern + read_ptn

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=read_phase0,
                read_phase1=read_phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class DDNGenerator(PatternGenerator):
    """Generate Pulse Pattern for Dynamical Decoupling measurement (sweepN).

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param tauconst: duration of free evolution
    :type tauconst: int
    :param readY: If True, readout by Y projection.
    :type readY: bool

    pattern0 => invert read (readout |0>)
    pattern1 => normal read (readout |1>)

    """

    def is_sweepN(self) -> bool:
        return True

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["tauconst"] = P.FloatParam(
            1e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, step=1e-9, doc="free evolution time."
        )
        pd["readY"] = P.BoolParam(False, doc="readout (apply pi/2 pulse) with phase Y.")
        pd["invertY"] = P.BoolParam(False, doc="invert Y phase.")
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, tauconst, readY = [
            pulse_params[k] for k in ["90pulse", "180pulse", "tauconst", "readY"]
        ]

        read_phase0 = mw_y_inv if readY else mw_x_inv
        read_phase1 = mw_y if readY else mw_x
        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, dummy, common_pulses, p0, p1 = K.round_pulses(
            self.freq, [0], common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [read_phase0, self.method]
        p1 = p1 + [read_phase1, self.method]

        def gen(n, p90, p180, tauconst, read_phase, method):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)
            init_ptn = [((mw_x, "mw"), p90), ((mw_x,), tau_f)]
            if self.mode() in (0, 2):
                read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), p90)]
            elif self.mode() == 1:
                phase = mw_uninvert(read_phase)
                read_ptn = [((phase,), tau_l), ((phase, "mw"), p90)]
                if mw_inverted(read_phase):
                    read_ptn.append(((phase, "mw"), p180))
                else:
                    read_ptn.append(((phase,), p180))
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

            pattern = []
            px = [((mw_x,), tau2_l), ((mw_x, "mw"), p180), ((mw_x,), tau2_f)]
            py = [((mw_y,), tau2_l), ((mw_y, "mw"), p180), ((mw_y,), tau2_f)]
            ix = [
                ((mw_x_inv,), tau2_l),
                ((mw_x_inv, "mw"), p180),
                ((mw_x_inv,), tau2_f),
            ]
            iy = [
                ((mw_y_inv,), tau2_l),
                ((mw_y_inv, "mw"), p180),
                ((mw_y_inv,), tau2_f),
            ]

            if method == "cpN":
                pattern = px
            elif method == "cpmgN":
                pattern = py
            elif method == "xy4N":
                pattern = px + py + px + py
            elif method == "xy8N":
                pattern = px + py + px + py + py + px + py + px
            elif method == "xy16N":
                pattern = (
                    px + py + px + py + py + px + py + px + ix + iy + ix + iy + iy + ix + iy + ix
                )
            else:
                raise ValueError(f"Unknown method {method}")
            pattern *= n

            pattern[0] = (pattern[0][0], tau_l)
            pattern[-1] = (pattern[-1][0], tau_f)

            return init_ptn + pattern + read_ptn

        if self.mode() in (0, 2):
            read_phase0_ = read_phase0
        elif self.mode() == 1:
            read_phase0_ = mw_uninvert(read_phase0)
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        blocks = [
            K.generate_blocks(
                i,
                n,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=read_phase0_,
                read_phase1=read_phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, n in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class PiTrainGenerator(PatternGenerator):
    """Generate Pulse Pattern for Pi pulse train measurement.

    :param tauconst: duration of free evolution
    :type tauconst: int
    :param Nconst: Number of pulse repeats
    :type Nconst: int

    pattern0 => ( mw(tau) - tauconst )*(Nconst-1) - mw(tau)
    pattern1 => no operation (T1 limited)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["tauconst"] = P.FloatParam(
            1e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, step=1e-9, doc="first inter-pulse time."
        )
        pd["Nconst"] = P.IntParam(1, 1, 10000, doc="Number of pulse repetitions.")
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        tauconst, Nconst = [pulse_params[k] for k in ["tauconst", "Nconst"]]

        p0 = [tauconst]
        p1 = [tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, True]
        p1 = p1 + [Nconst, False]

        def gen(v, tauconst, Nconst, operate):
            p = [((mw_x, "mw"), v), ((mw_x,), tauconst)]
            p *= Nconst

            flip = [((mw_x, "mw"), v)]
            wait = [((mw_x,), tauconst)]
            p = (flip + wait) * (Nconst - 1) + flip

            if operate:
                return p
            else:
                t = sum([pp[1] for pp in p])
                return [((mw_x,), t)]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=mw_x,
                read_phase1=mw_x,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class SEHalfPiSweepGenerator(PatternGenerator):
    """Generate Pulse Pattern for Spin Echo (pi/2 pulse sweep) measurement.

    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param tauconst: duration of free evolution
    :type tauconst: int

    pattern0 => SpinEcho (mw(tau) - tauconst - pi - tauconst - mw(tau))
    pattern1 => no operation (T1 limited)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["180pulse"] = P.FloatParam(
            10e-9, 1e-9, 1e-6, unit="s", SI_prefix=True, step=1e-9, doc="180 deg (pi) pulse width."
        )
        pd["tauconst"] = P.FloatParam(
            1e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, step=1e-9, doc="inter-pulse time."
        )
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p180, tauconst = [pulse_params[k] for k in ["180pulse", "tauconst"]]

        p0 = [p180, tauconst]
        p1 = [p180, tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [True]
        p1 = p1 + [False]

        def gen(v, mw_flip_width, tauconst, operate):
            p = [
                ((mw_x, "mw"), v),
                ((mw_x,), tauconst),
                ((mw_x, "mw"), mw_flip_width),
                ((mw_x,), tauconst),
                ((mw_x, "mw"), v),
            ]

            if operate:
                return p
            else:
                t = sum([pp[1] for pp in p])
                return [((mw_x,), t)]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=mw_x,
                read_phase1=mw_x,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class SpinLockGenerator(PatternGenerator):
    """Generate Pulse Pattern for Spin-Locking measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param iq_delay: duration of iq_delay
    :type iq_delay: float

    pattern0 => SpinLock( 90x - y(tau) - 90x_inv ), read |0>
    pattern1 => SpinLock( 90x - y(tau) - 90x     ), read |1>

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["iq_delay"] = P.FloatParam(10e-9, 1e-9, 1000e-9, unit="s", SI_prefix=True, step=1e-9)
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, iq_delay = [pulse_params[k] for k in ["90pulse", "iq_delay"]]

        p0 = [p90, iq_delay]
        p1 = [p90, iq_delay]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [mw_x_inv]
        p1 = p1 + [mw_x]

        def gen(v, p90, iq_delay, read_phase):
            if self.mode() in (0, 2):
                p = [
                    ((mw_x, "mw"), p90),
                    ((mw_y,), iq_delay),
                    ((mw_y, "mw"), v),
                    ((read_phase,), iq_delay),
                    ((read_phase, "mw"), p90),
                ]
                return p
            elif self.mode() == 1:
                p = [
                    ((mw_x, "mw"), p90),
                    ((mw_y,), iq_delay),
                    ((mw_y, "mw"), v),
                    ((mw_x,), iq_delay),
                    ((mw_x, "mw"), p90),
                ]
                # TODO maybe we'd better to add parameter p180
                if mw_inverted(read_phase):
                    p.append(((mw_x, "mw"), p90 * 2))
                else:
                    p.append(((mw_x,), p90 * 2))
                return p
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

        if self.mode() in (0, 2):
            read_phase0 = mw_x_inv
        elif self.mode() == 1:
            read_phase0 = mw_x
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=read_phase0,
                read_phase1=mw_x,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class XY8CorrelationGenerator(PatternGenerator):
    """Generate Pulse Pattern for XY8 correlation measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param tauconst: duration of free evolution
    :type tauconst: int
    :param Nconst: Number of pulse repeats
    :type Nconst: int
    :param reinitX: If True, reinit by X pulse.
    :type reinitX: bool
    :param readY: If True, readout by Y projection.
    :type readY: bool

    pattern0 => invert read (readout |0>)
    pattern1 => normal read (readout |1>)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["tauconst"] = P.FloatParam(
            1e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, step=1e-9, doc="first inter-pulse time."
        )
        pd["Nconst"] = P.IntParam(4, 1, 10000, doc="Number of pulse train repetitions.")
        pd["reinitX"] = P.BoolParam(False, doc="reinitialize X.")
        pd["readY"] = P.BoolParam(False, doc="readout (apply pi/2 pulse) with phase Y.")
        pd["invertY"] = P.BoolParam(False, doc="invert Y phase.")
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, tauconst, Nconst, reinitX, readY = [
            pulse_params[k]
            for k in ["90pulse", "180pulse", "tauconst", "Nconst", "reinitX", "readY"]
        ]

        read_phase0 = mw_y_inv if readY else mw_x_inv
        read_phase1 = mw_y if readY else mw_x
        reinit_phase = mw_x if reinitX else mw_y
        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, reinit_phase, read_phase0, self.method]
        p1 = p1 + [Nconst, reinit_phase, read_phase1, self.method]

        def gen(tau, p90, p180, tauconst, Nconst, reinit_phase, read_phase, method):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)

            init_ptn = [((mw_x, "mw"), p90), ((mw_x,), tau_f)]
            storage_ptn = [((mw_y,), tau_l), ((mw_y, "mw"), p90)]
            reinit_ptn = [((reinit_phase, "mw"), p90), ((reinit_phase,), tau_f)]
            if self.mode() in (0, 2):
                read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), p90)]
            elif self.mode() == 1:
                phase = mw_uninvert(read_phase)
                read_ptn = [((phase,), tau_l), ((phase, "mw"), p90)]
                if mw_inverted(read_phase):
                    read_ptn.append(((phase, "mw"), p180))
                else:
                    read_ptn.append(((phase,), p180))
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

            px = [((mw_x,), tau2_l), ((mw_x, "mw"), p180), ((mw_x,), tau2_f)]
            py = [((mw_y,), tau2_l), ((mw_y, "mw"), p180), ((mw_y,), tau2_f)]
            pattern = px + py + px + py + py + px + py + px
            pattern *= Nconst
            pattern[0] = (pattern[0][0], tau_l)
            pattern[-1] = (pattern[-1][0], tau_f)

            xy8_first = init_ptn + pattern + storage_ptn
            xy8_second = reinit_ptn + pattern + read_ptn

            v_f, v_l = K.split_int(tau, self.split_fraction)
            if method == "xy8cl":
                interlude = [((mw_y,), v_l), ((reinit_phase,), v_f)]
            elif method == "xy8cl1flip":
                interlude = [
                    ((mw_y,), v_l),
                    ((mw_x,), v_f),
                    ((mw_x, "mw"), p180),
                    ((mw_x,), v_l),
                    ((reinit_phase,), v_f),
                ]
            else:
                raise ValueError(f"Unknown method {method}")

            return xy8_first + interlude + xy8_second

        if self.mode() in (0, 2):
            read_phase0_ = read_phase0
        elif self.mode() == 1:
            read_phase0_ = mw_uninvert(read_phase0)
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=read_phase0_,
                read_phase1=read_phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class XY8CorrelationNflipGenerator(PatternGenerator):
    """Generate Pulse Pattern for XY8 correlation measurement (Nflip).

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param tauconst: duration of free evolution
    :type tauconst: int
    :param Nconst: Number of pulse repeats
    :type Nconst: int
    :param reinitX: If True, reinit by X pulse.
    :type reinitX: bool
    :param readY: If True, readout by Y projection.
    :type readY: bool

    pattern0 => invert read (readout |0>)
    pattern1 => normal read (readout |1>)

    """

    def is_sweepN(self) -> bool:
        return True

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["tauconst"] = P.FloatParam(
            1e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, step=1e-9, doc="first inter-pulse time."
        )
        pd["Nconst"] = P.IntParam(4, 1, 10000, doc="Number of pulse train repetitions.")
        pd["reinitX"] = P.BoolParam(False, doc="reinitialize X.")
        pd["readY"] = P.BoolParam(False, doc="readout (apply pi/2 pulse) with phase Y.")
        pd["invertY"] = P.BoolParam(False, doc="invert Y phase.")
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, tauconst, Nconst, reinitX, readY = [
            pulse_params[k]
            for k in ["90pulse", "180pulse", "tauconst", "Nconst", "reinitX", "readY"]
        ]

        read_phase0 = mw_y_inv if readY else mw_x_inv
        read_phase1 = mw_y if readY else mw_x
        reinit_phase = mw_x if reinitX else mw_y
        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, dummy, common_pulses, p0, p1 = K.round_pulses(
            self.freq, [0], common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, reinit_phase, read_phase0]
        p1 = p1 + [Nconst, reinit_phase, read_phase1]

        def gen(n, p90, p180, tauconst, Nconst, reinit_phase, read_phase):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)

            init_ptn = [((mw_x, "mw"), p90), ((mw_x,), tau_f)]
            storage_ptn = [((mw_y,), tau_l), ((mw_y, "mw"), p90), ((mw_y,), tau2_f)]
            reinit_ptn = [
                ((reinit_phase,), tau2_l),
                ((reinit_phase, "mw"), p90),
                ((reinit_phase,), tau_f),
            ]
            if self.mode() in (0, 2):
                read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), p90)]
            elif self.mode() == 1:
                phase = mw_uninvert(read_phase)
                read_ptn = [((phase,), tau_l), ((phase, "mw"), p90)]
                if mw_inverted(read_phase):
                    read_ptn.append(((phase, "mw"), p180))
                else:
                    read_ptn.append(((phase,), p180))
            else:
                raise ValueError(f"Unknown MW mode: {self.mode()}")

            px = [((mw_x,), tau2_l), ((mw_x, "mw"), p180), ((mw_x,), tau2_f)]
            py = [((mw_y,), tau2_l), ((mw_y, "mw"), p180), ((mw_y,), tau2_f)]
            pattern = px + py + px + py + py + px + py + px
            pattern *= Nconst
            pattern[0] = (pattern[0][0], tau_l)
            pattern[-1] = (pattern[-1][0], tau_f)

            xy8_first = init_ptn + pattern + storage_ptn
            xy8_second = reinit_ptn + pattern + read_ptn
            # Since timing misalignment of first/last are included in storage_ptn/reinit_ptn,
            # we don't need to fix them here.
            interlude = px * n

            return xy8_first + interlude + xy8_second

        if self.mode() in (0, 2):
            read_phase0_ = read_phase0
        elif self.mode() == 1:
            read_phase0_ = mw_uninvert(read_phase0)
        else:
            raise ValueError(f"Unknown MW mode: {self.mode()}")

        blocks = [
            K.generate_blocks(
                i,
                n,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=read_phase0_,
                read_phase1=read_phase1,
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, n in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class DDGateGenerator(PatternGenerator):
    """Generate Pulse Pattern for DD Gate measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param tauconst: duration of free evolution
    :type tauconst: int
    :param Nconst: Number of pulse repeats
    :type Nconst: int
    :param N2const: Number of pulse repeats
    :type N2const: int
    :param N3const: Number of pulse repeats
    :type N3const: int
    :param ddphase: Phase patterns
    :type ddphase: str

    pattern0 => invert read (readout |0>)
    pattern1 => normal read (readout |1>)

    """

    def __init__(
        self,
        freq: float = 2.0e9,
        reduce_start_divisor: int = 2,
        split_fraction: int = 4,
        minimum_block_length: int = 1000,
        block_base: int = 4,
        mw_modes: tuple[int] = (0,),
        iq_amplitude: float = 0.0,
        channel_remap: dict | None = None,
        print_fn=print,
        method: str = "",
    ):
        PatternGenerator.__init__(
            self,
            freq=freq,
            reduce_start_divisor=reduce_start_divisor,
            split_fraction=split_fraction,
            minimum_block_length=minimum_block_length,
            block_base=block_base,
            mw_modes=mw_modes,
            iq_amplitude=iq_amplitude,
            channel_remap=channel_remap,
            print_fn=print_fn,
            method=method,
        )
        c = "|".join(["X", "Y", "iX", "iY", "n"])
        p = "^({c:s}):({c:s}):({c:s}):({c:s}),({c:s}):({c:s}):({c:s}):({c:s})$".format(c=c)
        self.ddphase_pattern = re.compile(p)

    def parse_phase_for_ddgate(self, phase) -> tuple[list[Channels], list[Channels]]:
        """parse string representation of phase into mw on/off and phase (mw, mw_x, mw_y etc.).

        :returns: (channels0, channels1). first element is phase (mw_x etc.) in all the channels.

        """

        m = self.ddphase_pattern.match(phase.replace(" ", ""))
        if m is None:
            raise ValueError(f"Invalid phases {phase}")
        plist = m.groups()

        phase_dict = {
            "X": (mw_x, "mw"),
            "Y": (mw_y, "mw"),
            "iX": (mw_x_inv, "mw"),
            "iY": (mw_y_inv, "mw"),
            "n": (mw_x,),
        }  # assign phase only at this moment.
        p_ch = [phase_dict[p] for p in plist]
        pih_ch0 = p_ch[:4]
        pih_ch1 = p_ch[4:]

        return pih_ch0, pih_ch1

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["tauconst"] = P.FloatParam(
            1e-9,
            1e-9,
            1e-3,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="duration of free-evolution.",
        )
        pd["Nconst"] = P.IntParam(4, 1, 10000, doc="Number of pulse train repetitions.")
        pd["N2const"] = P.IntParam(2, 1, 10000, doc="Number of pulse train repetitions.")
        pd["N3const"] = P.IntParam(2, 1, 10000, doc="Number of pulse train repetitions.")
        pd["ddphase"] = P.StrParam("Y:X:Y:X,Y:X:Y:iX", doc="Phase patterns.")
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, tauconst, Nconst, N2const, N3const, ddphase = [
            pulse_params[k]
            for k in ["90pulse", "180pulse", "tauconst", "Nconst", "N2const", "N3const", "ddphase"]
        ]

        pih_ch0, pih_ch1 = self.parse_phase_for_ddgate(ddphase)

        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, N2const, N3const, pih_ch0]
        p1 = p1 + [Nconst, N2const, N3const, pih_ch1]

        def gen(tau, p90, p180, tauconst, Nconst, N2const, N3const, pih_ch):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)

            init_ptn = [(pih_ch[0], p90), ((mw_x,), tau_f)]
            storage_ptn = [((pih_ch[1][0],), tau_l), (pih_ch[1], p90)]
            reinit_ptn = [(pih_ch[2], p90), ((mw_x,), tau_f)]
            read_ptn = [((pih_ch[3][0],), tau_l), (pih_ch[3], p90)]

            px = [((mw_x,), tau2_l), ((mw_x, "mw"), p180), ((mw_x,), tau2_f)]
            py = [((mw_y,), tau2_l), ((mw_y, "mw"), p180), ((mw_y,), tau2_f)]
            pattern = px + py + px + py + py + px + py + px
            pattern *= Nconst
            first_wait = list(pattern[0])
            first_wait[1] = tau_l
            last_wait = list(pattern[-1])
            last_wait[1] = tau_f
            pattern[0] = tuple(first_wait)
            pattern[-1] = tuple(last_wait)

            xy8_first = init_ptn + pattern + storage_ptn
            xy8_second = reinit_ptn + pattern + read_ptn

            v_f, v_l = K.split_int(tau, self.split_fraction)
            interlude = [
                ((mw_y,), tau),
                ((mw_y, "mw"), p180),
                ((mw_y,), v_l),
                ((mw_y,), v_f),
            ] * N2const
            interlude[-1] = ((pih_ch[2][0],), v_f)

            return (xy8_first + interlude + xy8_second) * N3const

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=pih_ch0[-1][0],
                read_phase1=pih_ch1[-1][0],
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class DDNGateGenerator(DDGateGenerator):
    """Generate Pulse Pattern for DD Gate measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param tauconst: duration of free evolution
    :type tauconst: int
    :param tau2const: duration of free evolution
    :type tau2const: int
    :param Nconst: Number of pulse repeats
    :type Nconst: int
    :param ddphase: Phase patterns
    :type ddphase: str

    pattern0 => invert read (readout |0>)
    pattern1 => normal read (readout |1>)

    """

    def is_sweepN(self) -> bool:
        return True

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["90pulse"] = P.FloatParam(
            10e-9,
            1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="90 deg (pi/2) pulse width.",
        )
        pd["180pulse"] = P.FloatParam(
            -1e-9,
            -1e-9,
            1e-6,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="180 deg (pi) pulse width. Negative value means 2 * 90pulse.",
        )
        pd["tauconst"] = P.FloatParam(
            1e-9,
            1e-9,
            1e-3,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="duration of free-evolution.",
        )
        pd["tau2const"] = P.FloatParam(
            1e-9,
            1e-9,
            1e-3,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="duration of free-evolution.",
        )
        pd["Nconst"] = P.IntParam(4, 1, 10000, doc="Number of pulse train repetitions.")
        pd["ddphase"] = P.StrParam("Y:X:Y:X,Y:X:Y:iX", doc="Phase patterns.")
        return pd

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p90, p180, tauconst, tau2const, Nconst, ddphase = [
            pulse_params[k]
            for k in ["90pulse", "180pulse", "tauconst", "tau2const", "Nconst", "ddphase"]
        ]

        pih_ch0, pih_ch1 = self.parse_phase_for_ddgate(ddphase)

        p0 = [p90, p180, tauconst, tau2const]
        p1 = [p90, p180, tauconst, tau2const]
        freq, _, common_pulses, p0, p1 = K.round_pulses(
            self.freq, [0], common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, pih_ch0]
        p1 = p1 + [Nconst, pih_ch1]

        def gen(n, p90, p180, tauconst, tau2const, Nconst, pih_ch):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)

            init_ptn = [(pih_ch[0], p90), ((mw_x,), tau_f)]
            storage_ptn = [((pih_ch[1][0],), tau_l), (pih_ch[1], p90)]
            reinit_ptn = [(pih_ch[2], p90), ((mw_x,), tau_f)]
            read_ptn = [((pih_ch[3][0],), tau_l), (pih_ch[3], p90)]

            px = [((mw_x,), tau2_l), ((mw_x, "mw"), p180), ((mw_x,), tau2_f)]
            py = [((mw_y,), tau2_l), ((mw_y, "mw"), p180), ((mw_y,), tau2_f)]
            pattern = px + py + px + py + py + px + py + px
            pattern *= Nconst
            first_wait = list(pattern[0])
            first_wait[1] = tau_l
            last_wait = list(pattern[-1])
            last_wait[1] = tau_f
            pattern[0] = tuple(first_wait)
            pattern[-1] = tuple(last_wait)

            xy8_first = init_ptn + pattern + storage_ptn
            xy8_second = reinit_ptn + pattern + read_ptn

            v_f, v_l = K.split_int(tau2const, self.split_fraction)
            interlude = [
                ((mw_y,), tau2const),
                ((mw_y, "mw"), p180),
                ((mw_y,), v_l),
                ((mw_y,), v_f),
            ] * n
            interlude[-1] = ((pih_ch[2][0],), v_f)

            return xy8_first + interlude + xy8_second

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=pih_ch0[-1][0],
                read_phase1=pih_ch1[-1][0],
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class DRabiGenerator(PatternGenerator):
    """Generate Pulse Pattern for Rabi nutation (Double Resonance, ENDOR) measurement.

    :param 180pulse: duration of 180 deg (pi) pulse at mw channel 1
    :type 180pulse: float
    :param mw1_delay: small wait time between mw and mw1 pulses
    :type mw1_delay: float

    pattern0 => D-Rabi: mw(pi) - NOP(mw1_delay) - mw1(tau) - NOP(mw1_delay) - mw(pi)
    pattern1 => No mw1: mw(pi) - NOP(mw1_delay) - NOP      - NOP(mw1_delay) - mw(pi)

    """

    def pulse_params(self) -> P.ParamDict[str, P.PDValue]:
        pd = P.ParamDict()
        pd["180pulse"] = P.FloatParam(
            10e-9, 1e-9, 1e-6, unit="s", SI_prefix=True, step=1e-9, doc="180 deg (pi) pulse width."
        )
        pd["mw1_delay"] = P.FloatParam(
            1e-9,
            1e-9,
            1e-3,
            unit="s",
            SI_prefix=True,
            step=1e-9,
            doc="small wait time between mw and mw1 pulses.",
        )
        return pd

    def num_mw(self) -> int:
        return 2

    def _generate(
        self,
        xdata,
        common_pulses: list[float],
        pulse_params: dict,
        partial: int,
        reduce_start_divisor: int,
        fix_base_width: int | None,
    ):
        p180, delay = [pulse_params[k] for k in ["180pulse", "mw1_delay"]]

        p0 = [p180, delay]
        p1 = [p180, delay]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [True]
        p1 = p1 + [False]

        def gen(v, p180, delay, operate):
            p = [
                ((mw_x, mw1_x, "mw"), p180),
                ((mw_x, mw1_x), delay),
                ((mw_x, mw1_x, "mw1"), v),
                ((mw_x, mw1_x), delay),
                ((mw_x, mw1_x, "mw"), p180),
            ]
            if not operate:
                p[2] = ((mw_x, mw1_x), v)
            return p

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen,
                p0,
                p1,
                read_phase0=(mw_x, mw1_x),
                read_phase1=(mw_x, mw1_x),
                laser_phase=(mw_x, mw1_x),
                partial=partial,
                fix_base_width=fix_base_width,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


def make_generators(
    freq: float = 2.0e9,
    reduce_start_divisor: int = 2,
    split_fraction: int = 4,
    minimum_block_length: int = 1000,
    block_base: int = 4,
    mw_modes: tuple[int] = (0,),
    iq_amplitude: float = 0.0,
    channel_remap: dict | None = None,
    print_fn=print,
):
    args = (
        freq,
        reduce_start_divisor,
        split_fraction,
        minimum_block_length,
        block_base,
        mw_modes,
        iq_amplitude,
        channel_remap,
        print_fn,
    )
    generators = {
        "rabi": RabiGenerator(*args),
        "t1": T1Generator(*args),
        "fid": FIDGenerator(*args),
        "spinecho": SpinEchoGenerator(*args),
        "trse": TRSEGenerator(*args),
        "cp": DDGenerator(*args, method="cp"),
        "cpmg": DDGenerator(*args, method="cpmg"),
        "xy4": DDGenerator(*args, method="xy4"),
        "xy8": DDGenerator(*args, method="xy8"),
        "xy16": DDGenerator(*args, method="xy16"),
        "cpN": DDNGenerator(*args, method="cpN"),
        "cpmgN": DDNGenerator(*args, method="cpmgN"),
        "xy4N": DDNGenerator(*args, method="xy4N"),
        "xy8N": DDNGenerator(*args, method="xy8N"),
        "xy16N": DDNGenerator(*args, method="xy16N"),
        "180train": PiTrainGenerator(*args),
        "se90sweep": SEHalfPiSweepGenerator(*args),
        "spinlock": SpinLockGenerator(*args),
        "xy8cl": XY8CorrelationGenerator(*args, method="xy8cl"),
        "xy8cl1flip": XY8CorrelationGenerator(*args, method="xy8cl1flip"),
        "xy8clNflip": XY8CorrelationNflipGenerator(*args),
        "ddgate": DDGateGenerator(*args),
        "ddgateN": DDNGateGenerator(*args),
    }
    if len(mw_modes) > 1:
        # add sequences using multiple mw channels here
        generators["drabi"] = DRabiGenerator(*args)
    if all([m == 1 for m in mw_modes]):
        # these methods requires 4 phases (x, y, x_inv, y_inv) and unavailable in 2-phase mode.
        for key in ["xy16", "xy16N", "ddgate", "ddgateN"]:
            del generators[key]
    if any([m == 2 for m in mw_modes]):
        # Randomized DD. no rcpmg because it is equivalent to rcp.
        generators["rcp"] = RDDGenerator(*args, method="cp")
        generators["rxy4"] = RDDGenerator(*args, method="xy4")
        generators["rxy8"] = RDDGenerator(*args, method="xy8")
        generators["rxy16"] = RDDGenerator(*args, method="xy16")
    return generators
