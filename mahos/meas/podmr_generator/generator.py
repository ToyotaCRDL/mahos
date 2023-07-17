#!/usr/bin/env python3

"""
Pattern generators for Pulse ODMR.

.. This file is a part of MAHOS project.

"""

import typing as T
import re

import numpy as np
from . import generator_kernel as K

from ...msgs.inst_pg_msgs import Channels


class PatternGenerator(object):
    def __init__(
        self,
        freq: float = 2.0e9,
        reduce_start_divisor: int = 2,
        split_fraction: int = 4,
        minimum_block_length: int = 1000,
        block_base: int = 4,
        print_fn=print,
        method: str = "",
    ):
        self.freq = freq
        self.reduce_start_divisor = reduce_start_divisor
        self.split_fraction = split_fraction
        self.minimum_block_length = minimum_block_length
        self.block_base = block_base
        self.print_fn = print_fn
        self.method = method

    def pulse_params(self) -> T.List[str]:
        """Return list of names of required pulse params."""

        return []

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

    def get_pulse_params(self, params: dict):
        return [params[k] for k in self.pulse_params()]

    def generate(self, xdata, params: dict):
        """Generate the blocks."""

        if params.get("enable_reduce", False):
            reduce_start_divisor = self.reduce_start_divisor
        else:
            reduce_start_divisor = 0

        blocks, freq, common_pulses = self._generate(
            xdata,
            self.get_common_pulses(params),
            self.get_pulse_params(params),
            params.get("partial", -1),
            params.get("nomw", False),
            reduce_start_divisor,
            params.get("ignore_basewidth", False),
        )

        blocks, laser_timing = K.build_blocks(
            blocks,
            common_pulses,
            divide=params.get("divide_block", False),
            invertY=params.get("invertY", False),
            minimum_block_length=self.minimum_block_length,
            block_base=self.block_base,
        )
        return blocks, freq, laser_timing

    def generate_raw_blocks(self, xdata, params: dict):
        """Generate raw blocks without finishing it with build_blocks. Debug purpose."""

        if params.get("enable_reduce", False):
            reduce_start_divisor = self.reduce_start_divisor
        else:
            reduce_start_divisor = 0

        blocks, freq, common_pulses = self._generate(
            xdata,
            self.get_common_pulses(params),
            self.get_pulse_params(params),
            params.get("partial", -1),
            params.get("nomw", False),
            reduce_start_divisor,
            params.get("ignore_basewidth", False),
        )
        return blocks, freq

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        raise NotImplementedError("_generate() is not implemented.")


class RabiGenerator(PatternGenerator):
    """Generate Pulse Pattern for Rabi nutation measurement.

    pattern0 => Rabi
    pattern1 => no operation (T1 limited)

    """

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, [], [], reduce_start_divisor, self.print_fn
        )
        p0 = [True]
        p1 = [False]

        def gen_single_ptn_Rabi(v, operate):
            if operate:
                return [(("mw_x", "mw"), v)]
            else:
                return [(("mw_x",), v)]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_Rabi,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["90pulse"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90 = pulse_params[0]

        p0 = [p90]
        p1 = [p90]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + ["mw_x"]
        p1 = p1 + ["mw_x_inv"]

        def gen_single_ptn_FID(v, mw_width, mw_read_phase):
            v_f, v_l = K.split_int(v, self.split_fraction)
            return [
                (("mw_x", "mw"), mw_width),
                (("mw_x",), v_f),
                ((mw_read_phase,), v_l),
                ((mw_read_phase, "mw"), mw_width),
            ]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_FID,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x_inv",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse", "readY"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, readY = pulse_params

        read_phase0 = "mw_y" if readY else "mw_x"
        read_phase1 = "mw_y_inv" if readY else "mw_x_inv"
        p0 = [p90, p180]
        p1 = [p90, p180]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [read_phase0]
        p1 = p1 + [read_phase1]

        def gen_single_ptn_SpinEcho(v, mw_width, mw_flip_width, mw_read_phase):
            v_f, v_l = K.split_int(v, self.split_fraction)
            # mw_width_inv = mw_width + mw_flip_width
            return [
                (("mw_x", "mw"), mw_width),
                (("mw_x",), v),
                (("mw_x", "mw"), mw_flip_width),
                (("mw_x",), v_f),
                ((mw_read_phase,), v_l),
                ((mw_read_phase, "mw"), mw_width),
            ]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_SpinEcho,
                p0,
                p1,
                read_phase0=read_phase0,
                read_phase1=read_phase1,
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class DRamseyGenerator(PatternGenerator):
    """Generate Pulse Pattern for D Ramsey measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float

    """

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180 = pulse_params

        p0 = [p90, p180]
        p1 = [p90, p180]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + ["mw_x"]
        p1 = p1 + ["mw_x_inv"]

        def gen_single_ptn_SpinEcho(v, mw_width, mw_flip_width, mw_read_phase):
            v_f, v_l = K.split_int(v, self.split_fraction)
            mw_finalflip = mw_width
            return [
                (("mw_x", "mw2"), mw_width),
                (("mw_x",), v),
                (("mw_x", "mw2"), mw_flip_width),
                (("mw_x", "mw"), mw_flip_width),
                (("mw_x", "mw2"), mw_flip_width),
                (("mw_x",), v_f),
                ((mw_read_phase,), v_l),
                ((mw_read_phase, "mw"), mw_finalflip),
            ]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_SpinEcho,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x_inv",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class TEchoGenerator(PatternGenerator):
    """Generate Pulse Pattern for T Echo measurement.

    :param 90pulse: duration of 90 deg (pi/2) pulse
    :type 90pulse: float
    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float

    """

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180 = pulse_params

        p0 = [p90, p180]
        p1 = [p90, p180]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + ["mw_x"]
        p1 = p1 + ["mw_x_inv"]

        def gen_single_ptn_SpinEcho(v, mw_width, mw_flip_width, mw_read_phase):
            v_f, v_l = K.split_int(v, self.split_fraction)
            mw_finalflip = mw_width
            return [
                (("mw_x", "mw"), mw_width),
                (("mw_x",), v),
                (("mw_x", "mw2"), mw_flip_width),
                (("mw_x", "mw"), mw_flip_width),
                (("mw_x", "mw2"), mw_flip_width),
                (("mw_x",), v_f),
                ((mw_read_phase,), v_l),
                (("mw_x", "mw"), mw_flip_width),
                (("mw_x", "mw2"), mw_flip_width),
                (("mw_x", "mw"), mw_flip_width),
                ((mw_read_phase, "mw"), mw_finalflip),
            ]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_SpinEcho,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x_inv",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse", "tauconst"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, tauconst = pulse_params

        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + ["mw_x"]
        p1 = p1 + ["mw_x_inv"]

        def gen_single_ptn_TRSE(v, mw_width, mw_flip_width, tauconst, mw_read_phase):
            v_f, v_l = K.split_int(v, self.split_fraction)
            return [
                (("mw_x", "mw"), mw_width),
                (("mw_x",), tauconst),
                (("mw_x", "mw"), mw_flip_width),
                (("mw_x",), v_f),
                ((mw_read_phase,), v_l),
                ((mw_read_phase, "mw"), mw_width),
            ]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_TRSE,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x_inv",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        params = ["90pulse", "180pulse", "Nconst", "readY"]
        if self.method in ("xy8", "xy16"):
            params.append("supersample")
        return params

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        if self.method in ("xy8", "xy16"):
            supersample = pulse_params[-1]
        else:
            supersample = 1

        if supersample == 1:
            return self._generate_normal(
                xdata,
                common_pulses,
                pulse_params,
                partial,
                nomw,
                reduce_start_divisor,
                ignore_basewidth,
            )
        else:
            return self._generate_supersample(
                xdata,
                common_pulses,
                pulse_params,
                partial,
                nomw,
                reduce_start_divisor,
                ignore_basewidth,
            )

    def _generate_normal(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, Nconst, readY = pulse_params[:4]
        read_phase0 = {True: "mw_y_inv", False: "mw_x_inv"}[readY]
        read_phase1 = {True: "mw_y", False: "mw_x"}[readY]
        p0 = [p90, p180]
        p1 = [p90, p180]

        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, read_phase0, self.method]
        p1 = p1 + [Nconst, read_phase1, self.method]

        def gen_single_ptn_DD(tau, mw_width, mw_flip_width, Nconst, read_phase, method):
            tau_f, tau_l = K.split_int(tau, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tau * 2, self.split_fraction)
            init_ptn = [(("mw_x", "mw"), mw_width), (("mw_x",), tau_f)]
            read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), mw_width)]

            pattern = []
            px = [(("mw_x",), tau2_l), (("mw_x", "mw"), mw_flip_width), (("mw_x",), tau2_f)]
            py = [(("mw_y",), tau2_l), (("mw_y", "mw"), mw_flip_width), (("mw_y",), tau2_f)]
            ix = [
                (("mw_x_inv",), tau2_l),
                (("mw_x_inv", "mw"), mw_flip_width),
                (("mw_x_inv",), tau2_f),
            ]
            iy = [
                (("mw_y_inv",), tau2_l),
                (("mw_y_inv", "mw"), mw_flip_width),
                (("mw_y_inv",), tau2_f),
            ]

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

            first_wait = list(pattern[0])
            first_wait[1] = tau_l
            last_wait = list(pattern[-1])
            last_wait[1] = tau_f
            pattern[0] = tuple(first_wait)
            pattern[-1] = tuple(last_wait)

            return init_ptn + pattern + read_ptn

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_DD,
                p0,
                p1,
                read_phase0=read_phase0,
                read_phase1=read_phase1,
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses

    def _generate_supersample(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, Nconst, readY, supersample = pulse_params
        read_phase0 = {True: "mw_y_inv", False: "mw_x_inv"}[readY]
        read_phase1 = {True: "mw_y", False: "mw_x"}[readY]
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
            phase_list = ["mw_x", "mw_y", "mw_x", "mw_y", "mw_y", "mw_x", "mw_y", "mw_x"] * Nconst
        elif self.method == "xy16":
            pulse_num = Nconst * 16
            phase_list = [
                "mw_x",
                "mw_y",
                "mw_x",
                "mw_y",
                "mw_y",
                "mw_x",
                "mw_y",
                "mw_x",
                "mw_x_inv",
                "mw_y_inv",
                "mw_x_inv",
                "mw_y_inv",
                "mw_y_inv",
                "mw_x_inv",
                "mw_y_inv",
                "mw_x_inv",
            ] * Nconst
        else:
            raise ValueError(f"Unknown method {self.method}")

        sample = (np.linspace(0, 1, num=supersample + 1) * pulse_num).astype(np.int64)[
            :-1
        ] / pulse_num
        dt = abs(tau[0] - tau[1])
        p0 = p0 + [dt, read_phase0, pulse_num, phase_list]
        p1 = p1 + [dt, read_phase1, pulse_num, phase_list]

        def gen_single_ptn_DD_Qinterpolation(
            v, mw_width, mw_flip_width, dt, read_phase, pulse_num, phase_list
        ):
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
                (("mw_x", "mw"), mw_width),
                (("mw_x",), K.split_int(tau_list[0], self.split_fraction)[0]),
            ]
            read_ptn = [
                ((read_phase,), K.split_int(tau_list[-1], self.split_fraction)[1]),
                ((read_phase, "mw"), mw_width),
            ]
            pattern = []

            def mw_pattern(phase_before, tau_before, phase_next, tau_next):
                return [
                    ((phase_before,), K.split_int(tau_before, self.split_fraction)[1]),
                    ((phase_before, "mw"), mw_flip_width),
                    ((phase_next,), K.split_int(tau_next, self.split_fraction)[0]),
                ]

            phase_list_ = phase_list + [read_phase]
            for i in range(len(phase_list)):
                pattern += mw_pattern(
                    phase_list_[i], tau_list[i], phase_list_[i + 1], tau_list[i + 1]
                )

            return init_ptn + pattern + read_ptn

        ind = np.arange(len(tau) * len(sample)).reshape((len(tau), len(sample)))
        blocks = [
            K.generate_blocks(
                i,
                (tau[j], sample[k]),
                common_pulses,
                gen_single_ptn_DD_Qinterpolation,
                p0,
                p1,
                read_phase0=read_phase0,
                read_phase1=read_phase1,
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
            )
            for (j, k), i in np.ndenumerate(ind)
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

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse", "tauconst", "readY"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, tauconst, readY = pulse_params

        read_phase0 = {True: "mw_y_inv", False: "mw_x_inv"}[readY]
        read_phase1 = {True: "mw_y", False: "mw_x"}[readY]
        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, dummy, common_pulses, p0, p1 = K.round_pulses(
            self.freq, [0], common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [read_phase0, self.method]
        p1 = p1 + [read_phase1, self.method]

        def gen_single_ptn_DDN(n, mw_width, mw_flip_width, tauconst, read_phase, method):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)
            init_ptn = [(("mw_x", "mw"), mw_width), (("mw_x",), tau_f)]
            read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), mw_width)]

            pattern = []
            px = [(("mw_x",), tau2_l), (("mw_x", "mw"), mw_flip_width), (("mw_x",), tau2_f)]
            py = [(("mw_y",), tau2_l), (("mw_y", "mw"), mw_flip_width), (("mw_y",), tau2_f)]
            ix = [
                (("mw_x_inv",), tau2_l),
                (("mw_x_inv", "mw"), mw_flip_width),
                (("mw_x_inv",), tau2_f),
            ]
            iy = [
                (("mw_y_inv",), tau2_l),
                (("mw_y_inv", "mw"), mw_flip_width),
                (("mw_y_inv",), tau2_f),
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

            first_wait = list(pattern[0])
            first_wait[1] = tau_l
            last_wait = list(pattern[-1])
            last_wait[1] = tau_f
            pattern[0] = tuple(first_wait)
            pattern[-1] = tuple(last_wait)

            return init_ptn + pattern + read_ptn

        blocks = [
            K.generate_blocks(
                i,
                n,
                common_pulses,
                gen_single_ptn_DDN,
                p0,
                p1,
                read_phase0=read_phase0,
                read_phase1=read_phase1,
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["tauconst", "Nconst"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        tauconst, Nconst = pulse_params

        p0 = [tauconst]
        p1 = [tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, True]
        p1 = p1 + [Nconst, False]

        def gen_single_ptn_pitrain(v, tauconst, Nconst, operate):
            p = [(("mw_x", "mw"), v), (("mw_x",), tauconst)]
            p *= Nconst

            flip = [(("mw_x", "mw"), v)]
            wait = [(("mw_x",), tauconst)]
            p = (flip + wait) * (Nconst - 1) + flip

            if operate:
                return p
            else:
                t = sum([pp[1] for pp in p])
                return [(("mw_x",), t)]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_pitrain,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["180pulse", "tauconst"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p180, tauconst = pulse_params

        p0 = [p180, tauconst]
        p1 = [p180, tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [True]
        p1 = p1 + [False]

        def gen_single_ptn_SpinEcho90sweep(v, mw_flip_width, tauconst, operate):
            p = [
                (("mw_x", "mw"), v),
                (("mw_x",), tauconst),
                (("mw_x", "mw"), mw_flip_width),
                (("mw_x",), tauconst),
                (("mw_x", "mw"), v),
            ]

            if operate:
                return p
            else:
                t = sum([pp[1] for pp in p])
                return [(("mw_x",), t)]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_SpinEcho90sweep,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class RecoveryGenerator(PatternGenerator):
    """Generate Pulse Pattern for Recovery measurement.

    :param 180pulse: duration of 180 deg (pi) pulse
    :type 180pulse: float
    :param invertinit: If True, invert initialization.
    :type invertinit: bool

    If invertinit is False:
        pattern0 => |0> recovery ( tau     , read |0>)
        pattern1 => |0> recovery ( tau - pi, read |1>)
    If invertinit is True:
        pattern0 => |1> recovery ( pi - tau - pi, read |0>)
        pattern1 => |1> recovery ( pi - tau     , read |1>)

    """

    def pulse_params(self) -> T.List[str]:
        return ["180pulse", "invertinit"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p180, invertinit = pulse_params

        if invertinit:
            p0 = [p180, p180]
            p1 = [p180, 0]
        else:
            p0 = [0, 0]
            p1 = [0, p180]

        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )

        def gen_single_ptn_Recovery(v, mw_init_width, mw_read_width):
            return [
                (("mw_x", "mw"), mw_init_width),
                (("mw_x",), v),
                (("mw_x", "mw"), mw_read_width),
            ]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_Recovery,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "iq_delay"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, iq_delay = pulse_params

        p0 = [p90, iq_delay]
        p1 = [p90, iq_delay]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + ["mw_x_inv"]
        p1 = p1 + ["mw_x"]

        def gen_single_ptn_SpinLock(v, mw_width, iq_delay, mw_read_phase):
            return [
                (("mw_x", "mw"), mw_width),
                (("mw_y",), iq_delay),
                (("mw_y", "mw"), v),
                ((mw_read_phase,), iq_delay),
                ((mw_read_phase, "mw"), mw_width),
            ]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_SpinLock,
                p0,
                p1,
                read_phase0="mw_x_inv",
                read_phase1="mw_x",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse", "tauconst", "Nconst", "reinitX", "readY"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, tauconst, Nconst, reinitX, readY = pulse_params

        read_phase0 = {True: "mw_y_inv", False: "mw_x_inv"}[readY]
        read_phase1 = {True: "mw_y", False: "mw_x"}[readY]
        reinit_phase = {True: "mw_x", False: "mw_y"}[reinitX]
        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, reinit_phase, read_phase0, self.method]
        p1 = p1 + [Nconst, reinit_phase, read_phase1, self.method]

        def gen_single_ptn_xy8cl(
            tau, mw_width, mw_flip_width, tauconst, Nconst, reinit_phase, read_phase, method
        ):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)

            init_ptn = [(("mw_x", "mw"), mw_width), (("mw_x",), tau_f)]
            storage_ptn = [(("mw_y",), tau_l), (("mw_y", "mw"), mw_width)]
            reinit_ptn = [((reinit_phase, "mw"), mw_width), ((reinit_phase,), tau_f)]
            read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), mw_width)]

            px = [(("mw_x",), tau2_l), (("mw_x", "mw"), mw_flip_width), (("mw_x",), tau2_f)]
            py = [(("mw_y",), tau2_l), (("mw_y", "mw"), mw_flip_width), (("mw_y",), tau2_f)]
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
            if method == "xy8cl":
                interlude = [(("mw_y",), v_l), ((reinit_phase,), v_f)]
            elif method == "xy8cl1flip":
                interlude = [
                    (("mw_y",), v_l),
                    (("mw_x",), v_f),
                    (("mw_x", "mw"), mw_flip_width),
                    (("mw_x",), v_l),
                    ((reinit_phase,), v_f),
                ]
            else:
                raise ValueError(f"Unknown method {method}")

            return xy8_first + interlude + xy8_second

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_xy8cl,
                p0,
                p1,
                read_phase0=read_phase0,
                read_phase1=read_phase1,
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse", "tauconst", "Nconst", "reinitX", "readY"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, tauconst, Nconst, reinitX, readY = pulse_params

        read_phase0 = {True: "mw_y_inv", False: "mw_x_inv"}[readY]
        read_phase1 = {True: "mw_y", False: "mw_x"}[readY]
        reinit_phase = {True: "mw_x", False: "mw_y"}[reinitX]
        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, dummy, common_pulses, p0, p1 = K.round_pulses(
            self.freq, [0], common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, reinit_phase, read_phase0]
        p1 = p1 + [Nconst, reinit_phase, read_phase1]

        def gen_single_ptn_xy8clNflip(
            n, mw_width, mw_flip_width, tauconst, Nconst, reinit_phase, read_phase
        ):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)

            init_ptn = [(("mw_x", "mw"), mw_width), (("mw_x",), tau_f)]
            storage_ptn = [(("mw_y",), tau_l), (("mw_y", "mw"), mw_width), (("mw_y",), tau2_f)]
            reinit_ptn = [
                ((reinit_phase,), tau2_l),
                ((reinit_phase, "mw"), mw_width),
                ((reinit_phase,), tau_f),
            ]
            read_ptn = [((read_phase,), tau_l), ((read_phase, "mw"), mw_width)]

            px = [(("mw_x",), tau2_l), (("mw_x", "mw"), mw_flip_width), (("mw_x",), tau2_f)]
            py = [(("mw_y",), tau2_l), (("mw_y", "mw"), mw_flip_width), (("mw_y",), tau2_f)]
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
            # Since timing misalignment of first/last are included in storage_ptn/reinit_ptn,
            # we don't neet to fix them here.
            interlude = px * n

            return xy8_first + interlude + xy8_second

        blocks = [
            K.generate_blocks(
                i,
                n,
                common_pulses,
                gen_single_ptn_xy8clNflip,
                p0,
                p1,
                read_phase0=read_phase0,
                read_phase1=read_phase1,
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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
            print_fn=print_fn,
            method=method,
        )
        c = "|".join(["X", "Y", "iX", "iY", "n"])
        p = "^({c:s}):({c:s}):({c:s}):({c:s}),({c:s}):({c:s}):({c:s}):({c:s})$".format(c=c)
        self.ddphase_pattern = re.compile(p)

    def parse_phase_for_ddgate(self, phase) -> T.Tuple[T.List[Channels], T.List[Channels]]:
        """parse string representation of phase into mw on/off and phase (mw, mw_x, mw_y etc.).

        :returns: (channels0, channels1). first element is phase (mw_x etc.) in all the channels.

        """

        m = self.ddphase_pattern.match(phase.replace(" ", ""))
        if m is None:
            raise ValueError(f"Invalid phases {phase}")
        plist = m.groups()

        phase_dict = {
            "X": ("mw_x", "mw"),
            "Y": ("mw_y", "mw"),
            "iX": ("mw_x_inv", "mw"),
            "iY": ("mw_y_inv", "mw"),
            "n": ("mw_x",),
        }  # assign phase only at this moment.
        p_ch = [phase_dict[p] for p in plist]
        pih_ch0 = p_ch[:4]
        pih_ch1 = p_ch[4:]

        return pih_ch0, pih_ch1

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse", "tauconst", "Nconst", "N2const", "N3const", "ddphase"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, tauconst, Nconst, N2const, N3const, ddphase = pulse_params

        pih_ch0, pih_ch1 = self.parse_phase_for_ddgate(ddphase)

        p0 = [p90, p180, tauconst]
        p1 = [p90, p180, tauconst]
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, N2const, N3const, pih_ch0]
        p1 = p1 + [Nconst, N2const, N3const, pih_ch1]

        def gen_single_ptn_ddgate(
            tau, mw_width, mw_flip_width, tauconst, Nconst, N2const, N3const, pih_ch
        ):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)

            init_ptn = [(pih_ch[0], mw_width), (("mw_x",), tau_f)]
            storage_ptn = [((pih_ch[1][0],), tau_l), (pih_ch[1], mw_width)]
            reinit_ptn = [(pih_ch[2], mw_width), (("mw_x",), tau_f)]
            read_ptn = [((pih_ch[3][0],), tau_l), (pih_ch[3], mw_width)]

            px = [(("mw_x",), tau2_l), (("mw_x", "mw"), mw_flip_width), (("mw_x",), tau2_f)]
            py = [(("mw_y",), tau2_l), (("mw_y", "mw"), mw_flip_width), (("mw_y",), tau2_f)]
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
                (("mw_y",), tau),
                (("mw_y", "mw"), mw_flip_width),
                (("mw_y",), v_l),
                (("mw_y",), v_f),
            ] * N2const
            interlude[-1] = ((pih_ch[2][0],), v_f)

            return (xy8_first + interlude + xy8_second) * N3const

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_ddgate,
                p0,
                p1,
                read_phase0=pih_ch0[-1][0],
                read_phase1=pih_ch1[-1][0],
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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

    def pulse_params(self) -> T.List[str]:
        return ["90pulse", "180pulse", "tauconst", "tau2const", "Nconst", "ddphase"]

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        p90, p180, tauconst, tau2const, Nconst, ddphase = pulse_params

        pih_ch0, pih_ch1 = self.parse_phase_for_ddgate(ddphase)

        p0 = [p90, p180, tauconst, tau2const]
        p1 = [p90, p180, tauconst, tau2const]
        freq, _, common_pulses, p0, p1 = K.round_pulses(
            self.freq, [0], common_pulses, p0, p1, reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [Nconst, pih_ch0]
        p1 = p1 + [Nconst, pih_ch1]

        def gen_single_ptn_ddgate(n, mw_width, mw_flip_width, tauconst, tau2const, Nconst, pih_ch):
            tau_f, tau_l = K.split_int(tauconst, self.split_fraction)
            tau2_f, tau2_l = K.split_int(tauconst * 2, self.split_fraction)

            init_ptn = [(pih_ch[0], mw_width), (("mw_x",), tau_f)]
            storage_ptn = [((pih_ch[1][0],), tau_l), (pih_ch[1], mw_width)]
            reinit_ptn = [(pih_ch[2], mw_width), (("mw_x",), tau_f)]
            read_ptn = [((pih_ch[3][0],), tau_l), (pih_ch[3], mw_width)]

            px = [(("mw_x",), tau2_l), (("mw_x", "mw"), mw_flip_width), (("mw_x",), tau2_f)]
            py = [(("mw_y",), tau2_l), (("mw_y", "mw"), mw_flip_width), (("mw_y",), tau2_f)]
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
                (("mw_y",), tau2const),
                (("mw_y", "mw"), mw_flip_width),
                (("mw_y",), v_l),
                (("mw_y",), v_f),
            ] * n
            interlude[-1] = ((pih_ch[2][0],), v_f)

            return xy8_first + interlude + xy8_second

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_ddgate,
                p0,
                p1,
                read_phase0=pih_ch0[-1][0],
                read_phase1=pih_ch1[-1][0],
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
            )
            for i, v in enumerate(xdata)
        ]
        return blocks, freq, common_pulses


class CWODMRGenerator(PatternGenerator):
    """Generate Pulse Pattern for CW-ODMR measurement.

    pattern0 => only laser
    pattern1 => laser and MW

    """

    def _generate(
        self,
        xdata,
        common_pulses: T.List[float],
        pulse_params: list,
        partial: int,
        nomw: bool,
        reduce_start_divisor: int,
        ignore_basewidth: bool,
    ):
        freq, xdata, common_pulses, p0, p1 = K.round_pulses(
            self.freq, xdata, common_pulses, [], [], reduce_start_divisor, self.print_fn
        )
        p0 = p0 + [False]
        p1 = p1 + [True]

        def gen_single_ptn_ODMRdiff(v, operate):
            if operate:
                return [(("mw_x", "laser", "mw"), v)]
            else:
                return [(("mw_x", "laser"), v)]

        blocks = [
            K.generate_blocks(
                i,
                v,
                common_pulses,
                gen_single_ptn_ODMRdiff,
                p0,
                p1,
                read_phase0="mw_x",
                read_phase1="mw_x",
                partial=partial,
                nomw=nomw,
                ignore_basewidth=ignore_basewidth,
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
    print_fn=print,
):
    args = (freq, reduce_start_divisor, split_fraction, minimum_block_length, block_base, print_fn)
    generators = {
        "rabi": RabiGenerator(*args),
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
        "recovery": RecoveryGenerator(*args),
        "spinlock": SpinLockGenerator(*args),
        "xy8cl": XY8CorrelationGenerator(*args, method="xy8cl"),
        "xy8cl1flip": XY8CorrelationGenerator(*args, method="xy8cl1flip"),
        "xy8clNflip": XY8CorrelationNflipGenerator(*args),
        "ddgate": DDGateGenerator(*args),
        "ddgateN": DDNGateGenerator(*args),
    }
    return generators
