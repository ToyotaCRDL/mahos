#!/usr/bin/env python3

"""
Fitter for Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from functools import reduce
from operator import add

import numpy as np
from numpy.typing import NDArray
import lmfit as F

from ..msgs.podmr_msgs import PODMRData
from ..msgs.fit_msgs import PeakType
from ..msgs import param_msgs as P
from ..node.log import DummyLogger
from ..util.conv import real_fft
from .common_fitter import gaussian, lorentzian, voigt, BaseFitter
from .odmr_fitter import guess_single_peak, guess_multi_peak, guess_background


class Fitter(BaseFitter):
    def get_xydata(
        self, data: PODMRData, raw_params: dict[str, P.RawPDValue]
    ) -> tuple[NDArray, NDArray]:
        xdata = data.get_xdata()
        y0, _ = data.get_ydata()
        return xdata, y0


def rabi_decay_cos(x, A, T, f, dt):
    return A * np.exp(-(x - dt) / T) * np.cos(2.0 * np.pi * f * (x - dt))


class RabiFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(
                self.conf.get("c", 1.0),
                self.conf.get("c_min", 0.0),
                self.conf.get("c_max", 2.0),
                doc="constant baseline",
            ),
            A=self.make_model_param(
                self.conf.get("A", 0.15),
                self.conf.get("A_min", 0.001),
                self.conf.get("A_max", 1.0),
                doc="peak-to-peak amplitude of oscillation",
            ),
            T=self.make_model_param(
                400e-9, 1e-9, 1e3, unit="s", SI_prefix=True, doc="exponential decay T_2^*"
            ),
            f=self.make_model_param(4e6, 1e3, 100e6, unit="Hz", SI_prefix=True, doc="frequency"),
            dt=self.make_model_param(
                0.0,
                -15e-9,
                15e-9,
                unit="s",
                SI_prefix=True,
                fixable=True,
                doc="shift in time (x-axis)",
            ),
        )

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        ftau, fintensity = real_fft(xdata, ydata)

        if fit_params["c"].vary:
            fit_params["c"].set(np.average(ydata))
        if fit_params["A"].vary:
            A = np.max(ydata) - np.average(ydata)
            if fit_params["A"].value >= 0.0:
                fit_params["A"].set(A)
            else:
                fit_params["A"].set(-A)
        if fit_params["T"].vary:
            fit_params["T"].set(np.max(xdata))
        if fit_params["f"].vary:
            fit_params["f"].set(abs(ftau[np.argmax(fintensity)]))
        if fit_params["dt"].vary:
            fit_params["dt"].set(0.0)

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        baseline = F.models.ConstantModel()
        return baseline + F.Model(rabi_decay_cos)

    def additional_msg(self, popt: P.ParamDict[str, P.FloatParam]):
        f, dt, A, c = [popt[n].value() for n in ("f", "dt", "A", "c")]
        p90, p180 = 1 / 4 / f + dt, 1 / 2 / f + dt
        contrast = 2 * A / (c + A)
        return (
            f"Contrast: 2A/(c+A) = {contrast:.1%}, MW pulses: {p90*1E9:.1f} ns, {p180*1E9:.1f} ns"
        )


def T1_exp_decay(x, A, T1, m):
    return A * np.exp(-(m * x / T1))


class T1Fitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(0.0, -1.0, 1.0, fixable=True, doc="constant baseline"),
            A=self.make_model_param(0.5, 0.001, 1.0, doc="intensity at x = 0"),
            T1=self.make_model_param(
                10e-6, 1e-9, 1.0, unit="s", SI_prefix=True, doc="spin-lattice relaxation time"
            ),
            m=self.make_model_param(
                1.0,
                1.0,
                3.0,
                fixable=True,
                fixed=True,
                doc="coefficient for tau depending on energy-level structure",
            ),
        )

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        if fit_params["c"].vary:
            fit_params["c"].set(np.min(ydata))
        if fit_params["A"].vary:
            fit_params["A"].set(np.max(ydata))
        if fit_params["T1"].vary:
            fit_params["T1"].set(np.max(xdata) / 2.0)
        # don't guess m

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        baseline = F.models.ConstantModel()
        return baseline + F.Model(T1_exp_decay)


def spinecho_exp_decay(x, A, T2, n):
    return A * np.exp(-((x / T2) ** n))


class SpinEchoFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(0.0, -1.0, 1.0, fixable=True, doc="constant baseline"),
            A=self.make_model_param(0.5, 0.001, 1.0, doc="intensity at x = 0"),
            T2=self.make_model_param(
                10e-6, 1e-9, 1.0, unit="s", SI_prefix=True, doc="spin-spin relaxation time"
            ),
            n=self.make_model_param(2.0, 1.0, 3.0, fixable=True, doc=""),
        )

    def get_xydata(
        self, data: PODMRData, raw_params: dict[str, P.RawPDValue]
    ) -> tuple[NDArray, NDArray]:
        xdata = data.get_xdata(force_taumode="total")
        y0, _ = data.get_ydata()
        return xdata, y0

    def set_fit_data(self, data: PODMRData, fit_x, fit_y, raw_params, label, result_dict):
        # convert taumode from total to raw
        a, b = data.get_total_scale()
        x = (fit_x - b) / a

        data.set_fit_data(x, fit_y, raw_params, label, result_dict)

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        if fit_params["c"].vary:
            fit_params["c"].set(np.min(ydata))
        if fit_params["A"].vary:
            fit_params["A"].set(np.max(ydata))
        if fit_params["T2"].vary:
            fit_params["T2"].set(np.max(xdata) / 2.0)
        if fit_params["n"].vary:
            fit_params["n"].set(2.0)

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        baseline = F.models.ConstantModel()
        return baseline + F.Model(spinecho_exp_decay)


def fid_decay_cos(x, A, T2s, detuning):
    return A * (1 - np.exp(-x / T2s) * np.cos(2 * np.pi * detuning * x))


class FIDFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(0.0, -1.0, 1.0, doc="constant baseline"),
            A=self.make_model_param(-0.15, -1.0, 1.0, doc="intensity at x = 0"),
            T2s=self.make_model_param(
                10e-6, 1e-9, 1.0, unit="s", SI_prefix=True, doc="spin-spin relaxation time"
            ),
            detuning=self.make_model_param(
                1e6, 0.0, 1e9, unit="Hz", SI_prefix=True, doc="detuning frequency"
            ),
        )

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        if fit_params["c"].vary:
            fit_params["c"].set(np.max(ydata))
        if fit_params["A"].vary:
            fit_params["A"].set(np.max(ydata) - np.min(ydata))
        if fit_params["T2s"].vary:
            fit_params["T2s"].set(np.max(xdata) / 2.0)
        if fit_params["detuning"].vary:
            fit_params["detuning"].set(2.2e6)

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        baseline = F.models.ConstantModel()
        return baseline + F.Model(fid_decay_cos)


class SinglePeakFitter(Fitter):
    def param_dict(self) -> P.ParamDict[str, P.PDValue]:
        params = P.ParamDict(
            peak_type=P.EnumParam(PeakType, PeakType.Voigt),
            dip=P.BoolParam(
                self.conf.get("dip", True), doc="dip-shape instead of peak. used for guess."
            ),
            n_guess=P.IntParam(
                self.conf.get("n_guess", 20),
                1,
                1000,
                doc="number of data points in peak center guess.",
            ),
            n_guess_bg=P.IntParam(
                self.conf.get("n_guess_bg", 40),
                5,
                1000,
                doc="number of histogram bins in background guess.",
            ),
        )
        params.update(self.common_param_dict())
        return params

    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            slope=self.make_model_param(
                0.0, -1.0, 1.0, doc="baseline slope", SI_prefix=True, fixable=True, fixed=True
            ),
            intercept=self.make_model_param(
                0.0, -10, 10, doc="baseline y-intercept", fixable=True
            ),
            amplitude=self.make_model_param(0.1, -10, 10, doc="peak amplitude"),
            center=self.make_model_param(
                10e-6, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak center position"
            ),
            gamma=self.make_model_param(
                10e-9,
                1e-9,
                1e-3,
                unit="s",
                SI_prefix=True,
                doc="peak width param (Voigt or Lorentzian)",
            ),
            sigma=self.make_model_param(
                10e-9,
                1e-9,
                1e-3,
                unit="s",
                SI_prefix=True,
                doc="peak width param (Voigt or Gaussian)",
            ),
        )

    def model_params_for_fit(
        self, raw_params: dict[str, P.RawPDValue]
    ) -> P.ParamDict[str, P.PDValue]:
        params = self.model_params()
        if raw_params["peak_type"] == PeakType.Gaussian:
            del params["gamma"]
        elif raw_params["peak_type"] == PeakType.Lorentzian:
            del params["sigma"]
        return params

    def _guess_bg_ampl(self, ydata, fit_params, raw_params) -> float:
        dip = raw_params.get("dip", True)
        n_guess_bg = raw_params.get("n_guess_bg", 40)
        if n_guess_bg:
            bg = guess_background(ydata, n_guess_bg)
        else:
            if dip:
                bg = np.max(ydata)
            else:
                bg = np.min(ydata)

        if dip:
            fit_params["intercept"].set(bg)
            return np.min(ydata) - bg
        else:
            fit_params["intercept"].set(bg)
            return np.max(ydata) - bg

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        ampl = self._guess_bg_ampl(ydata, fit_params, raw_params)
        fit_params["amplitude"].set(ampl)
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        fit_params["center"].set(
            guess_single_peak(
                xdata, ydata, raw_params.get("dip", True), num=raw_params.get("n_guess", 20)
            ),
            min=np.min(xdata),
            max=np.max(xdata),
        )
        xrange = np.max(xdata) - np.min(xdata)

        if peak_type == PeakType.Voigt:
            fit_params["sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
            fit_params["gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        elif peak_type == PeakType.Gaussian:
            fit_params["sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        elif peak_type == PeakType.Lorentzian:
            fit_params["gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        baseline = F.models.LinearModel()
        peak_type = raw_params.get("peak_type", PeakType.Voigt)
        if peak_type == PeakType.Voigt:
            return baseline + F.Model(voigt)
        elif peak_type == PeakType.Gaussian:
            return baseline + F.Model(gaussian)
        elif peak_type == PeakType.Lorentzian:
            return baseline + F.Model(lorentzian)
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))


class MultiPeakFitter(SinglePeakFitter):
    def param_dict(self) -> P.ParamDict[str, P.PDValue]:
        params = P.ParamDict(
            n_peaks=P.IntParam(2, 2, 100, doc="number of peaks"),
            peak_type=P.EnumParam(PeakType, PeakType.Voigt),
            dip=P.BoolParam(
                self.conf.get("dip", True), doc="dip-shape instead of peak. used for guess."
            ),
            n_guess=P.IntParam(
                self.conf.get("n_guess", 20),
                1,
                1000,
                doc="number of data points in peak center guess.",
            ),
            n_guess_bg=P.IntParam(
                self.conf.get("n_guess_bg", 40),
                5,
                1000,
                doc="number of histogram bins in background guess.",
            ),
        )
        params.update(self.common_param_dict())
        return params

    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            slope=self.make_model_param(
                0.0, -1.0, 1.0, doc="baseline slope", SI_prefix=True, fixable=True, fixed=True
            ),
            intercept=self.make_model_param(
                0.0, -10, 10, doc="baseline y-intercept", fixable=True
            ),
            amplitude=self.make_model_param(0.1, -10, 10, doc="peak amplitude"),
            center=self.make_model_param(
                10e-6, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak center position"
            ),
            gamma=self.make_model_param(
                10e-9,
                1e-9,
                1e-3,
                unit="s",
                SI_prefix=True,
                doc="peak width param (Voigt or Lorentzian)",
            ),
            sigma=self.make_model_param(
                10e-9,
                1e-9,
                1e-3,
                unit="s",
                SI_prefix=True,
                doc="peak width param (Voigt or Gaussian)",
            ),
            # manually add many centers and filter them later
            p0_center=self.make_model_param(
                100e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak position"
            ),
            p1_center=self.make_model_param(
                100e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak position"
            ),
            p2_center=self.make_model_param(
                100e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak position"
            ),
            p3_center=self.make_model_param(
                100e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak position"
            ),
            p4_center=self.make_model_param(
                100e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak position"
            ),
            p5_center=self.make_model_param(
                100e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak position"
            ),
            p6_center=self.make_model_param(
                100e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak position"
            ),
            p7_center=self.make_model_param(
                100e-9, 1e-9, 1e-3, unit="s", SI_prefix=True, doc="peak position"
            ),
        )

    def model_params_for_fit(
        self, raw_params: dict[str, P.RawPDValue]
    ) -> P.ParamDict[str, P.PDValue]:
        params = P.ParamDict(
            slope=self.make_model_param(0.0, -1.0, 1.0, fixable=True, fixed=True),
            intercept=self.make_model_param(0.0, -1.0, 1.0, fixable=True),
        )
        for i in range(raw_params["n_peaks"]):
            params[f"p{i}_center"] = self.make_model_param(100e-9, 1e-9, 1e-3)
            params[f"p{i}_amplitude"] = self.make_model_param(0.1, -10, 10)

            if raw_params["peak_type"] == PeakType.Voigt:
                params[f"p{i}_gamma"] = self.make_model_param(10e-9, 1e-9, 1e-3)
                params[f"p{i}_sigma"] = self.make_model_param(10e-9, 1e-9, 1e-3)
            elif raw_params["peak_type"] == PeakType.Gaussian:
                params[f"p{i}_sigma"] = self.make_model_param(10e-9, 1e-9, 1e-3)
            elif raw_params["peak_type"] == PeakType.Lorentzian:
                params[f"p{i}_gamma"] = self.make_model_param(10e-9, 1e-9, 1e-3)
        return params

    def add_fit_params(
        self, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue], xdata, ydata
    ):
        for i in range(1, raw_params["n_peaks"]):
            fit_params.add(
                f"center_diff{i}", expr=f"p{i}_center-p{i-1}_center", min=0.0, max=np.max(xdata)
            )

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        ampl = self._guess_bg_ampl(ydata, fit_params, raw_params)
        n_peaks = raw_params["n_peaks"]
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        centers = guess_multi_peak(
            xdata,
            ydata,
            raw_params.get("dip", True),
            n_peaks=n_peaks,
            n_samples=raw_params.get("n_guess", 20),
        )
        for i, c in enumerate(centers):
            fit_params[f"p{i}_center"].set(c, min=np.min(xdata), max=np.max(xdata))
            fit_params[f"p{i}_amplitude"].set(ampl)

        xrange = np.max(xdata) - np.min(xdata)
        if peak_type == PeakType.Voigt:
            for i in range(n_peaks):
                fit_params[f"p{i}_sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
                fit_params[f"p{i}_gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        elif peak_type == PeakType.Gaussian:
            for i in range(n_peaks):
                fit_params[f"p{i}_sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        elif peak_type == PeakType.Lorentzian:
            for i in range(n_peaks):
                fit_params[f"p{i}_gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        n_peaks = raw_params["n_peaks"]
        peak_type = raw_params.get("peak_type", PeakType.Voigt)
        baseline = F.models.LinearModel()

        if peak_type == PeakType.Voigt:
            return reduce(
                add, [baseline] + [F.Model(voigt, prefix=f"p{i}_") for i in range(n_peaks)]
            )
        elif peak_type == PeakType.Gaussian:
            return reduce(
                add, [baseline] + [F.Model(gaussian, prefix=f"p{i}_") for i in range(n_peaks)]
            )
        elif peak_type == PeakType.Lorentzian:
            return reduce(
                add, [baseline] + [F.Model(lorentzian, prefix=f"p{i}_") for i in range(n_peaks)]
            )
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))


class PODMRFitter(object):
    def __init__(self, logger=None, conf=None):
        """Fitter for PODMR."""

        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger
        if conf is None:
            conf = {}

        self.fitters = {
            "rabi": RabiFitter(self.logger.info, conf.get("rabi")),
            "T1": T1Fitter(self.logger.info, conf.get("T1")),
            "spinecho": SpinEchoFitter(self.logger.info, conf.get("spinecho")),
            "fid": FIDFitter(self.logger.info, conf.get("fid")),
            "single_peak": SinglePeakFitter(self.logger.info, conf.get("single_peak")),
            "multi_peak": MultiPeakFitter(self.logger.info, conf.get("multi_peak")),
        }

    def get_param_dict_labels(self) -> list[str]:
        return list(self.fitters.keys())

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue]:
        """Get param dict for given label."""

        if label not in self.fitters:
            return P.ParamDict()
        return self.fitters[label].param_dict()

    def fit(self, data: PODMRData, params: dict, label: str) -> F.model.ModelResult | None:
        """Perform fitting. returns lmfit.model.ModelResult."""

        if label not in self.fitters:
            self.logger.error(f"Unknown label {label}")
            return None

        try:
            return self.fitters[label].fit(data, params, label)
        except Exception:
            self.logger.exception("Failed to fit.")
            return None

    def fitd(self, data: PODMRData, params: dict, label: str) -> dict:
        """Perform fitting. returns dict."""

        if label not in self.fitters:
            self.logger.error(f"Unknown label {label}")
            return {}

        try:
            return self.fitters[label].fitd(data, params, label)
        except Exception:
            self.logger.exception("Failed to fit.")
            return {}
