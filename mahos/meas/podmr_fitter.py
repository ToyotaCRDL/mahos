#!/usr/bin/env python3

"""
Fitter for Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import lmfit as F
import numpy as np
from numpy.typing import NDArray

from ..msgs.podmr_msgs import PODMRData
from ..msgs import param_msgs as P
from ..node.log import DummyLogger
from ..util.conv import real_fft
from .common_fitter import gaussian, lorentzian, BaseFitter


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
            c=self.make_model_param(1.0, 0.0, 2.0, doc="constant baseline"),
            A=self.make_model_param(0.15, 0.001, 1.0, doc="peak-to-peak amplitude of oscillation"),
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
            fit_params["A"].set(np.max(ydata) - np.average(ydata))
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


def T1_exp_decay(x, A, T1):
    return A * np.exp(-(x / T1))


class T1Fitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(0.0, -1.0, 1.0, fixable=True, doc="constant baseline"),
            A=self.make_model_param(0.5, 0.001, 1.0, doc="intensity at x = 0"),
            T1=self.make_model_param(
                10e-6, 1e-9, 1.0, unit="s", SI_prefix=True, doc="spin-lattice relaxation time"
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

    def set_fit_data(self, data: PODMRData, fit_x, fit_y, raw_params, result_dict):
        # convert taumode from total to raw
        a, b = data.get_total_scale()
        x = (fit_x - b) / a

        data.set_fit_data(x, fit_y, raw_params, result_dict)

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


class GaussianFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(0.0, -1.0, 1.0, doc="constant baseline"),
            amplitude=self.make_model_param(0.1, 0.0, 1.0, doc="peak amplitude"),
            center=self.make_model_param(
                10e-6, 1e-9, 1.0, unit="s", SI_prefix=True, doc="peak center position"
            ),
            sigma=self.make_model_param(
                10e-9, 0.0, 1e9, unit="s", SI_prefix=True, doc="peak width parameter"
            ),
        )

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        if fit_params["c"].vary:
            fit_params["c"].set(np.min(ydata))
        if fit_params["amplitude"].vary:
            fit_params["amplitude"].set(np.max(ydata) - np.min(ydata))
        if fit_params["center"].vary:
            fit_params["center"].set(xdata[np.argmax(ydata)])
        if fit_params["sigma"].vary:
            fit_params["sigma"].set(0.1 * (np.max(xdata) - np.min(xdata)))

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        baseline = F.models.ConstantModel()
        return baseline + F.Model(gaussian)


class LorentzianFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(0.0, -1.0, 1.0, doc="constant baseline"),
            amplitude=self.make_model_param(0.1, 0.0, 1.0, doc="peak amplitude"),
            center=self.make_model_param(
                10e-6, 1e-9, 1.0, unit="s", SI_prefix=True, doc="peak center position"
            ),
            gamma=self.make_model_param(
                10e-9, 0.0, 1e9, unit="s", SI_prefix=True, doc="peak HWHM"
            ),
        )

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        if fit_params["c"].vary:
            fit_params["c"].set(np.min(ydata))
        if fit_params["amplitude"].vary:
            fit_params["amplitude"].set(np.max(ydata) - np.min(ydata))
        if fit_params["center"].vary:
            fit_params["center"].set(xdata[np.argmax(ydata)])
        if fit_params["gamma"].vary:
            fit_params["gamma"].set(0.1 * (np.max(xdata) - np.min(xdata)))

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        baseline = F.models.ConstantModel()
        return baseline + F.Model(lorentzian)


class PODMRFitter(object):
    def __init__(self, logger=None):
        """Fitter for PODMR."""

        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

        self.fitters = {
            "rabi": RabiFitter(self.logger.info),
            "T1": T1Fitter(self.logger.info),
            "spinecho": SpinEchoFitter(self.logger.info),
            "fid": FIDFitter(self.logger.info),
            "gaussian": GaussianFitter(self.logger.info),
            "lorentzian": LorentzianFitter(self.logger.info),
        }

    def get_param_dict_labels(self) -> list[str]:
        return list(self.fitters.keys())

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue]:
        """Get param dict for given label."""

        if label not in self.fitters:
            return P.ParamDict()
        return self.fitters[label].param_dict()

    def fit(self, data: PODMRData, params: dict) -> F.model.ModelResult | None:
        """Perform fitting. returns lmfit.model.ModelResult."""

        m = params.get("method")
        if m not in self.fitters:
            self.logger.error(f"Unknown method {m}")
            return None

        try:
            return self.fitters[m].fit(data, params)
        except Exception:
            self.logger.exception("Failed to fit.")
            return None

    def fitd(self, data: PODMRData, params: dict) -> dict:
        """Perform fitting. returns dict."""

        m = params.get("method")
        if m not in self.fitters:
            self.logger.error(f"Unknown method {m}")
            return {}

        try:
            return self.fitters[m].fitd(data, params)
        except Exception:
            self.logger.exception("Failed to fit.")
            return {}
