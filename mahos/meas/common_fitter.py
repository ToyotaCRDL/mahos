#!/usr/bin/env python3

"""
Common Fitter functions.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import lmfit as F
import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz

from ..msgs.data_msgs import Data
from ..msgs import param_msgs as P


def gaussian(x, amplitude, center, sigma):
    """Gaussian function with given peak amplitude (at x = center) and s.d. of sigma.

    HWHM is sqrt(2 * ln(2)) * sigma.

    Note: While GaussianModel of lmfit is normalized as p.d.f. (integral is unity),
    this function is not.

    """

    return amplitude * np.exp(-np.power(x - center, 2) / (2 * np.power(sigma, 2)))


def lorentzian(x, amplitude, center, gamma):
    """Lorentzian function with given peak amplitude (at x = center) and HWHM of gamma.

    Note: While LorentzianModel of lmfit is normalized as p.d.f. (integral is unity),
    this function is not.

    """

    sq = np.power(gamma, 2)
    return amplitude * sq / (np.power(x - center, 2) + sq)


def norm_voigt(x, center, sigma, gamma):
    """Normalized Voigt function. Definition as p.d.f. (integral is unity)."""

    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2.0))
    return wofz(z).real / (sigma * np.sqrt(2 * np.pi))


def voigt(x, amplitude, center, sigma, gamma):
    """Voigt function with given peak amplitude (at x = center)."""

    return (
        amplitude * norm_voigt(x, center, sigma, gamma) / norm_voigt(center, center, sigma, gamma)
    )


class BaseFitter(object):
    def __init__(self, print_fn=print):
        self.print_fn = print_fn

    def make_model_param(
        self,
        value,
        minimum,
        maximum,
        unit: str = "",
        SI_prefix: bool = False,
        fixable: bool = False,
        doc: str = "",
    ) -> P.ParamDict[str, P.FloatParam]:
        ps = P.ParamDict(
            value=P.FloatParam(value, unit=unit, SI_prefix=SI_prefix, doc=doc),
            min=P.FloatParam(minimum, unit=unit, SI_prefix=SI_prefix, optional=True),
            max=P.FloatParam(maximum, unit=unit, SI_prefix=SI_prefix, optional=True),
        )
        if fixable:
            ps["fix"] = P.BoolParam(False)
        return ps

    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict()

    def model_params_for_fit(
        self, raw_params: dict[str, P.RawPDValue]
    ) -> P.ParamDict[str, P.PDValue]:
        """Generate model_params for actual fitting according to given parameters `raw_params`.

        Override this if the content of model_params can be changed
        according to the given parameters.

        """

        return self.model_params()

    def param_dict(self) -> P.ParamDict[str, P.PDValue]:
        return self.common_param_dict()

    def add_fit_params(
        self, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue], xdata, ydata
    ):
        pass

    def additional_msg(self, popt: P.ParamDict[str, P.FloatParam]):
        return ""

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        pass

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        pass

    def common_param_dict(self) -> P.ParamDict[str, P.PDValue]:
        """Generate common ParamDict with keys guess, xbounds, fit_xnum, and model."""

        return P.ParamDict(
            guess=P.BoolParam(True, doc="auto-guess initial parameters"),
            xbounds=[
                P.FloatParam(
                    0.0,
                    optional=True,
                    enable=False,
                    SI_prefix=True,
                    doc="lower bound for x value used for fitting",
                ),
                P.FloatParam(
                    1.0,
                    optional=True,
                    enable=False,
                    SI_prefix=True,
                    doc="upper bound for x value used for fitting",
                ),
            ],
            fit_xnum=P.IntParam(301, 2, 10000, doc="number of points in x to draw fit curve"),
            model=self.model_params(),
        )

    def make_fit_params(self, raw_params: dict[str, P.RawPDValue]) -> F.Parameters:
        ps = F.Parameters()
        model_params = raw_params.get("model", {})

        for name, p in self.model_params_for_fit(raw_params).unwrap().items():
            if name in model_params:
                p = model_params[name]
            if "fix" in p and p["fix"]:
                ps.add(name, value=p["value"], vary=False)
            else:
                ps.add(name, value=p["value"], min=p["min"], max=p["max"])
        return ps

    def _make_opt_params(
        self, res: F.model.ModelResult, raw_params: dict[str, P.RawPDValue]
    ) -> P.ParamDict[str, P.FloatParam]:
        popt = {}
        for name, p in self.model_params_for_fit(raw_params).items():
            p["value"].set(res.best_values[name])
            popt[name] = p["value"]
        return P.ParamDict(popt)

    def get_xydata(
        self, data: Data, raw_params: dict[str, P.RawPDValue]
    ) -> tuple[NDArray, NDArray]:
        xdata = data.get_xdata()
        ydata = data.get_ydata()
        return xdata, ydata

    def set_fit_data(self, data: Data, fit_x, fit_y, raw_params, result_dict):
        data.set_fit_data(fit_x, fit_y, raw_params, result_dict)

    def _filter_data(self, xdata, ydata, raw_params):
        if "xbounds" not in raw_params:
            return xdata, ydata
        mn, mx = raw_params["xbounds"]
        if mn is None:
            mn = -np.inf
        if mx is None:
            mx = +np.inf
        indices = np.where(np.logical_and(mn <= xdata, xdata <= mx))
        return xdata[indices], ydata[indices]

    def _fit_xy(
        self, xdata, ydata, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]
    ) -> tuple[F.model.ModelResult, dict]:
        return self._fit_core(xdata, ydata, None, params)

    def _fit(
        self,
        data: Data,
        params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue],
    ) -> tuple[F.model.ModelResult, dict]:
        xdata, ydata = self.get_xydata(data, P.unwrap(params))
        return self._fit_core(xdata, ydata, data, params)

    def _fit_core(
        self,
        xdata,
        ydata,
        data: Data | None,
        params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue],
    ) -> tuple[F.model.ModelResult, dict]:
        raw_params = P.unwrap(params)
        model = self.model(raw_params)
        fit_params = self.make_fit_params(raw_params)
        self.add_fit_params(fit_params, raw_params, xdata, ydata)
        if raw_params.get("guess", True):
            self.guess_fit_params(xdata, ydata, fit_params, raw_params)
        xdata_, ydata_ = self._filter_data(xdata, ydata, raw_params)

        if self.print_fn is not None:
            self.print_fn("Fit with parameters:\n" + fit_params.pretty_repr().rstrip())

        res = model.fit(ydata_, x=xdata_, params=fit_params)

        if self.print_fn is not None:
            self.print_fn("Fit report:\n" + res.fit_report())

        if not res.success:
            return res, {}

        x = np.linspace(min(xdata_), max(xdata_), raw_params.get("fit_xnum", 301))
        y = res.eval(x=x)

        popt = self._make_opt_params(res, raw_params)

        msg = res.fit_report() + "\n" * 2
        for name, param in popt.items():
            msg += "{}: {}\n".format(name, param.value_to_str())

        amsg = self.additional_msg(popt)
        if amsg:
            msg += "\n" + amsg

        res_dict = {"msg": msg, "popt": popt.unwrap(), "pcov": res.covar}

        if data is not None:
            self.set_fit_data(data, x, y, raw_params, res_dict)

        return res, res_dict

    def fit(self, data: Data, params: P.ParamDict[str, P.PDValue]) -> F.model.ModelResult:
        return self._fit(data, params)[0]

    def fitd(self, data: Data, params: P.ParamDict[str, P.PDValue]) -> dict:
        return self._fit(data, params)[1]

    def fit_xy(self, xdata, ydata, params: P.ParamDict[str, P.PDValue]) -> F.model.ModelResult:
        return self._fit_xy(xdata, ydata, params)[0]

    def fit_xyd(self, xdata, ydata, params: P.ParamDict[str, P.PDValue]) -> dict:
        return self._fit_xy(xdata, ydata, params)[1]
