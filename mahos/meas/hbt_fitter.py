#!/usr/bin/env python3

"""
Fitter for HBT.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import lmfit as F

from ..msgs import param_msgs as P
from ..msgs.hbt_msgs import HBTData
from ..node.log import DummyLogger
from .common_fitter import BaseFitter


class Fitter(BaseFitter):
    def param_dict(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            guess=P.BoolParam(True, doc="auto-guess initial parameters"),
            xbounds=[
                P.FloatParam(
                    -200e-9,
                    optional=True,
                    enable=True,
                    unit="s",
                    SI_prefix=True,
                    doc="lower bound for x value used for fitting",
                ),
                P.FloatParam(
                    200e-9,
                    optional=True,
                    enable=True,
                    unit="s",
                    SI_prefix=True,
                    doc="upper bound for x value used for fitting",
                ),
            ],
            fit_xnum=P.IntParam(501, 2, 10000, doc="number of points in x to draw fit curve"),
            model=self.model_params(),
        )

    def get_xydata(
        self, data: HBTData, raw_params: dict[str, P.RawPDValue]
    ) -> tuple[NDArray, NDArray]:
        xdata = data.get_xdata(normalize=True)
        ydata = data.get_ydata(normalize=True)

        return xdata, ydata

    def set_fit_data(self, data: HBTData, fit_x, fit_y, raw_params, result_dict):
        data.set_fit_data(
            data.denormalize_xdata(fit_x), data.denormalize_ydata(fit_y), raw_params, result_dict
        )


def three_level_g2(x, t1, t2, alpha):
    return 1.0 - (1.0 + alpha) * np.exp(-np.abs(x) / t1) + alpha * np.exp(-np.abs(x) / t2)


class ThreeLevelFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            t1=self.make_model_param(
                1e-9, 1e-12, 1e-6, unit="s", SI_prefix=True, doc="time constant (e.s. lifetime)"
            ),
            t2=self.make_model_param(
                1e-9, 1e-12, 1e-6, unit="s", SI_prefix=True, doc="time constant"
            ),
            alpha=self.make_model_param(1.0, 0.1, 10, doc="parameter"),
        )

    def model(self, raw_params: dict[str, P.RawPDValue]) -> F.Model:
        return F.Model(three_level_g2)


class HBTFitter(object):
    def __init__(self, logger=None, silent=False):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

        if silent:
            print_fn = self.print_null
        else:
            print_fn = self.logger.info

        self.fitters = {
            "threelevel": ThreeLevelFitter(print_fn),
        }

    def print_null(self, msg):
        pass

    def get_param_dict_names(self) -> list[str]:
        return list(self.fitters.keys())

    def get_param_dict(self, method: str) -> P.ParamDict[str, P.PDValue]:
        """Get param dict for given method."""

        if method not in self.fitters:
            return P.ParamDict()
        return self.fitters[method].param_dict()

    def fit(self, data: HBTData, params: dict) -> F.model.ModelResult | None:
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

    def fitd(self, data: HBTData, params: dict) -> dict:
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

    def fit_xy(self, xdata, ydata, params: dict) -> F.model.ModelResult | None:
        """Perform fitting using xdata and ydata. returns lmfit.model.ModelResult.

        :param xdata: time in sec
        :param ydata: events

        """

        m = params.get("method")
        if m not in self.fitters:
            self.logger.error(f"Unknown method {m}")
            return None

        try:
            return self.fitters[m].fit_xy(xdata, ydata, params)
        except Exception:
            self.logger.exception("Failed to fit.")
            return None
