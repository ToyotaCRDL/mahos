#!/usr/bin/env python3

"""
Fitter for Spectroscopy.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
from functools import reduce
from operator import add

import numpy as np
import lmfit as F
from sklearn.cluster import KMeans

from ..msgs import param_msgs as P
from ..msgs.spectroscopy_msgs import SpectroscopyData
from ..msgs.fit_msgs import PeakType
from ..node.log import DummyLogger
from .common_fitter import gaussian, lorentzian, voigt, BaseFitter


def guess_background(ydata, bins=40):
    """Guess background by ydata's mode."""

    hist, edges = np.histogram(ydata, bins=bins)
    i = np.argmax(hist)
    return (edges[i] + edges[i + 1]) / 2.0


def guess_single_peak(xdata, ydata, num: int = 10):
    """Guess position of single peak/dip in given spectrum."""

    idx = np.argpartition(ydata, kth=-num)[-num:]
    return np.mean(xdata[idx])


def guess_multi_peak(xdata, ydata, n_peaks: int, n_samples: int = 10):
    """Guess position of multiple peaks/dips in given spectrum."""

    idx = np.argpartition(ydata, kth=-n_samples)[-n_samples:]
    xs = xdata[idx]
    km = KMeans(n_clusters=n_peaks).fit(xs.reshape(-1, 1))
    return sorted(km.cluster_centers_[:, 0])


class Fitter(BaseFitter):
    def param_dict(self) -> P.ParamDict[str, P.PDValue]:
        params = P.ParamDict(
            peak_type=P.EnumParam(PeakType, PeakType.Voigt),
            n_guess=P.IntParam(20, 1, 1000, doc="number of data points in peak center guess."),
            n_guess_bg=P.IntParam(
                40, 5, 1000, doc="number of histogram bins in background guess."
            ),
        )
        params.update(self.common_param_dict())
        return params

    def _guess_bg(self, ydata, fit_params, raw_params) -> float:
        n_guess_bg = raw_params.get("n_guess_bg", 40)
        if n_guess_bg:
            bg = guess_background(ydata, n_guess_bg)
            fit_params["c"].set(bg, min=np.min(ydata), max=2 * bg)
        else:
            bg = fit_params["c"].value
        return bg


class SingleFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(1.0, 0.0, 1.0, doc="constant baseline"),
            amplitude=self.make_model_param(1.0, -1.0, 1.0, doc="amplitude"),
            center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
            gamma=self.make_model_param(
                20, 0.1, 1e3, unit="nm", doc="peak width param (Voigt or Lorentzian)"
            ),
            sigma=self.make_model_param(
                20, 0.1, 1e3, unit="nm", doc="peak width param (Voigt or Gaussian)"
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

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        bg = self._guess_bg(ydata, fit_params, raw_params)
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        fit_params["amplitude"].set(
            np.max(ydata) - bg, min=(np.max(ydata) - bg) * 0.1, max=np.max(ydata)
        )
        fit_params["center"].set(
            guess_single_peak(xdata, ydata, num=raw_params.get("n_guess", 20)),
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

    def model(self, raw_params: dict[str, P.RawPDValue]):
        baseline = F.models.ConstantModel()
        peak_type = raw_params.get("peak_type", PeakType.Voigt)
        if peak_type == PeakType.Voigt:
            return baseline + F.Model(voigt)
        elif peak_type == PeakType.Gaussian:
            return baseline + F.Model(gaussian)
        elif peak_type == PeakType.Lorentzian:
            return baseline + F.Model(lorentzian)
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))


class MultiFitter(Fitter):
    def param_dict(self) -> P.ParamDict[str, P.PDValue]:
        params = P.ParamDict(
            n_peaks=P.IntParam(2, 2, 100, doc="number of peaks"),
        )
        params.update(Fitter.param_dict(self))
        return params

    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(1.0, 0.0, 1.0, doc="constant baseline"),
            # shape parameters is considered common
            amplitude=self.make_model_param(1.0, -1.0, 1.0, doc="amplitude"),
            gamma=self.make_model_param(
                20, 0.1, 1e3, unit="nm", doc="peak width param (Voigt or Lorentzian)"
            ),
            sigma=self.make_model_param(
                20, 0.1, 1e3, unit="nm", doc="peak width param (Voigt or Gaussian)"
            ),
            # manually add many centers and filter them later
            p0_center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
            p1_center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
            p2_center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
            p3_center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
            p4_center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
            p5_center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
            p6_center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
            p7_center=self.make_model_param(600.0, 10.0, 10000.0, unit="nm", doc="peak position"),
        )

    def model_params_for_fit(
        self, raw_params: dict[str, P.RawPDValue]
    ) -> P.ParamDict[str, P.PDValue]:
        params = P.ParamDict(
            c=self.make_model_param(1.0, 0.0, 1.0, doc="constant baseline"),
        )
        for i in range(raw_params["n_peaks"]):
            params[f"p{i}_center"] = self.make_model_param(600.0, 10.0, 10000.0, unit="nm")
            params[f"p{i}_amplitude"] = self.make_model_param(1.0, -1.0, 1.0)

            if raw_params["peak_type"] == PeakType.Voigt:
                params[f"p{i}_gamma"] = self.make_model_param(20, 0.1, 1e3)
                params[f"p{i}_sigma"] = self.make_model_param(20, 0.1, 1e3)
            elif raw_params["peak_type"] == PeakType.Gaussian:
                params[f"p{i}_sigma"] = self.make_model_param(20, 0.1, 1e3)
            elif raw_params["peak_type"] == PeakType.Lorentzian:
                params[f"p{i}_gamma"] = self.make_model_param(20, 0.1, 1e3)
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
        bg = self._guess_bg(ydata, fit_params, raw_params)

        n_peaks = raw_params["n_peaks"]
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        centers = guess_multi_peak(
            xdata, ydata, n_peaks=n_peaks, n_samples=raw_params.get("n_guess", 20)
        )
        for i, c in enumerate(centers):
            fit_params[f"p{i}_center"].set(c, min=np.min(xdata), max=np.max(xdata))
        for i in range(n_peaks):
            fit_params[f"p{i}_amplitude"].set(
                np.max(ydata) - bg, min=(np.max(ydata) - bg) * 0.1, max=np.max(ydata)
            )

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

    def model(self, raw_params: dict[str, P.RawPDValue]):
        n_peaks = raw_params["n_peaks"]
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        baseline = F.models.ConstantModel()
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


class SpectroscopyFitter(object):
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
            "single": SingleFitter(print_fn),
            "multi": MultiFitter(print_fn),
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

    def fit(self, data: SpectroscopyData, params: dict) -> F.model.ModelResult | None:
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

    def fitd(self, data: SpectroscopyData, params: dict) -> dict:
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

        :param xdata: wavelength in nm
        :param ydata: intensity

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
