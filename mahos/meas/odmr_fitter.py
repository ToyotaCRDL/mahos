#!/usr/bin/env python3

"""
Fitter for ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
from functools import reduce
from operator import add

import numpy as np
from numpy.typing import NDArray
import lmfit as F
from sklearn.cluster import KMeans

from ..util.nv import peaks_of_B, peaks_of_B_aligned, gamma_MHz_mT
from ..msgs import param_msgs as P
from ..msgs.odmr_msgs import ODMRData
from ..msgs.fit_msgs import PeakType
from ..node.log import DummyLogger
from .common_fitter import gaussian, lorentzian, voigt, BaseFitter


def normalize(data: np.ndarray) -> np.ndarray:
    """normalize data within [0, 1] using max and min values."""

    max_ = np.max(data)
    min_ = np.min(data)

    # all values in data are exactly same. avoid nan by zero division.
    if max_ == min_:
        return np.ones_like(data)

    return (data - min_) / (max_ - min_)


def denormalize(data: np.ndarray, data_raw: np.ndarray) -> np.ndarray:
    """denormalize data."""

    max_ = np.max(data_raw)
    min_ = np.min(data_raw)

    return (max_ - min_) * data + min_


def guess_background(ydata, bins=40):
    """Guess background by ydata's mode."""

    hist, edges = np.histogram(ydata, bins=bins)
    i = np.argmax(hist)
    return (edges[i] + edges[i + 1]) / 2.0


def guess_single_peak(xdata, ydata, dip: bool, num: int = 10):
    """Guess position of single peak/dip in given spectrum."""

    if dip:
        idx = np.argpartition(ydata, kth=num)[:num]
    else:
        idx = np.argpartition(ydata, kth=-num)[-num:]
    return np.mean(xdata[idx])


def guess_multi_peak(xdata, ydata, dip: bool, n_peaks: int, n_samples: int = 10):
    """Guess position of multiple peaks/dips in given spectrum."""

    if dip:
        idx = np.argpartition(ydata, kth=n_samples)[:n_samples]
    else:
        idx = np.argpartition(ydata, kth=-n_samples)[-n_samples:]
    xs = xdata[idx]
    km = KMeans(n_clusters=n_peaks).fit(xs.reshape(-1, 1))
    return sorted(km.cluster_centers_[:, 0])


class Fitter(BaseFitter):
    def param_dict(self) -> P.ParamDict[str, P.PDValue]:
        params = P.ParamDict(
            peak_type=P.EnumParam(PeakType, PeakType.Voigt),
            dip=P.BoolParam(True, doc="dip-shape instead of peak. used for guess."),
            n_guess=P.IntParam(20, 1, 1000, doc="number of data points in peak center guess."),
            n_guess_bg=P.IntParam(
                40, 5, 1000, doc="number of histogram bins in background guess."
            ),
        )
        params.update(self.common_param_dict())
        return params

    def get_xydata(
        self, data: ODMRData, raw_params: dict[str, P.RawPDValue]
    ) -> tuple[NDArray, NDArray]:
        xdata = data.get_xdata()
        if data.has_background():
            # use background-normalized data when background is available
            ydata = data.get_ydata(normalize_n=1)[0]
        else:
            ydata = data.get_ydata()[0]
        # scale xdata to MHz, normalize ydata into [0, 1]
        return 1e-6 * xdata, normalize(ydata)

    def set_fit_data(self, data: ODMRData, fit_x, fit_y, raw_params, result_dict):
        # rescale xdata to Hz, denormalize ydata
        if data.has_background():
            ydata = data.get_ydata(normalize_n=1)[0]
        else:
            ydata = data.get_ydata()[0]
        data.set_fit_data(fit_x * 1e6, denormalize(fit_y, ydata), raw_params, result_dict)

    def _guess_bg(self, ydata, fit_params, raw_params) -> tuple[float]:
        """guess bg and return bg, ampl, ampl_min, ampl_max"""

        dip = raw_params.get("dip", True)
        n_guess_bg = raw_params.get("n_guess_bg", 40)
        if n_guess_bg:
            bg = guess_background(ydata, n_guess_bg)
            if dip:
                fit_params["c"].set(bg, min=2 * bg - 1, max=1.0)
            else:
                fit_params["c"].set(bg, min=0.0, max=2 * bg)
        else:
            bg = fit_params["c"].value

        if dip:
            return bg, -bg, -1.0, -bg * 0.1
        else:
            return bg, 1.0 - bg, (1.0 - bg) * 0.1, 1.0


class SingleFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(1.0, 0.0, 1.0, doc="constant baseline"),
            amplitude=self.make_model_param(1.0, -1.0, 1.0, doc="amplitude"),
            center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
            gamma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Lorentzian)"
            ),
            sigma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Gaussian)"
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
        bg, ampl, ampl_min, ampl_max = self._guess_bg(ydata, fit_params, raw_params)
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        fit_params["amplitude"].set(ampl, min=ampl_min, max=ampl_max)

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
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Lorentzian)"
            ),
            sigma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Gaussian)"
            ),
            # manually add many centers and filter them later
            p0_center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
            p1_center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
            p2_center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
            p3_center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
            p4_center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
            p5_center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
            p6_center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
            p7_center=self.make_model_param(2780, 0.1, 1e5, unit="MHz", doc="peak position"),
        )

    def model_params_for_fit(
        self, raw_params: dict[str, P.RawPDValue]
    ) -> P.ParamDict[str, P.PDValue]:
        params = P.ParamDict(
            c=self.make_model_param(1.0, 0.0, 1.0, doc="constant baseline"),
        )
        for i in range(raw_params["n_peaks"]):
            params[f"p{i}_center"] = self.make_model_param(2780, 0.0, 1e5)
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
        bg, ampl, ampl_min, ampl_max = self._guess_bg(ydata, fit_params, raw_params)
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
            fit_params[f"p{i}_amplitude"].set(ampl, min=ampl_min, max=ampl_max)

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


def B_lorentzians_many(
    x,
    B,
    theta,
    phi,
    p0_amplitude,
    p1_amplitude,
    p2_amplitude,
    p3_amplitude,
    p4_amplitude,
    p5_amplitude,
    p6_amplitude,
    p7_amplitude,
    p0_gamma,
    p1_gamma,
    p2_gamma,
    p3_gamma,
    p4_gamma,
    p5_gamma,
    p6_gamma,
    p7_gamma,
):
    centers = sorted(peaks_of_B(B, theta, phi))
    return (
        lorentzian(x, p0_amplitude, centers[0], p0_gamma)
        + lorentzian(x, p1_amplitude, centers[1], p1_gamma)
        + lorentzian(x, p2_amplitude, centers[2], p2_gamma)
        + lorentzian(x, p3_amplitude, centers[3], p3_gamma)
        + lorentzian(x, p4_amplitude, centers[4], p4_gamma)
        + lorentzian(x, p5_amplitude, centers[5], p5_gamma)
        + lorentzian(x, p6_amplitude, centers[6], p6_gamma)
        + lorentzian(x, p7_amplitude, centers[7], p7_gamma)
    )


def B_lorentzians(x, B, theta, phi, amplitude, gamma):
    centers = sorted(peaks_of_B(B, theta, phi))
    return (
        lorentzian(x, amplitude, centers[0], gamma)
        + lorentzian(x, amplitude, centers[1], gamma)
        + lorentzian(x, amplitude, centers[2], gamma)
        + lorentzian(x, amplitude, centers[3], gamma)
        + lorentzian(x, amplitude, centers[4], gamma)
        + lorentzian(x, amplitude, centers[5], gamma)
        + lorentzian(x, amplitude, centers[6], gamma)
        + lorentzian(x, amplitude, centers[7], gamma)
    )


def B_lorentzians_aligned(x, B, p03_amplitude, p12_amplitude, p03_gamma, p12_gamma):
    centers = sorted(peaks_of_B_aligned(B))
    return (
        lorentzian(x, p03_amplitude, centers[0], p03_gamma)
        + lorentzian(x, p12_amplitude, centers[1], p12_gamma)
        + lorentzian(x, p12_amplitude, centers[2], p12_gamma)
        + lorentzian(x, p03_amplitude, centers[3], p03_gamma)
    )


def B_gaussians_many(
    x,
    B,
    theta,
    phi,
    p0_amplitude,
    p1_amplitude,
    p2_amplitude,
    p3_amplitude,
    p4_amplitude,
    p5_amplitude,
    p6_amplitude,
    p7_amplitude,
    p0_sigma,
    p1_sigma,
    p2_sigma,
    p3_sigma,
    p4_sigma,
    p5_sigma,
    p6_sigma,
    p7_sigma,
):
    centers = sorted(peaks_of_B(B, theta, phi))
    return (
        gaussian(x, p0_amplitude, centers[0], p0_sigma)
        + gaussian(x, p1_amplitude, centers[1], p1_sigma)
        + gaussian(x, p2_amplitude, centers[2], p2_sigma)
        + gaussian(x, p3_amplitude, centers[3], p3_sigma)
        + gaussian(x, p4_amplitude, centers[4], p4_sigma)
        + gaussian(x, p5_amplitude, centers[5], p5_sigma)
        + gaussian(x, p6_amplitude, centers[6], p6_sigma)
        + gaussian(x, p7_amplitude, centers[7], p7_sigma)
    )


def B_gaussians(x, B, theta, phi, amplitude, sigma):
    centers = sorted(peaks_of_B(B, theta, phi))
    return (
        gaussian(x, amplitude, centers[0], sigma)
        + gaussian(x, amplitude, centers[1], sigma)
        + gaussian(x, amplitude, centers[2], sigma)
        + gaussian(x, amplitude, centers[3], sigma)
        + gaussian(x, amplitude, centers[4], sigma)
        + gaussian(x, amplitude, centers[5], sigma)
        + gaussian(x, amplitude, centers[6], sigma)
        + gaussian(x, amplitude, centers[7], sigma)
    )


def B_gaussians_aligned(x, B, p03_amplitude, p12_amplitude, p03_sigma, p12_sigma):
    centers = sorted(peaks_of_B_aligned(B))
    return (
        gaussian(x, p03_amplitude, centers[0], p03_sigma)
        + gaussian(x, p12_amplitude, centers[1], p12_sigma)
        + gaussian(x, p12_amplitude, centers[2], p12_sigma)
        + gaussian(x, p03_amplitude, centers[3], p03_sigma)
    )


def B_voigts(x, B, theta, phi, amplitude, sigma, gamma):
    centers = sorted(peaks_of_B(B, theta, phi))
    return (
        voigt(x, amplitude, centers[0], sigma, gamma)
        + voigt(x, amplitude, centers[1], sigma, gamma)
        + voigt(x, amplitude, centers[2], sigma, gamma)
        + voigt(x, amplitude, centers[3], sigma, gamma)
        + voigt(x, amplitude, centers[4], sigma, gamma)
        + voigt(x, amplitude, centers[5], sigma, gamma)
        + voigt(x, amplitude, centers[6], sigma, gamma)
        + voigt(x, amplitude, centers[7], sigma, gamma)
    )


def B_voigts_aligned(
    x, B, p03_amplitude, p12_amplitude, p03_sigma, p12_sigma, p03_gamma, p12_gamma
):
    centers = sorted(peaks_of_B_aligned(B))
    return (
        voigt(x, p03_amplitude, centers[0], p03_sigma, p03_gamma)
        + voigt(x, p12_amplitude, centers[1], p12_sigma, p12_gamma)
        + voigt(x, p12_amplitude, centers[2], p12_sigma, p12_gamma)
        + voigt(x, p03_amplitude, centers[3], p03_sigma, p03_gamma)
    )


class NVBFitter(Fitter):
    def param_dict(self) -> P.ParamDict[str, P.PDValue]:
        ps = Fitter.param_dict(self)
        ps["many"] = P.BoolParam(False, doc="use many parameter model")
        return ps

    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(1.0, 0.0, 1.0, doc="constant baseline"),
            # shape parameters is considered common
            amplitude=self.make_model_param(1.0, -1.0, 1.0, doc="amplitude"),
            gamma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Lorentzian)"
            ),
            sigma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Gaussian)"
            ),
            B=self.make_model_param(0.0, 0.0, 1e6, unit="mT", doc="B-field amplitude"),
            theta=self.make_model_param(np.pi / 4, 0.0, np.pi / 2.0, doc="B-field direction"),
            phi=self.make_model_param(np.pi / 4, 0.0, np.pi, doc="B-field direction"),
        )

    def model_params_for_fit(
        self, raw_params: dict[str, P.RawPDValue]
    ) -> P.ParamDict[str, P.PDValue]:
        many = raw_params.get("many", False)
        params = self.model_params()
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        if not many:
            if raw_params["peak_type"] == PeakType.Gaussian:
                del params["gamma"]
            elif raw_params["peak_type"] == PeakType.Lorentzian:
                del params["sigma"]
            return params

        del params["amplitude"]
        del params["gamma"]
        del params["sigma"]

        for i in range(8):
            params[f"p{i}_amplitude"] = self.make_model_param(1.0, -1.0, 1.0)

            if peak_type == PeakType.Voigt:
                raise ValueError("many = True and peak_type = PeakType.Voigt is not supported.")
            elif peak_type == PeakType.Gaussian:
                params[f"p{i}_sigma"] = self.make_model_param(20, 0.1, 1e3)
            elif peak_type == PeakType.Lorentzian:
                params[f"p{i}_gamma"] = self.make_model_param(20, 0.1, 1e3)
            else:
                raise ValueError("Unexpected peak_type: " + str(peak_type))
        return params

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        bg, ampl, ampl_min, ampl_max = self._guess_bg(ydata, fit_params, raw_params)
        n_peaks = 8
        many = raw_params.get("many", False)
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        centers = guess_multi_peak(
            xdata,
            ydata,
            raw_params.get("dip", True),
            n_peaks=n_peaks,
            n_samples=raw_params.get("n_guess", 20),
        )
        fit_params["B"].set(
            (centers[-1] - centers[0]) / (2 * gamma_MHz_mT),
            min=0.0,
            max=(xdata[-1] - xdata[0]) / (2 * gamma_MHz_mT),
        )
        if many:
            for i in range(n_peaks):
                fit_params[f"p{i}_amplitude"].set(ampl, min=ampl_min, max=ampl_max)
        else:
            fit_params["amplitude"].set(ampl, min=ampl_min, max=ampl_max)

        xrange = np.max(xdata) - np.min(xdata)
        if peak_type == PeakType.Voigt:
            if many:
                raise ValueError("many = True and peak_type = PeakType.Voigt is not supported.")
            fit_params["sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
            fit_params["gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        elif peak_type == PeakType.Gaussian:
            if many:
                for i in range(n_peaks):
                    fit_params[f"p{i}_sigma"].set(
                        xrange * 0.01, min=xdata[1] - xdata[0], max=xrange
                    )
            else:
                fit_params["sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        elif peak_type == PeakType.Lorentzian:
            if many:
                for i in range(n_peaks):
                    fit_params[f"p{i}_gamma"].set(
                        xrange * 0.01, min=xdata[1] - xdata[0], max=xrange
                    )
            else:
                fit_params["gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))

    def model(self, raw_params: dict[str, P.RawPDValue]):
        peak_type = raw_params.get("peak_type", PeakType.Voigt)
        many = raw_params.get("many", False)

        baseline = F.models.ConstantModel()
        if peak_type == PeakType.Voigt:
            if many:
                raise ValueError("many = True and peak_type = PeakType.Voigt is not supported.")
            return baseline + F.Model(B_voigts)
        elif peak_type == PeakType.Gaussian:
            if many:
                return baseline + F.Model(B_gaussians_many)
            else:
                return baseline + F.Model(B_gaussians)
        elif peak_type == PeakType.Lorentzian:
            if many:
                return baseline + F.Model(B_lorentzians_many)
            else:
                return baseline + F.Model(B_lorentzians)
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))


class NVBAlignedFitter(Fitter):
    def model_params(self) -> P.ParamDict[str, P.PDValue]:
        return P.ParamDict(
            c=self.make_model_param(1.0, 0.0, 1.0, doc="constant baseline"),
            # shape parameters is considered common
            p03_amplitude=self.make_model_param(1.0, -1.0, 1.0, doc="amplitude"),
            p12_amplitude=self.make_model_param(1.0, -1.0, 1.0, doc="amplitude"),
            p03_gamma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Lorentzian)"
            ),
            p12_gamma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Lorentzian)"
            ),
            p03_sigma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Gaussian)"
            ),
            p12_sigma=self.make_model_param(
                20, 0.1, 1e3, unit="MHz", doc="peak width param (Voigt or Gaussian)"
            ),
            B=self.make_model_param(0.0, 0.0, 1e6, unit="mT", doc="B-field amplitude"),
        )

    def model_params_for_fit(
        self, raw_params: dict[str, P.RawPDValue]
    ) -> P.ParamDict[str, P.PDValue]:
        params = self.model_params()
        if raw_params["peak_type"] == PeakType.Gaussian:
            del params["p03_gamma"]
            del params["p12_gamma"]
        elif raw_params["peak_type"] == PeakType.Lorentzian:
            del params["p03_sigma"]
            del params["p12_sigma"]
        return params

    def guess_fit_params(
        self, xdata, ydata, fit_params: F.Parameters, raw_params: dict[str, P.RawPDValue]
    ):
        bg, ampl, ampl_min, ampl_max = self._guess_bg(ydata, fit_params, raw_params)
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        centers = guess_multi_peak(
            xdata,
            ydata,
            raw_params.get("dip", True),
            n_peaks=4,
            n_samples=raw_params.get("n_guess", 20),
        )
        fit_params["B"].set(
            (centers[-1] - centers[0]) / (2 * gamma_MHz_mT),
            min=0.0,
            max=(xdata[-1] - xdata[0]) / (2 * gamma_MHz_mT),
        )
        fit_params["p03_amplitude"].set(ampl, min=ampl_min, max=ampl_max)
        fit_params["p12_amplitude"].set(ampl, min=ampl_min, max=ampl_max)

        xrange = np.max(xdata) - np.min(xdata)
        if peak_type == PeakType.Voigt:
            fit_params["p03_sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
            fit_params["p03_gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
            fit_params["p12_sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
            fit_params["p12_gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        elif peak_type == PeakType.Gaussian:
            fit_params["p03_sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
            fit_params["p12_sigma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        elif peak_type == PeakType.Lorentzian:
            fit_params["p03_gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
            fit_params["p12_gamma"].set(xrange * 0.01, min=xdata[1] - xdata[0], max=xrange)
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))

    def model(self, raw_params: dict[str, P.RawPDValue]):
        peak_type = raw_params.get("peak_type", PeakType.Voigt)

        baseline = F.models.ConstantModel()
        if peak_type == PeakType.Voigt:
            return baseline + F.Model(B_voigts_aligned)
        elif peak_type == PeakType.Gaussian:
            return baseline + F.Model(B_gaussians_aligned)
        elif peak_type == PeakType.Lorentzian:
            return baseline + F.Model(B_lorentzians_aligned)
        else:
            raise ValueError("Unexpected peak_type: " + str(peak_type))


class ODMRFitter(object):
    # scale xdata in MHz
    XCOEFF = 1e-6
    XUNIT = "MHz"

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
            "nvb": NVBFitter(print_fn),
            "nvba": NVBAlignedFitter(print_fn),
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

    def fit(self, data: ODMRData, params: dict) -> F.model.ModelResult | None:
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

    def fitd(self, data: ODMRData, params: dict) -> dict:
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

        :param xdata: frequency in MHz
        :param ydata: normalized intensity

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


# function interfaces for iodmr_fitter.


def fit_single(
    xdata,
    ydata,
    peak_type: PeakType = PeakType.Gaussian,
    n_guess: int = 20,
    dip: bool = True,
    silent: bool = False,
) -> F.model.ModelResult | None:
    """Fit ODMR spectrum with single-dip/peak shape. Wrapper around ODMRFitter."""

    fitter = ODMRFitter(silent=silent)
    params = {
        "method": "single",
        "guess": True,
        "peak_type": peak_type,
        "n_guess": n_guess,
        "dip": dip,
    }
    return fitter.fit_xy(xdata, ydata, params)


def fit_multi(
    xdata,
    ydata,
    n_peaks: int,
    peak_type: PeakType = PeakType.Gaussian,
    n_guess: int = 20,
    dip: bool = True,
    silent: bool = False,
) -> F.model.ModelResult | None:
    """Fit ODMR spectrum with n-dip/peak shape. Wrapper around ODMRFitter."""

    fitter = ODMRFitter(silent=silent)
    params = {
        "method": "multi",
        "guess": True,
        "n_peaks": n_peaks,
        "peak_type": peak_type,
        "n_guess": n_guess,
        "dip": dip,
    }
    return fitter.fit_xy(xdata, ydata, params)


def fit_double(
    xdata,
    ydata,
    peak_type: PeakType = PeakType.Gaussian,
    n_guess: int = 20,
    dip: bool = True,
    silent: bool = False,
) -> F.model.ModelResult | None:
    return fit_multi(xdata, ydata, 2, peak_type=peak_type, n_guess=n_guess, dip=dip, silent=silent)


def fit_quad(
    xdata,
    ydata,
    peak_type: PeakType = PeakType.Gaussian,
    n_guess: int = 20,
    dip: bool = True,
    silent: bool = False,
) -> F.model.ModelResult | None:
    return fit_multi(xdata, ydata, 4, peak_type=peak_type, n_guess=n_guess, dip=dip, silent=silent)


def fit_NVB(
    xdata,
    ydata,
    peak_type: PeakType = PeakType.Gaussian,
    n_guess: int = 20,
    dip: bool = True,
    many: bool = False,
    silent: bool = False,
) -> F.model.ModelResult | None:
    """Fit ODMR spectrum for NV in single crystal diamond with 8-dip/peak shape.

    Wrapper around ODMRFitter.

    """

    fitter = ODMRFitter(silent=silent)
    params = {
        "method": "nvb",
        "guess": True,
        "peak_type": peak_type,
        "n_guess": n_guess,
        "dip": dip,
        "many": many,
    }
    return fitter.fit_xy(xdata, ydata, params)


def fit_NVB_aligned(
    xdata,
    ydata,
    peak_type: PeakType = PeakType.Gaussian,
    n_guess: int = 20,
    dip: bool = True,
    silent: bool = False,
) -> F.model.ModelResult:
    """Fit ODMR spectrum for NV in single crystal diamond with 4-dip/peak shape.

    The positions of peaks are parametrized by B field aligned to one NV-axis.

    Wrapper around ODMRFitter.

    """

    fitter = ODMRFitter(silent=silent)
    params = {
        "method": "nvba",
        "guess": True,
        "peak_type": peak_type,
        "n_guess": n_guess,
        "dip": dip,
    }
    return fitter.fit_xy(xdata, ydata, params)
