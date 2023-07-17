#!/usr/bin/env python3

"""
Fitter for Imaging ODMR.

.. This file is a part of MAHOS project.

"""

import typing as T
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from functools import partial

try:
    import cv2
except ImportError:
    print("cv2 couldn't be imported. (in mahos.meas.iodmr_fitter)")

import numpy as np
from numpy.typing import NDArray
import lmfit as F
from lmfit.model import ModelResult

from . import odmr_fitter as OF

from ..node.log import DummyLogger
from ..msgs.iodmr_msgs import IODMRData
from ..util.timer import StopWatch
from ..util.nv import Dgs_MHz, gamma_MHz_mT


def load_modelresult(s: str):
    """Load ModelResult from dumped string."""

    res = ModelResult(F.Model(lambda x: x, None), F.Parameters())
    return res.loads(s)


# Wrap the fit functions to return dumped string because ModelResult is not picklable.
def _fit_single_dumps(xdata, ydata, peak_type, n_guess, dip, silent) -> str:
    res = OF.fit_single(xdata, ydata, peak_type=peak_type, n_guess=n_guess, dip=dip, silent=silent)
    return res.dumps()


def _fit_double_dumps(xdata, ydata, peak_type, n_guess, dip, silent) -> str:
    res = OF.fit_double(xdata, ydata, peak_type=peak_type, n_guess=n_guess, dip=dip, silent=silent)
    return res.dumps()


def _fit_quad_dumps(xdata, ydata, peak_type, n_guess, dip, silent) -> str:
    res = OF.fit_quad(xdata, ydata, peak_type=peak_type, n_guess=n_guess, dip=dip, silent=silent)
    return res.dumps()


def _fit_NVB_aligned_dumps(xdata, ydata, peak_type, n_guess, dip, silent) -> str:
    res = OF.fit_NVB_aligned(
        xdata, ydata, peak_type=peak_type, n_guess=n_guess, dip=dip, silent=silent
    )
    return res.dumps()


def _fit(
    fit_func,
    fit_func_dumps,
    xdata,
    image,
    peak_type: OF.PeakType = OF.PeakType.Gaussian,
    n_guess: int = 20,
    dip: bool = True,
    n_workers: int = 1,
    print_fn=print,
) -> T.List[ModelResult]:
    f, H, W = image.shape
    img = image.reshape((f, H * W), order="C")
    num = img.shape[1]

    if n_workers == 1:
        results = []
        done = 0
        for i in range(num):
            results.append(
                fit_func(
                    xdata,
                    OF.normalize(img[:, i]),
                    peak_type=peak_type,
                    n_guess=n_guess,
                    dip=dip,
                    silent=True,
                )
            )
            done += 1
            print_fn(f"[{done}/{num}] finished {i}")
        return results
    else:
        results = {}
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(
                    fit_func_dumps, xdata, OF.normalize(img[:, i]), peak_type, n_guess, dip, True
                ): i
                for i in range(num)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                done += 1
                try:
                    res = future.result()
                    results[i] = res
                    print_fn(f"[{done}/{num}] finished {i}")
                except Exception as e:
                    results[i] = None
                    print_fn(e)

        return [load_modelresult(results[i]) for i in range(num)]


fit_NVB_aligned = partial(_fit, OF.fit_NVB_aligned, _fit_NVB_aligned_dumps)
fit_single = partial(_fit, OF.fit_single, _fit_single_dumps)
fit_double = partial(_fit, OF.fit_double, _fit_double_dumps)
fit_quad = partial(_fit, OF.fit_quad, _fit_quad_dumps)


class IODMRFitResult(object):
    def __init__(self, params: dict, result: T.List[ModelResult]):
        self.params = params
        self.result = result

    def save(self, fn: str):
        with open(fn, "w") as f:
            f.write(json.dumps(self.params))
            f.write("\n")
            for r in self.result:
                f.write(r.dumps())
                f.write("\n")

    @classmethod
    def load(cls, fn: str):
        with open(fn, "r") as f:
            params = json.loads(f.readline())
            result = []
            for line in f.readlines():
                result.append(load_modelresult(line))
        return cls(params, result)

    def height(self) -> int:
        return self.params.get("resize").get("height") or self.params["size"]["height"]

    def width(self) -> int:
        return self.params.get("resize").get("width") or self.params["size"]["width"]

    def r(self, h, w) -> ModelResult:
        """Get result at height=h, width=w."""

        W = self.width()
        return self.result[W * h + w]

    def model_features(self, h, w):
        """Extract model's features of result at height=h, width=w."""

        res = self.r(h, w)
        d = {}
        for key, vals in zip(("init", "best"), (res.init_values, res.best_values)):
            if self.params["method"] == "nvba":
                centers = OF.peaks_of_B_aligned(vals["B"])
            else:
                centers = [v for k, v in vals.items() if k.endswith("center")]
            d[key] = {
                "background": vals["c"],
                "centers": centers,
            }
        return d

    def make_image(self, func: T.Callable[[ModelResult], float]) -> NDArray:
        image = []
        for h in range(self.height()):
            line = []
            for w in range(self.width()):
                line.append(func(self.r(h, w)))
            image.append(line)
        return np.array(image)

    def make_image_B(self) -> T.Optional[NDArray]:
        """Make image of B field."""

        if self.params["method"] == "nvba":
            return self.make_image(lambda r: r.best_values["B"])
        elif self.params["method"] == "single":
            f = self.make_image(lambda r: r.best_values["center"])
            return np.abs(f - Dgs_MHz) / gamma_MHz_mT
        elif self.params["method"] == "double":
            fh = self.make_image(lambda r: r.best_values["p1_center"])
            fl = self.make_image(lambda r: r.best_values["p0_center"])
            return (fh - fl) / (2 * gamma_MHz_mT)
        elif self.params["method"] == "quad":
            fh = self.make_image(lambda r: r.best_values["p3_center"])
            fl = self.make_image(lambda r: r.best_values["p0_center"])
            return (fh - fl) / (2 * gamma_MHz_mT)
        else:
            return None

    def make_image_freq(self) -> T.Optional[NDArray]:
        if self.params["method"] == "single":
            return self.make_image(lambda r: r.best_values["center"])
        else:
            return None

    def make_image_BIC(self) -> NDArray:
        """Make image of Bayesian Information Criteria."""

        return self.make_image(lambda r: r.bic)


class IODMRFitter(object):
    # scale xdata in MHz
    XCOEFF = 1e-6
    XUNIT = "MHz"

    def __init__(self, data: IODMRData, logger=None):
        self.data = data
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def fit(self, params: dict) -> T.Optional[IODMRFitResult]:
        """Perform fitting."""

        _, h, w = self.data.data_sum.shape
        # save original image size
        params["size"] = {"width": w, "height": h}

        pk = params.get("peak", "g").lower()
        if pk.startswith("g"):
            peak_type = OF.PeakType.Gaussian
        elif pk.startswith("l"):
            peak_type = OF.PeakType.Lorentzian
        elif pk.startswith("v"):
            peak_type = OF.PeakType.Voigt
        else:
            self.logger.exception(f"Unknown peak function type {pk}.")
            return None

        timer = StopWatch()
        try:
            xdata, image = self.make_data(params)
            self.logger.info("make data in " + timer.elapsed_str())
            n_guess = params.get("n_guess") or 20
            n_workers = params.get("n_workers") or 1

            m = params["method"]
            if m == "single":
                res = fit_single(
                    xdata,
                    image,
                    peak_type=peak_type,
                    n_guess=n_guess,
                    n_workers=n_workers,
                    print_fn=self.logger.info,
                )
            elif m == "double":
                res = fit_double(
                    xdata,
                    image,
                    peak_type=peak_type,
                    n_guess=n_guess,
                    n_workers=n_workers,
                    print_fn=self.logger.info,
                )
            elif m == "quad":
                res = fit_quad(
                    xdata,
                    image,
                    peak_type=peak_type,
                    n_guess=n_guess,
                    n_workers=n_workers,
                    print_fn=self.logger.info,
                )
            elif m == "nvba":
                res = fit_NVB_aligned(
                    xdata,
                    image,
                    peak_type=peak_type,
                    n_guess=n_guess,
                    n_workers=n_workers,
                    print_fn=self.logger.info,
                )
            else:
                raise RuntimeError("Unknown fit method.")
        except Exception:
            self.logger.exception("Failed to fit.")
            return None

        self.logger.info("fit done in " + timer.elapsed_str())

        return IODMRFitResult(params, res)

    def make_data(self, params: dict):
        if self.data.params.get("latest", False):
            d = self.data.data_latest
        else:  # mean
            d = self.data.data_sum / self.data.sweeps

        xdata = self.XCOEFF * np.linspace(
            self.data.params["start"], self.data.params["stop"], self.data.params["num"]
        )

        if "flim" in params:
            fl, fh = params["flim"]
            if fl is None:
                fl = -np.inf
            if fh is None:
                fh = +np.inf
            if fl < fh:
                indices = np.where(np.logical_and(fl <= xdata, xdata <= fh))[0]
            else:
                # this is special case to `remove center and use two intervals`
                indices = np.where(np.logical_or(fl <= xdata, xdata <= fh))[0]
            xdata = xdata[indices]
            d = d[indices]

        if "resize" in params:
            h = params["resize"].get("height", d.shape[1])
            w = params["resize"].get("width", d.shape[2])
            d = np.array([cv2.resize(img, (w, h)) for img in d])
        return xdata, d
