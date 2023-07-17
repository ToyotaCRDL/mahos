#!/usr/bin/env python3

"""
File I/O for Imaging ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import os
from os import path
from concurrent import futures
from itertools import cycle

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from lmfit.model import ModelResult

from ..msgs.iodmr_msgs import IODMRData
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle, load_h5
from ..util.image import save_image
from .iodmr_fitter import IODMRFitResult, IODMRFitter


class IODMRIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger
        self._executor = futures.ThreadPoolExecutor()

    def close(self):
        """Wait for saving thread and shutdown executor.

        Call this on app-closure if save_data_async() is used.

        ThreadPoolExecutor seemingly has auto-join mechanism on Python interpreter's exit.
        However, it could be better to call ThreadPoolExecutor.shutdown() explicitly.

        """

        self._executor.shutdown(wait=True)

    def save_data(self, fn, data: IODMRData, params: dict | None = None, note: str = "") -> bool:
        if params is None:
            params = {}

        if isinstance(fn, str) and (fn.endswith(".pkl") or fn.endswith(".pkl.bz2")):
            default_compression = None
        else:
            default_compression = "lzf"

        return save_pickle_or_h5(
            fn,
            data,
            IODMRData,
            self.logger,
            note=note,
            compression=params.get("compression", default_compression),
            compression_opts=params.get("compression_opts"),
        )

    def save_data_async(
        self, fn, data: IODMRData, params: dict | None = None, note: str = ""
    ) -> bool:
        """save data asynchronously using ThreadPoolExecutor."""

        self.logger.info(f"Saving (async) {fn}.")
        self._executor.submit(self.save_data, fn, data, params=params, note=note)
        return True

    def load_data(self, fn) -> IODMRData | None:
        if isinstance(fn, str) and (fn.endswith(".pkl") or fn.endswith(".pkl.bz2")):
            # try to load twice (bz2 and no compression) for backward compatibility
            # because old files are named *.pkl with bz2 compression.
            d = load_pickle(fn, IODMRData, self.logger, compression="bz2")
            if d is None:
                d = load_pickle(fn, IODMRData, self.logger, compression=None)
            return d
        else:
            return load_h5(fn, IODMRData, self.logger)

    def fit_data(self, data: IODMRData, params: dict) -> IODMRFitResult | None:
        fitter = IODMRFitter(data, self.logger)
        return fitter.fit(params)

    def fit_save_data(self, fn: str, data: IODMRData, params: dict) -> IODMRFitResult | None:
        res = self.fit_data(data, params)
        if res is not None:
            res.save(fn)
        return res

    def load_fit(self, fn) -> IODMRFitResult | None:
        res = IODMRFitResult.load(fn)
        self.logger.info(f"Loaded {fn}.")
        return res

    def export_data(
        self, fn, data: IODMRData | list[IODMRData], params: dict | None = None
    ) -> bool:
        if params is None:
            params = {}

        if isinstance(data, (list, tuple)) and isinstance(data[0], IODMRData):
            data_list = data
        elif isinstance(data, IODMRData):
            data_list = [data]
        else:
            self.logger.error(
                f"Given object ({data}) is not an IODMRData or Sequence of IODMRData."
            )
            return False

        ext = path.splitext(fn)[1]
        if ext in (".png", ".pdf", ".eps"):
            return self._export_data_image(fn, data_list, params) and self._export_freq_slice(
                fn, data_list, params
            )
        else:
            self.logger.error(f"Unknown extension to export data: {fn}")
            return False

    def export_fit(self, fn, fit: IODMRFitResult, params: dict | None = None):
        if params is None:
            params = {}

        head, ext = path.splitext(fn)
        figsize = params.get("figsize", (12, 12))
        dpi = params.get("dpi")
        fontsize = params.get("fontsize")
        cmap = params.get("cmap")

        img = fit.make_image_BIC()
        xfn = f"{head}_BIC{ext}"
        save_image(xfn, img, figsize=figsize, dpi=dpi, fontsize=fontsize, cmap=cmap)

        img = fit.make_image_B()
        xfn = f"{head}_B{ext}"
        if img is not None:
            save_image(
                xfn,
                img,
                figsize=figsize,
                dpi=dpi,
                fontsize=fontsize,
                cmap=cmap,
                vmin=params.get("vmin"),
                vmax=params.get("vmax"),
            )

        img = fit.make_image_freq()
        xfn = f"{head}_freq{ext}"
        if img is not None:
            save_image(xfn, img, figsize=figsize, dpi=dpi, fontsize=fontsize, cmap=cmap)

        if params.get("all", 0):
            self._export_fit_all(head + "_all", fit, params)

    def _export_fit(
        self, result: ModelResult, features: dict, h: int, w: int, dirname: str, params: dict
    ):
        fig = plt.figure(figsize=params.get("figsize", (12, 12)), dpi=params.get("dpi"))
        ax = fig.add_subplot(111)
        result.plot_fit(
            ax=ax,
            datafmt="o-",
            title=f"h{h:04d} w{w:04d}",
            xlabel="Frequency (MHz)",
            ylabel="Normalized Intensity",
        )

        # draw lines to visualize features
        s_init = {"colors": "0.5", "linestyles": "dotted"}
        s_best = {"colors": "0.4", "linestyles": "dashed"}
        f_init = features["init"]
        f_best = features["best"]
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.hlines(f_init["background"], xmin, xmax, **s_init)
        ax.hlines(f_best["background"], xmin, xmax, **s_best)
        for c in f_init["centers"]:
            ax.vlines(c, ymin, ymax, **s_init)
        for c in f_best["centers"]:
            ax.vlines(c, ymin, ymax, **s_best)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.get_legend().remove()
        plt.tight_layout()

        plt.savefig(path.join(dirname, f"h{h:04d}_w{w:04d}.png"))
        plt.clf()
        plt.close(fig)

    def _export_fit_all(self, dirname, fit, params):
        if not path.exists(dirname):
            os.makedirs(dirname)
        self.logger.info(f"Saving fit plot for all pixels in {dirname}")

        # avoid GUI which causes memory leak
        mpl.use("Agg")
        done = 0
        heights = range(0, fit.height(), params["all"])
        widths = range(0, fit.width(), params["all"])
        N = len(heights) * len(widths)
        for h in heights:
            for w in widths:
                self._export_fit(fit.r(h, w), fit.model_features(h, w), h, w, dirname, params)
                done += 1
                self.logger.info(f"[{done}/{N}] saved")

    def _parse_slice(self, s: str, length: int) -> slice:
        default = slice(0, length)
        if not s or s == ":":
            return default
        try:
            return slice(*[int(i) for i in s.split(":")])
        except (TypeError, ValueError):
            self.logger.exception("Error parsing slice. Using default.")
            return default

    def get_freq_space(self, data: IODMRData, params: dict):
        return np.linspace(data.params["start"], data.params["stop"], data.params["num"])

    def get_hwslice(self, data: IODMRData, params: dict):
        d = data.data_sum
        hslice = self._parse_slice(params.get("hslice", ""), d.shape[1])
        wslice = self._parse_slice(params.get("wslice", ""), d.shape[2])
        return hslice, wslice

    def get_xydata(self, data: IODMRData, params: dict):
        """get spectrum in ROI defined by params['wslice'] and params['hslice'].

        x: frequency, y: intensity.

        """

        if params.get("latest", False):
            d = data.data_latest
        else:  # mean
            d = data.data_sum / data.sweeps

        hslice, wslice = self.get_hwslice(data, params)
        ydata = np.mean(d[:, hslice, wslice], axis=(1, 2))
        return self.get_freq_space(data, params), ydata

    def get_freq_image(self, data: IODMRData, params: dict) -> tuple[float, NDArray]:
        """get an image at f = params['freq']."""

        freq_data = self.get_freq_space(data, params)

        if params.get("latest", False):
            d = data.data_latest
        else:  # mean
            d = data.data_sum / data.sweeps

        if not params.get("freq"):
            idx = 0
        else:
            idx = np.abs(freq_data - params["freq"]).argmin()
        return freq_data[idx], d[idx, :, :]

    def _export_freq_slice(self, fn, data_list: list[IODMRData], params: dict):
        for i, data in enumerate(data_list):
            freq, img = self.get_freq_image(data, params)
            hslice, wslice = self.get_hwslice(data, params)
            rect = Rectangle(
                (wslice.start, hslice.start),
                wslice.stop - wslice.start,
                hslice.stop - hslice.start,
                facecolor="none",
                edgecolor="#1E90FF",
            )
            head, ext = path.splitext(fn)
            freq_str = f"{freq*1E-6:.1f}MHz"
            xfn = f"{head}_d{i}f{freq_str}{ext}"
            save_image(
                xfn,
                img,
                patches=[rect],
                figsize=params.get("figsize", (12, 12)),
                dpi=params.get("dpi"),
                fontsize=params.get("fontsize"),
                title="f = " + freq_str,
            )

    def _export_data_image(self, fn, data_list: list[IODMRData], params: dict) -> bool:
        coeff = 1e-6
        xunit = "MHz"

        xs = [(data.params["start"], data.params["stop"]) for data in data_list]
        x_max = params.get("xmax") or max([max(x) for x in xs])
        x_min = params.get("xmin") or min([min(x) for x in xs])

        plt.rcParams["font.size"] = params.get("fontsize", 28)
        figsize = params.get("figsize", (12, 12))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_xlim(x_min * coeff, x_max * coeff)
        plt.xlabel("Frequency ({})".format(xunit))
        plt.ylabel("Intensity")

        label = params.get("label") or [f"data{i}" for i in range(len(data_list))]
        offset = params.get("offset") or [0.0] * len(data_list)
        color = cycle(params.get("color") or ["C0", "C1", "C2", "C3", "C4"])
        marker = cycle(params.get("marker") or ["o", "s", "^", "p", "*", "D", "h", "."])

        lines = []
        for data, l, ofs, col, mk in zip(data_list, label, offset, color, marker):
            x, y = self.get_xydata(data, params)
            lw = 1.0 if params.get("linewidth") is None else params.get("linewidth")
            lines.append(
                ax.plot(
                    x * coeff, y + ofs, label=l, marker=mk, linestyle="-", color=col, linewidth=lw
                )[0]
            )

        # set_ylim should be called after all plots to auto-scale properly
        ax.set_ylim(params.get("ymin"), params.get("ymax"))
        if params.get("legend") is not None:
            ax.legend(handles=lines, loc=params["legend"])
        plt.tight_layout()

        plt.savefig(fn, dpi=params.get("dpi"))
        plt.close()

        self.logger.info(f"Exported Data to {fn}.")
        return True
