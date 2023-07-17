#!/usr/bin/env python3

"""
File I/O for HBT Interferometer.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
from os import path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from ..meas.hbt_fitter import HBTFitter
from ..msgs.hbt_msgs import HBTData, update_data
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5


class HBTIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(self, filename: str, data: HBTData, note: str = "") -> bool:
        """Save data to filename. return True on success."""

        data.set_saved()
        return save_pickle_or_h5(filename, data, HBTData, self.logger, note=note)

    def load_data(self, filename: str) -> HBTData | None:
        """Load data from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, HBTData, self.logger)
        if d is not None:
            return update_data(d)

    def refit_data(self, params: dict, data: HBTData) -> bool:
        if "method" not in params["fit"]:
            self.logger.error("Undefined: params['fit']['method']")
            return False

        fitter = HBTFitter(self.logger)
        success = bool(fitter.fitd(data, params["fit"]))
        if not success:
            self.logger.error("Failed to fit.")
        return success

    def export_data(
        self, filename: str, data: HBTData | list[HBTData], params: dict | None = None
    ) -> bool:
        """Export the data to text or image files.

        :param filename: supported extensions: text: .txt and .csv. image: .png, .pdf, and .eps.
        :param data: single data or list of data
        :param params.normalize: set True to normalize data.
        :type params.normalize: bool
        :param params.offset: offset along y-axis
        :type params.offset: list[float]
        :param params.xmin: lower limit of x-axis
        :type params.xmin: float
        :param params.xmax: upper limit of x-axis
        :type params.xmax: float
        :param params.ymin: lower limit of y-axis
        :type params.ymin: float
        :param params.ymax: upper limit of y-axis
        :type params.ymax: float
        :param params.figsize: matplotlib figure size (w, h)
        :type params.figsize: tuple[float, float]
        :param params.fontsize: matplotlib fontsize
        :type params.fontsize: float
        :param params.label: matplotlib labels
        :type params.label: list[str]
        :param params.color_fit: matplotlib colors for fit data
        :type params.color_fit: list[str]
        :param params.color: matplotlib colors for data
        :type params.color: list[str]
        :param params.linewidth: matplotlib linewidths for data
        :type params.linewidth: list[float]
        :param params.linewidth_fit: matplotlib linewidths for fit data
        :type params.linewidth_fit: list[float]
        :param params.marker: matplotlib markers for data
        :type params.marker: list[str]
        :param params.show_fit: set True to plot fit data
        :type params.show_fit: bool
        :param params.legend: legend location (best|upper left|...). None to disable legend.
        :type params.legend: str|None

        """

        if params is None:
            params = {}

        if isinstance(data, (list, tuple)) and isinstance(data[0], HBTData):
            data_list = data
        elif isinstance(data, HBTData):
            data_list = [data]
        else:
            self.logger.error(f"Given object ({data}) is not an HBTData.")
            return False

        for d in data_list:
            if params.get("plot"):
                d.params["plot"].update(params["plot"])
            if params.get("fit") and not self.refit_data(params, d):
                return False

        ext = path.splitext(filename)[1]
        if ext in (".txt", ".csv"):
            return self._export_data_csv(filename, data)
        elif ext in (".png", ".pdf", ".eps"):
            return self._export_data_image(filename, data_list, params)
        else:
            self.logger.error(f"Unknown extension to export data: {filename}")
            return False

        return data.data[:, 0], data.data[:, 1]

    def _export_data_csv(self, fn, data: HBTData):
        xdata = data.get_xdata()
        ydata = data.get_ydata()
        with open(fn, "w", encoding="utf-8", newline="\n") as fo:
            fo.write("# HBT data taken by mahos.meas.hbt.\n")
            fo.write("# Time Events\n")
            fo.write("# ns counts\n")
        with open(fn, "ab") as fo:
            np.savetxt(fo, np.column_stack((xdata * 1e9, ydata)), delimiter=",")
        self.logger.info(f"Exported Data to {fn}.")
        return True

    def _export_data_image(self, fn, data_list: list[HBTData], params: dict):
        coeff = 1e9

        normalize = params.get("normalize", True)

        xs = [data.get_xdata(normalize=normalize) for data in data_list]
        x_max = params.get("xmax") or max([max(x) for x in xs]) * coeff
        x_min = params.get("xmin") or min([min(x) for x in xs]) * coeff

        plt.rcParams["font.size"] = params.get("fontsize", 28)

        label = params.get("label") or [f"data{i}" for i in range(len(data_list))]
        offset = params.get("offset") or [0.0] * len(data_list)
        color_fit = cycle(params.get("color_fit") or [f"C{i}" for i in range(10)])
        color = cycle(params.get("color") or [f"C{i}" for i in range(10)])
        marker = cycle(params.get("marker") or ["o", "s", "^", "p", "*", "D", "h", "."])

        fig = plt.figure(figsize=params.get("figsize", (12, 12)))
        ax = fig.add_subplot(111)

        lines = []
        if normalize:
            ax.hlines(0.5, x_min, x_max, color="#808080")
        for data, l, ofs, cf, col, mk in zip(data_list, label, offset, color_fit, color, marker):
            x = data.get_xdata(normalize=normalize) * coeff
            y = data.get_ydata(normalize=normalize)
            xfit = data.get_fit_xdata(normalize=normalize)
            yfit = data.get_fit_ydata(normalize=normalize)
            if params.get("show_fit", True) and xfit is not None and yfit is not None:
                lw_fit = params.get("linewidth_fit", 1.0)
                lw = 0.0 if params.get("linewidth") is None else params.get("linewidth")
                ax.plot(
                    xfit * coeff,
                    yfit + ofs,
                    label=l + "_fit",
                    linestyle="-",
                    color=cf,
                    linewidth=lw_fit,
                )
                lines.append(ax.plot(x, y + ofs, label=l, marker=mk, color=col, linewidth=lw)[0])
            else:
                lw = 1.0 if params.get("linewidth") is None else params.get("linewidth")
                lines.append(ax.plot(x, y + ofs, label=l, color=col, linewidth=lw)[0])

        ax.set_xlim(x_min, x_max)
        # set_ylim should be called after all plots to auto-scale properly
        ax.set_ylim(params.get("ymin"), params.get("ymax"))

        if params.get("legend") is not None:
            ax.legend(handles=lines, loc=params["legend"])

        plt.xlabel(r"Time, $\tau$ (ns)")
        if normalize:
            plt.ylabel(r"Second-order Correlation, $g^{(2)} (\tau)$")
        else:
            plt.ylabel("Events")

        plt.savefig(fn)
        plt.close()

        self.logger.info(f"Exported Data to {fn}.")
        return True
