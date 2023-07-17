#!/usr/bin/env python3

"""
File I/O for Spectroscopy.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
from os import path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from ..meas.spectroscopy_fitter import SpectroscopyFitter
from ..msgs.spectroscopy_msgs import SpectroscopyData, update_data
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5


class SpectroscopyIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(self, filename: str, data: SpectroscopyData, note: str = "") -> bool:
        """Save data to filename. return True on success."""

        data.set_saved()
        return save_pickle_or_h5(filename, data, SpectroscopyData, self.logger, note=note)

    def load_data(self, filename: str) -> SpectroscopyData | None:
        """Load data from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, SpectroscopyData, self.logger)
        if d is not None:
            return update_data(d)

    def refit_data(self, params: dict, data: SpectroscopyData) -> bool:
        if "method" not in params["fit"]:
            self.logger.error("Undefined: params['fit']['method']")
            return False

        fitter = SpectroscopyFitter(self.logger)
        success = bool(fitter.fitd(data, params["fit"]))
        if not success:
            self.logger.error("Failed to fit.")
        return success

    def export_data(
        self,
        filename: str,
        data: SpectroscopyData | list[SpectroscopyData],
        params: dict | None = None,
    ) -> bool:
        """Export the data to text or image files.

        :param filename: supported extensions: text: .txt and .csv. image: .png, .pdf, and .eps.
        :param data: single data or list of data
        :param params.last_n: use last n data. (0 to use all data. this is default.)
        :type params.last_n: int
        :param params.filter_n: outlier filter's order. data point exhibits value over n * sigma is
                                considered outlier. (0 to disable filter. this is default.)
        :type params.filter_n: float
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

        if isinstance(data, (list, tuple)) and isinstance(data[0], SpectroscopyData):
            data_list = data
        elif isinstance(data, SpectroscopyData):
            data_list = [data]
        else:
            self.logger.error(f"Given object ({data}) is not a SpectroscopyData.")
            return False

        for d in data_list:
            if params.get("fit") and not self.refit_data(params, d):
                return False

        ext = path.splitext(filename)[1]
        if ext in (".txt", ".csv"):
            # TODO: accept data_list ?
            return self._export_data_csv(filename, data, params)
        elif ext in (".png", ".pdf", ".eps"):
            return self._export_data_image(filename, data_list, params)
        else:
            self.logger.error(f"Unknown extension to export data: {filename}")
            return False

    def _export_data_csv(self, fn, data: SpectroscopyData, params: dict) -> bool:
        xdata = data.get_xdata()
        ydata = data.get_ydata(filter_n=params.get("filter_n", 0))
        with open(fn, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Spectroscopy data taken by mahos.meas.spectroscopy.\n")
            f.write("# Wavelength Intensity\n")
            f.write("# nm counts\n")
        with open(fn, "a") as f:
            np.savetxt(f, np.column_stack((xdata, ydata)), delimiter=",")
        self.logger.info(f"Exported Data to {fn}.")
        return True

    def _export_data_image(self, fn, data_list: list[SpectroscopyData], params: dict) -> bool:
        xs = [data.xdata for data in data_list]
        x_max = params.get("xmax") or max([max(x) for x in xs])
        x_min = params.get("xmin") or min([min(x) for x in xs])

        plt.rcParams["font.size"] = params.get("fontsize", 28)

        label = params.get("label") or [f"data{i}" for i in range(len(data_list))]
        offset = params.get("offset") or [0.0] * len(data_list)
        color_fit = cycle(params.get("color_fit") or [f"C{i}" for i in range(10)])
        color = cycle(params.get("color") or [f"C{i}" for i in range(10)])
        marker = cycle(params.get("marker") or ["o", "s", "^", "p", "*", "D", "h", "."])

        fig = plt.figure(figsize=params.get("figsize", (12, 12)))
        ax = fig.add_subplot(111)

        last_n = params.get("last_n", 0)
        filter_n = params.get("filter_n", 0)
        lines = []
        for data, l, ofs, cf, col, mk in zip(data_list, label, offset, color_fit, color, marker):
            if filter_n:
                self.logger.info(f"Plotting {l}")
                n = data.n_outliers(last_n=last_n, filter_n=filter_n)
                if n:
                    self.logger.info(f"{n} outliers removed")

            x = data.get_xdata()
            y = data.get_ydata(last_n=last_n, filter_n=filter_n)
            xfit = data.get_fit_xdata()
            yfit = data.get_fit_ydata()
            if params.get("show_fit", True) and xfit is not None and yfit is not None:
                lw_fit = params.get("linewidth_fit", 1.0)
                lw = 0.0 if params.get("linewidth") is None else params.get("linewidth")
                ax.plot(
                    xfit, yfit + ofs, label=l + "_fit", linestyle="-", color=cf, linewidth=lw_fit
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

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (counts)")

        plt.savefig(fn)
        plt.close()

        self.logger.info(f"Exported Data to {fn}.")
        return True
