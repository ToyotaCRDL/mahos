#!/usr/bin/env python3

"""
File I/O for ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from os import path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from ..meas.odmr_fitter import ODMRFitter
from ..msgs.odmr_msgs import ODMRData, update_data
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5
from ..util.plot import colors_tab20_pair


class ODMRIO(object):
    """IO class for ODMR."""

    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(self, filename: str, data: ODMRData, note: str = "") -> bool:
        """Save data to filename. return True on success."""

        data.set_saved()
        return save_pickle_or_h5(filename, data, ODMRData, self.logger, note=note)

    def load_data(self, filename: str) -> ODMRData | None:
        """Load data from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, ODMRData, self.logger)
        if d is not None:
            return update_data(d)

    def refit_data(self, params: dict, data: ODMRData) -> bool:
        if "complex_conv" not in params["fit"] and "complex_conv" in params["fit"]:
            params["fit"]["complex_conv"] = params["complex_conv"]

        fitter = ODMRFitter(self.logger)
        success = bool(fitter.fitd(data, params["fit"], params["fit_label"]))
        if not success:
            self.logger.error("Failed to fit.")
        return success

    def export_data(
        self, filename: str, data: ODMRData | list[ODMRData], params: dict | None = None
    ) -> bool:
        """Export the data to text or image files.

        :param filename: supported extensions: text: .txt and .csv. image: .png, .pdf, and .eps.
        :param data: single data or list of data
        :param params.normalize_n: number of points for normalization. (0 for no-normalization)
        :type params.normalize_n: int
        :param params.complex_conv: (real | imag | abs[olute] | angle)
            conversion method for complex data.
        :type params.complex_conv: str
        :param params.offset: offset along y-axis
        :type params.offset: list[float]
        :param params.base_line: set True to draw base lines
        :type params.base_line: bool
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
        :param params.dpi: matplotlib dpi
        :type params.dpi: float
        :param params.label: matplotlib labels
        :type params.label: list[str]
        :param params.color_fit: matplotlib colors for fit data
        :type params.color_fit: list[str]
        :param params.color: matplotlib colors for data
        :type params.color: list[str]
        :param params.color_bg: matplotlib colors for background data
        :type params.color_bg: list[str]
        :param params.linewidth: matplotlib linewidths for data
        :type params.linewidth: list[float]
        :param params.linewidth_fit: matplotlib linewidths for fit data
        :type params.linewidth_fit: list[float]
        :param params.marker: matplotlib markers for data
        :type params.marker: list[str]
        :param params.marker_bg: matplotlib markers for background data
        :type params.marker_bg: list[str]
        :param params.show_fit: set True to plot fit data
        :type params.show_fit: bool
        :param params.show_std: set True to plot estimated std. dev.
        :type params.show_std: bool
        :param params.legend: legend location (best|upper left|...). None to disable legend.
        :type params.legend: str|None

        """

        if params is None:
            params = {}

        if isinstance(data, (list, tuple)) and isinstance(data[0], ODMRData):
            data_list = data
        elif isinstance(data, ODMRData):
            data_list = [data]
        else:
            self.logger.error(f"Given object ({data}) is not an ODMRData or Sequence of ODMRData.")
            return False

        for d in data_list:
            if params.get("fit_label") and not self.refit_data(params, d):
                return False

        ext = path.splitext(filename)[1]
        if ext in (".txt", ".csv"):
            # TODO: accept data_list ?
            return self._export_data_csv(filename, data)
        elif ext in (".png", ".pdf", ".eps"):
            return self._export_data_image(filename, data_list, params)
        else:
            self.logger.error(f"Unknown extension to export data: {filename}")
            return False

    def _export_data_csv(self, fn, data: ODMRData) -> bool:
        xdata = data.get_xdata()
        ydata = data.get_ydata()

        with open(fn, "w", encoding="utf-8", newline="\n") as fo:
            fo.write("# ODMR data taken by mahos.meas.odmr.\n")
            fo.write(f"# {data.xlabel} {data.ylabel}\n")
            fo.write(f"# {data.xunit} {data.yunit}\n")
        with open(fn, "ab") as fo:
            np.savetxt(fo, np.column_stack((xdata, ydata)), delimiter=",")
        self.logger.info(f"Exported Data to {fn}.")
        return True

    def _export_data_image(self, fn, data_list: list[ODMRData], params: dict) -> bool:
        coeff = 1e-6
        unit = "MHz"

        xs = [(data.params["start"], data.params["stop"]) for data in data_list]
        x_max = params.get("xmax")
        if x_max is None:
            x_max = max([max(x) for x in xs])
        x_min = params.get("xmin")
        if x_min is None:
            x_min = min([min(x) for x in xs])

        figsize = params.get("figsize", (12, 12))
        plt.rcParams["font.size"] = params.get("fontsize", 28)
        normalize_n = params.get("normalize_n", 0)

        fig = plt.figure(figsize=figsize, dpi=params.get("dpi", 100))
        ax = fig.add_subplot(111)

        ax.set_xlim(x_min * coeff, x_max * coeff)
        plt.xlabel(f"Frequency ({unit})")
        if data_list[0].is_complex():
            suffix = " [" + params.get("complex_conv", "real") + "]"
        else:
            suffix = ""
        if normalize_n:
            plt.ylabel("Normalized Intensity" + suffix)
        else:
            plt.ylabel(f"Intensity ({data_list[0].yunit})" + suffix)

        label = params.get("label") or [f"data{i}" for i in range(len(data_list))]
        offset = params.get("offset") or [0.0] * len(data_list)
        c0, c1 = zip(*colors_tab20_pair())
        color_fit = cycle(params.get("color_fit") or c0)
        color = cycle(params.get("color") or c0)
        color_bg = cycle(params.get("color_bg") or c1)
        marker = cycle(params.get("marker") or ["o", "s", "^", "p", "*", "D", "h", "."])
        marker_bg = cycle(params.get("marker_bg") or ["v", ">", "<", "1", "2", "3", "4"])

        lines = []
        for data, l, ofs, cf, col, cb, mk, mb in zip(
            data_list, label, offset, color_fit, color, color_bg, marker, marker_bg
        ):
            x = data.get_xdata()
            y, y_bg = data.get_ydata(
                normalize_n=normalize_n, complex_conv=params.get("complex_conv", "real")
            )
            xfit = data.get_fit_xdata()
            yfit = data.get_fit_ydata(normalize_n=normalize_n)
            ystd, ybg_std = data.get_ydata(
                normalize_n=normalize_n, complex_conv=params.get("complex_conv", "real"), std=True
            )
            if normalize_n and params.get("base_line", False):
                ax.hlines(1.0 + ofs, x_min * coeff, x_max * coeff, "#A0A0A0", "dashed")

            _show_fit = params.get("show_fit", True) and xfit is not None and yfit is not None
            lw = (
                (0.5 if _show_fit else 1.0)
                if params.get("linewidth") is None
                else params.get("linewidth")
            )
            if params.get("show_std", False) and ystd is not None:
                lines.append(
                    ax.errorbar(
                        x * coeff,
                        y + ofs,
                        yerr=ystd,
                        label=l,
                        marker=mk,
                        linestyle="-",
                        color=col,
                        linewidth=lw,
                    )[0]
                )
            else:
                lines.append(
                    ax.plot(
                        x * coeff,
                        y + ofs,
                        label=l,
                        marker=mk,
                        linestyle="-",
                        color=col,
                        linewidth=lw,
                    )[0]
                )
            if y_bg is not None:
                if params.get("show_std", False) and ystd is not None:
                    lines.append(
                        ax.errorbar(
                            x * coeff,
                            y_bg + ofs,
                            yerr=ybg_std,
                            label=l,
                            marker=mb,
                            linestyle="-",
                            color=cb,
                            linewidth=lw,
                        )[0]
                    )
                else:
                    lines.append(
                        ax.plot(
                            x * coeff,
                            y_bg + ofs,
                            label=l,
                            marker=mb,
                            linestyle="-",
                            color=cb,
                            linewidth=lw,
                        )[0]
                    )

            if _show_fit:
                lw_fit = params.get("linewidth_fit", 2.0)
                ax.plot(
                    xfit * coeff,
                    yfit + ofs,
                    label=l + "_fit",
                    linestyle="-",
                    color=cf,
                    linewidth=lw_fit,
                )

        # set_ylim should be called after all plots to auto-scale properly
        ax.set_ylim(params.get("ymin"), params.get("ymax"))
        if params.get("legend") is not None:
            ax.legend(handles=lines, loc=params["legend"])
        plt.tight_layout()

        plt.savefig(fn)
        plt.close()

        self.logger.info(f"Exported Data to {fn}.")
        return True
