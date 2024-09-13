#!/usr/bin/env python3

"""
File I/O for Pulse ODMR with Slow detectors.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from os import path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from ..msgs.spodmr_msgs import SPODMRData, update_data
from ..meas.spodmr_worker import SPODMRDataOperator
from ..meas.podmr_fitter import PODMRFitter
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5
from ..util.unit import SI_scale
from ..util.conv import real_fft, real_fftfreq
from ..util.plot import colors_tab20_pair


class SPODMRIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(
        self, filename: str, data: SPODMRData, params: dict | None = None, note: str = ""
    ) -> bool:
        """Save data to filename. return True on success."""

        data.set_saved()
        return save_pickle_or_h5(filename, data, SPODMRData, self.logger, note=note)

    def load_data(self, filename: str) -> SPODMRData | None:
        """Load data from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, SPODMRData, self.logger)
        if d is not None:
            return update_data(d)

    def update_plot_params(self, plot_params: dict, data: SPODMRData):
        op = SPODMRDataOperator()
        pparams = data.params["plot"].copy()
        pparams.update(plot_params)
        op.update_plot_params(data, pparams)

    def refit_data(self, params: dict, data: SPODMRData) -> bool:
        fitter = PODMRFitter(self.logger)
        success = bool(fitter.fitd(data, params["fit"], params["fit_label"]))
        if not success:
            self.logger.error("Failed to fit.")
        return success

    def export_data(
        self, filename: str, data: SPODMRData | list[SPODMRData], params: dict | None = None
    ) -> bool:
        """Export the data to text or image files.

        :param filename: supported extensions: text: .txt and .csv. image: .png, .pdf, and .eps.
        :param data: single data or list of data
        :param params.plot: plot parameter. if given, reanalyze data.
        :type params.plot: dict|None
        :param params.fit: fit parameter. if given, refit data.
        :type params.fit: dict|None
        :param params.fft: perform FFT on data.
        :type params.fft: bool
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
        :param params.dpi: matplotlib dpi
        :type params.dpi: float
        :param params.label: matplotlib labels
        :type params.label: list[str]
        :param params.color_fit: matplotlib colors for fit data
        :type params.color_fit: list[str]
        :param params.color0: matplotlib colors for data0
        :type params.color0: list[str]
        :param params.color1: matplotlib colors for data1
        :type params.color1: list[str]
        :param params.linewidth: matplotlib linewidths for data
        :type params.linewidth: list[float]
        :param params.linewidth_fit: matplotlib linewidths for fit data
        :type params.linewidth_fit: list[float]
        :param params.marker0: matplotlib markers for data0
        :type params.marker0: list[str]
        :param params.marker1: matplotlib markers for data1
        :type params.marker1: list[str]
        :param params.show_fit: set True to plot fit data
        :type params.show_fit: bool
        :param params.show_std: set True to plot estimated std. dev.
        :type params.show_std: bool
        :param params.legend: legend location (best|upper left|...). None to disable legend.
        :type params.legend: str|None

        """

        if params is None:
            params = {}

        if isinstance(data, (list, tuple)) and isinstance(data[0], SPODMRData):
            data_list = data
        elif isinstance(data, SPODMRData):
            data_list = [data]
        else:
            self.logger.error(
                f"Given object ({data}) is not a SPODMRData or Sequence of SPODMRData."
            )
            return False

        for d in data_list:
            if params.get("plot"):
                self.update_plot_params(params["plot"], d)
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

    def _export_data_csv(self, fn, data: SPODMRData) -> bool:
        header = "# Pulse ODMR data taken by mahos.meas.spodmr.\n"
        header += f"# {data.xlabel} {data.ylabel}\n"
        header += f"# {data.xunit} {data.yunit}\n"

        x = data.get_xdata()
        y0, y1 = data.get_ydata()

        with open(fn, "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
        with open(fn, "ab") as f:
            if y1 is not None:
                np.savetxt(f, np.column_stack((x, y0, y1)))
            else:
                np.savetxt(f, np.column_stack((x, y0)))

        self.logger.info(f"Exported Data to {fn}.")
        return True

    def _fetch_data(self, data: SPODMRData, do_fft: bool):
        try:
            x = data.get_xdata()
            xfit = data.get_fit_xdata()
        except ValueError:
            self.logger.exception("Falied to get xdata")
            return None
        try:
            y0, y1 = data.get_ydata()
            yfit = data.get_fit_ydata()
            y0std, y1std = data.get_ydata(std=True)
        except ValueError:
            self.logger.exception("Falied to get ydata")
            return None

        if do_fft:
            x, y0 = real_fft(x, y0)
            y1 = None
            if xfit is not None and yfit is not None:
                xfit, yfit = real_fft(xfit, yfit)

        return x, xfit, y0, y1, yfit, y0std, y1std

    def _fft_freq(self, data: SPODMRData):
        try:
            return real_fftfreq(data.get_xdata())
        except ValueError:
            self.logger.exception("Falied to get xdata")
            return None

    def _export_data_image(self, fn, data_list: list[SPODMRData], params: dict) -> bool:
        d0: SPODMRData = data_list[0]
        do_fft = params.get("fft", False)

        if do_fft:
            xs = [self._fft_freq(data) for data in data_list]
            xunit = "Hz"
            xlabel = "Frequency dom. of " + d0.xlabel
            ylabel = "Amplitude"
        else:
            xs = [data.get_xdata() for data in data_list]
            xunit = d0.xunit
            xlabel = d0.xlabel
            ylabel = d0.ylabel
        x_max = params.get("xmax")
        if x_max is None:
            x_max = max([max(x) for x in xs])
        x_min = params.get("xmin")
        if x_min is None:
            x_min = min([min(x) for x in xs])
        if xunit:
            xcoeff, xprefix = SI_scale(x_max)
        else:
            xcoeff, xprefix = 1.0, ""

        figsize = params.get("figsize", (12, 12))
        plt.rcParams["font.size"] = params.get("fontsize", 28)

        fig = plt.figure(figsize=figsize, dpi=params.get("dpi", 100))
        ax = fig.add_subplot(111)

        ax.set_xlim(x_min * xcoeff, x_max * xcoeff)
        if xunit is not None:
            plt.xlabel(f"{xlabel} ({xprefix}{xunit})")
        else:
            plt.xlabel(xlabel)
        plt.xscale(d0.xscale)
        plt.yscale(d0.yscale)
        plt.ylabel(ylabel)

        label = params.get("label") or [f"data{i}" for i in range(len(data_list))]
        offset = params.get("offset") or [0.0] * len(data_list)
        c0, c1 = zip(*colors_tab20_pair())
        color_fit = cycle(params.get("color_fit") or c0)
        color0 = cycle(params.get("color0") or c0)
        color1 = cycle(params.get("color1") or c1)
        marker0 = cycle(params.get("marker0") or ["o", "s", "^", "p", "*", "D", "h", "."])
        marker1 = cycle(params.get("marker1") or ["v", ">", "<", "1", "2", "3", "4"])

        lines = []
        for data, l, ofs, cf, c0, c1, m0, m1 in zip(
            data_list, label, offset, color_fit, color0, color1, marker0, marker1
        ):
            d = self._fetch_data(data, do_fft)
            if d is None:
                return False
            x, xfit, y0, y1, yfit, y0std, y1std = d
            l0 = l + "_0" if y1 is not None else l
            l1 = l + "_1"

            _show_fit = params.get("show_fit", True) and xfit is not None and yfit is not None
            lw = (
                (0.5 if _show_fit else 1.0)
                if params.get("linewidth") is None
                else params.get("linewidth")
            )

            if params.get("show_std", False) and y0std is not None:
                lines.append(
                    ax.errorbar(
                        x * xcoeff,
                        y0 + ofs,
                        yerr=y0std,
                        label=l0,
                        marker=m0,
                        linestyle="-",
                        color=c0,
                        linewidth=lw,
                    )[0]
                )
            else:
                lines.append(
                    ax.plot(
                        x * xcoeff,
                        y0 + ofs,
                        label=l0,
                        marker=m0,
                        linestyle="-",
                        color=c0,
                        linewidth=lw,
                    )[0]
                )
            if y1 is not None:
                if params.get("show_std", False) and y1std is not None:
                    lines.append(
                        ax.errorbar(
                            x * xcoeff,
                            y1 + ofs,
                            yerr=y1std,
                            label=l1,
                            marker=m1,
                            linestyle="-",
                            color=c1,
                            linewidth=lw,
                        )[0]
                    )
                else:
                    lines.append(
                        ax.plot(
                            x * xcoeff,
                            y1 + ofs,
                            label=l1,
                            marker=m1,
                            linestyle="-",
                            color=c1,
                            linewidth=lw,
                        )[0]
                    )

            if params.get("show_fit", True) and xfit is not None and yfit is not None:
                lw_fit = params.get("linewidth_fit", 2.0)
                ax.plot(
                    xfit * xcoeff,
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
