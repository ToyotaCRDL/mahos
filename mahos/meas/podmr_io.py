#!/usr/bin/env python3

"""
File I/O for Pulse ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
from os import path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from ..msgs.podmr_msgs import PODMRData, update_data
from ..meas.podmr_worker import PODMRDataOperator
from ..meas.podmr_fitter import PODMRFitter
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5
from ..util.unit import SI_scale
from ..util.conv import real_fft, real_fftfreq
from ..util.plot import colors_tab20_pair


class PODMRIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(
        self, filename: str, data: PODMRData, params: dict | None = None, note: str = ""
    ) -> bool:
        """Save data to filename. return True on success."""

        if params is not None and "tmp" in params and "tmp":
            self.logger.info("Temporary save of PODMR data")
        else:
            data.set_saved()

        return save_pickle_or_h5(filename, data, PODMRData, self.logger, note=note)

    def load_data(self, filename: str) -> PODMRData | None:
        """Load data from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, PODMRData, self.logger)
        if d is not None:
            return update_data(d)

    def reanalyze_data(self, plot_params: dict, data: PODMRData):
        op = PODMRDataOperator()
        pparams = data.params["plot"].copy()
        pparams.update(plot_params)
        op.update_plot_params(data, pparams)
        op.get_marker_indices(data)
        op.analyze(data)

    def refit_data(self, params: dict, data: PODMRData) -> bool:
        if "method" not in params["fit"]:
            self.logger.error("Undefined: params['fit']['method']")
            return False

        fitter = PODMRFitter(self.logger)
        success = bool(fitter.fitd(data, params["fit"]))
        if not success:
            self.logger.error("Failed to fit.")
        return success

    def export_data(
        self, fn, data: PODMRData | list[PODMRData], params: dict | None = None
    ) -> bool:
        """Export the data to text or image files.

        :param filename: supported extensions: text: .txt and .csv. image: .png, .pdf, and .eps.
        :param data: single data or list of data
        :param params.plot: plot parameter. if given, reanalyze data.
        :type params.plot: dict|None
        :param params.fit: fit parameter. if given, refit data.
        :type params.fit: dict|None
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
        :param params.legend: legend location (best|upper left|...). None to disable legend.
        :type params.legend: str|None

        """

        if params is None:
            params = {}

        if isinstance(data, (list, tuple)) and isinstance(data[0], PODMRData):
            data_list = data
        elif isinstance(data, PODMRData):
            data_list = [data]
        else:
            self.logger.error(
                f"Given object ({data}) is not a PODMRData or Sequence of PODMRData."
            )
            return False

        for d in data_list:
            if params.get("plot"):
                self.reanalyze_data(params["plot"], d)
            if params.get("fit") and not self.refit_data(params, d):
                return False

        ext = path.splitext(fn)[1]
        if ext in (".txt", ".csv"):
            # TODO: accept data_list ?
            return self._export_data_csv(fn, data)
        elif ext in (".png", ".pdf", ".eps"):
            return self._export_data_image(fn, data_list, params)
        else:
            self.logger.error(f"Unknown extension to export data: {fn}")
            return False

    def _export_data_csv(self, fn, data: PODMRData) -> bool:
        header = "# Pulse ODMR data taken by mahos.meas.podmr.\n"
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

    def _fetch_data(self, data: PODMRData, do_fft: bool):
        try:
            x = data.get_xdata()
            xfit = data.get_fit_xdata()
        except ValueError:
            self.logger.exception("Falied to get xdata")
            return None
        try:
            y0, y1 = data.get_ydata()
            yfit = data.get_fit_ydata()
        except ValueError:
            self.logger.exception("Falied to get ydata")
            return None

        if do_fft:
            x, y0 = real_fft(x, y0)
            y1 = None
            if xfit is not None and yfit is not None:
                xfit, yfit = real_fft(xfit, yfit)

        return x, xfit, y0, y1, yfit

    def _fft_freq(self, data: PODMRData):
        try:
            return real_fftfreq(data.get_xdata())
        except ValueError:
            self.logger.exception("Falied to get xdata")
            return None

    def _export_data_image(self, fn, data_list: list[PODMRData], params: dict) -> bool:
        d0: PODMRData = data_list[0]
        do_fft = d0.params["plot"]["fft"]

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
        x_max = params.get("xmax") or max([max(x) for x in xs])
        x_min = params.get("xmin") or min([min(x) for x in xs])
        if xunit:
            xcoeff, xprefix = SI_scale(x_max)
        else:
            xcoeff, xprefix = 1.0, ""

        figsize = params.get("figsize", (12, 12))
        plt.rcParams["font.size"] = params.get("fontsize", 28)

        fig = plt.figure(figsize=figsize)
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
            x, xfit, y0, y1, yfit = d
            l0 = l + "_0" if y1 is not None else l
            l1 = l + "_1"
            if params.get("show_fit", True) and xfit is not None and yfit is not None:
                lw_fit = params.get("linewidth_fit", 1.0)
                lw = 0.0 if params.get("linewidth") is None else params.get("linewidth")
                ax.plot(
                    xfit * xcoeff,
                    yfit + ofs,
                    label=l + "_fit",
                    linestyle="-",
                    color=cf,
                    linewidth=lw_fit,
                )
                lines.append(
                    ax.plot(x * xcoeff, y0 + ofs, label=l0, marker=m0, color=c0, linewidth=lw)[0]
                )
                if y1 is not None:
                    lines.append(
                        ax.plot(x * xcoeff, y1 + ofs, label=l1, marker=m1, color=c1, linewidth=lw)[
                            0
                        ]
                    )
            else:
                lw = 1.0 if params.get("linewidth") is None else params.get("linewidth")
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

        # set_ylim should be called after all plots to auto-scale properly
        ax.set_ylim(params.get("ymin"), params.get("ymax"))
        if params.get("legend") is not None:
            ax.legend(handles=lines, loc=params["legend"])
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        self.logger.info(f"Exported Data to {fn}.")
        return True
