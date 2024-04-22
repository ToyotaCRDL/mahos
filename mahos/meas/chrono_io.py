#!/usr/bin/env python3

"""
File I/O for Chrono.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from os import path
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from ..msgs.chrono_msgs import ChronoData, update_data
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5


class ChronoIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(self, filename: str, data: ChronoData, note: str = "") -> bool:
        """Save data to filename. return True on success."""

        data.set_saved()
        return save_pickle_or_h5(filename, data, ChronoData, self.logger, note=note)

    def load_data(self, filename: str) -> ChronoData | None:
        """Load data from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, ChronoData, self.logger)
        if d is not None:
            return update_data(d)

    def export_data(self, filename: str, data: ChronoData, params: dict | None = None) -> bool:
        """Export the data to text or image files.

        :param filename: supported extensions: text: .txt and .csv. image: .png, .pdf, and .eps.
        :param data: single data or list of data
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
        :param params.color: matplotlib colors for data
        :type params.color: list[str]
        :param params.linewidth: matplotlib linewidths for data
        :type params.linewidth: list[float]
        :param params.marker: matplotlib markers for data
        :type params.marker: list[str]
        :param params.legend: legend location (best|upper left|...). None to disable legend.
        :type params.legend: str|None

        """

        if params is None:
            params = {}

        if not isinstance(data, ChronoData):
            self.logger.error(f"Given object ({data}) is not an ChronoData.")
            return False

        if params.get("plot"):
            data.params["plot"].update(params["plot"])

        ext = path.splitext(filename)[1]
        if ext in (".txt", ".csv"):
            return self._export_data_csv(filename, data)
        elif ext in (".png", ".pdf", ".eps"):
            return self._export_data_image(filename, data, params)
        else:
            self.logger.error(f"Unknown extension to export data: {filename}")
            return False

    def _export_data_csv(self, fn, data: ChronoData):
        raise NotImplementedError("export data to csv (txt) is not implemented")

    def _export_data_image(self, fn, data: ChronoData, params: dict):
        plt.rcParams["font.size"] = params.get("fontsize", 28)

        label = params.get("label") or list[data.units.keys()]
        offset = params.get("offset") or [0.0] * len(label)
        color = cycle(params.get("color") or [f"C{i}" for i in range(10)])
        marker = cycle(params.get("marker") or ["o", "s", "^", "p", "*", "D", "h", "."])

        fig = plt.figure(figsize=params.get("figsize", (12, 12)), dpi=params.get("dpi", 100))
        ax = fig.add_subplot(111)

        lines = []
        x = data.get_xdata()
        ys = data.get_ydata()
        for l, ofs, col, mk in zip(label, offset, color, marker):
            y = ys[l]
            lw = 1.0 if params.get("linewidth") is None else params.get("linewidth")
            lines.append(ax.plot(x, np.array(y) + ofs, label=l, color=col, linewidth=lw)[0])

        # set_xlim and set_ylim should be called after all plots to auto-scale properly
        ax.set_xlim(params.get("xmin"), params.get("xmax"))
        ax.set_ylim(params.get("ymin"), params.get("ymax"))

        if params.get("legend") is not None:
            ax.legend(handles=lines, loc=params["legend"])

        plt.xlabel(r"Time (s)")

        plt.savefig(fn)
        plt.close()

        self.logger.info(f"Exported Data to {fn}.")
        return True
