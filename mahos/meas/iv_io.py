#!/usr/bin/env python3

"""
File I/O for IV.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from os import path

import numpy as np
import matplotlib.pyplot as plt

from ..msgs.iv_msgs import IVData, update_data
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5

# from ..msgs.common_msgs import Reply, BinaryState
# from ..msgs.common_msgs import LoadDataReq


class IVIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(self, filename: str, data: IVData, note: str = "") -> bool:
        """Save data to filename. return True on success."""

        data.set_saved()
        return save_pickle_or_h5(filename, data, IVData, self.logger, note=note)

    def load_data(self, filename: str) -> IVData | None:
        """Load data from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, IVData, self.logger)
        if d is not None:
            return update_data(d)

    def export_data(self, filename: str, data: IVData, params: dict | None = None) -> bool:
        """Export the data to text or image files.

        :param filename: supported extensions: text: .txt and .csv. image: .png, .pdf, and .eps.
        :param data: single data or list of data
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

        if not isinstance(data, IVData):
            self.logger.error(f"Given object ({data}) is not an IVData.")
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

    def _export_data_csv(self, fn, data: IVData):
        xdata = data.get_xdata()
        ydata = data.get_ydata()
        with open(fn, "w", encoding="utf-8", newline="\n") as fo:
            fo.write("# IV  data taken by mahos.meas.iv.\n")
            fo.write("# Voltage (V)\n")
            fo.write("# Current (A)\n")
        with open(fn, "ab") as fo:
            np.savetxt(fo, np.column_stack((xdata, ydata)), delimiter=",")
        self.logger.info(f"Exported Data to {fn}.")
        return True

    def _export_data_image(self, fn, data: IVData, params: dict):
        plt.rcParams["font.size"] = params.get("fontsize", 28)

        fig = plt.figure(figsize=params.get("figsize", (12, 12)), dpi=params.get("dpi", 100))
        ax = fig.subplots()
        x = data.get_xdata()
        y = data.get_ydata()
        ax.plot(x, y)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")

        plt.savefig(fn)
        plt.close()

        self.logger.info(f"Exported Data to {fn}.")
        return True
