#!/usr/bin/env python3

"""
File I/O for Qdyne.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from os import path

import matplotlib.pyplot as plt

from mahos_dq.msgs.qdyne_msgs import QdyneData, update_data
from mahos.node.log import DummyLogger
from mahos.util.io import save_pickle_or_h5, load_pickle_or_h5


class QdyneIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(
        self, file_name: str, data: QdyneData, params: dict | None = None, note: str = ""
    ) -> bool:
        """Save data to file_name. return True on success."""

        if params is None:
            params = {}

        if params.get("remove_fft_data", True):
            data.remove_fft_data()
        data.set_saved()

        return save_pickle_or_h5(
            file_name,
            data,
            QdyneData,
            self.logger,
            note=note,
            compression=params.get("compression", "lzf"),
            compression_opts=params.get("compression_opts"),
        )

    def load_data(self, file_name: str) -> QdyneData | None:
        """Load data from file_name. return None if load is failed."""

        d = load_pickle_or_h5(file_name, QdyneData, self.logger)
        if d is not None:
            return update_data(d)

    def export_data(self, file_name: str, data: QdyneData, params: dict | None = None) -> bool:
        """

        :param file_name: supported extensions: .png, .pdf, and .eps.
        :param data: single data
        :param params.trace: If True, plot time-domain trace data too.
        :type params.trace: bool
        :param params.pulse: If True, plot a pulse-histogram data too. Requires raw_data.
        :type params.pulse: bool

        """

        if params is None:
            params = {}

        ext = path.splitext(file_name)[1]
        if ext in (".png", ".pdf", ".eps"):
            return self._export_data_image(file_name, data, params)
        else:
            self.logger.error(f"Unknown extension to export data: {file_name}")
            return False

    def _export_data_image(
        self, file_name: str, data: QdyneData, params: dict | None = None
    ) -> bool:
        head, ext = path.splitext(file_name)

        plt.plot(data.get_xdata(True), data.get_ydata(True))
        plt.xlabel("Frequency (Hz)")
        # plt.ylabel("")
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

        if params.get("trace", False):
            x = data.get_xdata(False) * data.get_bin()
            y = data.get_ydata(False)

            plt.plot(x, y, "C0.")
            plt.xlabel("Time (s)")
            plt.ylabel("Detected photons")
            plt.tight_layout()
            plt.savefig(head + "_trace" + ext)
            plt.close()

        if params.get("pulse", False):
            if not data.has_raw_data():
                self.logger.error("params.pulse is True but no raw_data.")
                return False

            T = data.get_period_bins()
            tbin = data.get_bin()
            mod_sec = (data.raw_data % T) * tbin
            coeff = 1e6
            N, bins, patches = plt.hist(mod_sec * coeff, bins=1000)
            sh = data.marker_indices[0] * tbin * coeff
            st = data.marker_indices[1] * tbin * coeff
            plt.vlines((sh, st), 0, max(N), "r")
            plt.xlabel("Time (us)")
            plt.ylabel("Events")
            plt.tight_layout()
            plt.savefig(head + "_pulse" + ext)
            plt.close()
        return True
