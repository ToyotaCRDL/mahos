#!/usr/bin/env python3

"""
GUI frontend of Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import math
import typing as T
import os
import time
from datetime import datetime
import uuid
from functools import partial

import numpy as np
import pyqtgraph as pg

from mahos.gui import Qt
from mahos.gui.Qt import QtCore, QtWidgets, QtGui

from mahos_dq.gui.ui.podmr import Ui_PODMR
from mahos_dq.gui.ui.podmr_nmr_table import Ui_NMRTable
from mahos_dq.gui.ui.podmr_autosave import Ui_PODMRAutoSave
from mahos_dq.gui.podmr_client import QPODMRClient

from mahos.msgs.common_msgs import BinaryStatus, BinaryState
from mahos.msgs.common_meas_msgs import Buffer
from mahos.msgs import param_msgs as P
import mahos.util.validation as V
from mahos_dq.msgs.podmr_msgs import PODMRData, TimingInfo
from mahos.node.global_params import GlobalParamsClient
from mahos.gui.gui_node import GUINode, get_gui_logger
from mahos.gui.common_widget import ClientWidget
from mahos.gui.fit_widget import FitWidget
from mahos.gui.param import (
    ParamDictComboBoxHandler,
    set_enabled,
    apply_widgets,
    apply_widget_bounds,
)
from mahos.gui.dialog import save_dialog, load_dialog, export_dialog
from mahos.node.node import local_conf, join_name
from mahos.util.plot import colors_tab20_pair
from mahos.util.timer import seconds_to_hms
from mahos.util.conv import real_fft
from mahos.util.unit import SI_format, dBm_to_Vpeak


Policy = QtWidgets.QSizePolicy.Policy


def decimals_for_period(period_ns: float, minimum: int = 1, maximum: int = 6) -> int:
    """Return display precision sufficient for a clock period in nanoseconds."""

    for decimals in range(minimum, maximum + 1):
        if math.isclose(round(period_ns, decimals), period_ns, rel_tol=1e-6, abs_tol=1e-9):
            return decimals
    return maximum


class QNumTableWidgetItem(QtWidgets.QTableWidgetItem):
    """TableWidgetItem for numerical sorting"""

    def __init__(self, value):
        super(QNumTableWidgetItem, self).__init__(str(value))

    def __lt__(self, other):
        if isinstance(other, QNumTableWidgetItem):
            try:
                selfv = float(self.data(QtCore.Qt.ItemDataRole.EditRole))
                otherv = float(other.data(QtCore.Qt.ItemDataRole.EditRole))
                return selfv < otherv
            except ValueError:
                return QtWidgets.QTableWidgetItem.__lt__(self, other)
        else:
            return QtWidgets.QTableWidgetItem.__lt__(self, other)


class Colors(T.NamedTuple):
    color0: str
    color1: str


_colors = [Colors(c0, c1) for c0, c1 in colors_tab20_pair()]


class PODMRFitWidget(FitWidget):
    def colors(self) -> list:
        return _colors

    def load_dialog(self, default_path: str) -> str:
        return load_dialog(self, default_path, "PODMR", ".podmr")


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, logger=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)
        self._auto_range = True

        self.init_ui()
        self.init_view()
        self.update_font_size()
        self.fontsizeBox.editingFinished.connect(self.update_font_size)

    def auto_range(self) -> bool:
        return self._auto_range

    def set_auto_range(self, enable: bool):
        self._auto_range = enable

    def sizeHint(self):
        return QtCore.QSize(1400, 1000)

    def init_ui(self):
        hl0 = QtWidgets.QHBoxLayout()
        self.fontsizeBox = QtWidgets.QSpinBox(parent=self)
        self.fontsizeBox.setPrefix("font size: ")
        self.fontsizeBox.setSuffix(" pt")
        self.fontsizeBox.setMinimum(1)
        self.fontsizeBox.setValue(12)
        self.fontsizeBox.setMaximum(99)
        for w in (self.fontsizeBox,):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)

        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        for w in (self.fontsizeBox,):
            hl0.addWidget(w)
        hl0.addItem(spacer)

        vl = QtWidgets.QVBoxLayout()
        self.graphicsView = pg.GraphicsView(parent=self)

        vl.addLayout(hl0)
        vl.addWidget(self.graphicsView)
        self.setLayout(vl)

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)
        self.plot = self.layout.addPlot(row=0, col=0, lockAspect=False)
        self.plot.showGrid(x=True, y=True)

    def plot_analyzed(self, data_list):
        self.plot.clearPlots()
        for data, fitdisp, c in data_list:
            try:
                x = data.get_xdata()
                xfit = data.get_fit_xdata()
            except ValueError:
                self.logger.exception("Failed to get x data for analyzed plot.")
                continue

            try:
                y0, y1 = data.get_ydata()
                if y0 is None:
                    return
                yfit = data.get_fit_ydata()
            except ValueError:
                self.logger.exception("Failed to get y data for analyzed plot.")
                continue

            if fitdisp and (xfit is not None) and (yfit is not None):
                self.plot.plot(
                    x,
                    y0,
                    pen=pg.mkPen(c.color0, width=0.5),
                    symbolPen=None,
                    symbol="o",
                    symbolSize=4,
                    symbolBrush=c.color0,
                )
                if y1 is not None:
                    self.plot.plot(
                        x,
                        y1,
                        pen=pg.mkPen(c.color1, width=0.5),
                        symbolPen=None,
                        symbol="o",
                        symbolSize=4,
                        symbolBrush=c.color1,
                    )
                self.plot.plot(xfit, yfit, pen=pg.mkPen(c.color0, width=2.0))

            else:
                self.plot.plot(
                    x,
                    y0,
                    pen=pg.mkPen(c.color0, width=1.0),
                    symbolPen=None,
                    symbol="o",
                    symbolSize=8,
                    symbolBrush=c.color0,
                )
                if y1 is not None:
                    self.plot.plot(
                        x,
                        y1,
                        pen=pg.mkPen(c.color1, width=1.0),
                        symbolPen=None,
                        symbol="o",
                        symbolSize=8,
                        symbolBrush=c.color1,
                    )

    def refresh(self, data_list, data: PODMRData):
        try:
            self.plot_analyzed(data_list)
        except TypeError:
            self.logger.exception("Failed to plot analyzed data.")

        self.update_label(data)

    def update_label(self, data: PODMRData):
        if not data.has_params():
            return

        self.plot.setLabel("bottom", data.xlabel, data.xunit)
        self.plot.setLabel("left", data.ylabel, data.yunit)
        self.plot.setLogMode(x=data.xscale == "log", y=data.yscale == "log")

    def update_font_size(self):
        font = QtGui.QFont()
        font.setPointSize(self.fontsizeBox.value())
        for p in ("bottom", "left"):
            self.plot.getAxis(p).label.setFont(font)
            self.plot.getAxis(p).setTickFont(font)

    def enable_auto_range(self):
        if self._auto_range:
            self.plot.enableAutoRange()


class AltPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, logger=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)

        self.init_ui()
        self.init_view()
        self.update_font_size()
        self.fontsizeBox.editingFinished.connect(self.update_font_size)

    def sizeHint(self):
        return QtCore.QSize(1400, 1000)

    def init_ui(self):
        hl0 = QtWidgets.QHBoxLayout()

        # TODO: mode is only FFT for now. We can add other modes to select here.
        self.modeBox = QtWidgets.QComboBox(parent=self)
        self.modeBox.addItem("None")
        self.modeBox.addItem("FFT")

        self.zeropadBox = QtWidgets.QSpinBox(parent=self)
        self.zeropadBox.setPrefix("FFT zero pad: ")
        self.zeropadBox.setSuffix(" points")
        self.zeropadBox.setMinimum(0)
        self.zeropadBox.setValue(0)
        self.zeropadBox.setMaximum(10000)

        self.removeDCBox = QtWidgets.QCheckBox("FFT remove DC", parent=self)
        self.removeDCBox.setChecked(True)

        self.fontsizeBox = QtWidgets.QSpinBox(parent=self)
        self.fontsizeBox.setPrefix("font size: ")
        self.fontsizeBox.setSuffix(" pt")
        self.fontsizeBox.setMinimum(1)
        self.fontsizeBox.setValue(12)
        self.fontsizeBox.setMaximum(99)

        for w in (self.zeropadBox, self.fontsizeBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)
        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        for w in (self.modeBox, self.zeropadBox, self.removeDCBox, self.fontsizeBox):
            hl0.addWidget(w)
        hl0.addItem(spacer)

        vl = QtWidgets.QVBoxLayout()
        self.graphicsView = pg.GraphicsView(parent=self)

        vl.addLayout(hl0)
        vl.addWidget(self.graphicsView)
        self.setLayout(vl)

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)
        self.plot = self.layout.addPlot(row=0, col=0, lockAspect=False)
        self.plot.showGrid(x=True, y=True)

        # TODO only for FFT
        self.plot.setLabel("left", "FFT spectrum")

    def get_mode(self) -> str:
        """Get current AltPlot's mode. One of (None, FFT)."""

        return self.modeBox.currentText()

    def real_fft(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        if self.removeDCBox.isChecked():
            y = y - np.mean(y)

        padN = self.zeropadBox.value()
        if padN:
            xstep = abs(x[1] - x[0])
            x = np.concatenate((x, np.linspace(x[-1] + xstep, x[-1] + xstep * padN, num=padN)))
            y = np.concatenate((y, np.zeros(padN)))

        # don't use remove_dc at util function side (already done above).
        return real_fft(x, y, remove_dc=False)

    def plot_fft(self, data_list):
        self.plot.clearPlots()

        # don't plot fit result for now
        for data, _, c in data_list:
            try:
                x = data.get_xdata()
            except ValueError:
                self.logger.exception("Failed to get x data for alternate plot.")
                continue
            if len(x) == 1:
                continue
            try:
                y0, y1 = data.get_ydata()
                if y0 is None:
                    return
            except ValueError:
                self.logger.exception("Failed to get y data for alternate plot.")
                continue

            f, S0 = self.real_fft(x, y0)
            self.plot.plot(
                f,
                S0,
                pen=pg.mkPen(c.color0, width=1.0),
                symbolPen=None,
                symbol="o",
                symbolSize=8,
                symbolBrush=c.color0,
            )
            if y1 is not None:
                f, S1 = self.real_fft(x, y1)
                self.plot.plot(
                    f,
                    S1,
                    pen=pg.mkPen(c.color1, width=1.0),
                    symbolPen=None,
                    symbol="o",
                    symbolSize=8,
                    symbolBrush=c.color1,
                )

    def refresh(self, data_list, data: PODMRData):
        if self.get_mode() == "None":
            self.plot.clearPlots()
            return

        try:
            self.plot_fft(data_list)
        except TypeError:
            self.logger.exception("Failed to plot FFT data.")

        self.update_label(data)

    def update_label(self, data: PODMRData):
        if not data.has_params():
            return

        # TODO only for FFT
        if data.xunit == "s":
            self.plot.setLabel("bottom", "Frequency", "Hz")
        elif data.xunit == "Hz":
            self.plot.setLabel("bottom", "Time", "s")
        else:
            self.plot.setLabel("bottom", "Unknown domain")
        # we shouldn't reflect data's attribute directly.
        # self.plot.setLogMode(x=data.xscale == "log", y=data.yscale == "log")

    def update_font_size(self):
        font = QtGui.QFont()
        font.setPointSize(self.fontsizeBox.value())
        for p in ("bottom", "left"):
            self.plot.getAxis(p).label.setFont(font)
            self.plot.getAxis(p).setTickFont(font)


class RawPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, logger=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)

        self.init_widgets()
        self.update_font_size()
        self.fontsizeBox.editingFinished.connect(self.update_font_size)

    def sizeHint(self):
        return QtCore.QSize(1400, 600)

    def init_widgets(self):
        glw = pg.GraphicsLayoutWidget(parent=self)
        self.raw_plot = glw.addPlot(lockAspect=False)
        self.raw_plot.showGrid(x=True, y=True)
        self.raw_plot.setLabel("bottom", "Time", "s")
        self.raw_plot.setLabel("left", "Events")

        # sig_head, sig_tail, ref_head, ref_tail
        marker_brushes = ((255, 0, 0), (255, 128, 0), (0, 0, 255), (0, 128, 255))
        laser_timing_pen = pg.mkPen((0, 180, 0), width=1, style=QtCore.Qt.PenStyle.DashLine)
        self._raw_item = self.raw_plot.plot([], [])
        self._marker_items = [
            self.raw_plot.plot(
                [],
                [],
                pen=None,
                symbolPen=None,
                symbol="o",
                symbolSize=8,
                symbolBrush=brush,
            )
            for brush in marker_brushes
        ]
        self._laser_timing_item = self.raw_plot.plot([], [], pen=laser_timing_pen)

        self.showBox = QtWidgets.QCheckBox("Show")
        self.showBox.setChecked(True)

        self.allBox = QtWidgets.QCheckBox("All")

        self.indexBox = QtWidgets.QSpinBox()
        self.indexBox.setPrefix("laser index: ")
        self.indexBox.setMinimum(0)
        self.indexBox.setMaximum(9999)

        self.numBox = QtWidgets.QSpinBox()
        self.numBox.setPrefix("num: ")
        self.numBox.setMinimum(1)
        self.numBox.setMaximum(9999)

        self.marginBox = QtWidgets.QSpinBox()
        self.marginBox.setPrefix("margin: ")
        self.marginBox.setMinimum(0)
        self.marginBox.setMaximum(3000)
        self.marginBox.setSingleStep(50)
        self.marginBox.setValue(150)

        self.fontsizeBox = QtWidgets.QSpinBox(parent=self)
        self.fontsizeBox.setPrefix("font size: ")
        self.fontsizeBox.setSuffix(" pt")
        self.fontsizeBox.setMinimum(1)
        self.fontsizeBox.setValue(12)
        self.fontsizeBox.setMaximum(99)

        for w in (self.indexBox, self.numBox, self.marginBox, self.fontsizeBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)

        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)

        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.showBox)
        hl.addWidget(self.allBox)
        hl.addWidget(self.indexBox)
        hl.addWidget(self.numBox)
        hl.addWidget(self.marginBox)
        hl.addWidget(self.fontsizeBox)
        hl.addItem(spacer)
        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl)
        vl.addWidget(glw)
        self.setLayout(vl)

    def _show_empty(self):
        self._raw_item.setData([0, 1], [0, 0])
        for item in self._marker_items:
            item.setData([], [])
        self._laser_timing_item.setData([], [])

    def _marker_data_roi(self, data: PODMRData, rx, ry, start, stop) -> list[tuple[list, list]]:
        marker_data = []
        rois = data.get_rois()
        for inds in data.marker_indices:
            x = []
            y = []
            for i, ((roi_start, _), idx) in enumerate(zip(rois[start:stop], inds[start:stop])):
                try:
                    px = rx[start + i][idx - roi_start]
                    py = ry[start + i][idx - roi_start]
                except IndexError:
                    continue
                else:
                    x.append(px)
                    y.append(py)
            marker_data.append((x, y))
        return marker_data

    def _marker_data_noroi(
        self, data: PODMRData, rx, ry, start, stop
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        marker_data = []
        for inds in data.marker_indices:
            x = np.array([rx[i] for i in inds[start:stop]])
            y = np.array([ry[i] for i in inds[start:stop]])
            marker_data.append((x, y))
        return marker_data

    def _set_marker_data(self, marker_data):
        for item, (x, y) in zip(self._marker_items, marker_data):
            item.setData(x, y)

    def _set_laser_timing_data(self, data: PODMRData, start, stop, plotted_y):
        if data.laser_timing is None or data.laser_timing_offset is None:
            self._laser_timing_item.setData([], [])
            return

        timing = np.asarray(data.laser_timing, dtype=np.float64)
        offset = np.asarray(data.laser_timing_offset, dtype=np.float64)
        if timing.ndim != 1 or offset.ndim != 1 or len(timing) != len(offset):
            self._laser_timing_item.setData([], [])
            return

        corrected = timing + offset
        corrected = corrected[start:stop]
        corrected = corrected[np.isfinite(corrected)]
        if len(corrected) == 0:
            self._laser_timing_item.setData([], [])
            return

        plotted_y = np.asarray(plotted_y)
        if plotted_y.size:
            ymax = float(np.nanmax(plotted_y))
        else:
            ymax = 1.0
        if not np.isfinite(ymax) or ymax <= 0.0:
            ymax = 1.0

        x = np.empty(len(corrected) * 3, dtype=np.float64)
        y = np.empty(len(corrected) * 3, dtype=np.float64)
        x[0::3] = corrected
        x[1::3] = corrected
        x[2::3] = np.nan
        y[0::3] = 0.0
        y[1::3] = ymax
        y[2::3] = np.nan
        self._laser_timing_item.setData(x, y)

    def plot_raw(self, data: PODMRData):
        if not self.isVisible():
            return

        if not data.has_raw_data() or not self.showBox.isChecked():
            self._show_empty()
            return

        rx = data.get_raw_xdata()
        ry = data.raw_data
        if rx is None or ry is None:
            self._show_empty()
            return

        laser_pulses = len(data.marker_indices[0])
        self.indexBox.setMaximum(laser_pulses - 1)
        self.numBox.setMaximum(laser_pulses)

        if self.allBox.isChecked():
            # show whole raw data
            if data.has_roi():
                x = np.concatenate(rx)
                y = np.concatenate(ry)
                self._set_marker_data(self._marker_data_roi(data, rx, ry, 0, None))
            else:
                x = rx
                y = ry
                self._set_marker_data(self._marker_data_noroi(data, rx, ry, 0, None))
            self._raw_item.setData(x, y)
            self._set_laser_timing_data(data, 0, laser_pulses, y)
            return

        lstart = self.indexBox.value()
        lstop = min(lstart + self.numBox.value() - 1, laser_pulses - 1)

        if data.has_roi():
            x = np.concatenate(rx[lstart : lstop + 1])
            y = np.concatenate(ry[lstart : lstop + 1])
            self._set_marker_data(self._marker_data_roi(data, rx, ry, lstart, lstop + 1))
        else:
            margin = self.marginBox.value()
            head = data.marker_indices[0][lstart] - margin
            tail = data.marker_indices[3][lstop] + margin
            x = rx[head:tail]
            y = ry[head:tail]
            self._set_marker_data(self._marker_data_noroi(data, rx, ry, lstart, lstop + 1))
        self._raw_item.setData(x, y)
        self._set_laser_timing_data(data, lstart, lstop + 1, y)

    def refresh(self, data: PODMRData):
        try:
            self.plot_raw(data)
        except TypeError:
            self.logger.exception("Failed to plot raw data.")

    def update_font_size(self):
        font = QtGui.QFont()
        font.setPointSize(self.fontsizeBox.value())
        for p in ("bottom", "left"):
            self.raw_plot.getAxis(p).label.setFont(font)
            self.raw_plot.getAxis(p).setTickFont(font)


class NMRTableWidget(QtWidgets.QWidget, Ui_NMRTable):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

    def init_connection(self, update_slot):
        for b in (
            self.B0Box,
            self.TiterBox,
            self.ind1HgammaBox,
            self.ind13CgammaBox,
            self.ind14NgammaBox,
            self.ind15NgammaBox,
            self.ind19FgammaBox,
            self.ind31PgammaBox,
            self.indacfreqBox,
            self.indac2freqBox,
        ):
            b.valueChanged.connect(update_slot)
        for b in (
            self.ind1HBox,
            self.ind13CBox,
            self.ind14NBox,
            self.ind15NBox,
            self.ind19FBox,
            self.ind31PBox,
            self.indacBox,
            self.indac2Box,
        ):
            b.toggled.connect(update_slot)
        for e in (
            self.ind1HpeakEdit,
            self.ind13CpeakEdit,
            self.ind14NpeakEdit,
            self.ind15NpeakEdit,
            self.ind19FpeakEdit,
            self.ind31PpeakEdit,
            self.indacpeakEdit,
            self.indac2peakEdit,
        ):
            e.editingFinished.connect(update_slot)

        self.tpimanualBox.toggled.connect(self.set_tpi_manual)
        self.tpimanualBox.toggled.connect(update_slot)
        self.tpiBox.editingFinished.connect(update_slot)

    def set_tpi_manual(self, manual: bool):
        self.tpiBox.setReadOnly(not manual)

    def downconv_freq(self, f, TL):
        """down convert frequency to (delta, LO)."""

        if TL == 0.0:
            return [float("nan")] * len(f), [float("nan")] * len(f)

        fTL = f * TL
        m = np.rint(fTL)
        d = np.abs(fTL - m) / TL
        f_LO = m / TL
        return d, f_LO

    def update(self, data: PODMRData):
        params = data.get_pulse_params()
        if "180pulse" in params and not self.tpimanualBox.isChecked():
            self.tpiBox.setValue(params["180pulse"] * 1e9)
        t_pi = self.tpiBox.value() * 1e-9
        t_iter = self.TiterBox.value() * 1e-6

        res = []  # No, Nuclei, Harmonics, Larmor freq., 2/(Larmor freq.), tau, folded freq.
        nuclei = ["1H", "13C", "14N", "15N", "19F", "31P", "ac", "ac2"]
        for nuc_label in nuclei:
            enabled = eval("self.ind%sBox" % nuc_label).isChecked()
            if not enabled:
                continue

            peaks, harm_label = self.calc_peak_position(nuc_label)
            peaks_arr = np.array(peaks)
            tau = 1.0 / (4.0 * peaks_arr) - t_pi / 2
            invfl = 2.0 / peaks_arr
            delta, LO = self.downconv_freq(peaks_arr, t_iter)

            for i in range(len(peaks)):
                res.append(
                    (
                        nuc_label,
                        harm_label[i],
                        "{:.6f}".format(peaks[i] / 1e6),
                        "{:.2f}".format(invfl[i] * 1e9),
                        "{:.2f}".format(tau[i] * 1e9),
                        "{:.3f}".format(delta[i] * 1e-3),
                    )
                )

        res = [(str(i + 1),) + r for i, r in enumerate(res)]  # insert the column of No.

        # rebuild table
        sorting = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(len(res))

        for (i, j), s in np.ndenumerate(res):
            self.tableWidget.setItem(i, j, QNumTableWidgetItem(s))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.setSortingEnabled(sorting)

    def calc_peak_position(self, nuc_label) -> list[tuple[float, str]]:
        """list of (Larmor freq: float, harmonics label: str)"""

        B0 = self.B0Box.value() * 1e-3  # mT to T

        if nuc_label in ["ac", "ac2"]:
            lamor_freq = abs(getattr(self, f"ind{nuc_label}freqBox").value() * 1e6)  # MHz to Hz
        else:
            gamma = getattr(self, f"ind{nuc_label}gammaBox").value() * 1e6  # MHz/T to Hz/T
            lamor_freq = abs(B0 * gamma)  # Hz

        peaks_str = getattr(self, f"ind{nuc_label}peakEdit").text()
        peaks = []
        harmonics = peaks_str.replace(" ", "").split(",")
        for harm in harmonics:
            try:
                if harm.find("/") >= 0:  # rational number
                    a, b = harm.split("/")
                    h = float(a) / float(b)
                else:  # integer or float
                    h = float(harm)
            except (TypeError, ValueError):
                pass
            peaks.append(lamor_freq * h)
        return peaks, harmonics


class PODMRAutoSaveWidget(QtWidgets.QWidget, Ui_PODMRAutoSave):
    AUTOSAVE_EXTENSION = ".podmr.h5"

    def __init__(self, cli, gparams_cli, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.extLabel.setText(self.AUTOSAVE_EXTENSION)

        self.cli = cli
        self.gparams_cli = gparams_cli

    def init_connection(self):
        self.browseButton.clicked.connect(self.browse_dir)

    def check_autosave(self, data: PODMRData):
        if not self.enableBox.isChecked():
            return
        if not data.has_data():
            # it's taking long for measurement to start actually.
            # start counting time after the start.
            self.last_saved_at = datetime.now()
            return

        t = datetime.now()
        dt = t - self.last_saved_at
        h, m, s = seconds_to_hms(dt.seconds)
        self.agoLabel.setText(f"({h:02d}:{m:02d}:{s:02d} ago)")

        if dt.seconds > self.intervalBox.value():
            self.request_save()
            self.last_saved_at = t
            self.lastLabel.setText("Last saved: " + t.strftime("%F %T"))

    def request_save(self):
        fn = self.get_file_name()
        self.cli.save_data(fn, params={"tmp": True})
        if self.exportPngBox.isChecked():
            self.cli.export_data(os.path.splitext(fn)[0] + ".png")
        self.suffixBox.setValue(self.suffixBox.value() + 1)

    def get_file_name(self):
        dirname = self.dirEdit.text()
        fname = "{:s}_{:04d}{:s}".format(
            self.fnEdit.text(), self.suffixBox.value(), self.AUTOSAVE_EXTENSION
        )
        return os.path.normpath(os.path.join(dirname, fname))

    def init_autosave(self):
        if not self.enableBox.isChecked():
            self.lastLabel.setText("Last saved: ")
            self.agoLabel.setText("()")
            return

        self.last_saved_at = datetime.now()
        if self.resetBox.isChecked():
            self.suffixBox.setValue(0)

        if os.path.isfile(self.get_file_name()):
            if not Qt.question_yn(
                self, "Overwrite file name?", "File name overlap for autosave. Overwrite?"
            ):
                self.enableBox.setChecked(False)
                return

        try:
            if not os.path.isdir(self.dirEdit.text()):
                os.makedirs(self.dirEdit.text())
        except IOError:
            QtWidgets.QMessageBox.warning(
                self, "Cannot autosave", "Failed to autosave directory. Cannot perform autosave"
            )
            self.enableBox.setChecked(False)

    def update_state(self, state: BinaryState, last_state: BinaryState):
        self.enableBox.setEnabled(state == BinaryState.IDLE)
        for w in (
            self.dirEdit,
            self.browseButton,
            self.fnEdit,
            self.suffixBox,
            self.resetBox,
            self.intervalBox,
            self.exportPngBox,
        ):
            w.setEnabled(state == BinaryState.IDLE and self.enableBox.isChecked())

    def browse_dir(self):
        current = str(self.gparams_cli.get_param("work_dir"))
        dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Select autosave directory", current)
        if dn:
            self.dirEdit.setText(dn)


class PODMRWidgetBase(ClientWidget):
    """Base widget for pulse-ODMR-like measurements."""

    CLIENT_CLASS = QPODMRClient
    DATA_CLASS = PODMRData
    FIT_WIDGET_CLASS = PODMRFitWidget
    AUTOSAVE_WIDGET_CLASS = PODMRAutoSaveWidget
    MEASUREMENT_NAME = "PODMR"
    FILE_EXTENSION = ".podmr"

    def __init__(
        self,
        gconf: dict,
        name,
        gparams_name,
        enable_resume: bool,
        plot: PlotWidget,
        alt_plot: AltPlotWidget,
        raw_plot: RawPlotWidget,
        context,
        parent=None,
        logger=None,
        file_transport_dir=None,
    ):
        ClientWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)
        self.setup_measurement_ui()

        self.conf = local_conf(gconf, name)

        self._finalizing = False
        self._has_fg = False
        self._found_sg1 = False
        self._timing_info: TimingInfo | None = None
        self._timing_params_key = None
        self._params = None
        self._meas_state = BinaryState.IDLE

        self.enableresumeBox.setChecked(enable_resume)
        self.init_radiobuttons()
        self.tabWidget.setCurrentIndex(0)
        self.plot = plot
        self.alt_plot = alt_plot
        self.raw_plot = raw_plot

        self.data = self.DATA_CLASS()

        self.cli = self.CLIENT_CLASS(
            gconf, name, context=context, parent=self, file_transport_dir=file_transport_dir
        )
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli, self.gparams_cli)

        self._fiTab_layout = QtWidgets.QVBoxLayout(self.fiTab)
        self.fit = self.FIT_WIDGET_CLASS(self.cli, self.gparams_cli, parent=self.fiTab)
        self._fiTab_layout.addWidget(self.fit)

        self._autosaveTab_layout = QtWidgets.QVBoxLayout(self.autosaveTab)
        self.autosave = self.AUTOSAVE_WIDGET_CLASS(
            self.cli, self.gparams_cli, parent=self.autosaveTab
        )
        self._autosaveTab_layout.addWidget(self.autosave)
        self.autosave.init_connection()

        self._table_layout = QtWidgets.QVBoxLayout(self.tableTab)
        self.table = NMRTableWidget(parent=self.tableTab)
        self._table_layout.addWidget(self.table)
        self.table.init_connection(self.update_table)

        self.setEnabled(False)

    def setup_measurement_ui(self):
        raise NotImplementedError

    def supports_discard(self) -> bool:
        """When True, UI must have discardButton and enablediscardBox."""

        return True

    def supports_find_laser_timing(self) -> bool:
        """When True, UI must have findLaserTimingButton etc."""

        return True

    def init_radiobuttons(self):
        def set_group(group, index, buttons):
            for b in buttons:
                group.addButton(b)
            group.setExclusive(True)
            buttons[index].setChecked(True)

        # options
        self.fg_buttons = QtWidgets.QButtonGroup(parent=self)
        set_group(self.fg_buttons, 0, [b for b, l in self.get_fg_mode_dict()])

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.methodBox.clear()
        methods = P.filter_out_label_prefix("fit", self.cli.get_param_dict_labels())
        self.methodBox.addItems(methods)

        self.init_widgets_with_params()
        self.init_connection()
        self.fit.init_with_status()

        self.fgGroupBox.setEnabled(self.has_fg())

        # update initial GUI state
        self.update_state(status.state, last_state=BinaryState.IDLE)
        self.switch_method()

        self.cli.stateUpdated.connect(self.update_state)
        self.cli.dataUpdated.connect(self.update_data)
        self.cli.bufferUpdated.connect(self.update_buffer)
        self.cli.stopped.connect(self.finalize)

        self.setEnabled(True)

    def init_connection(self):
        self.startButton.clicked.connect(self.request_start)
        self.validateButton.clicked.connect(self.request_validate)
        self.stopButton.clicked.connect(self.request_stop)
        self.saveButton.clicked.connect(self.save_data)
        self.exportButton.clicked.connect(self.export_data)
        self.exportaltButton.clicked.connect(self.export_alt_data)
        self.loadButton.clicked.connect(self.load_data)

        if self.supports_discard():
            self.discardButton.clicked.connect(self.discard_data)
        if self.supports_find_laser_timing():
            self.findLaserTimingButton.clicked.connect(self.find_laser_timing)
            self.clearLaserTimingButton.clicked.connect(self.clear_laser_timing)

        # main tab
        for w in (self.startBox, self.stepBox, self.numBox):
            w.valueChanged.connect(self.update_stop)
        for w in (self.NstartBox, self.NstepBox, self.NnumBox):
            w.valueChanged.connect(self.update_Nstop)
        self.powerBox.valueChanged.connect(partial(self.update_volt_label, self.voltLabel))
        self.power1Box.valueChanged.connect(partial(self.update_volt_label, self.volt1Label))

        self._method_handler = ParamDictComboBoxHandler(self.methodBox)
        self.methodBox.currentIndexChanged.connect(self.switch_method)
        self.partialBox.currentIndexChanged.connect(self.switch_partial)
        self.paramTable.paramsChanged.connect(self.update_timing_info)

        self._wire_plot_widgets(True)
        self.plotenableBox.toggled.connect(self.update_plot_enable)
        self.plotenableBox.setChecked(True)

        self.taumodeBox.currentIndexChanged.connect(self.plot.enable_auto_range)

        # extra tab
        if self.has_fg():
            self.fg_disableButton.toggled.connect(self.switch_fg)
            self.fg_cwButton.toggled.connect(self.switch_fg)
            self.fg_gateButton.toggled.connect(self.switch_fg)

    def init_widgets_with_params(self):
        params = self.cli.get_param_dict("rabi")

        if "fg" in params:
            self._has_fg = True
            apply_widgets(
                params["fg"],
                [
                    ("ampl", self.fg_amplBox),
                    ("freq", self.fg_freqBox, 1e-3),
                    ("phase", self.fg_phaseBox),
                ],
            )
        else:
            self._has_fg = False

        apply_widgets(
            params,
            [
                ("power", self.powerBox),
                ("freq", self.freqBox, 1e-6),
                ("nomw", self.nomwBox),
                ("interval", self.intervalBox, 1e3),
                ("timebin", self.binBox, 1e9),
                ("sweeps", self.sweepsBox),
                ("duration", self.durationBox),
                ("roi_head", self.roiheadBox, 1e9),
                ("roi_tail", self.roitailBox, 1e9),
                ("multi_histogram", self.multiHistoBox),
                ("divide_block", self.divideblockBox),
                ("base_width", self.basewidthBox, 1e9),
                ("laser_delay", self.ldelayBox, 1e9),
                ("laser_width", self.lwidthBox, 1e9),
                ("mw_delay", self.mdelayBox, 1e9),
                ("trigger_width", self.trigwidthBox, 1e9),
                ("init_delay", self.initdelayBox, 1e9),
                ("final_delay", self.finaldelayBox, 1e9),
                ("mw_offset", self.moffsetBox, 1e9),
            ],
        )
        if "freq1" in params:
            apply_widgets(
                params,
                [
                    ("power1", self.power1Box),
                    ("freq1", self.freq1Box, 1e-6),
                    ("nomw1", self.nomw1Box),
                ],
            )
        apply_widgets(
            params["plot"],
            [
                ("sigdelay", self.sigdelayBox, 1e9),
                ("sigwidth", self.sigwidthBox, 1e9),
                ("refdelay", self.refdelayBox, 1e9),
                ("refwidth", self.refwidthBox, 1e9),
            ],
        )

    # Widget status updates

    def update_table(self):
        self.table.update(self.data)

    def update_plot_enable(self, enable: bool):
        for w in (
            self.plotmodeBox,
            self.taumodeBox,
            self.refmodeBox,
            self.sigdelayBox,
            self.sigwidthBox,
            self.refdelayBox,
            self.refwidthBox,
            self.refavgBox,
            self.autoadjustBox,
            self.logXBox,
            self.logYBox,
            self.flipYBox,
        ):
            w.setEnabled(enable)

    def update_stop(self):
        stop = self.startBox.value() + self.stepBox.value() * (self.numBox.value() - 1)
        self.stopLabel.setText(f"stop: {stop:_.1f} ns")

    def update_Nstop(self):
        Nstop = self.NstartBox.value() + self.NstepBox.value() * (self.NnumBox.value() - 1)
        self.NstopLabel.setText(f"stop: {Nstop:_d}")

    def update_volt_label(self, widget, power_dBm: float):
        imp = 50
        Vp = SI_format(dBm_to_Vpeak(power_dBm, imp), suffix="Vp")
        widget.setText(f"{Vp} ({imp} Ω)")

    def switch_fg(self):
        disabled = self.fg_disableButton.isChecked()
        for w in (self.fg_waveBox, self.fg_freqBox, self.fg_amplBox):
            w.setEnabled(not disabled)
        self.fg_phaseBox.setEnabled(self.fg_gateButton.isChecked())

    def _apply_sg1(self, params: dict):
        """Check existence of SG1 (freq1 in params) and apply bound of SG1."""

        if self._found_sg1:
            return
        if "freq1" in params:
            apply_widgets(params, [("freq1", self.freq1Box, 1e-6), ("power1", self.power1Box)])
            self._found_sg1 = True

    def switch_method(self):
        params = self._method_handler.get(self.cli.get_param_dict)
        if params is None:
            return
        self._params = params
        self.update_cond_widgets()
        self._apply_sg1(self._params)
        apply_widget_bounds(
            self._params["pulse"],
            [
                ("90pulse", self.t90pulseBox, 1e9),
                ("180pulse", self.t180pulseBox, 1e9),
                ("270pulse", self.t270pulseBox, 1e9),
            ],
        )
        pp = P.ParamDict(
            pulse={
                k: v
                for k, v in self._params["pulse"].items()
                if k not in ("90pulse", "180pulse", "270pulse")
            }
        )
        if "pd" in self._params:
            pp["pd"] = self._params["pd"]
        if "awg" in self._params:
            pp["awg"] = self._params["awg"]
        self.paramTable.update_contents(pp)
        self.update_timing_info(force=True)
        self.reset_partial_modes(self._params["partial"].maximum())
        self.reset_plot_modes(
            self._params["plot"]["plotmode"].options(),
            self._params["plot"]["plotmode"].default(),
        )
        self.reset_tau_modes(self._params["plot"]["taumode"].options())
        self.plot.update_label(self.data)

    def switch_partial(self, index):
        # when partial is 0 (index 1), set plotmode to data0
        # when partial is 1 (index 2), set plotmode to data1
        # when partial is 2 (index 3), set plotmode to data2 if available (else data0)
        # when partial is 3 (index 4), set plotmode to data3 if available (else data0)
        if index == 1:
            self.set_plot_mode("data0")
        elif index == 2:
            self.set_plot_mode("data1")
        elif index == 3:
            if self.plotmodeBox.findText("data2") >= 0:
                self.set_plot_mode("data2")
            else:
                self.set_plot_mode("data0")
        elif index == 4:
            if self.plotmodeBox.findText("data3") >= 0:
                self.set_plot_mode("data3")
            else:
                self.set_plot_mode("data0")

    def _update_save_button(self):
        if self.data.is_finalized() and not self.data.is_saved():
            self.saveButton.setStyleSheet("background-color: #FF0000")
        else:
            self.saveButton.setStyleSheet("")

    def _wire_plot_timing_widgets(self, connect: bool):
        for b in (self.sigdelayBox, self.sigwidthBox, self.refdelayBox, self.refwidthBox):
            if connect:
                b.editingFinished.connect(self.update_plot_params)
            else:
                b.editingFinished.disconnect(self.update_plot_params)

    def _wire_plot_widgets(self, connect: bool):
        self._wire_plot_timing_widgets(connect)
        for cb in (self.plotmodeBox, self.taumodeBox, self.refmodeBox):
            if connect:
                cb.currentIndexChanged.connect(self.update_plot_params)
            else:
                cb.currentIndexChanged.disconnect(self.update_plot_params)
        for b in (self.refavgBox, self.logXBox, self.logYBox, self.flipYBox):
            if connect:
                b.toggled.connect(self.update_plot_params)
            else:
                b.toggled.disconnect(self.update_plot_params)

    def adjust_analysis_timing(self, tbin_ns: float):
        tbin = int(round(tbin_ns * 10))
        self._wire_plot_timing_widgets(False)
        for b in (self.sigdelayBox, self.sigwidthBox, self.refdelayBox, self.refwidthBox):
            v = int(round(b.value() * 10))
            v_ = v - v % tbin
            b.setValue(v_ / 10)
            b.setSingleStep(tbin / 10 + 0.01)
        self._wire_plot_timing_widgets(True)

    # data managements

    def save_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, self.MEASUREMENT_NAME, self.FILE_EXTENSION)
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.gparams_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)

        n = os.path.splitext(fn)[0] + ".png"
        params = {"show_fit": self.fit.show_current_data_fit()}
        self.cli.export_data(n, params=params)

    def export_data(self):
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = export_dialog(
            self, default_path, self.MEASUREMENT_NAME, (".png", ".pdf", ".eps", ".txt")
        )
        if not fn:
            return

        params = {}
        data = [d for (d, _, _) in data_list]
        params["color_fit"] = [color.color0 for (_, _, color) in data_list]
        params["color0"] = [color.color0 for (_, _, color) in data_list]
        params["color1"] = [color.color1 for (_, _, color) in data_list]
        params["show_fit"] = any([show_fit for (_, show_fit, _) in data_list])
        self.cli.export_data(fn, data=data, params=params)

    def export_alt_data(self):
        if self.alt_plot.get_mode() == "None":
            QtWidgets.QMessageBox.warning(self, "No alt plot", "Alt plot is disabled.")
            return

        # Assume FFT
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = export_dialog(
            self, default_path, self.MEASUREMENT_NAME, (".png", ".pdf", ".eps", ".txt")
        )
        if not fn:
            return

        params = {}
        data = [d for (d, _, _) in data_list]
        params["color_fit"] = [color.color0 for (_, _, color) in data_list]
        params["color0"] = [color.color0 for (_, _, color) in data_list]
        params["color1"] = [color.color1 for (_, _, color) in data_list]
        params["show_fit"] = False
        params["fft"] = True
        self.cli.export_data(fn, data=data, params=params)

    def load_data(self):
        if self.data.is_finalized() and not self.data.is_saved():
            if not Qt.question_yn(
                self, "Sure to load?", "Current data has not been saved. Are you sure to discard?"
            ):
                return

        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, self.MEASUREMENT_NAME, self.FILE_EXTENSION)
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.gparams_cli.set_param("loaded_note", data.note())

        self.data = data
        self.apply_meas_widgets()
        self.apply_fg_widgets()
        if self.plotenableBox.isChecked():
            self.apply_plot_widgets()
        self.refresh_plot()
        self.update_widgets()

        # self.update_data(data)

    # parameters

    def apply_param_table(self):
        p = self.data.params
        # optional pulse params
        pp = p["pulse"]
        for k, v in pp.items():
            if k not in ("90pulse", "180pulse", "270pulse"):
                self.paramTable.apply_value("pulse." + k, v)
        if "90pulse" in pp:
            self.t90pulseBox.setValue(pp["90pulse"] * 1e9)  # sec to ns
        if "180pulse" in pp:
            self.t180pulseBox.setValue(pp["180pulse"] * 1e9)  # sec to ns
        if "270pulse" in pp:
            self.t270pulseBox.setValue(pp["270pulse"] * 1e9)  # sec to ns
        # optional pd params
        if "pd" in p:
            for k, v in p["pd"].items():
                self.paramTable.apply_value("pd." + k, v)
        # optional awg params
        if "awg" in p:
            for k, v in p["awg"].items():
                self.paramTable.apply_value("awg." + k, v)

    def apply_meas_widgets(self):
        p = self.data.params

        # TDC
        self.binBox.setValue(p.get("timebin", 0.0) * 1e9)  # sec to ns
        self.intervalBox.setValue(int(round(p.get("interval", 0.0) * 1e3)))  # sec to ms
        self.sweepsBox.setValue(p.get("sweeps", 0))
        self.durationBox.setValue(p.get("duration", 0.0))
        self.roiheadBox.setValue(p.get("roi_head", -1e-9) * 1e9)  # sec to ns
        self.roitailBox.setValue(p.get("roi_tail", -1e-9) * 1e9)  # sec to ns
        self.multiHistoBox.setChecked(p.get("multi_histogram", False))

        # method
        self.set_method(self.data.label)

        self.apply_param_table()

        # MW
        self.freqBox.setValue(p.get("freq", 2740e6) * 1e-6)  # Hz to MHz
        self.powerBox.setValue(p.get("power", 0.0))
        self.nomwBox.setChecked(p.get("nomw", False))
        if "freq1" in p:
            self.freq1Box.setValue(p["freq1"] * 1e-6)  # Hz to MHz
        if "power1" in p:
            self.power1Box.setValue(p["power1"])
        if "nomw1" in p:
            self.nomw1Box.setChecked(p["nomw1"])

        # sequence parameters
        self.startBox.setValue(p.get("start", 0.0) * 1e9)  # sec to ns
        self.numBox.setValue(p.get("num", 1))
        self.stepBox.setValue(p.get("step", 0.0) * 1e9)  # sec to ns
        self.logBox.setChecked(p.get("log", False))
        self.NstartBox.setValue(p.get("Nstart", 1))
        self.NnumBox.setValue(p.get("Nnum", 1))
        self.NstepBox.setValue(p.get("Nstep", 1))

        # method params
        self.invertsweepBox.setChecked(p.get("invert_sweep", False))
        self.reduceBox.setChecked(p.get("enable_reduce", False))
        self.divideblockBox.setChecked(p.get("divide_block", False))
        partial = p.get("partial")
        if partial in (0, 1, 2, 3) and partial + 1 < self.partialBox.count():
            self.partialBox.setCurrentIndex(partial + 1)
        else:
            self.partialBox.setCurrentIndex(0)

        # sequence parameters (pulses) sec to ns
        self.basewidthBox.setValue(p.get("base_width", 0.0) * 1e9)
        self.ldelayBox.setValue(p.get("laser_delay", 0.0) * 1e9)
        self.lwidthBox.setValue(p.get("laser_width", 0.0) * 1e9)
        self.mdelayBox.setValue(p.get("mw_delay", 0.0) * 1e9)
        self.trigwidthBox.setValue(p.get("trigger_width", 0.0) * 1e9)
        self.initdelayBox.setValue(p.get("init_delay", 0.0) * 1e9)
        self.finaldelayBox.setValue(p.get("final_delay", 0.0) * 1e9)
        self.moffsetBox.setValue(p.get("mw_offset", 0.0) * 1e9)

    def apply_fg_widgets(self, params=None):
        if params is None:
            if "fg" in self.data.params:
                params = self.data.params["fg"]
            else:
                return

        for b in self.fg_buttons.buttons():
            b.setChecked(False)
        fgmodeButton = [
            b for b, m in self.get_fg_mode_dict() if m == params.get("mode", "disable")
        ][0]
        fgmodeButton.setChecked(True)

        wave_dict = {"Sinusoid": 0, "Square": 1}
        self.fg_waveBox.setCurrentIndex(wave_dict[params.get("wave", "Sinusoid")])
        self.fg_freqBox.setValue(params.get("freq", 1e6) * 1e-3)  # Hz to kHz
        self.fg_amplBox.setValue(params.get("ampl", 0.0))
        self.fg_phaseBox.setValue(params.get("phase", 0.0))

    def apply_plot_widgets(self):
        if "plot" in self.data.params:
            p = self.data.params["plot"]
        else:
            self.logger.error("params['plot'] is not found.")
            p = {}

        # avoid sending many requests
        self._wire_plot_widgets(False)

        self.set_plot_mode(p.get("plotmode", "data01"))
        self.set_tau_mode(p.get("taumode", "raw"))
        self.set_ref_mode(p.get("refmode", "ignore"))

        self.logXBox.setChecked(p.get("logX", False))
        self.logYBox.setChecked(p.get("logY", False))
        self.flipYBox.setChecked(p.get("flipY", False))

        self.sigdelayBox.setValue(p.get("sigdelay", 0.0) * 1e9)
        self.sigwidthBox.setValue(p.get("sigwidth", 100.0) * 1e9)
        self.refdelayBox.setValue(p.get("refdelay", 100.0) * 1e9)
        self.refwidthBox.setValue(p.get("refwidth", 100.0) * 1e9)
        self.refavgBox.setChecked(p.get("refaverage", False))

        self._wire_plot_widgets(True)

        self.update_plot_params()

    def add_param_table_params(self, params: dict):
        pt = P.unwrap(self.paramTable.params())
        params["pulse"] = pt["pulse"]
        if "90pulse" in self._params["pulse"]:
            params["pulse"]["90pulse"] = self.t90pulseBox.value() * 1e-9
        if "180pulse" in self._params["pulse"]:
            params["pulse"]["180pulse"] = self.t180pulseBox.value() * 1e-9
        if "270pulse" in self._params["pulse"]:
            params["pulse"]["270pulse"] = self.t270pulseBox.value() * 1e-9
        if "pd" in pt:
            params["pd"] = pt["pd"]
        if "awg" in pt:
            params["awg"] = pt["awg"]

    def get_params(self) -> tuple[dict, str]:
        label = self.methodBox.currentText()
        params = {}
        # fundamentals
        params["quick_resume"] = self.quickresumeBox.isChecked()
        params["freq"] = self.freqBox.value() * 1e6  # MHz to Hz
        params["power"] = self.powerBox.value()
        params["timebin"] = self.binBox.value() * 1e-9  # ns to sec
        params["interval"] = self.intervalBox.value() * 1e-3  # ms to sec
        params["sweeps"] = self.sweepsBox.value()
        params["duration"] = self.durationBox.value()
        params["roi_head"] = self.roiheadBox.value() * 1e-9  # ns to sec
        params["roi_tail"] = self.roitailBox.value() * 1e-9  # ns to sec
        params["multi_histogram"] = self.multiHistoBox.isChecked()

        if "freq1" in self._params:
            params["freq1"] = self.freq1Box.value() * 1e6  # MHz to Hz
            params["power1"] = self.power1Box.value()
            params["nomw1"] = self.nomw1Box.isChecked()

        # common_pulses
        params["base_width"] = self.basewidthBox.value() * 1e-9
        params["laser_delay"] = self.ldelayBox.value() * 1e-9
        params["laser_width"] = self.lwidthBox.value() * 1e-9
        params["mw_delay"] = self.mdelayBox.value() * 1e-9
        params["trigger_width"] = self.trigwidthBox.value() * 1e-9
        params["init_delay"] = self.initdelayBox.value() * 1e-9
        params["final_delay"] = self.finaldelayBox.value() * 1e-9
        ## global mw offset
        params["mw_offset"] = self.moffsetBox.value() * 1e-9

        # common switches
        params["invert_sweep"] = self.invertsweepBox.isChecked()
        params["nomw"] = self.nomwBox.isChecked()
        params["enable_reduce"] = self.reduceBox.isChecked()
        params["divide_block"] = self.divideblockBox.isChecked()
        params["partial"] = self.partialBox.currentIndex() - 1

        ## sweep params (tau / N)
        if "Nstart" in self._params:
            params["Nstart"] = self.NstartBox.value()
            params["Nnum"] = self.NnumBox.value()
            params["Nstep"] = self.NstepBox.value()
        else:
            params["start"] = self.startBox.value() * 1e-9  # ns to sec
            params["num"] = self.numBox.value()
            params["step"] = self.stepBox.value() * 1e-9  # ns to sec
            params["log"] = self.logBox.isChecked()

        self.add_param_table_params(params)
        params["plot"] = self.get_plot_params()
        params["fg"] = self.get_fg_params()

        return params, label

    def get_fg_params(self):
        params = {}
        params["mode"] = self.get_fg_mode()
        params["wave"] = self.fg_waveBox.currentText()
        params["freq"] = self.fg_freqBox.value() * 1e3  # kHz to Hz
        params["ampl"] = self.fg_amplBox.value()
        params["phase"] = self.fg_phaseBox.value()
        return params

    def get_plot_params(self):
        params = {}
        params["plotmode"] = self.plotmodeBox.currentText()
        params["taumode"] = self.taumodeBox.currentText()

        params["logX"] = self.logXBox.isChecked()
        params["logY"] = self.logYBox.isChecked()
        params["flipY"] = self.flipYBox.isChecked()

        # ns to sec
        params["sigdelay"] = self.sigdelayBox.value() * 1e-9
        params["sigwidth"] = self.sigwidthBox.value() * 1e-9
        params["refdelay"] = self.refdelayBox.value() * 1e-9
        params["refwidth"] = self.refwidthBox.value() * 1e-9
        params["refmode"] = self.refmodeBox.currentText()
        params["refaverage"] = self.refavgBox.isChecked()
        return params

    def get_fg_mode(self):
        return [m for b, m in self.get_fg_mode_dict() if b.isChecked()][0]

    def get_plottable_data(self) -> list[tuple[PODMRData, bool, Colors]]:
        return self.fit.get_plottable_data(self.data)

    # State managements
    def request_stop(self):
        self.cli.stop()

    def request_start(self):
        result = self.prepare_timing_params()
        if result is not None:
            self.start(*result)

    def request_validate(self):
        result = self.prepare_timing_params()
        if result is None:
            return
        if self.cli.validate(*result):
            QtWidgets.QMessageBox.information(
                self, "Pulse validation", "Pulse parameters are valid."
            )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Pulse validation failed",
                "Pulse parameters are invalid. See the log for details.",
            )

    def start(self, params: dict, label: str):
        """start the measurement."""

        title = "Continue?"
        body = (
            "Continue with current data?"
            + " Press No to clear current data and start a new measurement."
        )
        if (
            self.enableresumeBox.isChecked()
            and self.data.can_resume(params, label)
            and Qt.question_yn(self, title, body)
        ):
            params["resume"] = True
            params["ident"] = self.data.ident
        else:
            params["resume"] = False
            params["ident"] = uuid.uuid4()

        if not params["resume"] and self.data.is_finalized() and not self.data.is_saved():
            if not Qt.question_yn(
                self, "Sure to start?", "Current data has not been saved. Are you sure to discard?"
            ):
                return

        self.autosave.init_autosave()

        self.cli.start(params, label)

    def update_data(self, data: PODMRData):
        self.data = data
        if not self.data.has_data():
            return  # measurement has not been started yet or data has not been accumulated.

        self.refresh_plot()
        self.update_widgets()

        if self.data.running and self._meas_state == BinaryState.ACTIVE:
            # self._meas_state guards the case loaded data has running == True
            # such data is generated by autosave or user script that requests save of
            # data which is yet to be finalized.
            self.autosave.check_autosave(self.data)

    def update_buffer(self, buffer: Buffer[tuple[str, PODMRData]]):
        self.fit.update_buffer(buffer)
        self.refresh_plot()

    def _update_elapsed_time(self):
        if self.data is None:
            return

        t0 = self.data.start_time
        if self.data.finish_time is not None:
            t1 = self.data.finish_time
        elif self._meas_state == BinaryState.ACTIVE:
            t1 = time.time()
        else:  # unfinalized data is loaded and not running. skip real-time update.
            return

        elapsed = t1 - t0
        h, m, s = seconds_to_hms(elapsed)
        self.elapsedtimeLabel.setText(f"Elapsed time: {h:02d}:{m:02d}:{s:02d} ({elapsed:.1f} sec)")

    def _update_swept_label(self):
        if self.data.tdc_status is None:
            return
        sweep_num = int(self.data.tdc_status[1])
        self.sweptLabel.setText(f"{sweep_num} swept")

    def _update_range_bin(self):
        tbin = self.data.get_bin()
        trange = self.data.get_range()

        if not self.binBox.isEnabled():
            # if binBox is disabled, state is ACTIVE
            # Don't update these when the state is IDLE
            if tbin is not None:
                self.binBox.setValue(tbin * 1e9)  # sec to ns
            if trange is not None:
                self.rangeLabel.setText("range: {:.2f} us".format(trange * 1e6))

    def update_widgets(self):
        self._update_elapsed_time()
        self._update_swept_label()
        self._update_save_button()
        self._update_range_bin()

    def update_plot_params(self):
        if self.autoadjustBox.isChecked():
            tbin = self.data.get_bin()
            if tbin is not None:
                self.adjust_analysis_timing(self.data.get_bin() * 1e9)
        self.cli.update_plot_params(self.get_plot_params())

    def discard_data(self):
        if self.supports_discard():
            self.cli.discard()

    def find_laser_timing(self):
        if not self.supports_find_laser_timing():
            return
        scope = (self.scopeHeadBox.value() * 1e-9, self.scopeTailBox.value() * 1e-9)
        smooth_window = self.smoothWindowBox.value()
        fraction = self.fractionBox.value()
        monotonic = self.monotonicBox.isChecked()
        self.cli.find_laser_timing(scope, smooth_window, fraction, monotonic)

    def clear_laser_timing(self):
        if not self.supports_find_laser_timing():
            return
        self.cli.clear_laser_timing()

    def refresh_plot(self):
        self.plot.refresh(self.get_plottable_data(), self.data)
        self.alt_plot.refresh(self.get_plottable_data(), self.data)
        self.raw_plot.refresh(self.data)

    def finalize(self, data: PODMRData):
        if self._finalizing:
            return
        self._finalizing = True
        # NOP for now
        self._finalizing = False

    def update_cond_widgets(self, force_disable=False):
        """Update enable/disable state of widgets depending on param existence."""

        if self._params is None:
            return
        name_widgets = [
            ("Nstart", self.NstartBox),
            ("Nstep", self.NstepBox),
            ("Nnum", self.NnumBox),
            ("start", self.startBox),
            ("step", self.stepBox),
            ("num", self.numBox),
            ("log", self.logBox),
            ("freq1", self.freq1Box),
            ("power1", self.power1Box),
            ("nomw1", self.nomw1Box),
        ]
        if force_disable:
            for _, w in name_widgets:
                w.setEnabled(False)
        else:
            set_enabled(self._params, name_widgets)

        name_widgets = [
            ("90pulse", self.t90pulseBox),
            ("180pulse", self.t180pulseBox),
            ("270pulse", self.t270pulseBox),
        ]
        if force_disable:
            for _, w in name_widgets:
                w.setEnabled(False)
        else:
            set_enabled(self._params["pulse"], name_widgets)

    def _update_state_common(self, state: BinaryState, last_state: BinaryState):
        # Sweep widgets' enable/disable depends on selected method
        if state == BinaryState.IDLE:
            if last_state == BinaryState.ACTIVE:
                self.update_cond_widgets()
        else:
            self.update_cond_widgets(force_disable=True)

        if self.has_fg():
            for w in (self.fg_disableButton, self.fg_cwButton, self.fg_gateButton):
                w.setEnabled(state == BinaryState.IDLE)

            # Widgets enable/disable of which depends on selected fg mode
            if state == BinaryState.IDLE:
                self.switch_fg()
            else:
                for w in (self.fg_waveBox, self.fg_freqBox, self.fg_amplBox, self.fg_phaseBox):
                    w.setEnabled(False)

        if self.supports_discard():
            enable_discard = state == BinaryState.ACTIVE and self.enablediscardBox.isChecked()
            self.discardButton.setEnabled(enable_discard)

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)

        self.autosave.update_state(state, last_state)

        self._meas_state = state

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.startButton,
            self.validateButton,
            self.saveButton,
            self.exportButton,
            self.exportaltButton,
            self.loadButton,
            self.enableresumeBox,
            self.quickresumeBox,
            self.binBox,
            self.freqBox,
            self.powerBox,
            self.nomwBox,
            self.methodBox,
            self.partialBox,
            self.intervalBox,
            self.sweepsBox,
            self.durationBox,
            self.roiheadBox,
            self.roitailBox,
            self.multiHistoBox,
            self.invertsweepBox,
            self.reduceBox,
            self.divideblockBox,
            self.basewidthBox,
            self.ldelayBox,
            self.lwidthBox,
            self.mdelayBox,
            self.trigwidthBox,
            self.initdelayBox,
            self.finaldelayBox,
            self.moffsetBox,
            self.roundTimingBox,
            self.paramTable,
        ):
            w.setEnabled(state == BinaryState.IDLE)

        self._update_state_common(state, last_state)

    # helper functions

    def has_fg(self):
        return self._has_fg

    def is_awg_mode(self) -> bool:
        return self._params is not None and "awg" in self._params

    def timing_boxes(self):
        return (
            self.startBox,
            self.stepBox,
            self.basewidthBox,
            self.ldelayBox,
            self.lwidthBox,
            self.mdelayBox,
            self.trigwidthBox,
            self.initdelayBox,
            self.finaldelayBox,
            self.moffsetBox,
            self.t90pulseBox,
            self.t180pulseBox,
            self.t270pulseBox,
        )

    def timing_params_key(self):
        """Return the parameters that can affect timing information."""

        label = self.methodBox.currentText()
        params = P.unwrap(self.paramTable.params())
        awg_rate = params.get("awg", {}).get("rate")
        return label, awg_rate

    def update_timing_info(self, force: bool = False) -> TimingInfo | None:
        """Fetch timing information and update timing controls when necessary."""

        # The current non-AWG backend derives TimingInfo solely from its static pg_freq
        # configuration. Reuse a successful result to avoid requests while the node is busy.
        # Remove this short-circuit if non-AWG timing becomes parameter-dependent.
        if not self.is_awg_mode() and self._timing_info is not None:
            return self._timing_info

        key = self.timing_params_key()
        if not force and key == self._timing_params_key:
            return self._timing_info

        params, label = self.get_params()
        self._timing_params_key = key
        self._timing_info = self.cli.get_timing_info(params, label)
        self.update_timing_freq_label()
        self.update_timing_box_step()
        return self._timing_info

    def update_timing_freq_label(self):
        name = "AWG digital rate" if self.is_awg_mode() else "PG freq"
        if self._timing_info is None:
            self.pgfreqLabel.setText(f"{name}: Unknown")
            return

        freq = SI_format(self._timing_info.pg_freq, precision=4, suffix="Hz")
        period = SI_format(self._timing_info.period, precision=4, suffix="s")
        self.pgfreqLabel.setText(f"{name}: {freq} ({period} period)")

    def update_timing_box_step(self):
        if self._timing_info is None:
            return

        step = self._timing_info.period * 1e9
        decimals = decimals_for_period(step)
        for b in self.timing_boxes():
            b.setDecimals(max(b.decimals(), decimals))
            b.setSingleStep(step)

    def round_timing_box_values(self):
        """Round timing parameters to the current digital clock period."""

        if not self.roundTimingBox.isChecked() or self._timing_info is None:
            return

        period_ns = self._timing_info.period * 1e9
        for b in self.timing_boxes():
            b.setValue(round(b.value() / period_ns) * period_ns)

    def prepare_timing_params(self) -> tuple[dict, str] | None:
        """Fetch timing information, optionally round values, and rebuild parameters."""

        if self.update_timing_info(force=True) is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Timing information unavailable",
                "Failed to determine the digital timing resolution.",
            )
            return None

        self.round_timing_box_values()
        return self.get_params()

    def set_method(self, method: str):
        i = self.methodBox.findText(method)
        if i >= 0:
            self.methodBox.setCurrentIndex(i)
        else:
            self.logger.error(f"Unknown method: {method}")

    def set_plot_mode(self, mode: str):
        i = self.plotmodeBox.findText(mode)
        if i >= 0:
            self.plotmodeBox.setCurrentIndex(i)
        else:
            self.logger.error(f"Unknown plot mode: {mode}")

    def set_tau_mode(self, mode: str):
        i = self.taumodeBox.findText(mode)
        if i >= 0:
            self.taumodeBox.setCurrentIndex(i)
        else:
            self.logger.error(f"Unknown tau mode: {mode}")

    def reset_tau_modes(self, modes: list[str]):
        prev_mode = self.taumodeBox.currentText()
        if (prev_mode == "total" and "total" not in modes) or (
            prev_mode == "freq" and "freq" not in modes
        ):
            prev_mode = "raw"
        self.taumodeBox.currentIndexChanged.disconnect(self.update_plot_params)
        self.taumodeBox.clear()
        self.taumodeBox.addItems(modes)
        self.taumodeBox.currentIndexChanged.connect(self.update_plot_params)
        self.set_tau_mode(prev_mode)

    def reset_partial_modes(self, max_partial: int):
        """Reset partial selector entries for the current method.

        Worker currently exposes max_partial for N=2 or N=4 methods.

        """

        partial_labels = [
            "Complementary",
            "Pattern 0 only",
            "Pattern 1 only",
            "Pattern 2 only",
            "Pattern 3 only",
        ]
        max_partial = max(1, min(int(max_partial), 3))
        new_count = max_partial + 2

        prev_index = self.partialBox.currentIndex()
        self.partialBox.currentIndexChanged.disconnect(self.switch_partial)
        self.partialBox.clear()
        self.partialBox.addItems(partial_labels[:new_count])
        self.partialBox.currentIndexChanged.connect(self.switch_partial)

        if 0 <= prev_index < new_count:
            self.partialBox.setCurrentIndex(prev_index)
        else:
            self.partialBox.setCurrentIndex(0)

    def reset_plot_modes(self, modes: list[str], default_mode: str = "data01"):
        prev_mode = self.plotmodeBox.currentText()
        if prev_mode not in modes:
            prev_mode = default_mode if default_mode in modes else modes[0]
        self.plotmodeBox.currentIndexChanged.disconnect(self.update_plot_params)
        self.plotmodeBox.clear()
        self.plotmodeBox.addItems(modes)
        self.plotmodeBox.currentIndexChanged.connect(self.update_plot_params)
        self.set_plot_mode(prev_mode)

    def set_ref_mode(self, mode: str):
        i = self.refmodeBox.findText(mode)
        if i >= 0:
            self.refmodeBox.setCurrentIndex(i)
        else:
            self.logger.error(f"Unknown ref mode: {mode}")

    def get_fg_mode_dict(self):
        return [
            (self.fg_disableButton, "disable"),
            (self.fg_cwButton, "cw"),
            (self.fg_gateButton, "gate"),
        ]


class PODMRWidget(PODMRWidgetBase, Ui_PODMR):
    """Widget for Pulse ODMR."""

    def setup_measurement_ui(self):
        Ui_PODMR.setupUi(self, self)


class PODMRMainWindow(QtWidgets.QMainWindow):
    """MainWindow with PODMRWidget and PlotWidget."""

    def __init__(self, gconf: dict, name, context, parent=None, logger=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)

        lconf = local_conf(gconf, name)
        target = lconf["target"]
        enable_resume = V.check_bool(lconf.get("enable_resume", True), "enable_resume")

        self.plot = PlotWidget(parent=self, logger=self.logger)
        self.alt_plot = AltPlotWidget(parent=self, logger=self.logger)
        self.raw_plot = RawPlotWidget(parent=self, logger=self.logger)
        self.podmr = PODMRWidget(
            gconf,
            target["podmr"],
            target["gparams"],
            enable_resume,
            self.plot,
            self.alt_plot,
            self.raw_plot,
            context,
            parent=self,
            logger=self.logger,
            file_transport_dir=lconf.get("file_transport_dir"),
        )

        self.setWindowTitle(f"MAHOS.PODMRGUI ({join_name(target['podmr'])})")
        self.setAnimated(False)
        self.setCentralWidget(self.podmr)
        self.d_plot = QtWidgets.QDockWidget("Plot", parent=self)
        self.d_plot.setWidget(self.plot)
        self.d_alt_plot = QtWidgets.QDockWidget("Alt Plot", parent=self)
        self.d_alt_plot.setWidget(self.alt_plot)
        self.d_raw_plot = QtWidgets.QDockWidget("Raw Plot", parent=self)
        self.d_raw_plot.setWidget(self.raw_plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.d_raw_plot)
        self.tabifyDockWidget(self.d_raw_plot, self.d_alt_plot)
        self.tabifyDockWidget(self.d_alt_plot, self.d_plot)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_plot.toggleViewAction())
        self.view_menu.addAction(self.d_alt_plot.toggleViewAction())
        self.view_menu.addAction(self.d_raw_plot.toggleViewAction())

        self.option_menu = self.menuBar().addMenu("Option")
        act = QtGui.QAction("Auto range", parent=self.option_menu)
        act.setCheckable(True)
        act.setChecked(self.plot.auto_range())
        self.option_menu.addAction(act)
        act.toggled.connect(self.plot.set_auto_range)

    def closeEvent(self, event):
        """Close the clients on closeEvent."""

        self.podmr.close_clients()
        # self.plot.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)


class PODMRGUI(GUINode):
    """GUINode for Pulse ODMR using PODMRWidget.

    :param target.log: Optional LogBroker target for formal GUI logging.
    :type target.log: tuple[str, str] | str
    :param log_level: (default: ``"INFO"``) Optional logging level.
    :type log_level: str
    :param target.podmr: Target PODMR node full name.
    :type target.podmr: tuple[str, str] | str
    :param target.gparams: Target GlobalParams node full name.
    :type target.gparams: tuple[str, str] | str
    :param enable_resume: (default: True) Initial state of "Enable resume" CheckBox.
    :type enable_resume: bool
    :param file_transport_dir: Optional GUI-side directory for shared file transport.
    :type file_transport_dir: str

    """

    def init_widget(self, gconf: dict, name, context):
        return PODMRMainWindow(gconf, name, context, logger=self.logger)
