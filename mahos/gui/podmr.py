#!/usr/bin/env python3

"""
GUI frontend of Pulse ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import typing as T
import os
import time
from datetime import datetime
import uuid

import numpy as np
import pyqtgraph as pg

from . import Qt
from .Qt import QtCore, QtWidgets, QtGui

from .ui.podmr import Ui_PODMR
from .ui.podmr_nmr_table import Ui_NMRTable
from .ui.podmr_autosave import Ui_PODMRAutoSave
from .podmr_client import QPODMRClient

from ..msgs.common_msgs import BinaryState
from ..msgs.common_meas_msgs import Buffer
from ..msgs import param_msgs as P
from ..msgs.podmr_msgs import PODMRStatus, PODMRData
from ..node.global_params import GlobalParamsClient
from .gui_node import GUINode
from .common_widget import ClientWidget
from .fit_widget import FitWidget
from .param import set_enabled, apply_widgets
from .dialog import save_dialog, load_dialog, export_dialog
from ..node.node import local_conf, join_name
from ..util.plot import colors_tab20_pair
from ..util.timer import seconds_to_hms
from ..util.math_phys import round_halfint, round_evenint
from ..util.conv import real_fft


Policy = QtWidgets.QSizePolicy.Policy


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
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
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
            except ValueError as e:
                print("Error getting xdata: " + repr(e))
                continue

            try:
                y0, y1 = data.get_ydata()
                if y0 is None:
                    return
                yfit = data.get_fit_ydata()
            except ValueError as e:
                print("Error getting ydata: " + repr(e))
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
        except TypeError as e:
            # import sys, traceback; traceback.print_tb(sys.exc_info()[2])
            print("Error in plot_analyzed " + repr(e))

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
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

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
            except ValueError as e:
                print("Error getting xdata: " + repr(e))
                continue
            if len(x) == 1:
                continue
            try:
                y0, y1 = data.get_ydata()
                if y0 is None:
                    return
            except ValueError as e:
                print("Error getting ydata: " + repr(e))
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
        except TypeError as e:
            # import sys, traceback; traceback.print_tb(sys.exc_info()[2])
            print("Error in plot_fft " + repr(e))

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
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

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

    def plot_raw(self, data: PODMRData):
        # sig_head, sig_tail, ref_head, ref_tail
        BRUSHES = ((255, 0, 0), (255, 128, 0), (0, 0, 255), (0, 128, 255))

        def plot_markers_roi(rx, ry, start, stop):
            for inds, brush in zip(data.marker_indices, BRUSHES):
                x = []
                y = []
                try:
                    for i, ((roi_start, _), idx) in enumerate(
                        zip(data.get_rois()[start:stop], inds[start:stop])
                    ):
                        x.append(rx[start + i][idx - roi_start])
                        y.append(ry[start + i][idx - roi_start])
                except IndexError:
                    continue
                self.raw_plot.plot(
                    x, y, pen=None, symbolPen=None, symbol="o", symbolSize=8, symbolBrush=brush
                )

        def plot_markers(rx, ry, start, stop):
            for inds, brush in zip(data.marker_indices, BRUSHES):
                x = np.array([rx[i] for i in inds[start:stop]])
                y = np.array([ry[i] for i in inds[start:stop]])
                self.raw_plot.plot(
                    x, y, pen=None, symbolPen=None, symbol="o", symbolSize=8, symbolBrush=brush
                )

        if not self.isVisible():
            return

        self.raw_plot.clear()

        if not data.has_raw_data() or not self.showBox.isChecked():
            self.raw_plot.plot([0, 1], [0, 0])
            return

        rx = data.get_raw_xdata()
        ry = data.raw_data
        if rx is None or ry is None:
            return

        laser_pulses = len(data.marker_indices[0])
        self.indexBox.setMaximum(laser_pulses - 1)
        self.numBox.setMaximum(laser_pulses)

        if self.allBox.isChecked():
            # show whole raw data
            if data.has_roi():
                self.raw_plot.plot(np.concatenate(rx), np.concatenate(ry))
                plot_markers_roi(rx, ry, 0, None)
            else:
                self.raw_plot.plot(rx, ry)
                plot_markers(rx, ry, 0, None)
            return

        lstart = self.indexBox.value()
        lstop = min(lstart + self.numBox.value() - 1, laser_pulses - 1)

        if data.has_roi():
            self.raw_plot.plot(
                np.concatenate(rx[lstart : lstop + 1]), np.concatenate(ry[lstart : lstop + 1])
            )
            plot_markers_roi(rx, ry, lstart, lstop + 1)
        else:
            margin = self.marginBox.value()
            head = data.marker_indices[0][lstart] - margin
            tail = data.marker_indices[3][lstop] + margin
            self.raw_plot.plot(rx[head:tail], ry[head:tail])
            plot_markers(rx, ry, lstart, lstop + 1)

    def refresh(self, data: PODMRData):
        try:
            self.plot_raw(data)
        except TypeError as e:
            # import sys, traceback; traceback.print_tb(sys.exc_info()[2])
            print("Error in plot_raw " + repr(e))

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
    def __init__(self, cli, gparams_cli, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

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
        fn = self.get_filename()
        self.cli.save_data(fn, params={"tmp": True})
        self.suffixBox.setValue(self.suffixBox.value() + 1)

    def get_filename(self):
        dirname = self.dirEdit.text()
        fname = "{:s}_{:04d}.podmr.pkl".format(self.fnEdit.text(), self.suffixBox.value())
        return os.path.join(dirname, fname)

    def init_autosave(self):
        if not self.enableBox.isChecked():
            self.lastLabel.setText("Last saved: ")
            self.agoLabel.setText("()")
            return

        self.last_saved_at = datetime.now()
        if self.resetBox.isChecked():
            self.suffixBox.setValue(0)

        if os.path.isfile(self.get_filename()):
            if not Qt.question_yn(
                self, "Overwrite filename?", "File name overlap for autosave. Overwrite?"
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
        for w in (
            self.dirEdit,
            self.browseButton,
            self.fnEdit,
            self.suffixBox,
            self.resetBox,
            self.intervalBox,
            self.enableBox,
        ):
            w.setEnabled(state == BinaryState.IDLE)

    def browse_dir(self):
        current = str(self.gparams_cli.get_param("work_dir"))
        dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Select autosave directory", current)
        if dn and current != dn:
            self.dirEdit.setText(dn)


class PODMRWidget(ClientWidget, Ui_PODMR):
    """Widget for Pulse ODMR."""

    def __init__(
        self,
        gconf: dict,
        name,
        gparams_name,
        plot: PlotWidget,
        alt_plot: AltPlotWidget,
        raw_plot: RawPlotWidget,
        context,
        parent=None,
    ):
        ClientWidget.__init__(self, parent)
        self.setupUi(self)

        self.conf = local_conf(gconf, name)

        self._finalizing = False
        self._has_fg = False
        self._found_sg2 = False
        self._pg_freq = None
        self._params = None

        self.init_radiobuttons()
        self.tabWidget.setCurrentIndex(0)
        self.plot = plot
        self.alt_plot = alt_plot
        self.raw_plot = raw_plot

        self.data = PODMRData()

        self.cli = QPODMRClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli, self.gparams_cli)

        self._fiTab_layout = QtWidgets.QVBoxLayout(self.fiTab)
        self.fit = PODMRFitWidget(self.cli, self.gparams_cli, parent=self.fiTab)
        self._fiTab_layout.addWidget(self.fit)

        self._autosaveTab_layout = QtWidgets.QVBoxLayout(self.autosaveTab)
        self.autosave = PODMRAutoSaveWidget(self.cli, self.gparams_cli, parent=self.autosaveTab)
        self._autosaveTab_layout.addWidget(self.autosave)
        self.autosave.init_connection()

        self._table_layout = QtWidgets.QVBoxLayout(self.tableTab)
        self.table = NMRTableWidget(parent=self.tableTab)
        self._table_layout.addWidget(self.table)
        self.table.init_connection(self.update_table)

        self.setEnabled(False)

    def init_radiobuttons(self):
        def set_group(group, index, buttons):
            for b in buttons:
                group.addButton(b)
            group.setExclusive(True)
            buttons[index].setChecked(True)

        # options
        self.fg_buttons = QtWidgets.QButtonGroup(parent=self)
        set_group(self.fg_buttons, 0, [b for b, l in self.get_fg_mode_dict()])

    def init_with_status(self, status: PODMRStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self._pg_freq = status.pg_freq
        self.pgfreqLabel.setText(f"PG freq: {self._pg_freq*1e-9:.2f} GHz")
        self.update_timing_box_step()

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
        self.stopButton.clicked.connect(self.request_stop)
        self.saveButton.clicked.connect(self.save_data)
        self.exportButton.clicked.connect(self.export_data)
        self.exportaltButton.clicked.connect(self.export_alt_data)
        self.loadButton.clicked.connect(self.load_data)

        self.discardButton.clicked.connect(self.discard_data)

        # main tab
        for w in (self.startBox, self.stepBox, self.numBox):
            w.valueChanged.connect(self.update_stop)
        for w in (self.NstartBox, self.NstepBox, self.NnumBox):
            w.valueChanged.connect(self.update_Nstop)

        self.methodBox.currentIndexChanged.connect(self.switch_method)
        self.partialBox.currentIndexChanged.connect(self.switch_partial)

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
                ("divide_block", self.divideblockBox),
                ("base_width", self.basewidthBox, 1e9),
                ("laser_delay", self.ldelayBox, 1e9),
                ("laser_width", self.lwidthBox, 1e9),
                ("mw_delay", self.mdelayBox, 1e9),
                ("trigger_width", self.trigwidthBox, 1e9),
                ("init_delay", self.initdelayBox, 1e9),
                ("final_delay", self.finaldelayBox, 1e9),
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
        self.stopLabel.setText(f"stop: {stop:.1f} ns")

    def update_Nstop(self):
        Nstop = self.NstartBox.value() + self.NstepBox.value() * (self.NnumBox.value() - 1)
        self.NstopLabel.setText(f"stop: {Nstop:d}")

    def switch_fg(self):
        disabled = self.fg_disableButton.isChecked()
        for w in (self.fg_waveBox, self.fg_freqBox, self.fg_amplBox):
            w.setEnabled(not disabled)
        self.fg_phaseBox.setEnabled(self.fg_gateButton.isChecked())

    def _apply_sg2(self, params: dict):
        """Check existence of SG2 (freq2 in params) and apply bound of SG2."""

        if self._found_sg2:
            return
        if "freq2" in params:
            apply_widgets(params, [("freq2", self.freq2Box, 1e-6), ("power2", self.power2Box)])
            self._found_sg2 = True

    def switch_method(self):
        method = self.methodBox.currentText()
        self._params = self.cli.get_param_dict(method)
        self.update_cond_widgets()
        self._apply_sg2(self._params)
        self.paramTable.update_contents(self._params["pulse"])
        self.reset_tau_modes(self._params["plot"]["taumode"].options())
        self.plot.update_label(self.data)

    def switch_partial(self, index):
        # when partial is 0 (index 1), plotmode to data0 (index 1)
        # when partial is 1 (index 2), plotmode to data1 (index 2)
        # when partial is 2 (index 3), plotmode to data0 (index 1)

        if index in (1, 2):
            self.plotmodeBox.setCurrentIndex(index)
        elif index == 3:
            self.plotmodeBox.setCurrentIndex(1)

    def update_save_button(self, saved):
        if saved:
            self.saveButton.setStyleSheet("")
        else:
            self.saveButton.setStyleSheet("background-color: #FF0000")

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
        fn = save_dialog(self, default_path, "PODMR", ".podmr")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.gparams_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)
        self.update_save_button(True)

        n = os.path.splitext(fn)[0] + ".png"
        params = {"show_fit": self.fit.show_current_data_fit()}
        self.cli.export_data(n, params=params)

    def export_data(self):
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "PODMR", (".png", ".pdf", ".eps", ".txt"))
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
        fn = export_dialog(self, default_path, "PODMR", (".png", ".pdf", ".eps", ".txt"))
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
        self.update_save_button(True)

        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "PODMR", ".podmr")
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
    def apply_meas_widgets(self):
        p = self.data.params

        # TDC
        self.binBox.setValue(p.get("timebin", 0.0) * 1e9)  # sec to ns
        self.intervalBox.setValue(int(round(p.get("interval", 0.0) * 1e3)))  # sec to ms
        self.sweepsBox.setValue(p.get("sweeps", 0))
        self.durationBox.setValue(p.get("duration", 0.0))
        self.roiheadBox.setValue(p.get("roi_head", -1e-9) * 1e9)  # sec to ns
        self.roitailBox.setValue(p.get("roi_tail", -1e-9) * 1e9)  # sec to ns

        # method
        self.set_method(self.data.label)

        for k, v in p["pulse"].items():
            self.paramTable.apply_value(k, v)

        # MW
        self.freqBox.setValue(p.get("freq", 2740e6) * 1e-6)  # Hz to MHz
        self.powerBox.setValue(p.get("power", 0.0))
        self.nomwBox.setChecked(p.get("nomw", False))
        if "freq2" in p:
            self.freq2Box.setValue(p["freq2"] * 1e-6)  # Hz to MHz
        if "power2" in p:
            self.power2Box.setValue(p["power2"])
        if "nomw2" in p:
            self.nomw2Box.setChecked(p["nomw2"])

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
        if partial in (0, 1):
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

    def apply_fg_widgets(self, params=None):
        if params is None:
            if "fg" in self.data.params:
                params = self.data.params["fg"]
            else:
                print("params['fg'] is not found.")
                params = {}

        for b in self.fg_buttons.buttons():
            b.setChecked(False)
        fgmodeButton = [
            b for b, m in self.get_fg_mode_dict() if m == params.get("mode", "disable")
        ][0]
        fgmodeButton.setChecked(True)

        wave_dict = {"Sinusoid": 0, "Square": 1}
        self.fg_waveBox.setCurrentIndex(wave_dict[params.get("wave", 0)])
        self.fg_freqBox.setValue(params.get("freq", 1e6) * 1e-3)  # Hz to kHz
        self.fg_amplBox.setValue(params.get("ampl", 0.0))
        self.fg_phaseBox.setValue(params.get("phase", 0.0))

    def apply_plot_widgets(self):
        if "plot" in self.data.params:
            p = self.data.params["plot"]
        else:
            print("params['plot'] is not found.")
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

        if "freq2" in self._params:
            params["freq2"] = self.freq2Box.value() * 1e6  # MHz to Hz
            params["power2"] = self.power2Box.value()
            params["nomw2"] = self.nomw2Box.isChecked()

        # common_pulses
        params["base_width"] = self.basewidthBox.value() * 1e-9
        params["laser_delay"] = self.ldelayBox.value() * 1e-9
        params["laser_width"] = self.lwidthBox.value() * 1e-9
        params["mw_delay"] = self.mdelayBox.value() * 1e-9
        params["trigger_width"] = self.trigwidthBox.value() * 1e-9
        params["init_delay"] = self.initdelayBox.value() * 1e-9
        params["final_delay"] = self.finaldelayBox.value() * 1e-9

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

        params["pulse"] = P.unwrap(self.paramTable.params())
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
        self.round_timing_box_values()

        if self.validate_pulse():
            self.start()

    def validate_pulse(self):
        return self.cli.validate(*self.get_params())

    def start(self):
        """start the measurement."""

        params, label = self.get_params()
        title = "Continue?"
        body = (
            "Continue with current data?"
            + " Press No to clear current data and start a new measurement."
        )
        if self.data.can_resume(params, label) and Qt.question_yn(self, title, body):
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

        self.update_save_button(True)

    def update_data(self, data: PODMRData):
        self.data = data
        if not self.data.has_data():
            return  # measurement has not been started yet or data has not been accumulated.

        self.refresh_plot()
        self.update_widgets()

        if self.data.running:
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
        else:
            t1 = time.time()

        elapsed = t1 - t0
        h, m, s = seconds_to_hms(elapsed)
        self.elapsedtimeLabel.setText(f"Elapsed time: {h:02d}:{m:02d}:{s:02d} ({elapsed:.1f} sec)")

    def _update_swept_label(self):
        if self.data.tdc_status is None:
            return
        sweep_num = int(self.data.tdc_status[1])
        self.sweptLabel.setText(f"{sweep_num} swept")

    def update_widgets(self):
        self._update_elapsed_time()
        self._update_swept_label()
        tbin = self.data.get_bin()
        trange = self.data.get_range()

        if not self.binBox.isEnabled():
            # if binBox is disabled, state is ACTIVE
            # Don't update these when the state is IDLE
            if tbin is not None:
                self.binBox.setValue(tbin * 1e9)  # sec to ns
            if trange is not None:
                self.rangeLabel.setText("range: {:.2f} us".format(trange * 1e6))

    def update_plot_params(self):
        if self.autoadjustBox.isChecked():
            tbin = self.data.get_bin()
            if tbin is not None:
                self.adjust_analysis_timing(self.data.get_bin() * 1e9)
        self.cli.update_plot_params(self.get_plot_params())

    def discard_data(self):
        self.cli.discard()

    def refresh_plot(self):
        self.plot.refresh(self.get_plottable_data(), self.data)
        self.alt_plot.refresh(self.get_plottable_data(), self.data)
        self.raw_plot.refresh(self.data)

    def finalize(self, data: PODMRData):
        if self._finalizing:
            return
        self._finalizing = True
        self.update_save_button(False)
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
            ("freq2", self.freq2Box),
            ("power2", self.power2Box),
            ("nomw2", self.nomw2Box),
        ]
        set_enabled(self._params, name_widgets)

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.startButton,
            self.saveButton,
            self.exportButton,
            self.exportaltButton,
            self.loadButton,
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
            self.paramTable,
        ):
            w.setEnabled(state == BinaryState.IDLE)

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

        self.discardButton.setEnabled(
            self.enablediscardBox.isChecked() and state == BinaryState.ACTIVE
        )
        self.stopButton.setEnabled(state == BinaryState.ACTIVE)

        self.autosave.update_state(state, last_state)

    # helper functions

    def has_fg(self):
        return self._has_fg

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
        )

    def update_timing_box_step(self):
        if round(self._pg_freq) == round(2.0e9):
            step = 0.5
        elif round(self._pg_freq) == round(1.0e9):
            step = 1.0
        elif round(self._pg_freq) == round(0.5e9):
            step = 2.0
        else:
            print(f"Cannot determine timing box step with PG freq {self._pg_freq*1e-9:.2f} GHz")
            step = 1.0

        for b in self.timing_boxes():
            b.setSingleStep(step)

    def round_timing_box_values(self):
        """check and round values of QDoubleSpinBox for timing parameters."""

        if round(self._pg_freq) == round(2.0e9):
            _round = round_halfint
        elif round(self._pg_freq) == round(1.0e9):
            _round = round
        elif round(self._pg_freq) == round(0.5e9):
            _round = round_evenint
        else:
            print(f"Cannot determine round method with PG freq {self._pg_freq*1e-9:.2f} GHz")
            _round = round

        for b in self.timing_boxes():
            b.setValue(_round(b.value()))

    def set_method(self, method: str):
        i = self.methodBox.findText(method)
        if i >= 0:
            self.methodBox.setCurrentIndex(i)
        else:
            print(f"[ERROR] unknown method: {method}")

    def set_plot_mode(self, mode: str):
        i = self.plotmodeBox.findText(mode)
        if i >= 0:
            self.plotmodeBox.setCurrentIndex(i)
        else:
            print(f"[ERROR] unknown plot mode: {mode}")

    def set_tau_mode(self, mode: str):
        i = self.taumodeBox.findText(mode)
        if i >= 0:
            self.taumodeBox.setCurrentIndex(i)
        else:
            print(f"[ERROR] unknown tau mode: {mode}")

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

    def set_ref_mode(self, mode: str):
        i = self.refmodeBox.findText(mode)
        if i >= 0:
            self.refmodeBox.setCurrentIndex(i)
        else:
            print(f"[ERROR] unknown ref mode: {mode}")

    def get_fg_mode_dict(self):
        return [
            (self.fg_disableButton, "disable"),
            (self.fg_cwButton, "cw"),
            (self.fg_gateButton, "gate"),
        ]


class PODMRMainWindow(QtWidgets.QMainWindow):
    """MainWindow with PODMRWidget and PlotWidget."""

    def __init__(self, gconf: dict, name, context, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        lconf = local_conf(gconf, name)
        target = lconf["target"]

        self.plot = PlotWidget(parent=self)
        self.alt_plot = AltPlotWidget(parent=self)
        self.raw_plot = RawPlotWidget(parent=self)
        self.podmr = PODMRWidget(
            gconf,
            target["podmr"],
            target["gparams"],
            self.plot,
            self.alt_plot,
            self.raw_plot,
            context,
            parent=self,
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
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.d_alt_plot)
        self.tabifyDockWidget(self.d_alt_plot, self.d_plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_raw_plot)

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
    """GUINode for Pulse ODMR using PODMRWidget."""

    def init_widget(self, gconf: dict, name, context):
        return PODMRMainWindow(gconf, name, context)
