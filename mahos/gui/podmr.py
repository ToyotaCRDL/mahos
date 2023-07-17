#!/usr/bin/env python3

"""
GUI frontend of Pulse ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import typing as T
import os
import time
import re
from datetime import datetime
import uuid

import numpy as np
import pyqtgraph as pg

from . import Qt
from .Qt import QtCore, QtWidgets

from .ui.podmr import Ui_PODMR
from .ui.podmr_indicator import Ui_PODMRIndicator
from .ui.podmr_autosave import Ui_PODMRAutoSave
from .podmr_client import QPODMRClient

from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import FloatParam
from ..msgs.podmr_msgs import PODMRData, is_CPlike
from ..node.param_server import ParamClient
from .gui_node import GUINode
from .common_widget import ClientWidget
from .fit_widget import FitWidget
from .param import set_enabled
from .dialog import save_dialog, load_dialog, export_dialog
from ..node.node import local_conf, join_name
from ..util.plot import colors_tab20_pair
from ..util.timer import seconds_to_hms
from ..util.math_phys import round_halfint


Policy = QtWidgets.QSizePolicy.Policy


class QSortingTableWidgetItem(QtWidgets.QTableWidgetItem):
    """for numerical sorting"""

    def __init__(self, value):
        super(QSortingTableWidgetItem, self).__init__("%s" % value)

    def __lt__(self, other):
        if isinstance(other, QSortingTableWidgetItem):
            selfDataValue = float(self.data(QtCore.Qt.EditRole))
            otherDataValue = float(other.data(QtCore.Qt.EditRole))
            return selfDataValue < otherDataValue
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


class PlotWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        pg.GraphicsLayoutWidget.__init__(self, parent)

        self.init_view()

    def sizeHint(self):
        return QtCore.QSize(1400, 1000)

    def init_view(self):
        self.plot = self.addPlot(row=0, col=0, lockAspect=False)
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
                    x, y0, pen=None, symbolPen=None, symbol="o", symbolSize=4, symbolBrush=c.color0
                )
                if y1 is not None:
                    self.plot.plot(
                        x,
                        y1,
                        pen=None,
                        symbolPen=None,
                        symbol="o",
                        symbolSize=4,
                        symbolBrush=c.color1,
                    )
                self.plot.plot(xfit, yfit, pen=c.color0, width=1)

            else:
                self.plot.plot(
                    x,
                    y0,
                    pen=c.color0,
                    width=1,
                    symbolPen=None,
                    symbol="o",
                    symbolSize=8,
                    symbolBrush=c.color0,
                )
                if y1 is not None:
                    self.plot.plot(
                        x,
                        y1,
                        pen=c.color1,
                        width=1,
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

    def update_fft_mode(self, enable):
        self.plot.ctrl.fftCheck.setChecked(enable)

    def enable_auto_range(self):
        self.plot.enableAutoRange()


class RawPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.init_widgets()

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

        for w in (self.indexBox, self.numBox, self.marginBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)

        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)

        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.showBox)
        hl.addWidget(self.allBox)
        hl.addWidget(self.indexBox)
        hl.addWidget(self.numBox)
        hl.addWidget(self.marginBox)
        hl.addItem(spacer)
        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl)
        vl.addWidget(glw)
        self.setLayout(vl)

    def plot_raw(self, data: PODMRData):
        def plot_markers(start, stop):
            brushes = ((255, 0, 0), (255, 128, 0), (0, 0, 255), (0, 128, 255))
            for inds, brush in zip(data.marker_indices, brushes):
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
            self.raw_plot.plot(rx, ry)
            plot_markers(0, None)
            return

        lstart = self.indexBox.value()
        lstop = min(lstart + self.numBox.value() - 1, laser_pulses - 1)
        margin = self.marginBox.value()
        head = data.marker_indices[0][lstart] - margin
        tail = data.marker_indices[3][lstop] + margin
        self.raw_plot.plot(rx[head:tail], ry[head:tail])
        plot_markers(lstart, lstop + 1)

    def refresh(self, data: PODMRData):
        try:
            self.plot_raw(data)
        except TypeError as e:
            # import sys, traceback; traceback.print_tb(sys.exc_info()[2])
            print("Error in plot_raw " + repr(e))


class PODMRIndicatorWidget(QtWidgets.QWidget, Ui_PODMRIndicator):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

    def init_connection(self, update_slot):
        for b in (
            self.staticfieldBox,
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
            e.textChanged.connect(update_slot)

    def convert_freq(f, TL):
        """frequency ==> delta, LO"""

        fTL = f * TL
        m = np.rint(fTL)
        d = np.abs(fTL - m) / TL
        f_LO = m / TL
        return d, f_LO

    def update(self, data: PODMRData):
        res = []  # No, Nuclei, Harmonics, Larmor freq., 2/(Larmor freq.), corresponding tau
        nuclei = ["1H", "13C", "14N", "15N", "19F", "31P", "ac", "ac2"]
        for nuc_label in nuclei:
            enabled = eval("self.ind%sBox" % nuc_label).isChecked()
            if not enabled:
                continue

            peaks, harmonics = self.calc_peak_position(nuc_label)
            t_pi = data.get_pulse_params()["180pulse"]
            tau = 1 / 4 / np.array(peaks) - t_pi / 2
            invfl = 2 / np.array(peaks)

            t_iter = self.TiterBox.value() * 1e-6
            delta, LO = self.convert_freq(np.array(peaks), t_iter)

            for i in range(len(peaks)):
                res.append(
                    (
                        nuc_label,
                        harmonics[i],
                        "%.6f" % (peaks[i] / 1e6),
                        "%.2f" % (invfl[i] * 1e9),
                        "%.2f" % (tau[i] * 1e9),
                        "%.3f" % (delta[i] * 1e-3),
                    )
                )

        res = [(str(i),) + r for i, r in enumerate(res)]  # insert the column of No.

        # rebuild table
        sorting = self.indtableWidget.isSortingEnabled()
        self.indtableWidget.setSortingEnabled(False)
        for i in range(self.indtableWidget.rowCount()):
            self.indtableWidget.removeRow(0)
        self.indtableWidget.setRowCount(len(res))

        for (i, j), s in np.ndenumerate(res):
            self.indtableWidget.setItem(i, j, QSortingTableWidgetItem(s))

        self.indtableWidget.resizeColumnsToContents()
        self.indtableWidget.setSortingEnabled(sorting)

    def calc_peak_position(self, nuc_label):
        """[(Larmor freq, 'harmonics label'), ...]"""
        staticfield = self.staticfieldBox.value() * 1e-3  # mT to T

        if nuc_label in ["ac", "ac2"]:
            lamor_freq = abs(eval("self.ind%sfreqBox" % nuc_label).value() * 1e6)  # MHz to Hz
        else:
            gamma = eval("self.ind%sgammaBox" % nuc_label).value() * 1e6  # MHz/T to Hz/T
            lamor_freq = abs(staticfield * gamma)  # [Hz]

        peaks_str = eval("self.ind%speakEdit" % nuc_label).text()
        peaks = []
        harmonics = peaks_str.replace(" ", "").split(",")
        for harm in harmonics:
            if harm.find("/") >= 0:  # rational number
                a, b = harm.split("/")
                h = float(a) / float(b)
            else:  # integer or float
                h = float(harm)

            peaks.append(lamor_freq * h)
        return peaks, harmonics


class PODMRAutoSaveWidget(QtWidgets.QWidget, Ui_PODMRAutoSave):
    def __init__(self, cli, param_cli, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.cli = cli
        self.param_cli = param_cli

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
        current = str(self.param_cli.get_param("work_dir"))
        dn = QtWidgets.QFileDialog.getExistingDirectory(self, "Select autosave directory", current)
        if dn and current != dn:
            self.dirEdit.setText(dn)


class PODMRWidget(ClientWidget, Ui_PODMR):
    """Widget for Pulse ODMR."""

    def __init__(
        self,
        gconf: dict,
        name,
        param_server_name,
        plot: PlotWidget,
        raw_plot: RawPlotWidget,
        context,
        parent=None,
    ):
        ClientWidget.__init__(self, parent)
        self.setupUi(self)

        self.conf = local_conf(gconf, name)

        self._finalizing = False
        self._has_fg = False

        self.init_radiobuttons()
        self.init_widgets()
        self.plot = plot
        self.raw_plot = raw_plot

        self.data = PODMRData()

        self.cli = QPODMRClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.param_cli = ParamClient(gconf, param_server_name, context=context)

        self.add_clients(self.cli, self.param_cli)

        self._fiTab_layout = QtWidgets.QVBoxLayout(self.fiTab)
        self.fit = PODMRFitWidget(self.cli, self.param_cli, parent=self.fiTab)
        self._fiTab_layout.addWidget(self.fit)

        self._autosaveTab_layout = QtWidgets.QVBoxLayout(self.autosaveTab)
        self.autosave = PODMRAutoSaveWidget(self.cli, self.param_cli, parent=self.autosaveTab)
        self._autosaveTab_layout.addWidget(self.autosave)
        self.autosave.init_connection()

        self._indicaTab_layout = QtWidgets.QVBoxLayout(self.indicaTab)
        self.indicator = PODMRIndicatorWidget(parent=self.indicaTab)
        self._indicaTab_layout.addWidget(self.indicator)
        self.indicator.init_connection(self.update_indicator)

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

    def init_widgets(self):
        self.tabWidget.setCurrentIndex(0)

        self.startBox.setValue(1)
        self.stepBox.setValue(1)
        self.numBox.setValue(100)
        self.tauconstBox.setValue(0)

        self.NstartBox.setValue(1)
        self.NstepBox.setValue(1)
        self.NnumBox.setValue(100)
        self.NconstBox.setValue(1)

        self.basewidthBox.setValue(320)
        self.ldelayBox.setValue(45)
        self.lwidthBox.setValue(5000)
        self.mdelayBox.setValue(1000)
        self.trigwidthBox.setValue(20)
        self.initdelayBox.setValue(0)
        self.finaldelayBox.setValue(5000)

        self.tpwidthBox.setValue(10)
        self.tp2widthBox.setValue(-1)
        self.iqdelayBox.setValue(10)

        self.init_delay()

    def init_delay(self):
        self.sigdelayBox.setValue(190)
        self.sigwidthBox.setValue(300)
        self.refdelayBox.setValue(2200)
        self.refwidthBox.setValue(2400)

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.update_inst_bounds()
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
        self.loadButton.clicked.connect(self.load_data)

        self.discardButton.clicked.connect(self.discard_data)

        # main tab
        for w in (self.startBox, self.stepBox, self.numBox):
            w.valueChanged.connect(self.update_stop)
        for w in (self.NstartBox, self.NstepBox, self.NnumBox):
            w.valueChanged.connect(self.update_Nstop)

        self.methodBox.currentIndexChanged.connect(self.switch_method)
        self.sweepNBox.toggled.connect(self.switch_method)
        self.partialBox.currentIndexChanged.connect(self.switch_partial)

        self._wire_plot_widgets(True)
        self.plotenableBox.toggled.connect(self.update_plot_enable)
        self.plotenableBox.setChecked(True)

        self.fftBox.toggled.connect(self.plot.update_fft_mode)
        self.taumodeBox.currentIndexChanged.connect(self.plot.enable_auto_range)
        self.ddgatephaseEdit.textChanged.connect(self.check_ddphase)

        # extra tab
        if self.has_fg():
            self.fg_disableButton.toggled.connect(self.switch_fg)
            self.fg_cwButton.toggled.connect(self.switch_fg)
            self.fg_gateButton.toggled.connect(self.switch_fg)

    # Widget status updates

    def update_indicator(self):
        self.indicator.update(self.data)

    def update_inst_bounds(self):
        params = self.cli.get_param_dict("rabi")
        if "power" in params:
            power: FloatParam = params["power"]
            self.powerBox.setMinimum(power.minimum())
            self.powerBox.setMaximum(power.maximum())
            self.powerBox.setValue(power.value())
        else:
            print(f"[ERROR] no key 'power' in params: {params}")

        if "freq" in params:
            freq: FloatParam = params["freq"]
            self.freqBox.setMinimum(freq.minimum() * 1e-6)  # Hz -> MHz
            self.freqBox.setMaximum(freq.maximum() * 1e-6)
            self.freqBox.setValue(freq.value() * 1e-6)
        else:
            print(f"[ERROR] no key 'freq' in params: {params}")

        if "fg" in params:
            self._has_fg = True
            ampl: FloatParam = params["fg"]["ampl"]
            self.fg_amplBox.setMinimum(ampl.minimum())
            self.fg_amplBox.setMaximum(ampl.maximum())
            self.fg_amplBox.setValue(ampl.value())
            freq: FloatParam = params["fg"]["freq"]
            self.fg_freqBox.setMinimum(freq.minimum() * 1e-6)
            self.fg_freqBox.setMaximum(freq.maximum() * 1e-6)
            self.fg_freqBox.setValue(freq.value() * 1e-6)
        else:
            self._has_fg = False

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
            self.xlogBox,
            self.ylogBox,
            self.fftBox,
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

    def switch_method(self):
        method = self.get_method()
        self.sweepNBox.setEnabled(is_CPlike(method))
        params = self.cli.get_param_dict(method)
        name_widgets = [
            ("Nstart", self.NstartBox),
            ("Nnum", self.NnumBox),
            ("Nstep", self.NstepBox),
            ("start", self.startBox),
            ("num", self.numBox),
            ("step", self.stepBox),
            ("log", self.logBox),
            ("supersample", self.qisampleBox),
            ("invertsweep", self.invertsweepBox),
            ("90pulse", self.tpwidthBox),
            ("180pulse", self.tp2widthBox),
            ("tauconst", self.tauconstBox),
            ("tau2const", self.tau2constBox),
            ("iq_delay", self.iqdelayBox),
            ("Nconst", self.NconstBox),
            ("N2const", self.N2constBox),
            ("N3const", self.N3constBox),
            ("ddphase", self.ddgatephaseEdit),
            ("invertinit", self.invertinitBox),
            ("invertY", self.invertYBox),
            ("readY", self.readYBox),
            ("reinitX", self.reinitXBox),
        ]
        set_enabled(params, name_widgets)

        self.reset_tau_modes(params["plot"]["taumode"].options())
        self.plot.update_label(self.data)

    def switch_partial(self, index):
        # when partial is 0 (index 1), plotmode to data0 (index 1)
        # when partial is 1 (index 2), plotmode to data1 (index 2)

        if index in (1, 2):
            self.plotmodeBox.setCurrentIndex(index)

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
        for b in (self.refavgBox, self.xlogBox, self.ylogBox, self.fftBox):
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

    def check_ddphase(self):
        """Validate input of ddgatephaseEdit. If input is invalid, set background color.

        Available labels: X Y iX iY n
        , is the single delimiter between data0 and data1: data0, data1
        : is the delimiters for each π/2 pulses
        All the whitespaces are ignored.

        Example:
        X/2--DDa--Y/2--DDb--X/2--DDa--Y/2 and the last pulse is inverted for data1:
        "X:Y:X:Y,X:Y:X:iY"

        """

        # parse
        c = "|".join(["X", "Y", "iX", "iY", "n"])
        p = "^({c:s}):({c:s}):({c:s}):({c:s}),({c:s}):({c:s}):({c:s}):({c:s})$".format(c=c)

        current = self.ddgatephaseEdit.text().replace(" ", "")
        m = re.match(p, current)

        # validation
        ## empty str is OK because it will be substituted by self.ddgatephaseEdit.placeholderText()
        bg = "none" if (m is not None or current == "") else "pink"
        self.ddgatephaseEdit.setStyleSheet("background: {:s};".format(bg))

    # data managements

    def save_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "PODMR", ".podmr")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.param_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)
        self.update_save_button(True)

        return fn

    def export_data(self):
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.param_cli.get_param("work_dir"))
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

        return fn

    def load_data(self):
        if self.data.is_finalized() and not self.data.is_saved():
            if not Qt.question_yn(
                self, "Sure to load?", "Current data has not been saved. Are you sure to discard?"
            ):
                return
        self.update_save_button(True)

        default_path = str(self.param_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "PODMR", ".podmr")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.param_cli.set_param("loaded_note", data.note())

        self.data = data
        self.apply_meas_widgets()
        self.apply_fg_widgets()
        if self.plotenableBox.isChecked():
            self.apply_plot_widgets()
        self.switch_method()
        self.refresh_plot()
        self.update_widgets()

        # self.update_data(data)

    # parameters
    def apply_meas_widgets(self, params=None):
        if params is None:
            p = self.data.params
        else:
            p = params

        # pulser
        self.binBox.setValue(p.get("timebin", 0.0) * 1e9)  # sec to ns

        self.intervalBox.setValue(int(round(p.get("interval", 0.0) * 1e3)))  # sec to ms
        self.sweepsBox.setValue(p.get("sweeps", 0))

        # method
        method = p.get("method", "rabi")
        self.set_method(method)

        # MW
        self.freqBox.setValue(p.get("freq", 2740e6) * 1e-6)  # [Hz] ==> [MHz]
        self.powerBox.setValue(p.get("power", 0.0))

        # sequence parameters
        self.startBox.setValue(p.get("start", 0.0) * 1e9)  # [sec] ==> [ns]
        self.numBox.setValue(p.get("num", 1))
        self.stepBox.setValue(p.get("step", 0.0) * 1e9)  # [sec] ==> [ns]
        self.logBox.setChecked(p.get("log", False))
        self.tauconstBox.setValue(p.get("tauconst", 0.0) * 1e9)  # [sec] ==> [ns]
        self.NstartBox.setValue(p.get("Nstart", 1))
        self.NnumBox.setValue(p.get("Nnum", 1))
        self.NstepBox.setValue(p.get("Nstep", 1))
        self.NconstBox.setValue(p.get("Nconst", 1))
        self.qisampleBox.setValue(p.get("supersample", 1))
        self.tau2constBox.setValue(p.get("tau2const", 0.0) * 1e9)  # [sec] ==> [ns]
        self.N2constBox.setValue(p.get("N2const", 1))
        self.N3constBox.setValue(p.get("N3const", 1))
        self.ddgatephaseEdit.setText(p.get("ddphase", ""))

        # method params
        self.invertsweepBox.setChecked(p.get("invertsweep", False))
        self.invertYBox.setChecked(p.get("invertY", False))
        self.reinitXBox.setChecked(p.get("reinitX", False))
        self.readYBox.setChecked(p.get("readY", False))
        self.nomwBox.setChecked(p.get("nomw", False))
        self.invertinitBox.setChecked(p.get("invertinit", False))
        self.reduceBox.setChecked(p.get("enable_reduce", False))
        self.divideblockBox.setChecked(p.get("divide_block", True))
        partial = p.get("partial")
        if partial in (0, 1):
            self.partialBox.setCurrentIndex(partial + 1)
        else:
            self.partialBox.setCurrentIndex(0)

        # sequence parameters (pulses)  [sec] ==> [ns]
        self.basewidthBox.setValue(p.get("base_width", 0.0) * 1e9)
        self.ldelayBox.setValue(p.get("laser_delay", 0.0) * 1e9)
        self.lwidthBox.setValue(p.get("laser_width", 0.0) * 1e9)
        self.mdelayBox.setValue(p.get("mw_delay", 0.0) * 1e9)
        self.trigwidthBox.setValue(p.get("trigger_width", 0.0) * 1e9)
        self.initdelayBox.setValue(p.get("init_delay", 0.0) * 1e9)
        self.finaldelayBox.setValue(p.get("final_delay", 0.0) * 1e9)
        self.iqdelayBox.setValue(p.get("iq_delay", 0.0) * 1e9)
        self.tpwidthBox.setValue(p.get("90pulse", 0.0) * 1e9)
        self.tp2widthBox.setValue(p.get("180pulse", 0.0) * 1e9)

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
        self.fg_freqBox.setValue(params.get("freq", 1e6) * 1e-6)  # [Hz] ==> [MHz]
        self.fg_amplBox.setValue(params.get("amp", 0.0))
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

        self.xlogBox.setChecked(p.get("xlogscale", False))
        self.ylogBox.setChecked(p.get("ylogscale", False))
        self.fftBox.setChecked(p.get("fft", False))

        self.sigdelayBox.setValue(p.get("sigdelay", 0.0) * 1e9)
        self.sigwidthBox.setValue(p.get("sigwidth", 100.0) * 1e9)
        self.refdelayBox.setValue(p.get("refdelay", 100.0) * 1e9)
        self.refwidthBox.setValue(p.get("refwidth", 100.0) * 1e9)
        self.refavgBox.setChecked(p.get("refaverage", False))

        self._wire_plot_widgets(True)

        self.update_plot_params()

    def get_params(self) -> dict:
        params = {}
        params["method"] = self.get_method()
        params["freq"] = self.freqBox.value() * 1e6  # [MHz] ==> [Hz]
        params["power"] = self.powerBox.value()
        params["timebin"] = self.binBox.value() * 1e-9  # [ns] ==> [sec]

        params["interval"] = self.intervalBox.value() * 1e-3  # [ms] ==> [sec]
        params["sweeps"] = self.sweepsBox.value()

        params["start"] = self.startBox.value() * 1e-9  # [ns] ==> [sec]
        params["num"] = self.numBox.value()
        params["step"] = self.stepBox.value() * 1e-9  # [ns] ==> [sec]
        params["log"] = self.logBox.isChecked()
        params["tauconst"] = self.tauconstBox.value() * 1e-9  # [ns] ==> [sec]

        params["Nstart"] = self.NstartBox.value()
        params["Nnum"] = self.NnumBox.value()
        params["Nstep"] = self.NstepBox.value()
        params["Nconst"] = self.NconstBox.value()
        params["supersample"] = self.qisampleBox.value()

        params["tau2const"] = self.tau2constBox.value() * 1e-9  # [ns] ==> [sec]
        params["N2const"] = self.N2constBox.value()
        params["N3const"] = self.N3constBox.value()
        params["ddphase"] = self.ddgatephaseEdit.text()
        if not params["ddphase"]:
            params["ddphase"] = self.ddgatephaseEdit.placeholderText()

        params["invertsweep"] = self.invertsweepBox.isChecked()
        params["invertY"] = self.invertYBox.isChecked()
        params["reinitX"] = self.reinitXBox.isChecked()
        params["readY"] = self.readYBox.isChecked()
        params["partial"] = self.partialBox.currentIndex() - 1
        params["nomw"] = self.nomwBox.isChecked()
        params["invertinit"] = self.invertinitBox.isChecked()
        params["enable_reduce"] = self.reduceBox.isChecked()
        params["divide_block"] = self.divideblockBox.isChecked()

        # [ns] ==> [sec]
        params["base_width"] = self.basewidthBox.value() * 1e-9
        params["laser_delay"] = self.ldelayBox.value() * 1e-9
        params["laser_width"] = self.lwidthBox.value() * 1e-9
        params["mw_delay"] = self.mdelayBox.value() * 1e-9
        params["trigger_width"] = self.trigwidthBox.value() * 1e-9
        params["init_delay"] = self.initdelayBox.value() * 1e-9
        params["final_delay"] = self.finaldelayBox.value() * 1e-9
        params["iq_delay"] = self.iqdelayBox.value() * 1e-9
        params["90pulse"] = self.tpwidthBox.value() * 1e-9
        params["180pulse"] = self.tp2widthBox.value() * 1e-9

        params["plot"] = self.get_plot_params()
        params["fg"] = self.get_fg_params()

        return params

    def get_fg_params(self):
        params = {}
        params["mode"] = self.get_fg_mode()
        params["wave"] = self.fg_waveBox.currentText()
        params["freq"] = self.fg_freqBox.value() * 1.0e6  # MHz ==> Hz
        params["ampl"] = self.fg_amplBox.value()
        params["phase"] = self.fg_phaseBox.value()
        return params

    def get_plot_params(self):
        params = {}
        params["plotmode"] = self.plotmodeBox.currentText()
        params["taumode"] = self.taumodeBox.currentText()

        params["xlogscale"] = self.xlogBox.isChecked()
        params["ylogscale"] = self.ylogBox.isChecked()
        params["fft"] = self.fftBox.isChecked()

        # [us] or [ns] ==> [sec]
        params["sigdelay"] = self.sigdelayBox.value() * 1e-9
        params["sigwidth"] = self.sigwidthBox.value() * 1e-9
        params["refdelay"] = self.refdelayBox.value() * 1e-9
        params["refwidth"] = self.refwidthBox.value() * 1e-9
        params["refmode"] = self.refmodeBox.currentText()
        params["refaverage"] = self.refavgBox.isChecked()
        return params

    def get_fg_mode(self):
        return [m for b, m in self.get_fg_mode_dict() if b.isChecked()][0]

    def get_plottable_data(self) -> T.List[T.Tuple[PODMRData, bool, Colors]]:
        return self.fit.get_plottable_data(self.data)

    # State managements
    def request_stop(self):
        self.cli.stop()

    def request_start(self):
        self.round_box_values()

        if self.validate_pulse():
            self.start()

    def validate_pulse(self):
        return self.cli.validate(self.get_params())

    def start(self):
        """start the measurement."""

        params = self.get_params()
        title = "Continue?"
        body = (
            "Continue with current data?"
            + " Press No to clear current data and start a new measurement."
        )
        if self.data.can_resume(params) and Qt.question_yn(self, title, body):
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

        self.cli.start(params)

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
                self.binBox.setValue(tbin * 1.0e9)  # sec to ns
            if trange is not None:
                self.rangeLabel.setText("range: {:d}".format(int(round(trange))))

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
        self.raw_plot.refresh(self.data)

    def finalize(self, data: PODMRData):
        if self._finalizing:
            return
        self._finalizing = True
        self.update_save_button(False)
        self._finalizing = False

    def update_state(self, state: BinaryState, last_state: BinaryState):
        for w in (
            self.startButton,
            self.saveButton,
            self.exportButton,
            self.loadButton,
            self.binBox,
            self.freqBox,
            self.powerBox,
            self.nomwBox,
            self.methodBox,
            self.partialBox,
            self.intervalBox,
            self.sweepsBox,
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
        ):
            w.setEnabled(state == BinaryState.IDLE)

        # Widgets enable/disable of which depends on selected method
        if state == BinaryState.IDLE:
            if last_state == BinaryState.ACTIVE:
                self.switch_method()
        else:
            for w in (
                self.startBox,
                self.stepBox,
                self.numBox,
                self.logBox,
                self.tauconstBox,
                self.qisampleBox,
                self.sweepNBox,
                self.NstartBox,
                self.NstepBox,
                self.NnumBox,
                self.NconstBox,
                self.invertYBox,
                self.reinitXBox,
                self.readYBox,
                self.invertinitBox,
                self.iqdelayBox,
                self.tpwidthBox,
                self.tp2widthBox,
                self.tau2constBox,
                self.N2constBox,
                self.N3constBox,
                self.ddgatephaseEdit,
            ):
                w.setEnabled(False)

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

    def round_box_values(self):
        """check and round to half-integer values of QDoubleSpinBox for timing parameters."""

        widthboxes = (
            self.startBox,
            self.stepBox,
            self.tauconstBox,
            self.basewidthBox,
            self.ldelayBox,
            self.lwidthBox,
            self.mdelayBox,
            self.iqdelayBox,
            self.tpwidthBox,
            self.tp2widthBox,
        )
        for b in widthboxes:
            b.setValue(round_halfint(b.value()))

    METHODS = (
        "rabi",
        "fid",
        "spinecho",
        "trse",
        "cp",
        "cpmg",
        "xy4",
        "xy8",
        "xy16",
        "180train",
        "se90sweep",
        "recovery",
        "spinlock",
        "xy8cl",
        "xy8cl1flip",
        "xy8clNflip",
        "ddgate",
    )

    def get_method(self) -> str:
        i = self.methodBox.currentIndex()
        m = self.METHODS[i]
        if m in ("cp", "cpmg", "xy4", "xy8", "xy16", "ddgate") and self.sweepNBox.isChecked():
            m = m + "N"
        return m

    def set_method(self, method: str):
        is_N_sweep = re.match("(cp|cpmg|xy4|xy8|xy16)N", method)
        if is_N_sweep is not None:
            method = is_N_sweep.group(1)
            self.sweepNBox.setChecked(True)

        try:
            self.methodBox.setCurrentIndex(self.METHODS.index(method))
        except ValueError:
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

    def reset_tau_modes(self, modes: T.List[str]):
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
        self.raw_plot = RawPlotWidget(parent=self)
        self.podmr = PODMRWidget(
            gconf,
            target["podmr"],
            target["param_server"],
            self.plot,
            self.raw_plot,
            context,
            parent=self,
        )

        self.setWindowTitle(f"MAHOS.PODMRGUI ({join_name(target['podmr'])})")
        self.setAnimated(False)
        self.setCentralWidget(self.podmr)
        self.d_plot = QtWidgets.QDockWidget("Plot", parent=self)
        self.d_plot.setWidget(self.plot)
        self.d_raw_plot = QtWidgets.QDockWidget("Raw Plot", parent=self)
        self.d_raw_plot.setWidget(self.raw_plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.d_plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_raw_plot)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_plot.toggleViewAction())
        self.view_menu.addAction(self.d_raw_plot.toggleViewAction())

    def closeEvent(self, event):
        """Close the clients on closeEvent."""

        self.podmr.close_clients()
        # self.plot.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)


class PODMRGUI(GUINode):
    """GUINode for Pulse ODMR using PODMRWidget."""

    def init_widget(self, gconf: dict, name, context):
        return PODMRMainWindow(gconf, name, context)
