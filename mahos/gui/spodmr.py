#!/usr/bin/env python3

"""
GUI frontend of Pulse ODMR with Slow detectors.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import typing as T
import os
import re
import uuid

import numpy as np
import pyqtgraph as pg

from . import Qt
from .Qt import QtCore, QtGui, QtWidgets

from .ui.spodmr import Ui_SPODMR
from .spodmr_client import QSPODMRClient

from ..msgs.common_msgs import BinaryState, BinaryStatus
from ..msgs.common_meas_msgs import Buffer
from ..msgs.param_msgs import FloatParam
from ..msgs.spodmr_msgs import SPODMRData, is_CPlike
from ..node.global_params import GlobalParamsClient
from .gui_node import GUINode
from .common_widget import ClientWidget
from .fit_widget import FitWidget
from .param import set_enabled, apply_widgets
from .dialog import save_dialog, load_dialog, export_dialog
from ..node.node import local_conf, join_name
from ..util.plot import colors_tab20_pair
from ..util.math_phys import round_halfint


Policy = QtWidgets.QSizePolicy.Policy


class Colors(T.NamedTuple):
    color0: str
    color1: str


_colors = [Colors(c0, c1) for c0, c1 in colors_tab20_pair()]


class SPODMRFitWidget(FitWidget):
    def colors(self) -> list:
        return _colors

    def load_dialog(self, default_path: str) -> str:
        return load_dialog(self, default_path, "SPODMR", ".spodmr")


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.init_ui()
        self.init_view()
        self.update_font_size()
        self.fontsizeBox.editingFinished.connect(self.update_font_size)

    def sizeHint(self):
        return QtCore.QSize(1400, 700)

    def init_ui(self):
        hl0 = QtWidgets.QHBoxLayout()
        self.showimgBox = QtWidgets.QCheckBox("Show image", parent=self)
        self.showimgBox.setChecked(True)
        self.showstdBox = QtWidgets.QCheckBox("Show std. dev.", parent=self)
        self.showstdBox.setChecked(False)
        self.lastnBox = QtWidgets.QSpinBox(parent=self)
        self.lastnBox.setPrefix("last_n: ")
        self.lastnBox.setMinimum(0)
        self.lastnBox.setMaximum(10000)
        self.lastnimgBox = QtWidgets.QSpinBox(parent=self)
        self.lastnimgBox.setPrefix("last_n (img): ")
        self.lastnimgBox.setMinimum(0)
        self.lastnimgBox.setMaximum(10000)
        self.fontsizeBox = QtWidgets.QSpinBox(parent=self)
        self.fontsizeBox.setPrefix("font size: ")
        self.fontsizeBox.setSuffix(" pt")
        self.fontsizeBox.setMinimum(1)
        self.fontsizeBox.setValue(12)
        self.fontsizeBox.setMaximum(99)
        for w in (self.lastnBox, self.lastnimgBox, self.fontsizeBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)
        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        for w in (
            self.showimgBox,
            self.showstdBox,
            self.lastnBox,
            self.lastnimgBox,
            self.fontsizeBox,
        ):
            hl0.addWidget(w)
        hl0.addItem(spacer)

        hl = QtWidgets.QHBoxLayout()
        self.graphicsView = pg.GraphicsView(parent=self)
        self.histo = pg.HistogramLUTWidget(parent=self)
        hl.addWidget(self.graphicsView)
        hl.addWidget(self.histo)

        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl0)
        vl.addLayout(hl)
        self.setLayout(vl)

    def init_view(self):
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)
        self.histo.gradient.loadPreset("inferno")

        self.plot = self.layout.addPlot(row=0, col=0, lockAspect=False)

        self.img = pg.ImageItem()
        self.img_plot = self.layout.addPlot(row=1, col=0, lockAspect=False)
        self.img_plot.addItem(self.img)
        self.histo.setImageItem(self.img)

        self.plot.showGrid(x=True, y=True)

        self.img_plot.setLabel("bottom", "Data point")
        self.img_plot.setLabel("left", "Number of accumulation")

        self.showimgBox.toggled.connect(self.toggle_image)

    def update_plot(self, data_list: list[tuple[SPODMRData, bool, str]]):
        self.plot.clear()
        for data, show_fit, c in data_list:
            try:
                x = data.get_xdata()
                xfit = data.get_fit_xdata()
            except ValueError as e:
                print("Error getting xdata: " + repr(e))
                continue

            try:
                y0, y1 = data.get_ydata(last_n=self.lastnBox.value())
                if self.showstdBox.isChecked():
                    y0std, y1std = data.get_ydata(last_n=self.lastnBox.value(), std=True)
                else:
                    y0std = y1std = None
                if y0 is None:
                    return
                yfit = data.get_fit_ydata()
            except ValueError as e:
                print("Error getting ydata: " + repr(e))
                continue

            plot_fit = show_fit and (xfit is not None) and (yfit is not None)
            width = 0.5 if plot_fit else 1.0
            symbolSize = 4 if plot_fit else 8
            if y0std is not None:
                self.plot.addItem(
                    pg.ErrorBarItem(x=x, y=y0, height=2 * y0std, pen=pg.mkPen(c.color0, width=0.5))
                )
            if y1std is not None:
                self.plot.addItem(
                    pg.ErrorBarItem(x=x, y=y1, height=2 * y1std, pen=pg.mkPen(c.color1, width=0.5))
                )
            self.plot.plot(
                x,
                y0,
                pen=pg.mkPen(c.color0, width=width),
                symbolPen=None,
                symbol="o",
                symbolSize=symbolSize,
                symbolBrush=c.color0,
            )
            if y1 is not None:
                self.plot.plot(
                    x,
                    y1,
                    pen=pg.mkPen(c.color1, width=width),
                    symbolPen=None,
                    symbol="o",
                    symbolSize=symbolSize,
                    symbolBrush=c.color1,
                )
            if plot_fit:
                self.plot.plot(xfit, yfit, pen=pg.mkPen(c.color0, width=2.0))

    def refresh_all(self, data_list: list[tuple[SPODMRData, bool, str]], data: SPODMRData):
        try:
            self.update_plot(data_list)
        except TypeError as e:
            # import sys, traceback; traceback.print_tb(sys.exc_info()[2])
            print("Error in plot_analyzed " + repr(e))

        try:
            self.update_image(data)
        except TypeError as e:
            # import sys, traceback; traceback.print_tb(sys.exc_info()[2])
            print("Error in update_image " + repr(e))

        self.update_label(data)

    def refresh_plot(self, data_list: list[tuple[SPODMRData, bool, str]]):
        try:
            self.update_plot(data_list)
        except TypeError as e:
            # import sys, traceback; traceback.print_tb(sys.exc_info()[2])
            print("Error in plot_analyzed " + repr(e))

    def update_label(self, data: SPODMRData):
        if not data.has_params():
            return

        self.plot.setLabel("bottom", data.xlabel, data.xunit)
        self.plot.setLabel("left", data.ylabel, data.yunit)
        self.plot.setLogMode(x=data.xscale == "log", y=data.yscale == "log")

    def update_image(self, data: SPODMRData, setlevel=True):
        if not data.has_data():
            return

        img = data.get_image(last_n=self.lastnimgBox.value())
        self.img.updateImage(img)
        if setlevel:
            mn, mx = np.nanmin(img), np.nanmax(img)
            if mn == mx:
                mn, mx = mn - 0.1, mx + 0.1
            self.histo.setLevels(mn, mx)
        # self.img.resetTransform()
        # self.img.setPos(0.0, 0.0)
        # self.img.setTransform(QtGui.QTransform.fromScale(1.0, 1.0))

    def update_fft_mode(self, enable):
        self.plot.ctrl.fftCheck.setChecked(enable)

    def enable_auto_range(self):
        self.plot.enableAutoRange()

    def toggle_image(self, show):
        if show:
            self.layout.addItem(self.img_plot, row=1, col=0)
        else:
            self.layout.removeItem(self.img_plot)

    def update_font_size(self):
        font = QtGui.QFont()
        font.setPointSize(self.fontsizeBox.value())
        for p in (self.plot, self.img_plot):
            for l in ("bottom", "left"):
                p.getAxis(l).label.setFont(font)
                p.getAxis(l).setTickFont(font)


class SPODMRWidget(ClientWidget, Ui_SPODMR):
    """Widget for Pulse ODMR with Slow detectors."""

    def __init__(
        self,
        gconf: dict,
        name,
        gparams_name,
        plot: PlotWidget,
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

        self.data = SPODMRData()

        self.cli = QSPODMRClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)

        self.add_clients(self.cli, self.gparams_cli)

        self._fiTab_layout = QtWidgets.QVBoxLayout(self.fiTab)
        self.fit = SPODMRFitWidget(self.cli, self.gparams_cli, parent=self.fiTab)
        self._fiTab_layout.addWidget(self.fit)

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
        self.numBox.setValue(50)
        self.tauconstBox.setValue(0)

        self.NstartBox.setValue(1)
        self.NstepBox.setValue(1)
        self.NnumBox.setValue(50)
        self.NconstBox.setValue(1)

        self.ldelayBox.setValue(45)
        self.lwidthBox.setValue(3000)
        self.mdelayBox.setValue(1000)

        self.tpwidthBox.setValue(10)
        self.tp2widthBox.setValue(-1)
        self.iqdelayBox.setValue(10)

        self.offsetBox.setOpts(siPrefix=True)

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

        apply_widgets(
            params,
            [
                ("accum_window", self.accumwindowBox, 1e3),  # sec to ms
                ("accum_rep", self.accumrepBox),
                ("drop_rep", self.droprepBox),
                ("lockin_rep", self.lockinrepBox),
                ("pd_rate", self.pdrateBox, 1e-3),  # Hz to kHz
                ("pd_bounds", [self.pd_lbBox, self.pd_ubBox]),
            ],
        )

    def update_plot_enable(self, enable: bool):
        for w in (
            self.plotmodeBox,
            self.taumodeBox,
            self.logXBox,
            self.logYBox,
            self.fftBox,
            self.flipYBox,
            self.normalizeBox,
            self.offsetBox,
            self.complexBox,
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
            ("flip_head", self.flipheadBox),
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

    def _wire_plot_widgets(self, connect: bool):
        for cb in (self.plotmodeBox, self.taumodeBox, self.complexBox):
            if connect:
                cb.currentIndexChanged.connect(self.update_plot_params)
            else:
                cb.currentIndexChanged.disconnect(self.update_plot_params)
        for b in (self.logXBox, self.logYBox, self.fftBox, self.flipYBox, self.normalizeBox):
            if connect:
                b.toggled.connect(self.update_plot_params)
            else:
                b.toggled.disconnect(self.update_plot_params)
        if connect:
            self.offsetBox.editingFinished.connect(self.update_plot_params)
        else:
            self.offsetBox.editingFinished.disconnect(self.update_plot_params)

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
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "SPODMR", ".spodmr")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.gparams_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)
        self.update_save_button(True)

        n = os.path.splitext(fn)[0] + ".png"
        params = {
            "show_fit": self.fit.show_current_data_fit(),
            "show_std": self.plot.showstdBox.isChecked(),
        }
        self.cli.export_data(n, params=params)

    def export_data(self):
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "SPODMR", (".png", ".pdf", ".eps", ".txt"))
        if not fn:
            return

        params = {}
        data = [d for (d, _, _) in data_list]
        params["color_fit"] = [color.color0 for (_, _, color) in data_list]
        params["color0"] = [color.color0 for (_, _, color) in data_list]
        params["color1"] = [color.color1 for (_, _, color) in data_list]
        params["show_fit"] = any([show_fit for (_, show_fit, _) in data_list])
        params["show_std"] = self.plot.showstdBox.isChecked()
        self.cli.export_data(fn, data=data, params=params)

    def load_data(self):
        if self.data.is_finalized() and not self.data.is_saved():
            if not Qt.question_yn(
                self, "Sure to load?", "Current data has not been saved. Are you sure to discard?"
            ):
                return
        self.update_save_button(True)

        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "SPODMR", ".spodmr")
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
        self.switch_method()
        self.refresh_all()

        # self.update_data(data)

    # parameters
    def apply_meas_widgets(self):
        p = self.data.params

        # pulser
        self.sweepsBox.setValue(p.get("sweeps", 0))
        self.accumwindowBox.setValue(p.get("accum_window", 1e-3) * 1e3)  # [sec] ==> [ms]
        self.accumrepBox.setValue(p.get("accum_rep", 1))
        if "drop_rep" in p:
            self.droprepBox.setValue(p["drop_rep"])
        if "lockin_rep" in p:
            self.lockinrepBox.setValue(p["lockin_rep"])
        self.pdrateBox.setValue(round(p.get("pd_rate", 1e5) * 1e-3))
        if "pd_bounds" in p:
            lb, ub = p["pd_bounds"]
            self.pd_lbBox.setValue(lb)
            self.pd_ubBox.setValue(ub)

        # method
        self.set_method(self.data.label)

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
        self.flipheadBox.setChecked(p.get("flip_head", False))
        self.nomwBox.setChecked(p.get("nomw", False))
        self.invertinitBox.setChecked(p.get("invertinit", False))
        self.reduceBox.setChecked(p.get("enable_reduce", False))
        partial = p.get("partial")
        if partial in (-1, 0, 1, 2):
            self.partialBox.setCurrentIndex(partial + 1)
        else:
            self.partialBox.setCurrentIndex(0)

        # sequence parameters (pulses)  [sec] ==> [ns]
        self.ldelayBox.setValue(p.get("laser_delay", 0.0) * 1e9)
        self.lwidthBox.setValue(p.get("laser_width", 0.0) * 1e9)
        self.mdelayBox.setValue(p.get("mw_delay", 0.0) * 1e9)
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

        self.logXBox.setChecked(p.get("logX", False))
        self.logYBox.setChecked(p.get("logY", False))
        self.fftBox.setChecked(p.get("fft", False))
        self.flipYBox.setChecked(p.get("flipY", False))
        self.offsetBox.setValue(p.get("offset", 0.0))

        self._wire_plot_widgets(True)

        self.update_plot_params()

    def get_params(self) -> tuple[dict, str]:
        label = self.get_method()
        params = {}
        params["partial"] = self.partialBox.currentIndex() - 1
        params["freq"] = self.freqBox.value() * 1e6  # [MHz] ==> [Hz]
        params["power"] = self.powerBox.value()

        params["sweeps"] = self.sweepsBox.value()
        params["accum_window"] = self.accumwindowBox.value() * 1e-3  # [ms] ==> [sec]
        params["accum_rep"] = self.accumrepBox.value()
        params["drop_rep"] = self.droprepBox.value()
        if params["partial"] == 2:
            params["lockin_rep"] = self.lockinrepBox.value()
        params["pd_rate"] = self.pdrateBox.value() * 1e3  # [kHz] ==> [Hz]
        params["pd_bounds"] = [self.pd_lbBox.value(), self.pd_ubBox.value()]

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
        params["flip_head"] = self.flipheadBox.isChecked()
        params["nomw"] = self.nomwBox.isChecked()
        params["invertinit"] = self.invertinitBox.isChecked()
        params["enable_reduce"] = self.reduceBox.isChecked()

        # [ns] ==> [sec]
        params["laser_delay"] = self.ldelayBox.value() * 1e-9
        params["laser_width"] = self.lwidthBox.value() * 1e-9
        params["mw_delay"] = self.mdelayBox.value() * 1e-9
        params["iq_delay"] = self.iqdelayBox.value() * 1e-9
        params["90pulse"] = self.tpwidthBox.value() * 1e-9
        params["180pulse"] = self.tp2widthBox.value() * 1e-9

        params["plot"] = self.get_plot_params()
        params["fg"] = self.get_fg_params()

        return params, label

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

        params["logX"] = self.logXBox.isChecked()
        params["logY"] = self.logYBox.isChecked()
        params["fft"] = self.fftBox.isChecked()
        params["flipY"] = self.flipYBox.isChecked()

        params["complex_conv"] = self.complexBox.currentText()
        params["normalize"] = self.normalizeBox.isChecked()
        params["offset"] = self.offsetBox.value()

        return params

    def get_fg_mode(self):
        return [m for b, m in self.get_fg_mode_dict() if b.isChecked()][0]

    def get_plottable_data(self) -> list[tuple[SPODMRData, bool, Colors]]:
        return self.fit.get_plottable_data(self.data)

    # State managements
    def request_stop(self):
        self.cli.stop()

    def request_start(self):
        self.round_box_values()
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

        self.cli.start(params, label)

        self.update_save_button(True)

    def update_data(self, data: SPODMRData):
        self.data = data
        if not self.data.has_data():
            return  # measurement has not been started yet or data has not been accumulated.

        self.refresh_all()

    def update_buffer(self, buffer: Buffer[tuple[str, SPODMRData]]):
        self.fit.update_buffer(buffer)
        self.plot.refresh_plot(self.get_plottable_data())

    def update_plot_params(self):
        self.cli.update_plot_params(self.get_plot_params())

    def refresh_all(self):
        self.plot.refresh_all(self.get_plottable_data(), self.data)

    def finalize(self, data: SPODMRData):
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
            self.freqBox,
            self.powerBox,
            self.nomwBox,
            self.methodBox,
            self.partialBox,
            self.sweepsBox,
            self.accumwindowBox,
            self.accumrepBox,
            self.droprepBox,
            self.pdrateBox,
            self.pd_lbBox,
            self.pd_ubBox,
            self.invertsweepBox,
            self.reduceBox,
            self.ldelayBox,
            self.lwidthBox,
            self.mdelayBox,
        ):
            w.setEnabled(state == BinaryState.IDLE)

        self.lockinrepBox.setEnabled(
            state == BinaryState.IDLE and self.partialBox.currentIndex() == 3
        )

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
                self.flipheadBox,
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

        self.stopButton.setEnabled(state == BinaryState.ACTIVE)

    # helper functions

    def has_fg(self):
        return self._has_fg

    def round_box_values(self):
        """check and round to half-integer values of QDoubleSpinBox for timing parameters."""

        widthboxes = (
            self.startBox,
            self.stepBox,
            self.tauconstBox,
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
        "t1",
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
        # "recovery",
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

    def get_fg_mode_dict(self):
        return [
            (self.fg_disableButton, "disable"),
            (self.fg_cwButton, "cw"),
            (self.fg_gateButton, "gate"),
        ]


class SPODMRMainWindow(QtWidgets.QMainWindow):
    """MainWindow with SPODMRWidget and PlotWidget."""

    def __init__(self, gconf: dict, name, context, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        lconf = local_conf(gconf, name)
        target = lconf["target"]

        self.plot = PlotWidget(parent=self)
        self.spodmr = SPODMRWidget(
            gconf,
            target["spodmr"],
            target["gparams"],
            self.plot,
            context,
            parent=self,
        )

        self.setWindowTitle(f"MAHOS.SPODMRGUI ({join_name(target['spodmr'])})")
        self.setAnimated(False)
        self.setCentralWidget(self.spodmr)
        self.d_plot = QtWidgets.QDockWidget("Plot", parent=self)
        self.d_plot.setWidget(self.plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_plot)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_plot.toggleViewAction())

    def closeEvent(self, event):
        """Close the clients on closeEvent."""

        self.spodmr.close_clients()
        # self.plot.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)


class SPODMRGUI(GUINode):
    """GUINode for Pulse ODMR with Slow detectors using SPODMRWidget."""

    def init_widget(self, gconf: dict, name, context):
        return SPODMRMainWindow(gconf, name, context)
