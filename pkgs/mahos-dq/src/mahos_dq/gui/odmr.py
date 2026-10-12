#!/usr/bin/env python3

"""
GUI frontend of ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import typing as T
import uuid
import os

import numpy as np
import pyqtgraph as pg

from mahos.gui import Qt
from mahos.gui.Qt import QtCore, QtWidgets, QtGui

from mahos_dq.gui.ui.odmr import Ui_ODMR
from mahos_dq.gui.ui.odmr_peaks import Ui_ODMRPeaks
from mahos_dq.gui.odmr_client import QODMRClient

from mahos.msgs.common_msgs import BinaryState, BinaryStatus
from mahos.msgs.common_meas_msgs import Buffer
from mahos_dq.msgs.odmr_msgs import ODMRData
from mahos.msgs import param_msgs as P
from mahos.node.global_params import GlobalParamsClient
from mahos_dq.meas.confocal import ConfocalIORequester
from mahos.util import conv
from mahos_dq.util import nv
from mahos.util.plot import colors_tab20_pair
from mahos.gui.gui_node import GUINode, get_gui_logger
from mahos.gui.common_widget import ClientWidget
from mahos.gui.fit_widget import FitWidget
from mahos.gui.param import ParamDictComboBoxHandler, apply_widgets
from mahos.gui.dialog import save_dialog, load_dialog, export_dialog
from mahos.node.node import local_conf, join_name


Policy = QtWidgets.QSizePolicy.Policy


class Colors(T.NamedTuple):
    color: str
    color_bg: str


_colors = [Colors(c0, c1) for c0, c1 in colors_tab20_pair()]


class ODMRFitWidget(FitWidget):
    def colors(self) -> list:
        return _colors

    def load_dialog(self, default_path: str) -> str:
        return load_dialog(self, default_path, "ODMR", ".odmr")


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self._normalize_updated = False
        self._yunit: str = ""
        self.init_ui()
        self.init_view()
        self.update_font_size()
        self.fontsizeBox.editingFinished.connect(self.update_font_size)
        self._error_bar_items = []

    def sizeHint(self):
        return QtCore.QSize(1600, 1200)

    def init_ui(self):
        hl0 = QtWidgets.QHBoxLayout()
        self.showimgBox = QtWidgets.QCheckBox("Show image", parent=self)
        self.showimgBox.setChecked(True)
        self.showstdBox = QtWidgets.QCheckBox("Show std. dev.", parent=self)
        self.showstdBox.setChecked(False)
        self.lastnBox = QtWidgets.QSpinBox(parent=self)
        self.lastnBox.setPrefix("last_n: ")
        self.lastnBox.setMinimum(-1_000_000)
        self.lastnBox.setMaximum(1_000_000)
        self.lastnBox.setValue(0)
        self.lastnimgBox = QtWidgets.QSpinBox(parent=self)
        self.lastnimgBox.setPrefix("last_n (img): ")
        self.lastnimgBox.setMinimum(-1_000_000)
        self.lastnimgBox.setMaximum(1_000_000)
        self.lastnimgBox.setValue(0)
        self.normalizenBox = QtWidgets.QSpinBox(parent=self)
        self.normalizenBox.setPrefix("normalize_n: ")
        self.normalizenBox.setMinimum(-100)
        self.normalizenBox.setMaximum(100)
        self.fontsizeBox = QtWidgets.QSpinBox(parent=self)
        self.fontsizeBox.setPrefix("font size: ")
        self.fontsizeBox.setSuffix(" pt")
        self.fontsizeBox.setMinimum(1)
        self.fontsizeBox.setValue(12)
        self.fontsizeBox.setMaximum(99)
        for w in (self.lastnBox, self.lastnimgBox, self.normalizenBox, self.fontsizeBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)
        self.complexBox = QtWidgets.QComboBox(parent=self)
        self.complexBox.addItems(["real", "imag", "abs", "angle"])
        self.complexBox.setCurrentIndex(0)
        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        for w in (
            self.showimgBox,
            self.showstdBox,
            self.lastnBox,
            self.lastnimgBox,
            self.normalizenBox,
            self.fontsizeBox,
            self.complexBox,
        ):
            hl0.addWidget(w)
        hl0.addItem(spacer)

        hl = QtWidgets.QHBoxLayout()
        self.graphicsView = pg.GraphicsView(parent=self)
        self.histo = pg.HistogramLUTWidget(parent=self)
        hl.addWidget(self.graphicsView)
        hl.addWidget(self.histo)

        hl_bottom = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel("Ready")
        hl_bottom.addWidget(self.label)

        vl = QtWidgets.QVBoxLayout()
        vl.addLayout(hl0)
        vl.addLayout(hl)
        vl.addLayout(hl_bottom)
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
        self.plot.setLabel("bottom", "Microwave frequency", "Hz")
        self.plot.setLabel("left", "Intensity", "")
        self.img_plot.setLabel("bottom", "Microwave frequency", "Hz")
        self.img_plot.setLabel("left", "Number of accumulation")

        self.plot.scene().sigMouseMoved.connect(self.update_pos)
        self.showimgBox.toggled.connect(self.toggle_image)
        self.normalizenBox.valueChanged.connect(self.update_normalize)

    def refresh_all(self, data_list: list[tuple[ODMRData, bool, str]], data: ODMRData):
        if self._yunit != data.yunit:
            self._yunit = data.yunit
            self.update_ylabel(self.normalizenBox.value())

        if data.has_data():
            self.update_image(data)
        self.update_plot(data_list)

    def refresh_plot(self, data_list: list[tuple[ODMRData, bool, str]]):
        self.update_plot(data_list)

    def update_image(self, data: ODMRData, setlevel=True):
        img = data.get_image(
            last_n=self.lastnimgBox.value(),
            complex_conv=self.complexBox.currentText(),
            allow_nan=True,
        )
        if img.size == 0:
            return
        self.img.updateImage(img)
        if setlevel:
            mn, mx = np.nanmin(img), np.nanmax(img)
            if mn == mx:
                mn, mx = mn - 0.1, mx + 0.1
            self.histo.setLevels(mn, mx)
        self.img.resetTransform()
        self.img.setPos(data.params["start"], 0.0)
        self.img.setTransform(QtGui.QTransform.fromScale(data.step(), 1.0))

    def update_pos(self, pos):
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        point = self.plot.getViewBox().mapSceneToView(pos)
        x, y = point.x(), point.y()
        self.update_label(x, y)

    def update_label(self, x, y):
        sx = pg.siFormat(x, precision=6, suffix="Hz")
        sy = pg.siFormat(y, precision=6, suffix=self._yunit)
        self.label.setText(f"{sx}, {sy}")

    def add_error_bar(self, x, y, height, pen):
        item = pg.ErrorBarItem(x=x, y=y, height=height, pen=pen)
        self.plot.addItem(item)
        self._error_bar_items.append(item)

    def clear_error_bars(self):
        if not self._error_bar_items:
            return

        for item in self._error_bar_items:
            self.plot.removeItem(item)
        self._error_bar_items = []

    def update_plot(self, data_list: list[tuple[ODMRData, bool, Colors]]):
        self.plot.clearPlots()
        self.clear_error_bars()

        for data, show_fit, c in data_list:
            x = data.get_xdata()
            y, y_bg = data.get_ydata(
                last_n=self.lastnBox.value(),
                normalize_n=self.normalizenBox.value(),
                complex_conv=self.complexBox.currentText(),
            )
            if y is None:
                continue
            if self.showstdBox.isChecked():
                ystd, ybg_std = data.get_ydata(
                    last_n=self.lastnBox.value(),
                    normalize_n=self.normalizenBox.value(),
                    complex_conv=self.complexBox.currentText(),
                    std=True,
                )
            else:
                ystd = ybg_std = None
            xfit = data.get_fit_xdata()
            yfit = data.get_fit_ydata(
                last_n=self.lastnBox.value(), normalize_n=self.normalizenBox.value()
            )

            _show_fit = show_fit and (xfit is not None) and (yfit is not None)
            w = 0.5 if _show_fit else 1
            pen = pg.mkPen(c.color, width=w)
            pen_bg = pg.mkPen(c.color_bg, width=w)

            if ystd is not None:
                self.add_error_bar(x, y, 2 * ystd, pen)
            if ybg_std is not None:
                self.add_error_bar(x, y_bg, 2 * ybg_std, pen_bg)

            self.plot.plot(
                x,
                y,
                pen=pen,
                symbolPen=None,
                symbol="o",
                symbolSize=8,
                symbolBrush=c.color,
            )
            if y_bg is not None:
                self.plot.plot(
                    x,
                    y_bg,
                    pen=pen_bg,
                    symbolPen=None,
                    symbol="o",
                    symbolSize=8,
                    symbolBrush=c.color_bg,
                )

            if _show_fit:
                self.plot.plot(xfit, yfit, pen=pg.mkPen(c.color, width=2))

        if self._normalize_updated:
            self.plot.autoRange()
            self._normalize_updated = False

    def toggle_image(self, show):
        if show:
            self.layout.addItem(self.img_plot, row=1, col=0)
        else:
            self.layout.removeItem(self.img_plot)

    def update_ylabel(self, normalize_n: int):
        if normalize_n:
            self.plot.setLabel("left", "Normalized Intensity")
        else:
            self.plot.setLabel("left", "Intensity", self._yunit)

    def update_normalize(self, normalize_n: int):
        self.update_ylabel(normalize_n)
        self._normalize_updated = True

    def update_font_size(self):
        font = QtGui.QFont()
        font.setPointSize(self.fontsizeBox.value())
        for p in (self.plot, self.img_plot):
            for l in ("bottom", "left"):
                p.getAxis(l).label.setFont(font)
                p.getAxis(l).setTickFont(font)


class ODMRPeaksWidget(QtWidgets.QWidget, Ui_ODMRPeaks):
    def __init__(self, plot, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.plot = plot

        peak_cols = ["#FF0000", "#FFA500", "#00CBFF", "#0000FF"]
        self.gs_peak_pairs = []
        for col in peak_cols:
            l0 = self.plot.addLine(x=0, pen=col, movable=False)
            l1 = self.plot.addLine(x=0, pen=col, movable=False)
            self.plot.removeItem(l0)
            self.plot.removeItem(l1)
            self.gs_peak_pairs.append([l0, l1])
        self.es_peak_pairs = []

        self.es_peak_pairs = []
        for col in peak_cols:
            p = pg.mkPen(color=col, style=QtCore.Qt.PenStyle.DashLine)
            l0 = self.plot.addLine(x=0, pen=p, movable=False)
            l1 = self.plot.addLine(x=0, pen=p, movable=False)
            self.plot.removeItem(l0)
            self.plot.removeItem(l1)
            self.es_peak_pairs.append([l0, l1])

        self.init_connections()

    def init_connections(self):
        self.substrate_buttons = QtWidgets.QButtonGroup(parent=self)
        for b in (self.sub100Button, self.sub111Button):
            self.substrate_buttons.addButton(b)
            b.toggled.connect(self.update_peak_positions)
            b.toggled.connect(self.update_B_angle_label)
            b.toggled.connect(self.update_E_angle_label)
        self.substrate_buttons.setExclusive(True)
        self.sub100Button.setChecked(True)
        for b in (
            self.BfieldBox,
            self.BthetaBox,
            self.BphiBox,
            self.EfieldBox,
            self.EthetaBox,
            self.EphiBox,
            self.DgsBox,
            self.DesBox,
            self.gammaBox,
            self.kperpBox,
        ):
            b.valueChanged.connect(self.update_peak_positions)
            b.valueChanged.connect(self.update_B_angle_label)
            b.valueChanged.connect(self.update_E_angle_label)
        self.stepBox.valueChanged.connect(self.update_peak_step)
        self.B111Button.clicked.connect(lambda: self.set_B_directions(1, 1, 1))
        self.B100Button.clicked.connect(lambda: self.set_B_directions(1, 0, 0))
        self.B110Button.clicked.connect(lambda: self.set_B_directions(1, 1, 0))
        self.E111Button.clicked.connect(lambda: self.set_E_directions(1, 1, 1))
        self.E100Button.clicked.connect(lambda: self.set_E_directions(1, 0, 0))
        self.E110Button.clicked.connect(lambda: self.set_E_directions(1, 1, 0))

        self.set_B_directions(1, 1, 1)
        self.set_E_directions(1, 1, 1)

        self.showgsBox.toggled.connect(self.toggle_gs_peaks)
        self.showesBox.toggled.connect(self.toggle_es_peaks)

    def toggle_gs_peaks(self, checked):
        if checked:
            for l0, l1 in self.gs_peak_pairs:
                self.plot.addItem(l0)
                self.plot.addItem(l1)
        else:
            for l0, l1 in self.gs_peak_pairs:
                self.plot.removeItem(l0)
                self.plot.removeItem(l1)

    def toggle_es_peaks(self, checked):
        if checked:
            for l0, l1 in self.es_peak_pairs:
                self.plot.addItem(l0)
                self.plot.addItem(l1)
        else:
            for l0, l1 in self.es_peak_pairs:
                self.plot.removeItem(l0)
                self.plot.removeItem(l1)

    def get_fields(self, unit=False):
        B = self.BfieldBox.value()
        theta_B = np.radians(self.BthetaBox.value())
        phi_B = np.radians(self.BphiBox.value())
        vB = np.array(
            [np.sin(theta_B) * np.cos(phi_B), np.sin(theta_B) * np.sin(phi_B), np.cos(theta_B)]
        )
        if not unit:
            vB = B * vB

        E = self.EfieldBox.value() * 1e-3  # V / um -> V / nm
        theta_E = np.radians(self.EthetaBox.value())
        phi_E = np.radians(self.EphiBox.value())
        vE = np.array(
            [np.sin(theta_E) * np.cos(phi_E), np.sin(theta_E) * np.sin(phi_E), np.cos(theta_E)]
        )
        if not unit:
            vE = E * vE

        if self.sub100Button.isChecked():
            vB = nv.project_from100(vB)
            vE = nv.project_from100(vE)

        return vB, vE

    def get_field_intensities(self) -> tuple[float, float]:
        return self.BfieldBox.value(), self.EfieldBox.value() * 1e-3

    def update_peak_positions(self):
        Dgs = self.DgsBox.value()
        Des = self.DesBox.value()
        gamma_e = self.gammaBox.value()
        k_perp = self.kperpBox.value()  # kHz um / V = MHz nm / V

        B, E = self.get_fields()

        # Ground state
        freq = nv.peaks_of_BE(B, E, D=Dgs, gamma=gamma_e, k_perp=k_perp)
        for i in range(4):
            l0, l1 = self.gs_peak_pairs[i]
            l0.setValue(freq[i * 2] * 1e6)
            l1.setValue(freq[i * 2 + 1] * 1e6)

        if self.sortgsBox.isChecked():
            f_str = sorted(set([f"{f:.3f}" for f in freq]))
        else:
            f_str = [f"{f:.3f}" for f in freq]
        txt = ", ".join(f_str)
        self.gsEdit.setText(txt + " MHz")

        # Excited state
        freq = nv.peaks_of_B_es(B, D=Des, gamma=gamma_e)
        for i in range(4):
            l0, l1 = self.es_peak_pairs[i]
            l0.setValue(freq[i * 2] * 1e6)
            l1.setValue(freq[i * 2 + 1] * 1e6)

        if self.sortesBox.isChecked():
            f_str = sorted(set([f"{f:.3f}" for f in freq]))
        else:
            f_str = [f"{f:.3f}" for f in freq]
        txt = ", ".join(f_str)
        self.esEdit.setText(txt + " MHz")

    def update_B_angle_label(self):
        vec_nv = nv.projector.AXIS
        vB, _ = self.get_fields(unit=True)
        B, _ = self.get_field_intensities()

        angles = np.degrees([np.arccos(np.dot(v, vB)) for v in vec_nv])
        min_angle = abs(angles[np.argmin(angles**2)])
        txt = "angle to <111>: {:.2f} degree".format(min_angle)
        self.BangleLabel.setText(txt)

        angle_rad = np.radians(min_angle)
        txtB = "B//: {:.3f} mT, B⊥: {:.3f} mT".format(B * np.cos(angle_rad), B * np.sin(angle_rad))
        self.BfieldLabel.setText(txtB)

    def update_E_angle_label(self):
        vec_nv = nv.projector.AXIS
        _, vE = self.get_fields(unit=True)
        _, E = self.get_field_intensities()

        angles = np.degrees([np.arccos(np.dot(v, vE)) for v in vec_nv])
        min_angle = abs(angles[np.argmin(angles**2)])
        txt = "angle to <111>: {:.2f} degree".format(min_angle)
        self.EangleLabel.setText(txt)

    def get_direction(self, k, l, m):
        d = np.array([k, l, m])
        d = d / np.linalg.norm(d)
        if self.sub100Button.isChecked():
            return d
        elif self.sub111Button.isChecked():
            # for (1, 1, 1) the small error is seen in phi = atan2(y, x).
            # return exact value to avoid this.
            if (k, l, m) == (1, 1, 1):
                return (0, 0, 1)
            return nv.project_from100(d)

    def set_B_directions(self, k, l, m):
        x, y, z = self.get_direction(k, l, m)

        theta = np.arccos(np.clip(z, -1, 1))
        phi = np.arctan2(y, x)

        self.BthetaBox.setValue(np.degrees(theta))
        self.BphiBox.setValue(np.degrees(phi))

    def set_E_directions(self, k, l, m):
        x, y, z = self.get_direction(k, l, m)

        theta = np.arccos(np.clip(z, -1, 1))
        phi = np.arctan2(y, x)

        self.EthetaBox.setValue(np.degrees(theta))
        self.EphiBox.setValue(np.degrees(phi))

    def update_peak_step(self, value):
        if value <= 0:
            return

        for w in (
            self.BfieldBox,
            self.BthetaBox,
            self.BphiBox,
            self.EfieldBox,
            self.EthetaBox,
            self.EphiBox,
        ):
            w.setSingleStep(value)


class ODMRWidget(ClientWidget, Ui_ODMR):
    """Widget for ODMR."""

    def __init__(
        self,
        gconf: dict,
        name,
        gparams_name,
        confocal_name,
        plot: PlotWidget,
        context,
        parent=None,
        logger=None,
        file_transport_dir=None,
        confocal_file_transport_dir=None,
    ):
        ClientWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)
        self.setupUi(self)

        self.conf = local_conf(gconf, name)

        self.plot = plot
        self.data = ODMRData()

        self.cli = QODMRClient(
            gconf, name, context=context, parent=self, file_transport_dir=file_transport_dir
        )
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)
        if confocal_name:
            self.confocal_cli = ConfocalIORequester(
                gconf,
                confocal_name,
                context=context,
                file_transport_dir=confocal_file_transport_dir,
            )
        else:
            self.confocal_cli = None

        self.add_clients(self.cli, self.gparams_cli, self.confocal_cli)

        self._finalizing = False

        self._fiTab_layout = QtWidgets.QVBoxLayout(self.fiTab)
        self.fit = ODMRFitWidget(self.cli, self.gparams_cli, parent=self.fiTab)
        self._fiTab_layout.addWidget(self.fit)

        self._peaksTab_layout = QtWidgets.QVBoxLayout(self.peaksTab)
        self.peaks = ODMRPeaksWidget(self.plot.plot, parent=self.peaksTab)
        self._peaksTab_layout.addWidget(self.peaks)

        self.setEnabled(False)

    def init_with_status(self, status: BinaryStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        self.init_widgets()
        self.init_connection()
        self.fit.init_with_status()

        self.update_step_label()

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
        self.validateButton.clicked.connect(self.request_validate)

        for w in (self.startBox, self.stopBox, self.numBox):
            w.valueChanged.connect(self.update_step_label)
        for w in (
            self.startBox,
            self.stopBox,
            self.numBox,
            self.powerBox,
            self.sweepsBox,
            self.delayBox,
            self.bgdelayBox,
            self.finaldelayBox,
        ):
            w.valueChanged.connect(self.reset_pulse_label)
        for w in (self.backgroundBox, self.mwcontBox):
            w.toggled.connect(self.reset_pulse_label)

        self._method_handler = ParamDictComboBoxHandler(self.methodBox)
        self.methodBox.currentIndexChanged.connect(self.switch_method)
        self.paramTable.paramsChanged.connect(self.reset_pulse_label)

    def init_widgets(self):
        self.tabWidget.setCurrentIndex(0)

        labels = self.cli.get_param_dict_labels()
        self.methodBox.clear()
        methods = P.filter_out_label_prefix("fit", labels)
        self.methodBox.addItems(methods)

        params = self.cli.get_param_dict("cw")
        apply_widgets(
            params,
            [
                ("power", self.powerBox),
                ("num", self.numBox),
                ("start", self.startBox, 1e-6),
                ("stop", self.stopBox, 1e-6),
                ("delay", self.delayBox, 1e3),
                ("background_delay", self.bgdelayBox, 1e3),
                ("final_delay", self.finaldelayBox, 1e3),
            ],
        )

        self.saveconfocalBox.setEnabled(self.confocal_cli is not None)
        self.saveconfocalBox.setChecked(self.confocal_cli is not None)

    def set_method(self, method: str) -> bool:
        i = self.methodBox.findText(method)
        if i >= 0:
            self.methodBox.setCurrentIndex(i)
            return True
        else:
            self.logger.error(f"Unknown method: {method}")
            return False

    def switch_method(self):
        params = self._method_handler.get(self.cli.get_param_dict)
        if params is None:
            return
        ps = P.ParamDict()
        for key in ("timing", "pd", "mod"):
            if key in params:
                ps[key] = params[key]
        self.paramTable.update_contents(ps)
        self.reset_pulse_label()

    def update_step_label(self):
        self.stepLabel.setText(
            "step: {:.4f} MHz".format(
                conv.num_to_step(self.startBox.value(), self.stopBox.value(), self.numBox.value())
            )
        )

    def reset_pulse_label(self):
        self.pulseLabel.setText("Not validated yet")

    def apply_widgets(self, data: ODMRData):
        start, stop = data.bounds()
        self.startBox.setValue(start * 1e-6)
        self.stopBox.setValue(stop * 1e-6)
        self.numBox.setValue(data.params["num"])
        self.powerBox.setValue(data.params["power"])
        bg = data.params.get("background", False)
        self.backgroundBox.setChecked(bg)
        if "sweeps" in data.params:
            self.sweepsBox.setValue(data.params["sweeps"])
        if "delay" in data.params:
            self.delayBox.setValue(data.params["delay"] * 1e3)
        if bg and "background_delay" in data.params:
            self.bgdelayBox.setValue(data.params["background_delay"] * 1e3)
        if "final_delay" in data.params:
            self.finaldelayBox.setValue(data.params["final_delay"] * 1e3)

        if not self.set_method(data.label):
            return

        ps = {}
        for key in ("timing", "pd", "mod"):
            if key in data.params:
                ps[key] = data.params[key]
        for k, v in P.flatten(ps).items():
            self.paramTable.apply_value(k, v)

    def save_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "ODMR", ".odmr")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.gparams_cli.get_param("note", "")
        self.cli.save_data(fn, note=note)
        n = os.path.splitext(fn)[0] + ".png"
        params = {}
        params["normalize_n"] = self.plot.normalizenBox.value()
        params["complex_conv"] = self.plot.complexBox.currentText()
        params["show_std"] = self.plot.showstdBox.isChecked()
        self.cli.export_data(n, params=params)

        if self.confocal_cli is not None and self.saveconfocalBox.isChecked():
            n = os.path.splitext(fn)[0] + ".pos.png"
            self.confocal_cli.export_view(n)

    def load_data(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "ODMR", ".odmr")
        if not fn:
            return

        self.gparams_cli.set_param("work_dir", os.path.split(fn)[0])
        data = self.cli.load_data(fn)
        if data is None:
            return
        if data.note():
            self.gparams_cli.set_param("loaded_note", data.note())

        self.refresh_all()
        self.apply_widgets(data)

    def export_data(self):
        data_list = self.get_plottable_data()
        if not data_list:
            return

        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "ODMR", (".png", ".pdf", ".eps", ".txt"))
        if not fn:
            return

        data = [d for (d, _, _) in data_list]
        params = {}
        params["color"] = params["color_fit"] = [c.color for (_, _, c) in data_list]
        params["color_bg"] = [c.color_bg for (_, _, c) in data_list]
        params["normalize_n"] = self.plot.normalizenBox.value()
        params["complex_conv"] = self.plot.complexBox.currentText()
        params["show_std"] = self.plot.showstdBox.isChecked()
        self.cli.export_data(fn, data=data, params=params)

    # State managements
    def request_stop(self):
        self.cli.stop()

    def get_params(self):
        params = {}
        params["start"] = self.startBox.value() * 1e6  # MHz to Hz
        params["stop"] = self.stopBox.value() * 1e6
        params["num"] = self.numBox.value()
        params["power"] = self.powerBox.value()
        params["sweeps"] = self.sweepsBox.value()
        params["background"] = self.backgroundBox.isChecked()
        params["delay"] = self.delayBox.value() * 1e-3  # ms to s
        params["background_delay"] = self.bgdelayBox.value() * 1e-3  # ms to s
        params["final_delay"] = self.finaldelayBox.value() * 1e-3  # ms to s
        params["continue_mw"] = self.mwcontBox.isChecked()

        params.update(P.unwrap(self.paramTable.params()))
        label = self.methodBox.currentText()

        return params, label

    def request_start(self):
        if self.request_validate():
            self.start_sweep()

    def request_validate(self) -> bool:
        params, label = self.get_params()
        success, error_message, pulse_summary = self.cli.validate(params, label)
        if not success:
            self.pulseLabel.setText(f"Validation failed: {error_message}")
        elif label == "pulse":
            self.pulseLabel.setText(pulse_summary)
        else:
            self.pulseLabel.setText("Parameters are valid")
        return success

    def start_sweep(self):
        params, label = self.get_params()

        title = "Continue sweep?"
        body = (
            "Continue sweep with current data?"
            + " Press No to clear current data and start a new sweep."
        )
        if self.data.can_resume(params, label) and Qt.question_yn(self, title, body):
            params["resume"] = True
            params["ident"] = self.data.ident
        else:
            params["resume"] = False
            params["ident"] = uuid.uuid4()

        self.cli.start(params, label)

    def update_data(self, data: ODMRData):
        self.data = data
        self.refresh_all()

    def update_buffer(self, buffer: Buffer[tuple[str, ODMRData]]):
        self.fit.update_buffer(buffer)
        self.plot.refresh_plot(self.get_plottable_data())

    def refresh_all(self):
        self.plot.refresh_all(self.get_plottable_data(), self.data)

    def get_plottable_data(self) -> list[tuple[ODMRData, bool, Colors]]:
        return self.fit.get_plottable_data(self.data)

    def finalize(self, data: ODMRData):
        if self._finalizing:
            return
        self._finalizing = True
        # Auto save is temporarily disabled.
        self._finalizing = False

    def update_state(self, state: BinaryState, last_state: BinaryState):
        is_IDLE = state == BinaryState.IDLE
        for w in (
            self.startButton,
            self.saveButton,
            self.exportButton,
            self.loadButton,
            self.validateButton,
            self.mwcontBox,
            self.startBox,
            self.stopBox,
            self.numBox,
            self.powerBox,
            self.sweepsBox,
            self.backgroundBox,
            self.delayBox,
            self.finaldelayBox,
            self.methodBox,
            self.paramTable,
        ):
            w.setEnabled(is_IDLE)

        self.bgdelayBox.setEnabled(is_IDLE and self.backgroundBox.isChecked())
        self.stopButton.setEnabled(state == BinaryState.ACTIVE)


class ODMRMainWindow(QtWidgets.QMainWindow):
    """MainWindow with ODMRWidget and PlotWidget."""

    def __init__(self, gconf: dict, name, context, parent=None, logger=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)

        lconf = local_conf(gconf, name)
        target = lconf["target"]

        self.plot = PlotWidget(parent=self)
        self.odmr = ODMRWidget(
            gconf,
            target["odmr"],
            target["gparams"],
            target.get("confocal"),
            self.plot,
            context,
            parent=self,
            logger=self.logger,
            file_transport_dir=lconf.get("file_transport_dir"),
            confocal_file_transport_dir=lconf.get("confocal_file_transport_dir"),
        )

        self.setWindowTitle(f"MAHOS.ODMRGUI ({join_name(target['odmr'])})")
        self.setAnimated(False)
        self.setCentralWidget(self.odmr)
        self.d_plot = QtWidgets.QDockWidget("Plot", parent=self)
        self.d_plot.setWidget(self.plot)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_plot)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_plot.toggleViewAction())

    def closeEvent(self, event):
        """Close the clients on closeEvent."""

        self.odmr.close_clients()
        # self.plot.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)


class ODMRGUI(GUINode):
    """GUINode for ODMR using ODMRWidget.

    :param target.log: Optional LogBroker target for formal GUI logging.
    :type target.log: tuple[str, str] | str
    :param log_level: (default: ``"INFO"``) Optional logging level.
    :type log_level: str
    :param target.odmr: Target ODMR node full name.
    :type target.odmr: tuple[str, str] | str
    :param target.gparams: Target GlobalParams node full name.
    :type target.gparams: tuple[str, str] | str
    :param target.confocal: Optional Confocal node used for view export on save.
    :type target.confocal: tuple[str, str] | str
    :param file_transport_dir: Optional GUI-side directory for shared file transport.
    :type file_transport_dir: str
    :param confocal_file_transport_dir: Optional GUI-side directory for shared file transport
        with the auxiliary Confocal node.
    :type confocal_file_transport_dir: str

    """

    def init_widget(self, gconf: dict, name, context):
        return ODMRMainWindow(gconf, name, context, logger=self.logger)
