#!/usr/bin/env python3

"""
GUI client for Pulse-based measurement to visualize the Pulse Pattern.

.. This file is a part of MAHOS project.

"""

import numpy as np
import pyqtgraph as pg

from .Qt import QtCore, QtWidgets

from .gui_node import GUINode
from .common_widget import ClientTopWidget
from .pulse_monitor_client import QPulseClient

from ..node.node import local_conf, join_name
from ..msgs.pulse_msgs import PulsePattern


Policy = QtWidgets.QSizePolicy.Policy


class PulseMonitorWidget(ClientTopWidget):
    """Top widget for PulseMonitor"""

    def __init__(self, gconf: dict, name, context):
        ClientTopWidget.__init__(self)

        self.conf = local_conf(gconf, name)
        target = self.conf["target"]["pulse"]
        self.cli = QPulseClient(gconf, target, context=context)
        self.add_client(self.cli)
        self.pulse = None
        self.markers = None

        # layout input UIs in hl
        hl = QtWidgets.QHBoxLayout()

        self.downsampleBox = QtWidgets.QSpinBox()
        self.downsampleBox.setPrefix("downsample: ")
        self.downsampleBox.setMinimum(1)
        self.downsampleBox.setMaximum(100)
        self.downsampleBox.setValue(10)

        self.realtimeBox = QtWidgets.QCheckBox("Real time")
        self.usemarkerBox = QtWidgets.QCheckBox("Use marker")
        self.usemarkerBox.setChecked(True)
        self.regionLabel = QtWidgets.QLabel("Fit region")

        self.indexBox = QtWidgets.QSpinBox()
        self.indexBox.setPrefix("index: ")
        self.indexBox.setMinimum(0)
        self.indexBox.setMaximum(100)
        self.indexBox.setValue(0)

        self.numBox = QtWidgets.QSpinBox()
        self.numBox.setPrefix("num: ")
        self.numBox.setMinimum(1)
        self.numBox.setValue(1)
        self.numBox.setMaximum(9999)

        for w in (self.downsampleBox, self.indexBox, self.numBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)

        for w in (
            self.downsampleBox,
            self.realtimeBox,
            self.usemarkerBox,
            self.regionLabel,
            self.indexBox,
            self.numBox,
        ):
            hl.addWidget(w)
        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        hl.addItem(spacer)

        self.downsampleBox.editingFinished.connect(self.update_plot)
        self.realtimeBox.toggled.connect(self.update_plot)
        self.usemarkerBox.toggled.connect(self.update_plot)
        self.indexBox.valueChanged.connect(self.set_region)
        self.numBox.valueChanged.connect(self.set_region)

        vl = QtWidgets.QVBoxLayout()
        glw = pg.GraphicsLayoutWidget()
        self.plot: pg.PlotItem = glw.addPlot(row=0, col=0)
        self.plot.setMouseEnabled(False, False)
        self.plot.addLegend()
        self.plot_sub: pg.PlotItem = glw.addPlot(row=1, col=0)
        self.plot_sub.setMouseEnabled(True, False)
        self.lines = []
        self.lines_sub = []
        self.cmap = pg.colormap.get(self.conf.get("colormap", "viridis"))

        self.lr = pg.LinearRegionItem([0, 1])
        self.lr.setZValue(-10)
        self.plot.addItem(self.lr)
        self.lr.sigRegionChanged.connect(self.update_plot_sub)
        self.plot_sub.sigXRangeChanged.connect(self.update_lr)

        vl.addLayout(hl)
        vl.addWidget(glw)
        self.setLayout(vl)
        self.setWindowTitle(f"MAHOS.PulseMonitor ({join_name(target)})")

        self.cli.pulseUpdated.connect(self.update)

    def update_plot_sub(self):
        self.plot_sub.setXRange(*self.lr.getRegion(), padding=0)

    def update_lr(self):
        self.lr.setRegion(self.plot_sub.getViewBox().viewRange()[0])

    def sizeHint(self):
        return QtCore.QSize(1400, 900)

    def update(self, pulse: PulsePattern):
        self.pulse = pulse
        self.update_plot()

    def update_markers(self):
        if self.usemarkerBox.isChecked() and self.pulse.markers:
            if self.pulse.markers[0]:
                markers = [0] + self.pulse.markers
            else:
                markers = self.pulse.markers
            self.markers = np.array(markers, dtype=np.uint64)
        else:
            p = np.array([0] + [b.total_length() for b in self.pulse.blocks], dtype=np.uint64)
            self.markers = np.cumsum(p)
        if self.realtimeBox.isChecked():
            self.markers = self.markers.astype(np.float64) / self.pulse.freq

        for l in self.lines:
            self.plot.removeItem(l)
        for ls in self.lines_sub:
            self.plot_sub.removeItem(ls)

        pen = pg.mkPen(0.3)
        for x in self.markers:
            l = pg.InfiniteLine(x, pen=pen)
            l.setZValue(-20)
            ls = pg.InfiniteLine(x, pen=pen)
            ls.setZValue(-20)
            self.plot.addItem(l)
            self.lines.append(l)
            self.plot_sub.addItem(ls)
            self.lines_sub.append(ls)

    def update_plot(self):
        if self.pulse is None:
            return

        channels, patterns = self.pulse.blocks.decode_all()

        if not channels:
            print("[ERROR] empty pulse pattern")
            return

        num = len(channels)
        step = self.downsampleBox.value()

        self.plot.clearPlots()
        self.plot_sub.clearPlots()

        x = np.arange(len(patterns[0]))[::step]
        if self.realtimeBox.isChecked():
            x = x.astype(np.float64) / self.pulse.freq
            self.plot.setLabel("bottom", "time", "s")
            self.plot_sub.setLabel("bottom", "time", "s")
        else:
            self.plot.setLabel("bottom", "sampling point")
            self.plot_sub.setLabel("bottom", "sampling point")

        for i, (ch, pat) in enumerate(zip(channels, patterns)):
            offset = (num - 1 - i) * 1.05
            if num > 1:
                pen = self.cmap.map(i / (num - 1))
            else:
                pen = self.cmap.map(0.5)

            pat = pat[::step]  # downsample
            self.plot.plot(x, pat + offset, name=ch, pen=pen)
            self.plot_sub.plot(x, pat + offset, name=ch, pen=pen)
        self.update_markers()

        self.plot_sub.autoRange()
        self.set_region()

    def set_region(self):
        if self.markers is None:
            return
        n_markers = len(self.markers)
        self.indexBox.setMaximum(max(0, n_markers - 2))
        self.numBox.setMaximum(n_markers - 1)
        head = self.indexBox.value()
        tail = min(head + self.numBox.value(), n_markers - 1)
        self.lr.setRegion((self.markers[head], self.markers[tail]))


class PulseMonitor(GUINode):
    def init_widget(self, gconf: dict, name, context):
        return PulseMonitorWidget(gconf, name, context)
