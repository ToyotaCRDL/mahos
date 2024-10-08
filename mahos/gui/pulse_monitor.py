#!/usr/bin/env python3

"""
GUI client for Pulse-based measurement to visualize the Pulse Pattern.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

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
        self.add_clients(self.cli)
        self.pulse = None
        self.markers = None

        # layout input UIs in hl
        hl = QtWidgets.QHBoxLayout()

        self.maxpointsBox = QtWidgets.QSpinBox()
        self.maxpointsBox.setPrefix("limit: ")
        self.maxpointsBox.setSuffix(" kpts")
        self.maxpointsBox.setMinimum(10)
        self.maxpointsBox.setMaximum(10_000)
        self.maxpointsBox.setValue(500)

        self.realtimeBox = QtWidgets.QCheckBox("Real time")
        self.realtimeBox.setChecked(True)
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

        for w in (self.maxpointsBox, self.indexBox, self.numBox):
            w.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            w.setMaximumWidth(200)

        for w in (
            self.maxpointsBox,
            self.realtimeBox,
            self.usemarkerBox,
            self.regionLabel,
            self.indexBox,
            self.numBox,
        ):
            hl.addWidget(w)
        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        hl.addItem(spacer)

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
        self._mx = 0.0
        self._my = 0.0
        self.cursor0 = None
        self.cursor1 = None
        self.cmap = pg.colormap.get(self.conf.get("colormap", "viridis"))

        self.lr = pg.LinearRegionItem([0, 1])
        self.lr.setZValue(-10)
        self.plot.addItem(self.lr)
        self.lr.sigRegionChanged.connect(self.update_plot_sub)
        self.plot_sub.sigXRangeChanged.connect(self.update_lr)
        self.plot_sub.scene().sigMouseMoved.connect(self.update_pos)
        self.plot_sub.scene().sigMouseClicked.connect(self.update_cursor)

        hl_bottom = QtWidgets.QHBoxLayout()
        self.clearButton = QtWidgets.QPushButton("Clear Cursors")
        self.clearButton.clicked.connect(self.clear_cursors)
        self.label = QtWidgets.QLabel("Ready")
        hl_bottom.addWidget(self.clearButton)
        hl_bottom.addWidget(self.label)
        spacer = QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum)
        hl_bottom.addItem(spacer)
        hint = QtWidgets.QLabel("Shift+Click to put C0, Ctrl+Click to put C1")
        hl_bottom.addWidget(hint)

        vl.addLayout(hl)
        vl.addWidget(glw)
        vl.addLayout(hl_bottom)
        self.setLayout(vl)
        self.setWindowTitle(f"MAHOS.PulseMonitor ({join_name(target)})")

        self.cli.pulseUpdated.connect(self.update)

    def update_plot_sub(self):
        self.plot_sub.setXRange(*self.lr.getRegion(), padding=0)

    def update_lr(self):
        self.lr.setRegion(self.plot_sub.getViewBox().viewRange()[0])

    def clear_cursors(self):
        if self.cursor0 is not None:
            for l in self.cursor0:
                self.plot_sub.removeItem(l)
            self.cursor0 = None
        if self.cursor1 is not None:
            for l in self.cursor1:
                self.plot_sub.removeItem(l)
            self.cursor1 = None
        self.update_label()

    def update_label(self):
        scale, prefix = pg.siScale(self._mx)
        s = f"M({scale*self._mx:7.3f} {prefix}s, {self._my:7.3f})"
        if self.cursor0 is not None:
            x0 = self.cursor0[0].value()
            y0 = self.cursor0[1].value()
            scale, prefix = pg.siScale(x0)
            s += f"  C0({scale*x0:7.3f} {prefix}s, {y0:7.3f})"
        if self.cursor1 is not None:
            x1 = self.cursor1[0].value()
            y1 = self.cursor1[1].value()
            scale, prefix = pg.siScale(x1)
            s += f"  C1({scale*x1:7.3f} {prefix}s, {y1:7.3f})"
        if self.cursor0 is not None and self.cursor1 is not None:
            scale, prefix = pg.siScale(x1 - x0)
            s += f"  Delta({scale*(x1-x0):7.3f} {prefix}s, {y1-y0:7.3f})"
        self.label.setText(s)

    def update_pos(self, pos):
        if not self.plot_sub.sceneBoundingRect().contains(pos):
            return
        point = self.plot_sub.getViewBox().mapSceneToView(pos)
        self._mx = point.x()
        self._my = point.y()
        self.update_label()

    def update_cursor(self, ev):
        pos = ev.scenePos()
        if not self.plot_sub.sceneBoundingRect().contains(pos):
            return
        point = self.plot_sub.getViewBox().mapSceneToView(pos)
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            if self.cursor0 is None:
                pen = pg.mkPen("r", style=QtCore.Qt.PenStyle.DashLine)
                vline = pg.InfiniteLine(pos=point.x(), angle=90, pen=pen)
                hline = pg.InfiniteLine(pos=point.y(), angle=0, pen=pen)
                # vline.setZValue(-20)
                self.plot_sub.addItem(vline)
                self.plot_sub.addItem(hline)
                self.cursor0 = (vline, hline)
            else:
                self.cursor0[0].setPos(point.x())
                self.cursor0[1].setPos(point.y())
            self.update_label()
        elif ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            if self.cursor1 is None:
                pen = pg.mkPen("r")
                vline = pg.InfiniteLine(pos=point.x(), angle=90, pen=pen)
                hline = pg.InfiniteLine(pos=point.y(), angle=0, pen=pen)
                # vline.setZValue(-20)
                self.plot_sub.addItem(vline)
                self.plot_sub.addItem(hline)
                self.cursor1 = (vline, hline)
            else:
                self.cursor1[0].setPos(point.x())
                self.cursor1[1].setPos(point.y())
            self.update_label()

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

    def plottable_all_d(self, blocks, max_len):
        channels = blocks.digital_channels()
        try:
            channels = list(sorted(channels))
        except TypeError:
            channels = list(channels)
        patterns = [blocks.plottable_digital(ch, max_len=max_len) for ch in channels]
        return channels, patterns

    def plottable_all_a(self, blocks, max_len):
        channels = blocks.analog_channels()
        channels = list(sorted(channels))
        patterns = [blocks.plottable_analog(ch, max_len=max_len) for ch in channels]
        return channels, patterns

    def update_plot(self):
        if self.pulse is None:
            return

        max_len = self.maxpointsBox.value() * 1000
        d_channels, d_patterns = self.plottable_all_d(self.pulse.blocks, max_len)
        a_channels, a_patterns = self.plottable_all_a(self.pulse.blocks, max_len)

        if not d_channels and not a_channels:
            print("[ERROR] empty pulse pattern")
            return

        self.plot.clearPlots()
        self.plot_sub.clearPlots()

        x = self.pulse.blocks.plottable_time(max_len=max_len)
        if self.realtimeBox.isChecked():
            x = x.astype(np.float64) / self.pulse.freq
            self.plot.setLabel("bottom", "time", "s")
            self.plot_sub.setLabel("bottom", "time", "s")
        else:
            self.plot.setLabel("bottom", "sampling point")
            self.plot_sub.setLabel("bottom", "sampling point")

        num = len(a_channels) + len(d_channels)
        for i, (ch, pat) in enumerate(zip(d_channels, d_patterns)):
            offset = num - 1 - i
            pen = self.cmap.map(i / (num - 1)) if num > 1 else self.cmap.map(0.5)
            self.plot.plot(x, pat * 0.8 + offset, name=ch, pen=pen)
            self.plot_sub.plot(x, pat * 0.8 + offset, name=ch, pen=pen)
        pen_base = pg.mkPen(0.3)
        for i, (ch, pat) in enumerate(zip(a_channels, a_patterns)):
            j = i + len(d_channels)
            offset = num - 1 - j
            pen = self.cmap.map(j / (num - 1)) if num > 1 else self.cmap.map(0.5)
            # draw baseline for analog channels only
            # plot() baseline instead of InfiniteLine,
            # which won't be erased by clearPlots().
            self.plot.plot([x[0], x[-1]], [offset, offset], pen=pen_base)
            self.plot.plot(x, pat + offset, name=ch, pen=pen)
            self.plot_sub.plot([x[0], x[-1]], [offset, offset], pen=pen_base)
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
