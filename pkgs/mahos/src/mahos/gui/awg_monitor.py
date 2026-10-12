#!/usr/bin/env python3

"""GUI client for visualizing bounded AWG waveform previews."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from mahos.gui.Qt import QtCore, QtWidgets
from mahos.gui.awg_monitor_client import QAWGClient
from mahos.gui.common_widget import ClientTopWidget
from mahos.gui.gui_node import GUINode, get_gui_logger
from mahos.msgs.inst.awg_msgs import AWGWaveform
from mahos.node.node import NAME_DELIM, local_conf, split_name


Policy = QtWidgets.QSizePolicy.Policy


def _targets(conf: dict):
    targets = conf["target"]["wave"]
    if isinstance(targets, str):
        return [targets]
    if (
        isinstance(targets, (list, tuple))
        and len(targets) == 2
        and all(isinstance(target, str) for target in targets)
        and all(NAME_DELIM not in target for target in targets)
    ):
        return [tuple(targets)]
    if isinstance(targets, tuple):
        targets = list(targets)
    if not targets:
        raise ValueError("target.wave must not be empty.")
    return targets


def _rle_steps(runs: list[tuple[bool, int]], stop: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    position = 0
    for value, length in runs:
        if position >= stop:
            break
        end = min(position + length, stop)
        x.extend((position, end))
        y.extend((float(value), float(value)))
        position = end
    return np.asarray(x, dtype=np.uint64), np.asarray(y, dtype=np.float32)


class AWGMonitorWidget(ClientTopWidget):
    """Top widget for :class:`AWGMonitor`."""

    def __init__(self, gconf: dict, name, context, parent=None, logger=None):
        ClientTopWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)
        self.conf = local_conf(gconf, name)
        self.targets = _targets(self.conf)
        self.clis = [QAWGClient(gconf, target, context=context) for target in self.targets]
        self.add_clients(*self.clis)
        self.wave = None
        self.markers = None
        self._marker_lines = []
        self._mx = 0.0
        self._my = 0.0
        self.cursor0 = None
        self.cursor1 = None
        self.cmap = pg.colormap.get(self.conf.get("colormap", "viridis"))

        controls = QtWidgets.QHBoxLayout()
        self.maxpointsBox = QtWidgets.QSpinBox()
        self.maxpointsBox.setPrefix("display limit: ")
        self.maxpointsBox.setSuffix(" kpts")
        self.maxpointsBox.setRange(10, 10_000)
        self.maxpointsBox.setValue(500)
        self.realtimeBox = QtWidgets.QCheckBox("Real time")
        self.realtimeBox.setChecked(True)
        self.usemarkerBox = QtWidgets.QCheckBox("Use marker")
        self.usemarkerBox.setChecked(True)
        self.regionLabel = QtWidgets.QLabel("Fit region")
        self.indexBox = QtWidgets.QSpinBox()
        self.indexBox.setPrefix("index: ")
        self.indexBox.setRange(0, 100)
        self.numBox = QtWidgets.QSpinBox()
        self.numBox.setPrefix("num: ")
        self.numBox.setRange(1, 9999)
        for widget in (self.maxpointsBox, self.indexBox, self.numBox):
            widget.setSizePolicy(Policy.MinimumExpanding, Policy.Minimum)
            widget.setMaximumWidth(220)
        for widget in (
            self.maxpointsBox,
            self.realtimeBox,
            self.usemarkerBox,
            self.regionLabel,
            self.indexBox,
            self.numBox,
        ):
            controls.addWidget(widget)
        controls.addItem(QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum))

        glw = pg.GraphicsLayoutWidget()
        self.plot = glw.addPlot(row=0, col=0, title="Analog overview")
        self.plot_d = glw.addPlot(row=1, col=0, title="Digital overview")
        self.plot_sub = glw.addPlot(row=2, col=0, title="Analog detail")
        self.plot_sub_d = glw.addPlot(row=3, col=0, title="Digital detail")
        self.plots = (self.plot, self.plot_d, self.plot_sub, self.plot_sub_d)
        self.plot.addLegend()
        self.plot_d.addLegend()
        self.plot_sub.addLegend()
        self.plot_sub_d.addLegend()
        self.plot.setMouseEnabled(False, False)
        self.plot_d.setMouseEnabled(False, False)
        self.plot_sub.setMouseEnabled(True, False)
        self.plot_sub_d.setMouseEnabled(True, False)
        self.plot_d.setXLink(self.plot)
        self.plot_sub_d.setXLink(self.plot_sub)
        self.plot_d.setMaximumHeight(180)
        self.plot_sub_d.setMaximumHeight(180)
        self.plot.setLabel("left", "output", "mV")
        self.plot_sub.setLabel("left", "output", "mV")
        self.plot_d.setLabel("left", "digital")
        self.plot_sub_d.setLabel("left", "digital")

        self.lr = pg.LinearRegionItem([0, 1])
        self.lr.setZValue(-10)
        self.plot.addItem(self.lr)
        self.lr.sigRegionChanged.connect(self.update_plot_sub)
        self.plot_sub.sigXRangeChanged.connect(self.update_lr)
        self.plot_sub.scene().sigMouseMoved.connect(self.update_pos)
        self.plot_sub.scene().sigMouseClicked.connect(self.update_cursor)

        bottom = QtWidgets.QHBoxLayout()
        self.clearButton = QtWidgets.QPushButton("Clear Cursors")
        self.clearButton.clicked.connect(self.clear_cursors)
        self.positionLabel = QtWidgets.QLabel("Ready")
        self.previewLabel = QtWidgets.QLabel("Waiting for AWG waveform")
        bottom.addWidget(self.clearButton)
        bottom.addWidget(self.positionLabel)
        bottom.addItem(QtWidgets.QSpacerItem(40, 20, Policy.Expanding, Policy.Minimum))
        bottom.addWidget(self.previewLabel)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(glw)
        layout.addLayout(bottom)
        self.setLayout(layout)
        node_names = [split_name(target)[1] for target in self.targets]
        self.setWindowTitle(f"MAHOS.AWGMonitor ({', '.join(node_names)})")

        self.maxpointsBox.valueChanged.connect(self.update_plot)
        self.realtimeBox.toggled.connect(self.update_plot)
        self.usemarkerBox.toggled.connect(self.update_plot)
        self.indexBox.valueChanged.connect(self.set_region)
        self.numBox.valueChanged.connect(self.set_region)
        for cli in self.clis:
            cli.awgUpdated.connect(self.update)

    def sizeHint(self):
        return QtCore.QSize(1400, 1000)

    def _scale_x(self, x):
        if self.realtimeBox.isChecked():
            return np.asarray(x, dtype=np.float64) / self.wave.rate
        return x

    def update(self, wave: AWGWaveform):
        self.wave = wave
        self.update_plot()

    def update_plot_sub(self):
        self.plot_sub.setXRange(*self.lr.getRegion(), padding=0)

    def update_lr(self):
        self.lr.setRegion(self.plot_sub.getViewBox().viewRange()[0])

    def _clear_marker_lines(self):
        for plot, line in self._marker_lines:
            plot.removeItem(line)
        self._marker_lines.clear()

    def _update_markers(self):
        if self.usemarkerBox.isChecked() and self.wave.markers:
            markers = list(self.wave.markers)
            if markers[0] != 0:
                markers.insert(0, 0)
            if markers[-1] != self.wave.preview_samples:
                markers.append(self.wave.preview_samples)
        else:
            markers = [0, self.wave.preview_samples]
        self.markers = self._scale_x(np.asarray(sorted(set(markers)), dtype=np.uint64))
        pen = pg.mkPen(0.3)
        for x in self.markers:
            for plot in self.plots:
                line = pg.InfiniteLine(x, pen=pen)
                line.setZValue(-20)
                plot.addItem(line)
                self._marker_lines.append((plot, line))
        if self.wave.rendered_samples < self.wave.actual_samples:
            boundary = self.wave.rendered_samples
            if boundary <= self.wave.preview_samples:
                boundary = self._scale_x(boundary)
                pen = pg.mkPen("y", style=QtCore.Qt.PenStyle.DashLine)
                for plot in self.plots:
                    line = pg.InfiniteLine(boundary, pen=pen)
                    line.setZValue(-15)
                    plot.addItem(line)
                    self._marker_lines.append((plot, line))

    def update_plot(self):
        if self.wave is None:
            return
        self._clear_marker_lines()
        for plot in self.plots:
            plot.clearPlots()

        unit = ("s",) if self.realtimeBox.isChecked() else None
        for plot in self.plots:
            if unit:
                plot.setLabel("bottom", "time", *unit)
            else:
                plot.setLabel("bottom", "sampling point")

        limit = self.maxpointsBox.value() * 1000
        channels = sorted(self.wave.analog)
        for i, channel in enumerate(channels):
            indices, normalized = self.wave.analog[channel]
            indices = indices[:limit]
            normalized = normalized[:limit]
            x = self._scale_x(indices)
            y = normalized * self.wave.amplitude_mV[channel]
            pen = (
                self.cmap.map(i / (len(channels) - 1)) if len(channels) > 1 else self.cmap.map(0.5)
            )
            name = f"CH{channel}"
            self.plot.plot(x, y, name=name, pen=pen)
            self.plot_sub.plot(x, y, name=name, pen=pen)

        digital_names = sorted(self.wave.digital)
        for i, name in enumerate(digital_names):
            x, value = _rle_steps(self.wave.digital[name], self.wave.preview_samples)
            x = x[:limit]
            value = value[:limit]
            x = self._scale_x(x)
            offset = len(digital_names) - 1 - i
            y = value * 0.8 + offset
            pen = (
                self.cmap.map(i / (len(digital_names) - 1))
                if len(digital_names) > 1
                else self.cmap.map(0.5)
            )
            self.plot_d.plot(x, y, name=name, pen=pen)
            self.plot_sub_d.plot(x, y, name=name, pen=pen)

        self._update_markers()
        preview_points = max((len(values) for _, values in self.wave.analog.values()), default=0)
        shown_points = min(preview_points, limit)
        truncated = self.wave.preview_samples < self.wave.actual_samples
        suffix = ", prefix truncated" if truncated else ""
        if self.wave.rendered_samples < self.wave.actual_samples:
            if self.wave.rendered_samples > self.wave.preview_samples:
                suffix += f", padding starts at sample {self.wave.rendered_samples:_d}"
            else:
                suffix += ", yellow line = padding boundary"
        self.previewLabel.setText(
            f"showing {shown_points:_d}/{preview_points:_d} pts/ch from first "
            f"{self.wave.preview_samples:_d} of {self.wave.actual_samples:_d} samples "
            f"({self.wave.reduction} {self.wave.reduction_factor:.1f}x{suffix})"
        )
        self.plot_sub.autoRange()
        self.plot_sub_d.autoRange()
        self.set_region()

    def set_region(self):
        if self.markers is None or len(self.markers) < 2:
            return
        n_markers = len(self.markers)
        self.indexBox.setMaximum(max(0, n_markers - 2))
        self.numBox.setMaximum(n_markers - 1)
        head = self.indexBox.value()
        tail = min(head + self.numBox.value(), n_markers - 1)
        self.lr.setRegion((self.markers[head], self.markers[tail]))

    def clear_cursors(self):
        for cursor in (self.cursor0, self.cursor1):
            if cursor is not None:
                for line in cursor:
                    self.plot_sub.removeItem(line)
        self.cursor0 = self.cursor1 = None
        self.update_label()

    def _format_x(self, value: float) -> str:
        if self.realtimeBox.isChecked():
            scale, prefix = pg.siScale(value)
            return f"{scale * value:7.3f} {prefix}s"
        return f"{value:,.1f} samples"

    def update_label(self):
        text = f"M({self._format_x(self._mx)}, {self._my:7.3f} mV)"
        for name, cursor in (("C0", self.cursor0), ("C1", self.cursor1)):
            if cursor is not None:
                text += (
                    f"  {name}({self._format_x(cursor[0].value())}, {cursor[1].value():.3f} mV)"
                )
        if self.cursor0 is not None and self.cursor1 is not None:
            dx = self.cursor1[0].value() - self.cursor0[0].value()
            text += (
                f"  Delta({self._format_x(dx)}, "
                f"{self.cursor1[1].value() - self.cursor0[1].value():.3f} mV)"
            )
        self.positionLabel.setText(text)

    def update_pos(self, pos):
        if self.plot_sub.sceneBoundingRect().contains(pos):
            point = self.plot_sub.getViewBox().mapSceneToView(pos)
            self._mx, self._my = point.x(), point.y()
            self.update_label()

    def update_cursor(self, event):
        if not self.plot_sub.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plot_sub.getViewBox().mapSceneToView(event.scenePos())
        modifiers = event.modifiers()
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.cursor0 = self._set_cursor(self.cursor0, point, dashed=True)
        elif modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.cursor1 = self._set_cursor(self.cursor1, point, dashed=False)
        self.update_label()

    def _set_cursor(self, cursor, point, dashed: bool):
        if cursor is None:
            style = QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine
            pen = pg.mkPen("r", style=style)
            cursor = (
                pg.InfiniteLine(pos=point.x(), angle=90, pen=pen),
                pg.InfiniteLine(pos=point.y(), angle=0, pen=pen),
            )
            for line in cursor:
                self.plot_sub.addItem(line)
        else:
            cursor[0].setPos(point.x())
            cursor[1].setPos(point.y())
        return cursor


class AWGMonitor(GUINode):
    """GUINode that visualizes bounded AWG waveform previews.

    :param target.log: Optional LogBroker target for formal GUI logging.
    :type target.log: tuple[str, str] | str
    :param log_level: (default: ``"INFO"``) Optional logging level.
    :type log_level: str
    :param target.awg: Target AWG-waveform publisher node name(s).
    :type target.awg: str | tuple[str, str] | list[str]
    :param colormap: PyQtGraph colormap name used for traces.
    :type colormap: str

    """

    def init_widget(self, gconf: dict, name, context):
        return AWGMonitorWidget(gconf, name, context, logger=self.logger)
