#!/usr/bin/env python3

"""
GUI frontend of Confocal.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import os
from functools import partial
import uuid
import typing as T

from . import Qt
from .Qt import QtCore, QtGui, QtWidgets

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.functions import mkPen

from .ui.confocal import Ui_Confocal
from .ui.scanDialog import Ui_scanDialog
from .ui.trackDialog import Ui_trackDialog
from .ui.imageDialog import Ui_imageDialog
from .ui.traceView import Ui_traceView

from ..msgs.common_msgs import BinaryState
from ..msgs.confocal_msgs import (
    Axis,
    ConfocalStatus,
    ConfocalState,
    ScanDirection,
    ScanMode,
    LineMode,
    Image,
    Trace,
)
from ..msgs.confocal_msgs import direction_to_str, str_to_direction, mode_to_str, str_to_mode
from ..msgs.confocal_msgs import line_mode_to_str, str_to_line_mode
from ..msgs.confocal_tracker_msgs import OptMode
from ..meas.confocal_worker import DEFAULT_TRACER_SIZE
from .confocal_client import QConfocalClient, QConfocalTrackerClient, QTracerClient
from ..node.param_server import ParamClient
from ..node.node import local_conf, join_name
from ..util import conv
from ..util.plot import colors_tab20_pair
from ..util.stat import simple_moving_average
from ..util.timer import FPSCounter
from .gui_node import GUINode
from .dialog import save_dialog, load_dialog, export_dialog
from .common_widget import ClientWidget


class imageDialog(QtWidgets.QDialog, Ui_imageDialog):
    """Dialog for Image save condition."""

    def __init__(self, levels, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.minBox.setValue(levels[0])
        self.maxBox.setValue(levels[1])

        self.minBox.valueChanged.connect(self.update_maxBox)
        self.maxBox.valueChanged.connect(self.update_minBox)

    def validate(self):
        if self.minBox.value() < self.maxBox.value():
            return (True, "")
        else:
            return (False, "Max value must be greater than min value.")

    def update_minBox(self):
        self.minBox.setMaximum(self.maxBox.value())

    def update_maxBox(self):
        self.maxBox.setMinimum(self.minBox.value())

    def get_params(self):
        p = {}
        p["vmin"] = self.minBox.value()
        p["vmax"] = self.maxBox.value()
        p["figsize"] = (self.sizexBox.value(), self.sizeyBox.value())
        p["dpi"] = self.dpiBox.value()
        p["fontsize"] = self.fontsizeBox.value()
        p["aspect"] = "equal" if self.aspectBox.isChecked() else "auto"
        p["cmap"] = self.cmapBox.currentText()
        p["set_pos"] = self.posBox.isChecked()
        p["only"] = self.onlyBox.isChecked()
        return p


class scanDialog(QtWidgets.QDialog, Ui_scanDialog):
    """Dialog for Scanning function."""

    def __init__(
        self,
        cli,
        direction,
        xbound,
        ybound,
        zbound,
        roibound,
        pos,
        scan_capability,
        default_path,
        parent=None,
    ):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.cli = cli
        self.xmin, self.xmax = xbound
        self.ymin, self.ymax = ybound
        self.zmin, self.zmax = zbound
        self.roibound = roibound
        x, y, z = pos
        self.xy_pos = x, y

        self.direction = direction
        if self.direction == ScanDirection.XZ:
            self.msgLabel.setText("Set XZ Scan parameter")
            self.yLabel.setText("Z")
            self.ysizeBox.setPrefix("Z size: ")
            self.zBox.setPrefix("Y: ")
        elif self.direction == ScanDirection.YZ:
            self.msgLabel.setText("Set YZ Scan parameter")
            self.xLabel.setText("Y")
            self.yLabel.setText("Z")
            self.xsizeBox.setPrefix("Y size: ")
            self.ysizeBox.setPrefix("Z size: ")
            self.zBox.setPrefix("X: ")

        self.roiButton.clicked.connect(self.set_roi)
        self.roicenterButton.clicked.connect(self.set_roi_center)
        self.fullButton.clicked.connect(self.set_full)
        self.sizeButton.clicked.connect(self.set_size)
        self.loadButton.clicked.connect(self.load_parameter)

        for w in (self.xminBox, self.xmaxBox):
            w.setMinimum(self.xmin)
            w.setMaximum(self.xmax)
        for w in (self.yminBox, self.ymaxBox):
            w.setMinimum(self.ymin)
            w.setMaximum(self.ymax)
        self.xsizeBox.setMaximum(self.xmax)
        self.ysizeBox.setMaximum(self.ymax)
        self.zBox.setMinimum(self.zmin)
        self.zBox.setMaximum(self.zmax)

        for w in (self.xminBox, self.xmaxBox, self.xnumBox):
            w.valueChanged.connect(self.update_xstep_size)
        for w in (self.yminBox, self.ymaxBox, self.ynumBox):
            w.valueChanged.connect(self.update_ystep_size)

        self.modeBox.clear()
        for mode in scan_capability:
            self.modeBox.addItem(mode_to_str(mode))

        self.modeBox.currentIndexChanged.connect(self.update_modal_boxes)
        self.modeBox.setCurrentIndex(0)

        self.update_modal_boxes()
        self.update_xstep_size()
        self.update_ystep_size()

        self.set_roi()
        self.zBox.setValue(z)
        self.default_path = default_path

    def _set_with_clip(self, xmin, xmax, ymin, ymax):
        self.xminBox.setValue(max(self.xmin, xmin))
        self.xmaxBox.setValue(min(self.xmax, xmax))
        self.yminBox.setValue(max(self.ymin, ymin))
        self.ymaxBox.setValue(min(self.ymax, ymax))

    def set_roi(self):
        self.xminBox.setValue(self.roibound.x())
        self.xmaxBox.setValue(self.roibound.x() + self.roibound.width())
        self.yminBox.setValue(self.roibound.y())
        self.ymaxBox.setValue(self.roibound.y() + self.roibound.height())

    def set_roi_center(self):
        x, y = self.xy_pos
        w = self.roibound.width() / 2.0
        h = self.roibound.height() / 2.0
        self._set_with_clip(x - w, x + w, y - h, y + h)

    def set_full(self):
        self.xminBox.setValue(self.xmin)
        self.xmaxBox.setValue(self.xmax)
        self.yminBox.setValue(self.ymin)
        self.ymaxBox.setValue(self.ymax)

    def set_size(self):
        x, y = self.xy_pos
        w = self.xsizeBox.value() / 2.0
        h = self.ysizeBox.value() / 2.0
        self._set_with_clip(x - w, x + w, y - h, y + h)

    def load_parameter(self):
        d_str = direction_to_str(self.direction)
        fn = load_dialog(self, self.default_path, f"{d_str} Scan", "." + d_str.lower() + ".scan")
        if not fn:
            return

        image = self.cli.load_image(fn)
        if self.direction != image.direction:
            msg = "Current direction {} is not matching with loaded image {}".format(
                self.direction, image.direction
            )
            QtWidgets.QMessageBox.warning(self, "Direction doesn't match", msg)
            return

        params = image.params
        self.xminBox.setValue(params["xmin"])
        self.xmaxBox.setValue(params["xmax"])
        self.xnumBox.setValue(params["xnum"])
        self.yminBox.setValue(params["ymin"])
        self.ymaxBox.setValue(params["ymax"])
        self.ynumBox.setValue(params["ynum"])
        self.set_window(params["time_window"])
        mode = params["mode"]
        self.modeBox.setCurrentIndex(self.modeBox.findText(mode_to_str(mode)))
        self.linemodeBox.setCurrentIndex(
            self.linemodeBox.findText(line_mode_to_str(params.get("line_mode", LineMode.ASCEND)))
        )
        if mode == ScanMode.ANALOG:
            self.dummysampleBox.setValue(params.get("dummy_samples", 1))
            self.oversampleBox.setValue(params.get("oversample", 1))
        if mode == ScanMode.COM_DIPOLL:
            self.pollsampleBox.setValue(params.get("poll_samples", 1000))
        if mode != ScanMode.ANALOG:
            self.set_delay(params.get("delay", 0.0))

        if self.direction == ScanDirection.XY:
            _z = "Z"
        elif self.direction == ScanDirection.XZ:
            _z = "Y"
        else:
            _z = "X"

        yes = Qt.question_yn(
            self,
            f"Load {_z} as well?",
            "Do you want to load {} value ({:.4f}) from the file?".format(_z, params["z"]),
        )
        if yes:
            self.zBox.setValue(params["z"])

    def update_xstep_size(self):
        self.xstepLabel.setText(
            "{:.4f}".format(
                conv.num_to_step(self.xminBox.value(), self.xmaxBox.value(), self.xnumBox.value())
            )
        )
        self.xsizeLabel.setText("{:.4f}".format(self.xmaxBox.value() - self.xminBox.value()))

    def update_ystep_size(self):
        self.ystepLabel.setText(
            "{:.4f}".format(
                conv.num_to_step(self.yminBox.value(), self.ymaxBox.value(), self.ynumBox.value())
            )
        )
        self.ysizeLabel.setText("{:.4f}".format(self.ymaxBox.value() - self.yminBox.value()))

    def update_modal_boxes(self):
        self.dummysampleBox.setEnabled(str_to_mode(self.modeBox.currentText()) == ScanMode.ANALOG)
        self.oversampleBox.setEnabled(str_to_mode(self.modeBox.currentText()) == ScanMode.ANALOG)
        self.pollsampleBox.setEnabled(
            str_to_mode(self.modeBox.currentText()) == ScanMode.COM_DIPOLL
        )
        self.delayBox.setEnabled(str_to_mode(self.modeBox.currentText()) != ScanMode.ANALOG)

    def validate(self):
        if (
            self.xminBox.value() < self.xmaxBox.value()
            and self.yminBox.value() < self.ymaxBox.value()
        ):
            return (True, "")
        else:
            return (False, "Max value must be greater than min value.")

    def get_bounds(self):
        xbound = (self.xminBox.value(), self.xmaxBox.value())
        ybound = (self.yminBox.value(), self.ymaxBox.value())
        return (xbound, ybound)

    def get_nums(self):
        return (self.xnumBox.value(), self.ynumBox.value())

    def get_z(self):
        return self.zBox.value()

    def get_window(self):
        return self.timeBox.value() * 1e-3  # convert ms to s here

    def set_window(self, v):
        self.timeBox.setValue(v * 1e3)  # convert s to ms here

    def get_mode(self):
        return str_to_mode(self.modeBox.currentText())

    def get_line_mode(self):
        return str_to_line_mode(self.linemodeBox.currentText())

    def get_dummy_samples(self):
        return self.dummysampleBox.value()

    def get_oversample(self):
        return self.oversampleBox.value()

    def get_poll_samples(self):
        return self.pollsampleBox.value()

    def get_delay(self):
        return self.delayBox.value() * 1e-3  # convert ms to s here

    def set_delay(self, v):
        self.delayBox.setValue(v * 1e3)  # convert s to ms here

    def get_params(self):
        params = {}
        params["xmin"] = self.xminBox.value()
        params["xmax"] = self.xmaxBox.value()
        params["ymin"] = self.yminBox.value()
        params["ymax"] = self.ymaxBox.value()
        params["xnum"] = self.xnumBox.value()
        params["ynum"] = self.ynumBox.value()
        params["z"] = self.get_z()
        params["time_window"] = self.get_window()
        params["ident"] = uuid.uuid4()
        params["mode"] = self.get_mode()
        params["line_mode"] = self.get_line_mode()

        if params["mode"] == ScanMode.ANALOG:
            params["dummy_samples"] = self.get_dummy_samples()
            params["oversample"] = self.get_oversample()
        if params["mode"] == ScanMode.COM_DIPOLL:
            params["poll_samples"] = self.get_poll_samples()
        if params["mode"] != ScanMode.ANALOG:
            params["delay"] = self.get_delay()

        return params


class trackDialog(QtWidgets.QDialog, Ui_trackDialog):
    """Dialog for Tracking function."""

    def __init__(self, xbound, ybound, zbound, scan_capability, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.xmin, self.xmax = xbound
        self.ymin, self.ymax = ybound
        self.zmin, self.zmax = zbound

        for w in (self.xyxlenBox, self.xyxnumBox):
            w.valueChanged.connect(self.update_xyxstep)
        for w in (self.xyylenBox, self.xyynumBox):
            w.valueChanged.connect(self.update_xyystep)
        for w in (self.xzxlenBox, self.xzxnumBox):
            w.valueChanged.connect(self.update_xzxstep)
        for w in (self.xzzlenBox, self.xzznumBox):
            w.valueChanged.connect(self.update_xzzstep)
        for w in (self.yzylenBox, self.yzynumBox):
            w.valueChanged.connect(self.update_yzystep)
        for w in (self.yzzlenBox, self.yzznumBox):
            w.valueChanged.connect(self.update_yzzstep)

        self.modeBox.clear()
        for mode in scan_capability:
            self.modeBox.addItem(mode_to_str(mode))

        self.modeBox.currentIndexChanged.connect(self.update_modal_boxes)
        self.modeBox.setCurrentIndex(0)

        self.update_modal_boxes()

        self.update_xyxstep()
        self.update_xyystep()
        self.update_xzxstep()
        self.update_xzzstep()
        self.update_yzystep()
        self.update_yzzstep()

        self.upButton.clicked.connect(self.up_order_item)
        self.downButton.clicked.connect(self.down_order_item)

        self.defaultButton.clicked.connect(partial(self.load_parameter, params=None))

    def update_xyxstep(self):
        self.xyxstepLabel.setText(
            "{:.4f}".format(conv.num_to_step(0, self.xyxlenBox.value(), self.xyxnumBox.value()))
        )

    def update_xyystep(self):
        self.xyystepLabel.setText(
            "{:.4f}".format(conv.num_to_step(0, self.xyylenBox.value(), self.xyynumBox.value()))
        )

    def update_xzxstep(self):
        self.xzxstepLabel.setText(
            "{:.4f}".format(conv.num_to_step(0, self.xzxlenBox.value(), self.xzxnumBox.value()))
        )

    def update_xzzstep(self):
        self.xzzstepLabel.setText(
            "{:.4f}".format(conv.num_to_step(0, self.xzzlenBox.value(), self.xzznumBox.value()))
        )

    def update_yzystep(self):
        self.yzystepLabel.setText(
            "{:.4f}".format(conv.num_to_step(0, self.yzylenBox.value(), self.yzynumBox.value()))
        )

    def update_yzzstep(self):
        self.yzzstepLabel.setText(
            "{:.4f}".format(conv.num_to_step(0, self.yzzlenBox.value(), self.yzznumBox.value()))
        )

    def up_order_item(self):
        i = self.orderList.currentRow()
        if i <= 0:
            return
        self.orderList.insertItem(i - 1, self.orderList.takeItem(i))
        self.orderList.setCurrentRow(i - 1)

    def down_order_item(self):
        i = self.orderList.currentRow()
        if i == -1 or i >= self.orderList.count() - 1:
            return
        self.orderList.insertItem(i + 1, self.orderList.takeItem(i))
        self.orderList.setCurrentRow(i + 1)

    def update_modal_boxes(self):
        self.dummysampleBox.setEnabled(str_to_mode(self.modeBox.currentText()) == ScanMode.ANALOG)
        self.oversampleBox.setEnabled(str_to_mode(self.modeBox.currentText()) == ScanMode.ANALOG)
        self.pollsampleBox.setEnabled(
            str_to_mode(self.modeBox.currentText()) == ScanMode.COM_DIPOLL
        )
        self.delayBox.setEnabled(str_to_mode(self.modeBox.currentText()) != ScanMode.ANALOG)

    def validate(self):
        if not self.get_order():
            return (False, "All directions are disabled.")

        return (True, "")

    def get_opt_mode(self, direction):
        if direction == ScanDirection.XY:
            m = self.xymodeBox.currentIndex()
        elif direction == ScanDirection.XZ:
            m = self.xzmodeBox.currentIndex()
        elif direction == ScanDirection.YZ:
            m = self.yzmodeBox.currentIndex()
        else:
            raise ValueError(f"Invalid direction {direction}")
        return OptMode(m) if m else None

    def get_lens(self, direction):
        if direction == ScanDirection.XY:
            return (self.xyxlenBox.value(), self.xyylenBox.value())
        elif direction == ScanDirection.XZ:
            return (self.xzxlenBox.value(), self.xzzlenBox.value())
        elif direction == ScanDirection.YZ:
            return (self.yzylenBox.value(), self.yzzlenBox.value())
        else:
            raise ValueError(f"Invalid direction {direction}")

    def get_nums(self, direction):
        if direction == ScanDirection.XY:
            return (self.xyxnumBox.value(), self.xyynumBox.value())
        elif direction == ScanDirection.XZ:
            return (self.xzxnumBox.value(), self.xzznumBox.value())
        elif direction == ScanDirection.YZ:
            return (self.yzynumBox.value(), self.yzznumBox.value())
        else:
            raise ValueError(f"Invalid direction {direction}")

    def get_offsets(self, direction):
        if direction == ScanDirection.XY:
            return (self.xyxoffsetBox.value(), self.xyyoffsetBox.value())
        elif direction == ScanDirection.XZ:
            return (self.xzxoffsetBox.value(), self.xzzoffsetBox.value())
        elif direction == ScanDirection.YZ:
            return (self.yzyoffsetBox.value(), self.yzzoffsetBox.value())
        else:
            raise ValueError(f"Invalid direction {direction}")

    def get_interval(self):
        return self.intervalBox.value()

    def get_window(self):
        return self.timeBox.value() * 1e-3  # convert ms to s here

    def set_window(self, v):
        self.timeBox.setValue(v * 1e3)  # convert s to ms here

    def get_scan_mode(self):
        return str_to_mode(self.modeBox.currentText())

    def get_line_mode(self):
        return str_to_line_mode(self.linemodeBox.currentText())

    def get_dummy_samples(self):
        return self.dummysampleBox.value()

    def get_oversample(self):
        return self.oversampleBox.value()

    def get_poll_samples(self):
        return self.pollsampleBox.value()

    def get_delay(self):
        return self.delayBox.value() * 1e-3  # convert ms to s here

    def set_delay(self, v):
        self.delayBox.setValue(v * 1e3)  # convert s to ms here

    def get_order(self):
        order = [
            str_to_direction(self.orderList.item(i).text()) for i in range(self.orderList.count())
        ]
        return [d for d in order if self.get_opt_mode(d) is not None]

    def get_save_enable(self):
        return self.saveBox.isChecked()

    def get_params(self):
        params = {}
        params["order"] = self.get_order()
        for d in params["order"]:
            params[d] = {}
            params[d]["xlen"], params[d]["ylen"] = self.get_lens(d)
            params[d]["xnum"], params[d]["ynum"] = self.get_nums(d)
            params[d]["xoffset"], params[d]["yoffset"] = self.get_offsets(d)
            params[d]["opt_mode"] = self.get_opt_mode(d)
        params["interval_sec"] = self.get_interval()
        params["time_window"] = self.get_window()
        params["mode"] = self.get_scan_mode()
        params["line_mode"] = self.get_line_mode()

        if params["mode"] == ScanMode.ANALOG:
            params["dummy_samples"] = self.get_dummy_samples()
            params["oversample"] = self.get_oversample()
        if params["mode"] == ScanMode.COM_DIPOLL:
            params["poll_samples"] = self.get_poll_samples()
        if params["mode"] != ScanMode.ANALOG:
            params["delay"] = self.get_delay()

        return params

    def load_parameter(self, params: dict):
        """Load track parameters"""

        if params is None:
            params = {}

        self.intervalBox.setValue(params.get("interval_sec", 180))
        self.saveBox.setChecked(params.get("autosave", False))
        self.set_window(params.get("time_window", 10e-3))
        if "mode" in params:
            mode = params["mode"]
            self.modeBox.setCurrentIndex(self.modeBox.findText(mode_to_str(mode)))
            self.linemodeBox.setCurrentIndex(
                self.linemodeBox.findText(
                    line_mode_to_str(params.get("line_mode", LineMode.ASCEND))
                )
            )
            if mode == ScanMode.ANALOG:
                self.dummysampleBox.setValue(params.get("dummy_samples", 1))
                self.oversampleBox.setValue(params.get("oversample", 1))
            if mode == ScanMode.COM_DIPOLL:
                self.pollsampleBox.setValue(params.get("poll_samples", 1000))
            if mode != ScanMode.ANALOG:
                self.set_delay(params.get("delay", 0.0))

        if ScanDirection.XY in params:
            c = params[ScanDirection.XY]
            self.xymodeBox.setCurrentIndex(c.get("opt_mode", OptMode.Disable).value)
            self.xyxlenBox.setValue(c.get("xlen", 2))
            self.xyxnumBox.setValue(c.get("xnum", 21))
            self.xyxoffsetBox.setValue(c.get("xoffset", 0))
            self.xyylenBox.setValue(c.get("ylen", 2))
            self.xyynumBox.setValue(c.get("ynum", 21))
            self.xyyoffsetBox.setValue(c.get("yoffset", 0))
        if ScanDirection.XZ in params:
            c = params[ScanDirection.XZ]
            self.xzmodeBox.setCurrentIndex(c.get("opt_mode", OptMode.Disable).value)
            self.xzxlenBox.setValue(c.get("xlen", 2))
            self.xzxnumBox.setValue(c.get("xnum", 21))
            self.xzxoffsetBox.setValue(c.get("xoffset", 0))
            self.xzzlenBox.setValue(c.get("ylen", 2))
            self.xzznumBox.setValue(c.get("ynum", 21))
            self.xzzoffsetBox.setValue(c.get("yoffset", 0))
        if ScanDirection.YZ in params:
            c = params[ScanDirection.YZ]
            self.yzmodeBox.setCurrentIndex(c.get("opt_mode", OptMode.Disable).value)
            self.yzylenBox.setValue(c.get("xlen", 2))
            self.yzynumBox.setValue(c.get("xnum", 21))
            self.yzyoffsetBox.setValue(c.get("xoffset", 0))
            self.yzzlenBox.setValue(c.get("ylen", 2))
            self.yzznumBox.setValue(c.get("ynum", 21))
            self.yzzoffsetBox.setValue(c.get("yoffset", 0))

        for i in range(self.orderList.count()):
            self.orderList.takeItem(0)
        dir_all = (ScanDirection.XY, ScanDirection.XZ, ScanDirection.YZ)
        dir_conf = params.get("order", dir_all)
        for d in dir_conf:
            self.orderList.addItem(direction_to_str(d))
        rest = set(dir_all) - set(dir_conf)
        for d in rest:
            self.orderList.addItem(direction_to_str(d))


class subImage(pg.ImageItem):
    sigLeftClicked = QtCore.pyqtSignal(object)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.drawKernel is not None:
                self.drawAt(ev.pos(), ev)
            else:
                self.sigLeftClicked.emit(ev.scenePos())
                ev.accept()


class mainImage(pg.ImageItem):
    """pg.ImageItem with a bit of additonal properties to express pseudo-color scan image."""

    def __init__(self, image: Image, **kargs):
        pg.ImageItem.__init__(self, image=None, **kargs)

        self.direction = image.direction
        self.params = image.params
        self.ident = image.ident
        self.data = image.image

        self.convert_param_types()

        self.xstep = conv.num_to_step(
            self.params["xmin"], self.params["xmax"], self.params["xnum"]
        )
        self.ystep = conv.num_to_step(
            self.params["ymin"], self.params["ymax"], self.params["ynum"]
        )

    def get_nums(self):
        return (self.params["xnum"], self.params["ynum"])

    def convert_param_types(self):
        for k in ("xmin", "xmax", "ymin", "ymax", "z", "time_window", "delay"):
            if k in self.params:
                self.params[k] = float(self.params[k])

        for k in ("xnum", "ynum", "poll_samples", "dummy_samples"):
            if k in self.params:
                self.params[k] = int(self.params[k])

    def get_pos(self):
        return (self.params["xmin"], self.params["ymin"])

    def get_size(self):
        return (
            self.params["xmax"] - self.params["xmin"],
            self.params["ymax"] - self.params["ymin"],
        )

    def get_steps(self):
        return (self.xstep, self.ystep)

    def has_same_ident(self, other_image: Image) -> bool:
        """Take another image and check if self and other has same identifier."""

        return self.ident == other_image.ident

    def is_equally_sized(self, other) -> bool:
        """Take another instance of mainImage and check if 2 images are equally sized or not."""

        if self.get_nums() == other.get_nums() and self.get_steps() == other.get_steps():
            return True
        else:
            return False


class scanItem(QtCore.QObject):
    sigSubUpdated = QtCore.pyqtSignal(object)
    sigXChanged = QtCore.pyqtSignal(float)
    sigYChanged = QtCore.pyqtSignal(float)

    IMAGE_MAXNUM = 1.0e5

    def __init__(
        self,
        direction: ScanDirection,
        pi,
        pi_sub,
        xbound,
        ybound,
        histo,
        pos,
        style: dict,
        roi_size=(20.0, 20.0),
    ):
        QtCore.QObject.__init__(self)
        self.xmin, self.xmax = xbound
        self.ymin, self.ymax = ybound
        self.histo = histo

        self.direction = direction
        self.pi = pi
        self.pi_sub = pi_sub
        # self.gi = pg.GridItem()
        # self.gi_sub = pg.GridItem()

        self.pi.disableAutoRange(axis="xy")
        self.pi_sub.disableAutoRange(axis="xy")

        xybound = QtCore.QRectF(self.xmin, self.ymin, self.xmax, self.ymax)
        roi_x, roi_y = pos[0] - roi_size[0] / 2.0, pos[1] - roi_size[1] / 2.0
        if roi_x < self.xmin:
            roi_x = self.xmin
        if roi_x > self.xmax - roi_size[0]:
            roi_x = self.xmax - roi_size[0]
        if roi_y < self.ymin:
            roi_y = self.ymin
        if roi_y > self.ymax - roi_size[1]:
            roi_y = self.ymax - roi_size[1]
        s = style.get("roi", {})
        w = s.get("width", 2.0)
        p = mkPen(s.get("color", (30, 144, 255)), width=w)
        hp = mkPen(s.get("handle_color", (0, 255, 255)), width=w)
        vp = mkPen(s.get("hover_color", (255, 255, 0)), width=w)
        self.roi = pg.ROI(
            (roi_x, roi_y),
            size=pg.Point(roi_size),
            pen=p,
            hoverPen=vp,
            handlePen=hp,
            handleHoverPen=vp,
            maxBounds=xybound,
        )
        self.roi.handleSize = s.get("handle_size", 8)
        self.roi.translatable = False

        # bg_nums = (conv.step_to_num(self.xmin, self.xmax, 10.0),
        # conv.step_to_num(self.ymin, self.ymax, 10.0))
        # bg_data = np.fromfunction(lambda x, y: (x + y) % 2, bg_nums)
        bg_nums = (11, 11)
        bg_nums_r = (bg_nums[0] - 1, bg_nums[1] - 1)
        bg_data = np.ones(bg_nums_r) * -1
        xs = conv.num_to_step(self.xmin, self.xmax, bg_nums[0])
        ys = conv.num_to_step(self.ymin, self.ymax, bg_nums[1])
        im = Image(
            {
                "xmin": self.xmin,
                "xmax": self.xmax - xs,
                "ymin": self.ymin,
                "ymax": self.ymax - ys,
                "xnum": bg_nums_r[0],
                "ynum": bg_nums_r[1],
                "z": 0.0,
                "direction": self.direction,
            }
        )
        im.finalize(False)
        im.image = bg_data

        self.img_bg = mainImage(im)
        self.img_bg.setZValue(-1.0)

        self.img = self.img_bg
        self.pi.addItem(self.img)
        self.histo.setImageItem(self.img)
        self.histo.gradient.loadPreset("inferno")

        self.img_list = []  # img_bg is NOT in the img_list
        self.img_sub = subImage()

        s = style.get("crosshair", {})
        p = mkPen(s.get("color", (43, 179, 43)), width=s.get("width", 2.0))
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=p, bounds=xbound)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=p, bounds=ybound)
        self.vline_sub = pg.InfiniteLine(angle=90, movable=False, pen=p, bounds=xbound)
        self.hline_sub = pg.InfiniteLine(angle=0, movable=False, pen=p, bounds=ybound)

        items = (self.vline, self.hline, self.roi)
        for i, item in enumerate(items):
            item.setZValue(float(i + 1) * scanItem.IMAGE_MAXNUM)
            self.pi.addItem(item)

        items = (self.img_sub, self.vline_sub, self.hline_sub)
        for i, item in enumerate(items):
            item.setZValue(float(i))
            self.pi_sub.addItem(item)

        self.roi.sigRegionChanged.connect(self.update_sub)
        self.histo.item.sigLookupTableChanged.connect(self.update_sub)
        self.histo.item.sigLevelsChanged.connect(self.update_sub)

        self._update_image()

    def contains(self, point):
        """Return if given point is in current ROI."""

        rect = self.roi.parentBounds()
        return rect.contains(pg.Point(point))

    def update_sub(self):
        """Update sub image. If crosshairs went outside of ROI, move them to center of ROI."""

        if self.img.data is None:
            return
        lut = self.histo.item.getLookupTable(img=self.img.data)
        levels = self.histo.item.getLevels()
        self.img_sub.setImage(
            self.roi.getArrayRegion(self.img.data, self.img), lut=lut, levels=levels
        )
        # roi pos & size can be used instead, but rect is more convenient
        # (with contains() and center() methods.)
        rect = self.roi.parentBounds()
        self.img_sub.setRect(rect)
        if not rect.contains(pg.Point(self.get_crosshair())):
            p = rect.center()
            self.move_crosshair(p)
            self.sigXChanged.emit(p.x())
            self.sigYChanged.emit(p.y())

        self.pi_sub.autoRange()
        self.sigSubUpdated.emit(rect)

    def sub_clicked(self, scene_pos):
        """This function is intended to be used as slot of subImage.sigLeftClicked."""

        p = self.pi_sub.getViewBox().mapSceneToView(scene_pos)
        self.move_crosshair(p)
        self.sigXChanged.emit(p.x())
        self.sigYChanged.emit(p.y())

    def set_crosshair(self, point):
        """Move crosshair and emit sig{X,Y}Changed to propagate the change to the other scanItem.
        Use this function instead of move_crosshair() if you want to set position properly
        by calling method from only one scanItem (XY or XZ).
        """

        self.move_crosshair(point)
        if hasattr(point, "x"):
            x, y = point.x(), point.y()
        else:
            x, y = point
        self.sigXChanged.emit(x)
        self.sigYChanged.emit(y)

    def set_crosshair_X(self, x):
        """Move X value of crosshair and emit sigXChanged.

        sigXChanged is to propagate the change to the other scanItem.

        """

        self.move_crosshair_X(x)
        self.sigXChanged.emit(x)

    def set_crosshair_Y(self, y):
        """Move Y value of crosshair and emit sigYChanged.

        sigYChanged is to propagate the change to the other scanItem.

        """

        self.move_crosshair_Y(y)
        self.sigYChanged.emit(y)

    def move_crosshair(self, point):
        if hasattr(point, "x") and hasattr(point, "y"):
            x, y = point.x(), point.y()
        else:
            x, y = point

        self.vline.setPos(x)
        self.hline.setPos(y)
        self.vline_sub.setPos(x)
        self.hline_sub.setPos(y)

    def move_crosshair_X(self, x):
        self.vline.setPos(x)
        self.vline_sub.setPos(x)

    def move_crosshair_Y(self, y):
        self.hline.setPos(y)
        self.hline_sub.setPos(y)

    def get_crosshair(self):
        return (self.vline.value(), self.hline.value())

    def get_crosshair_X(self):
        return self.vline.value()

    def get_crosshair_Y(self):
        return self.hline.value()

    def maximize_roi(self):
        self.roi.setPos((self.xmin, self.ymin), update=False)
        self.roi.setSize((self.xmax, self.ymax), update=True)

    def fit_roi(self):
        """fit roi size to current image."""

        self.roi.setPos(self.img.get_pos(), update=False)
        self.roi.setSize(self.img.get_size(), update=True)

    def center_roi(self, size=(20, 20)):
        """move roi region to the center of current crosshair position."""

        x = self.get_crosshair_X() - size[0] / 2
        y = self.get_crosshair_Y() - size[1] / 2

        self.roi.setPos((x, y), update=False)
        self.roi.setSize(size, update=True)

    def switch_interaction(self, enable):
        if enable:
            self.img_sub.sigLeftClicked.connect(self.sub_clicked)
            self.roi.translatable = True
            self.roi.addScaleHandle((1, 1), (0, 0))
            self.roi.addScaleHandle((0, 0), (1, 1))
        else:
            self.img_sub.sigLeftClicked.disconnect(self.sub_clicked)
            self.roi.translatable = False
            for i in range(2):
                self.roi.removeHandle(0)

    def auto_range(self):
        self.pi.autoRange()
        self.pi_sub.autoRange()

    def clear(self):
        for item in self.img_list:
            self.pi.removeItem(item)

        self.img_list = []

        self.img = self.img_bg
        self.histo.setImageItem(self.img)
        self._set_levels()

    def pop(self):
        if self.img_list:
            item = self.img_list.pop()
            self.pi.removeItem(item)

        if self.img_list:
            self.img = self.img_list[-1]
        else:
            self.img = self.img_bg

        self.histo.setImageItem(self.img)
        self._set_levels()

    def append(self, image: Image):
        """append an image to the buffer, without fitting ROI to it."""

        self._append(image)
        # self.fit_roi()
        self._update_image()

    def _append(self, image: Image):
        self.img = mainImage(image)

        self.img_list.append(self.img)
        self.img.setZValue(float(self.img_list.index(self.img)))

        self.pi.addItem(self.img)
        self.histo.setImageItem(self.img)

    def update_image(self, image: Image):
        if not self.img.has_same_ident(image):
            self._append(image)
            self.fit_roi()
        else:
            self.img.data = image.image
        self._update_image()

    def _set_levels(self):
        """set histogram levels to fit current img."""

        mn, mx = np.nanmin(self.img.data), np.nanmax(self.img.data)
        if mn == mx:
            mn, mx = mn - 0.1 * abs(mn), mn + 0.9 * abs(mn)  # aims somewhat 'dark' looks
        self.histo.setLevels(mn, mx)

    def _set_img_transform(self):
        self.img.resetTransform()
        self.img.setPos(*self.img.get_pos())
        self.img.setTransform(QtGui.QTransform.fromScale(*self.img.get_steps()))

    def _update_image(self, set_levels=True):
        # not necessary because auto range is enabled by default.
        # self.histo.setHistogramRange(np.min(self.img.data), np.max(self.img.data))

        # self.img.setImage(self.img.data) # not so much difference with this.
        self.img.updateImage(self.img.data)

        if set_levels:
            self._set_levels()
        self._set_img_transform()
        self.update_sub()

    def get_inverted(self):
        return (self.pi.getViewBox().xInverted(), self.pi.getViewBox().yInverted())


class MoveBinder(QtCore.QObject):
    def __init__(
        self,
        cli: QConfocalClient,
        pos: T.Tuple[float, float, float],
        interval_ms: int,
        parent=None,
    ):
        QtCore.QObject.__init__(self, parent=parent)
        self.cli = cli
        self.x, self.y, self.z = pos
        self.interval_ms = interval_ms
        self.updated = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.check)

    def start(self):
        self.timer.start(self.interval_ms)

    def check(self):
        try:
            if not self.updated:
                return
            self.cli.move([Axis.X, Axis.Y, Axis.Z], [self.x, self.y, self.z])
            self.updated = False
        except KeyboardInterrupt:
            # avoid printing Traceback to stdout on exit by KeyboardInterrupt.
            pass

    def move_X(self, line):
        self.x = line.value()
        self.updated = True

    def move_Y(self, line):
        self.y = line.value()
        self.updated = True

    def move_Z(self, line):
        self.z = line.value()
        self.updated = True


class ConfocalWidget(ClientWidget, Ui_Confocal):
    """Main widget for Confocal (scan and interact)"""

    def __init__(
        self,
        gconf: dict,
        name,
        tracker_name,
        param_server_name,
        style: dict,
        invert,
        move_interval_ms,
        context,
        parent=None,
    ):
        ClientWidget.__init__(self, parent)
        self.setupUi(self)

        self.confocal_conf = local_conf(gconf, name)
        self._style = style
        self._invert = invert

        self._finalizing = False
        self._scan_capability = None
        self._prev_pos = None
        self._move_interval_ms = move_interval_ms

        self.cli = QConfocalClient(gconf, name, context=context, parent=self)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.tracker_cli = QConfocalTrackerClient(
            gconf, tracker_name, context=context, parent=self
        )
        self.tracker_cli.stateUpdated.connect(self.update_tracker_state)

        self.param_cli = ParamClient(gconf, param_server_name, context=context)

        self.add_clients(self.cli, self.tracker_cli, self.param_cli)

        self.setEnabled(False)

    def init_with_status(self, status: ConfocalStatus):
        """Initialize the widget after receiving first status."""

        if not status.pos.has_range_and_target():
            return
        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        x, y, z = status.pos.x_tgt, status.pos.y_tgt, status.pos.z_tgt

        self.XTgtBox.setValue(x)
        self.YTgtBox.setValue(y)
        self.ZTgtBox.setValue(z)

        self.init_view(status.pos)
        self.load_buffers(status.pos)
        self.init_connections()
        self.init_move(status.pos)

        self.XY.auto_range()
        self.XZ.auto_range()
        self.YZ.auto_range()

        self.move_crosshair((x, y, z))

        # Cannot call self.set_home() here because incorrect value of position
        # may be returned by self.get_pos().
        self.home_pos = (x, y, z)
        self.homeLabel.setText("Home x: {:.4f} y: {:.4f} z: {:.4f}".format(*self.home_pos))

        # update initial GUI state
        self.update_state(status.state, last_state=ConfocalState.IDLE)

        # try to store scan_capability
        self.get_scan_capability()

        self.cli.stateUpdated.connect(self.update_state)
        self.cli.xyImageUpdated.connect(self.XY.update_image)
        self.cli.xzImageUpdated.connect(self.XZ.update_image)
        self.cli.yzImageUpdated.connect(self.YZ.update_image)
        self.cli.scanFinished.connect(self.finalize_scan)

        self.setEnabled(True)

        # defer init_splitter() because direct call here won't respect real size.
        QtCore.QTimer.singleShot(200, QtCore.Qt.TimerType.PreciseTimer, self.init_splitter)

    def init_view(self, pos):
        """Initialize views."""

        x, y, z = pos.x_tgt, pos.y_tgt, pos.z_tgt
        self.layout = pg.GraphicsLayout()
        self.graphicsView.setCentralItem(self.layout)
        self.graphicsScene = self.graphicsView.scene()

        self.layout.addLabel("XY", row=0, col=0)
        main_XY = self.layout.addPlot(row=1, col=0)
        sub_XY = self.layout.addPlot(row=2, col=0)
        self.layout.addLabel("XZ", row=0, col=1)
        main_XZ = self.layout.addPlot(row=1, col=1)
        sub_XZ = self.layout.addPlot(row=2, col=1)
        self.layout.addLabel("YZ", row=0, col=2)
        main_YZ = self.layout.addPlot(row=1, col=2)
        sub_YZ = self.layout.addPlot(row=2, col=2)

        for pi in (main_XY, sub_XY, main_XZ, sub_XZ, main_YZ, sub_YZ):
            pi.setAspectLocked(True)
            pi.showGrid(x=True, y=True)

        iX, iY, iZ = self._invert
        for xy in (main_XY, sub_XY):
            xy.invertX(iX)
            xy.invertY(iY)
        for xz in (main_XZ, sub_XZ):
            xz.invertX(iX)
            xz.invertY(iZ)
        for xy in (main_YZ, sub_YZ):
            xy.invertX(iY)
            xy.invertY(iZ)

        self.xbound = pos.x_range
        self.ybound = pos.y_range
        self.zbound = pos.z_range

        self.XY = scanItem(
            ScanDirection.XY,
            main_XY,
            sub_XY,
            self.xbound,
            self.ybound,
            self.histo_XY,
            (x, y),
            self._style,
        )
        self.XZ = scanItem(
            ScanDirection.XZ,
            main_XZ,
            sub_XZ,
            self.xbound,
            self.zbound,
            self.histo_XZ,
            (x, z),
            self._style,
        )
        self.YZ = scanItem(
            ScanDirection.YZ,
            main_YZ,
            sub_YZ,
            self.ybound,
            self.zbound,
            self.histo_YZ,
            (y, z),
            self._style,
        )

    def init_splitter(self):
        sizes = self.splitter.sizes()
        w = self._style.get("lut_width", 120)
        self.splitter.setSizes([w, sum(sizes) - 3 * w, w, w])

    def load_buffers(self, pos):
        """get the image buffers from confocal and load all the images."""

        for si in (self.XY, self.XZ, self.YZ):
            buf = self.cli.get_all_buffer(si.direction)
            for img in buf:
                si.append(img)

    def init_move(self, pos):
        if self._move_interval_ms:
            p = pos.x_tgt, pos.y_tgt, pos.z_tgt
            self.binder = MoveBinder(self.cli, p, self._move_interval_ms, parent=self)
            self.XY.vline_sub.sigPositionChanged.connect(self.binder.move_X)
            self.XY.hline_sub.sigPositionChanged.connect(self.binder.move_Y)
            self.XZ.hline_sub.sigPositionChanged.connect(self.binder.move_Z)
            self.binder.start()
        else:
            # Direct client call
            self.XY.vline_sub.sigPositionChanged.connect(self.request_move_X)
            self.XY.hline_sub.sigPositionChanged.connect(self.request_move_Y)
            # below doubles X move.
            # self.XZ.vline_sub.sigPositionChanged.connect(self.request_move_X)
            self.XZ.hline_sub.sigPositionChanged.connect(self.request_move_Z)
            # no need to add self.YZ ones.

    def init_connections(self):
        """Connect permanent signals.

        Note the signals among XY, XZ and YZ to syncronize.

        """

        # among XY, XZ and YZ
        self.XY.roi.sigRegionChangeFinished.connect(self.update_roiXY)
        self.XZ.roi.sigRegionChangeFinished.connect(self.update_roiXZ)
        self.YZ.roi.sigRegionChangeFinished.connect(self.update_roiYZ)

        self.XY.sigXChanged.connect(self.XZ.move_crosshair_X)
        self.XY.sigYChanged.connect(self.YZ.move_crosshair_X)

        self.XZ.sigXChanged.connect(self.XY.move_crosshair_X)
        self.XZ.sigYChanged.connect(self.YZ.move_crosshair_Y)

        self.YZ.sigXChanged.connect(self.XY.move_crosshair_Y)
        self.YZ.sigYChanged.connect(self.XZ.move_crosshair_Y)

        # Status updates from confocal
        self.cli.xposChanged.connect(self.update_XPosEdit)
        self.cli.yposChanged.connect(self.update_YPosEdit)
        self.cli.zposChanged.connect(self.update_ZPosEdit)
        self.cli.xtgtChanged.connect(self.XTgtBox.setValue)
        self.cli.ytgtChanged.connect(self.YTgtBox.setValue)
        self.cli.ztgtChanged.connect(self.ZTgtBox.setValue)
        self.cli.xtgtChanged.connect(self.XY.set_crosshair_X)
        self.cli.ytgtChanged.connect(self.XY.set_crosshair_Y)
        self.cli.ztgtChanged.connect(self.XZ.set_crosshair_Y)

        # Other widgets
        self.XY.sigSubUpdated.connect(self.update_XYbox)
        self.XZ.sigSubUpdated.connect(self.update_XZbox)
        self.YZ.sigSubUpdated.connect(self.update_YZbox)

        self.xyfitButton.clicked.connect(self.XY.fit_roi)
        self.xzfitButton.clicked.connect(self.XZ.fit_roi)
        self.yzfitButton.clicked.connect(self.YZ.fit_roi)
        self.xymaxButton.clicked.connect(self.XY.maximize_roi)
        self.xzmaxButton.clicked.connect(self.XZ.maximize_roi)
        self.yzmaxButton.clicked.connect(self.YZ.maximize_roi)

        self.xypopButton.clicked.connect(self.XY.pop)
        self.xzpopButton.clicked.connect(self.XZ.pop)
        self.yzpopButton.clicked.connect(self.YZ.pop)
        self.xypopButton.clicked.connect(self._request_pop_xy)
        self.xzpopButton.clicked.connect(self._request_pop_xz)
        self.yzpopButton.clicked.connect(self._request_pop_yz)

        self.xyclearButton.clicked.connect(self.XY.clear)
        self.xzclearButton.clicked.connect(self.XZ.clear)
        self.yzclearButton.clicked.connect(self.YZ.clear)
        self.xyclearButton.clicked.connect(self._request_clear_xy)
        self.xzclearButton.clicked.connect(self._request_clear_xz)
        self.yzclearButton.clicked.connect(self._request_clear_yz)

        self.xysaveButton.clicked.connect(self.save_image_xy)
        self.xzsaveButton.clicked.connect(self.save_image_xz)
        self.yzsaveButton.clicked.connect(self.save_image_yz)
        self.xyexportButton.clicked.connect(self.export_image_xy)
        self.xzexportButton.clicked.connect(self.export_image_xz)
        self.yzexportButton.clicked.connect(self.export_image_yz)

        self.exportviewButton.clicked.connect(self.export_view)
        self.loadButton.clicked.connect(self.load_image)

        # Home pos functions
        self.sethomeButton.clicked.connect(self.set_home)
        self.gohomeButton.clicked.connect(self.go_home)
        self.centerButton.clicked.connect(self.go_center)

        # State requests
        self.idleButton.clicked.connect(self.request_idle)
        self.piezoButton.clicked.connect(self.request_piezo)
        self.interactButton.clicked.connect(self.request_interact)
        self.xyscanButton.clicked.connect(self.request_xyscan)
        self.xzscanButton.clicked.connect(self.request_xzscan)
        self.yzscanButton.clicked.connect(self.request_yzscan)
        self.trackstartButton.clicked.connect(self.request_track_start)
        self.trackstopButton.clicked.connect(self.request_track_stop)
        self.tracknowButton.clicked.connect(self.request_track_now)

    def _request_pop_xy(self):
        self.cli.pop_buffer(ScanDirection.XY)

    def _request_pop_xz(self):
        self.cli.pop_buffer(ScanDirection.XZ)

    def _request_pop_yz(self):
        self.cli.pop_buffer(ScanDirection.YZ)

    def _request_clear_xy(self):
        self.cli.clear_buffer(ScanDirection.XY)

    def _request_clear_xz(self):
        self.cli.clear_buffer(ScanDirection.XZ)

    def _request_clear_yz(self):
        self.cli.clear_buffer(ScanDirection.YZ)

    def move_crosshair(self, target_pos):
        """Move crosshair to given target position."""

        x, y, z = target_pos
        self.XY.move_crosshair((x, y))
        self.XZ.move_crosshair((x, z))
        self.YZ.move_crosshair((y, z))

    def get_pos(self):
        """Get crosshair position."""

        x, y = self.XY.get_crosshair()
        z = self.XZ.get_crosshair_Y()
        return x, y, z

    def set_home(self):
        """Store current position as the home position."""

        self.home_pos = self.get_pos()
        self.homeLabel.setText("Home x: %.4f y: %.4f z: %.4f" % self.home_pos)

    def go_home(self):
        """Move to stored home position."""

        msgs = [
            "Are you sure to move piezo position to the home position:",
            "({:.4f}, {:.4f}, {:.4f}) ?".format(*self.home_pos),
        ]

        x, y, z = self.home_pos
        if self.XY.contains((x, y)) and self.XZ.contains((x, z)) and self.YZ.contains((y, z)):
            if Qt.question_yn(self, "Sure to go home?", "\n".join(msgs)):
                self.move_crosshair(self.home_pos)
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Home position out of bound",
                "Cannot go home because Home position is outside ROI.",
            )

    def go_center(self):
        """Move to the center of boundaries."""

        x = (self.xbound[0] + self.xbound[1]) / 2
        y = (self.ybound[0] + self.ybound[1]) / 2
        z = (self.zbound[0] + self.zbound[1]) / 2

        msgs = [
            "Are you sure to move piezo position to the center:",
            "({:.4f}, {:.4f}, {:.4f}) ?".format(x, y, z),
            "Note that this may result in LARGE MOTION.",
        ]

        if not Qt.question_yn(self, "Sure to go to center?", "\n".join(msgs)):
            return

        for si in (self.XY, self.XZ, self.YZ):
            si.maximize_roi()

        self.move_crosshair((x, y, z))

        for si in (self.XY, self.XZ, self.YZ):
            si.center_roi()

    def update_XYbox(self, rect):
        self.XTgtBox.setMinimum(rect.x())
        self.XTgtBox.setMaximum(rect.x() + rect.width())
        self.YTgtBox.setMinimum(rect.y())
        self.YTgtBox.setMaximum(rect.y() + rect.height())

    def update_XZbox(self, rect):
        self.XTgtBox.setMinimum(rect.x())
        self.XTgtBox.setMaximum(rect.x() + rect.width())
        self.ZTgtBox.setMinimum(rect.y())
        self.ZTgtBox.setMaximum(rect.y() + rect.height())

    def update_YZbox(self, rect):
        self.YTgtBox.setMinimum(rect.x())
        self.YTgtBox.setMaximum(rect.x() + rect.width())
        self.ZTgtBox.setMinimum(rect.y())
        self.ZTgtBox.setMaximum(rect.y() + rect.height())

    def update_roiXY(self):
        """Transmit change of XYroi to XZroi and YZroi"""

        p, s = self.XY.roi.pos(), self.XY.roi.size()
        x, y, w, h = p.x(), p.y(), s.x(), s.y()

        self.XZ.roi.setPos((x, self.XZ.roi.pos().y()), update=False)
        self.XZ.roi.setSize((w, self.XZ.roi.size().y()), finish=False)

        self.YZ.roi.setPos((y, self.YZ.roi.pos().y()), update=False)
        self.YZ.roi.setSize((h, self.YZ.roi.size().y()), finish=False)

    def update_roiXZ(self):
        """Transmit change of XZroi to XYroi and YZroi"""

        p, s = self.XZ.roi.pos(), self.XZ.roi.size()
        x, y, w, h = p.x(), p.y(), s.x(), s.y()

        self.XY.roi.setPos((x, self.XY.roi.pos().y()), update=False)
        self.XY.roi.setSize((w, self.XY.roi.size().y()), finish=False)

        self.YZ.roi.setPos((self.YZ.roi.pos().x(), y), update=False)
        self.YZ.roi.setSize((self.YZ.roi.size().x(), h), finish=False)

    def update_roiYZ(self):
        """Transmit change of YZroi to XYroi and XZroi"""

        p, s = self.YZ.roi.pos(), self.YZ.roi.size()
        x, y, w, h = p.x(), p.y(), s.x(), s.y()

        self.XY.roi.setPos((self.XY.roi.pos().x(), x), update=False)
        self.XY.roi.setSize((self.XY.roi.size().x(), w), finish=False)

        self.XZ.roi.setPos((self.XZ.roi.pos().x(), y), update=False)
        self.XZ.roi.setSize((self.XZ.roi.size().x(), h), finish=False)

    def request_move_X(self, line):
        val = line.value()
        self.cli.move(Axis.X, val)

    def request_move_Y(self, line):
        val = line.value()
        self.cli.move(Axis.Y, val)

    def request_move_Z(self, line):
        val = line.value()
        self.cli.move(Axis.Z, val)

    def update_XPosEdit(self, val, on_tgt):
        self.update_PosEdit(val, on_tgt, self.XPosEdit)

    def update_YPosEdit(self, val, on_tgt):
        self.update_PosEdit(val, on_tgt, self.YPosEdit)

    def update_ZPosEdit(self, val, on_tgt):
        self.update_PosEdit(val, on_tgt, self.ZPosEdit)

    def update_PosEdit(self, val, on_tgt, edit):
        edit.setText("%.4f" % val)
        if on_tgt:
            edit.setStyleSheet("QLineEdit {background: #005500;}")
        else:
            edit.setStyleSheet("QLineEdit {background: #800000;}")

    def X_edited(self):
        val = self.XTgtBox.value()
        # print("X edited: %.4f" % (val))
        self.XY.move_crosshair_X(val)
        self.XZ.move_crosshair_X(val)

    def Y_edited(self):
        val = self.YTgtBox.value()
        # print("Y edited: %.4f" % (val))
        self.XY.move_crosshair_Y(val)
        self.YZ.move_crosshair_X(val)

    def Z_edited(self):
        val = self.ZTgtBox.value()
        # print("Z edited: %.4f" % (val))
        self.XZ.move_crosshair_Y(val)
        self.YZ.move_crosshair_Y(val)

    def _scan_file_pre_ext(self, direction: T.Optional[ScanDirection] = None):
        ext = ".scan"
        if direction is None:
            direction = self.cli.get_direction()
        if direction is None:
            return ext
        else:
            return ".{}{}".format(direction_to_str(direction).lower(), ext)

    def save_image_xy(self):
        self.save_image(ScanDirection.XY)

    def save_image_xz(self):
        self.save_image(ScanDirection.XZ)

    def save_image_yz(self):
        self.save_image(ScanDirection.YZ)

    def export_image_xy(self):
        self.export_image(ScanDirection.XY)

    def export_image_xz(self):
        self.export_image(ScanDirection.XZ)

    def export_image_yz(self):
        self.export_image(ScanDirection.YZ)

    def save_image(self, direction: T.Optional[ScanDirection] = None):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Last Scan", self._scan_file_pre_ext(direction))
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.param_cli.get_param("note", "")
        self.cli.save_image(fn, direction=direction, note=note)
        return fn

    def export_image(self, direction: T.Optional[ScanDirection] = None):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = export_dialog(
            self, default_path, "Last Scan", (".png", ".pdf", ".eps", ".csv", ".txt")
        )
        if not fn:
            return

        # don't change work_dir after export.
        # self.param_cli.set_param("work_dir", os.path.split(fn)[0])

        ext = os.path.splitext(fn)[1]
        if ext not in (".csv", ".txt"):
            params = self.get_export_params()
            if params is None:
                return
        else:
            params = None
        self.cli.export_image(fn, direction=direction, params=params)

    def export_view(self, fn):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = export_dialog(self, default_path, "View", (".png", ".pdf", ".eps"))
        if not fn:
            return

        params = []
        for d in (ScanDirection.XY, ScanDirection.XZ, ScanDirection.YZ):
            p = {}
            p["invX"], p["invY"] = self.direction_to_scanitem(d).get_inverted()
            params.append(p)
        self.cli.export_view(fn, params=params)

    def get_export_params(self):
        si = self.direction_to_scanitem(self.cli.get_direction())
        d = imageDialog(si.histo.item.getLevels())
        if not d.exec():
            return None
        valid, msg = d.validate()
        if not valid:
            QtWidgets.QMessageBox.warning(self, "Intensity levels invalid.", msg)
            return None

        params = d.get_params()
        params["invX"], params["invY"] = si.get_inverted()

        return params

    def load_image(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "Scan", ".scan")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        image = self.cli.load_image(fn)
        if image is None:
            return
        if image.note():
            self.param_cli.set_param("loaded_note", image.note())
        si = self.direction_to_scanitem(image.direction)
        si.update_image(image)

    # State managements
    def request_idle(self):
        self.cli.change_state(ConfocalState.IDLE)

    def request_piezo(self):
        self.cli.change_state(ConfocalState.PIEZO)

    def request_interact(self):
        self.cli.change_state(ConfocalState.INTERACT)

    def request_xyscan(self):
        scan_capability = self.get_scan_capability()
        if scan_capability is None:
            return
        x, y, z = self.get_pos()
        d = scanDialog(
            self.cli,
            ScanDirection.XY,
            self.xbound,
            self.ybound,
            self.zbound,
            self.XY.roi.parentBounds(),
            (x, y, z),
            scan_capability,
            str(self.param_cli.get_param("work_dir")),
            parent=self,
        )

        self.start_scan(d, ScanDirection.XY)

    def request_xzscan(self):
        scan_capability = self.get_scan_capability()
        if scan_capability is None:
            return
        # Note permutation (X, Y, Z) -> (X, Z, Y)
        x, y, z = self.get_pos()
        d = scanDialog(
            self.cli,
            ScanDirection.XZ,
            self.xbound,
            self.zbound,
            self.ybound,
            self.XZ.roi.parentBounds(),
            (x, z, y),
            scan_capability,
            str(self.param_cli.get_param("work_dir")),
            parent=self,
        )

        self.start_scan(d, ScanDirection.XZ)

    def request_yzscan(self):
        scan_capability = self.get_scan_capability()
        if scan_capability is None:
            return
        # Note permutation (X, Y, Z) -> (Y, Z, X)
        x, y, z = self.get_pos()
        d = scanDialog(
            self.cli,
            ScanDirection.YZ,
            self.ybound,
            self.zbound,
            self.xbound,
            self.YZ.roi.parentBounds(),
            (y, z, x),
            scan_capability,
            str(self.param_cli.get_param("work_dir")),
            parent=self,
        )

        self.start_scan(d, ScanDirection.YZ)

    def start_scan(self, dialog, direction: ScanDirection):
        if not dialog.exec():
            return

        valid, msg = dialog.validate()
        if not valid:
            QtWidgets.QMessageBox.warning(self, "Scan parameter invalid.", msg)
            return

        params = dialog.get_params()
        params["direction"] = direction
        dialog.deleteLater()

        self._prev_pos = self.get_pos()
        self.cli.change_state(ConfocalState.SCAN, params)

    def get_scan_capability(self):
        if self._scan_capability is None:
            params = self.cli.get_param_dict("scan")
            if params is None or "mode" not in params:
                print("[ERROR] Failed to get scan capability")
                return None
            self._scan_capability = params["mode"].options()
        return self._scan_capability

    def request_track_start(self):
        scan_capability = self.get_scan_capability()
        if scan_capability is None:
            return
        d = trackDialog(self.xbound, self.ybound, self.zbound, scan_capability, parent=self)
        c = self.tracker_cli.load_params()
        if c is not None:
            d.load_parameter(c)
        if not d.exec():
            return
        valid, msg = d.validate()
        if not valid:
            QtWidgets.QMessageBox.warning(self, "Track parameter invalid.", msg)
            return

        params = d.get_params()
        self.tracker_cli.save_params(params)
        self.tracker_cli.start(params)

    def request_track_stop(self):
        self.tracker_cli.stop()

    def request_track_now(self):
        self.tracker_cli.track_now()

    def direction_to_scanitem(self, direction: ScanDirection) -> scanItem:
        if direction == ScanDirection.XY:
            return self.XY
        elif direction == ScanDirection.XZ:
            return self.XZ
        else:
            return self.YZ

    def finalize_scan(self, image: Image):
        # block this function against sequential signals
        # without this, dialog interaction might cause racing-like behaviour
        if self._finalizing:
            return
        self._finalizing = True

        if self.tracker_cli.get_state() == BinaryState.ACTIVE:
            self.finalize_track(image)
        else:
            self.finalize_manual_scan(image)

        self._finalizing = False

    def finalize_manual_scan(self, image: Image):
        si = self.direction_to_scanitem(image.direction)

        if si.img.data is None:  # aborted before aquiring one line
            si.pop()
        else:
            fn = self.save_image()
            if fn:
                params = self.get_export_params()
                if params:
                    png_fn = os.path.splitext(fn)[0] + ".png"
                    self.cli.export_image(png_fn, direction=None, params=params)

        if not image.aborted:
            self.request_interact()
            if self._prev_pos is not None:
                self.move_crosshair(self._prev_pos)

    def finalize_track(self, image: Image):
        pass
        # TODO: maybe automatic save at meas/Tracker?
        # if self.dialog.get_save_enable():
        #     tim = time.strftime("%Y%m%d_%H%M%S", time.localtime(finish))
        #     fn = str(self.pathEdit.text()) + '/track_' + tim + '.' + direction.lower() + 's'
        #     print('Auto saving %s scan image to %s' % (direction, fn))
        #     si.save_image(fn)
        #     si.export_image(fn, dialog=False, pos=True)

    def update_state(self, state: ConfocalState, last_state: ConfocalState):
        self.switch_interaction(state, last_state)
        self.switch_buttons(state)
        self.switch_PosEdit(state)
        self.set_background(state)

    def update_tracker_state(self, state: BinaryState, last_state: BinaryState):
        self.tracknowButton.setEnabled(state == BinaryState.ACTIVE)
        self.trackstopButton.setEnabled(state == BinaryState.ACTIVE)
        self.trackstartButton.setEnabled(state == BinaryState.IDLE)

    def switch_buttons(self, state):
        for b in (
            self.xyscanButton,
            self.xyfitButton,
            self.xymaxButton,
            self.xypopButton,
            self.xyclearButton,
            self.xysaveButton,
            self.xyexportButton,
            self.xzscanButton,
            self.xzfitButton,
            self.xzmaxButton,
            self.xzpopButton,
            self.xzclearButton,
            self.xzsaveButton,
            self.xzexportButton,
            self.yzscanButton,
            self.yzfitButton,
            self.yzmaxButton,
            self.yzpopButton,
            self.yzclearButton,
            self.yzsaveButton,
            self.yzexportButton,
            self.exportviewButton,
            self.loadButton,
        ):
            b.setEnabled(state != ConfocalState.SCAN)

        for b in (self.sethomeButton, self.gohomeButton, self.centerButton):
            b.setEnabled(state in (ConfocalState.PIEZO, ConfocalState.INTERACT))

        if state == ConfocalState.IDLE:
            self.idleButton.setChecked(True)
        if state == ConfocalState.PIEZO:
            self.piezoButton.setChecked(True)
        if state == ConfocalState.INTERACT:
            self.interactButton.setChecked(True)

        if state == ConfocalState.SCAN:
            d = self.cli.get_direction()
            if d == ScanDirection.XY:
                self.xyscanButton.setChecked(True)
            elif d == ScanDirection.XZ:
                self.xzscanButton.setChecked(True)
            elif d == ScanDirection.YZ:
                self.yzscanButton.setChecked(True)

    def switch_interaction(self, state, last_state):
        # This state change must avoid `changing to current state`.
        def active(s):
            return s in (ConfocalState.PIEZO, ConfocalState.INTERACT)

        if not active(last_state) and active(state):
            self.XY.switch_interaction(True)
            self.XZ.switch_interaction(True)
            self.YZ.switch_interaction(True)
            self.switch_TgtBox(True)
        elif active(last_state) and not active(state):
            self.XY.switch_interaction(False)
            self.XZ.switch_interaction(False)
            self.YZ.switch_interaction(False)
            self.switch_TgtBox(False)

    def switch_TgtBox(self, editable):
        for ax in ("X", "Y", "Z"):
            box = getattr(self, ax + "TgtBox")
            box.setReadOnly(not editable)
            if editable:
                box.editingFinished.connect(getattr(self, ax + "_edited"))
            else:
                box.editingFinished.disconnect(getattr(self, ax + "_edited"))

    def switch_PosEdit(self, state):
        """Set PosEdit string for states without polling."""

        if state == ConfocalState.IDLE:
            for w in (self.XPosEdit, self.YPosEdit, self.ZPosEdit):
                w.setText("Idle")
                w.setStyleSheet("QLineEdit {background: #800000;}")
        if state == ConfocalState.SCAN:
            for w in (self.XPosEdit, self.YPosEdit, self.ZPosEdit):
                w.setText("Scanning")
                w.setStyleSheet("QLineEdit {background: #AA8800;}")

    def set_background(self, state):
        if state == ConfocalState.SCAN:
            bg = "#AA8800"
        else:
            bg = "black"

        for w in [self.XTgtBox, self.YTgtBox, self.ZTgtBox]:
            w.setStyleSheet("background: {bg:s};".format(bg=bg))


class traceView(ClientWidget, Ui_traceView):
    """Widget for Confocal trace view."""

    def __init__(self, gconf: dict, name, param_server_name, context, parent=None):
        ClientWidget.__init__(self, parent)
        self.setupUi(self)

        self._single_channel = None

        self.conf = local_conf(gconf, name)

        self.pi = pg.PlotItem()
        self.graphicsView.setCentralItem(self.pi)

        self.cli = QTracerClient(gconf, name, context=context, parent=self)
        self.param_cli = ParamClient(gconf, param_server_name, context=context)
        self.add_clients(self.cli, self.param_cli)

        self.fps_counter = FPSCounter()

        self.init_plot()
        self.init_connection()

        self.timestampBox.setChecked(True)
        self.update_fontsize()

    def init_plot(self):
        try:
            size = self.conf["tracer"]["size"]
        except KeyError:
            size = DEFAULT_TRACER_SIZE

        colors = colors_tab20_pair()
        sma_width = 3
        self._c0 = colors[0][0]
        self._c1 = colors[1][0]
        self._ctotal = colors[2][0]
        self.curve0 = self.pi.plot(np.zeros(size), pen=colors[0][1])
        self.curve1 = self.pi.plot(np.zeros(size), pen=colors[1][1])
        self.curve_total = self.pi.plot(np.zeros(size), pen=colors[2][1])
        self.curve0_sma = self.pi.plot(
            np.zeros(size),
            pen=mkPen(self._c0, width=sma_width),
        )
        self.curve1_sma = self.pi.plot(
            np.zeros(size),
            pen=mkPen(self._c1, width=sma_width),
        )
        self.curve_total_sma = self.pi.plot(
            np.zeros(size),
            pen=mkPen(self._ctotal, width=sma_width),
        )

        self.pi.showGrid(x=True, y=True)
        self.pi.setLabel("bottom", "Data point")
        self.pi.setLabel("left", "Intensity", "")

    def init_connection(self):
        self.saveButton.clicked.connect(self.save_data)
        self.pauseButton.clicked.connect(self.cli.pause)
        self.resumeButton.clicked.connect(self.cli.resume)
        self.clearButton.clicked.connect(self.clear)

        self.cli.traceUpdated.connect(self.update)
        self.cli.paused.connect(self.update_buttons)

        self.showtotalBox.toggled.connect(self.toggle_total)
        self.show0Box.toggled.connect(self.toggle0)
        self.show1Box.toggled.connect(self.toggle1)
        self.showrawBox.toggled.connect(self.toggle_raw)
        self.fontsizeBox.editingFinished.connect(self.update_fontsize)
        self.timestampBox.toggled.connect(self.update_xaxis)

    def update_xaxis(self, use_ts: bool):
        if use_ts:
            ai = pg.DateAxisItem(text="Date time")
            self.pi.setAxisItems({"bottom": ai})
        else:
            ai = pg.AxisItem("bottom", text="Data point")
            self.pi.setAxisItems({"bottom": ai})
        ai.showLabel()
        self.pi.enableAutoRange()

    def update_fontsize(self):
        fs = self.fontsizeBox.value()
        self.labeltotal.setStyleSheet(f"QLabel {{font: bold {fs}px; background: {self._ctotal};}}")
        self.label0.setStyleSheet(f"QLabel {{font: bold {fs}px; background: {self._c0};}}")
        self.label1.setStyleSheet(f"QLabel {{font: bold {fs}px; background: {self._c1};}}")

    def update_buttons(self, is_paused: bool):
        self.pauseButton.setEnabled(not is_paused)
        self.resumeButton.setEnabled(is_paused)

    def _sum_traces(self, trace: Trace) -> tuple[np.ndarray, np.ndarray]:
        df0, df1 = trace.as_dataframes()
        rdf1 = df1.reindex(df0.index, method="nearest")
        cdf = pd.concat([df0, rdf1])
        sdf = cdf.groupby(cdf.index).sum()
        return sdf.index.values, sdf[0].values

    def init_with_first_data(self, trace: Trace):
        self.pi.setLabel("left", "Intensity", trace.yunit)

        chs = trace.channels()
        if chs == 1:
            self._single_channel = True
            for b in (self.show1Box, self.showtotalBox):
                b.setChecked(False)
                b.setEnabled(False)
            self.show0Box.setChecked(True)
            self.show0Box.setEnabled(False)
            for l in (self.label1, self.labeltotal):
                l.setVisible(False)
        elif chs == 2:
            self._single_channel = False
        else:
            raise ValueError(f"Unexpected number of channels: {chs}")

    def update(self, trace: Trace):
        if self._single_channel is None:
            self.init_with_first_data(trace)
        if self._single_channel:
            self.update_single(trace)
        else:
            self.update_dual(trace)

    def update_single(self, trace: Trace):
        N = self.smaBox.value()
        raw = self.showrawBox.isChecked()
        if self.timestampBox.isChecked():
            s0, t0 = trace.valid_trace(0)
            s0 = s0.astype(np.float64) * 1e-9
            tsma0 = simple_moving_average(t0, N)
            mean0 = tsma0[-1] if len(tsma0) else 0.0
            if raw:
                self.curve0.setData(x=s0, y=t0)
            self.curve0_sma.setData(x=simple_moving_average(s0, N), y=tsma0)
        else:
            tsma0 = simple_moving_average(trace.traces[0], N)
            mean0 = tsma0[-1] if len(tsma0) else 0.0
            if raw:
                self.curve0.setData(trace.traces[0])
            self.curve0_sma.setData(tsma0)

        self.label0.setText("PD0: {:.2e}".format(mean0))
        self.fpsLabel.setText("{:.1f} fps".format(self.fps_counter.tick()))

    def update_dual(self, trace: Trace):
        N = self.smaBox.value()
        raw = self.showrawBox.isChecked()
        if self.timestampBox.isChecked():
            s0, t0 = trace.valid_trace(0)
            s0 = s0.astype(np.float64) * 1e-9
            tsma0 = simple_moving_average(t0, N)
            mean0 = tsma0[-1] if len(tsma0) else 0.0
            s1, t1 = trace.valid_trace(1)
            s1 = s1.astype(np.float64) * 1e-9
            tsma1 = simple_moving_average(t1, N)
            mean1 = tsma1[-1] if len(tsma1) else 0.0
            if self.showtotalBox.isChecked():
                s, t = self._sum_traces(trace)
                s = s.astype(np.float64) * 1e-9
                if raw:
                    self.curve_total.setData(x=s, y=t)
                self.curve_total_sma.setData(
                    x=simple_moving_average(s, N), y=simple_moving_average(t, N)
                )
            if self.show0Box.isChecked():
                if raw:
                    self.curve0.setData(x=s0, y=t0)
                self.curve0_sma.setData(x=simple_moving_average(s0, N), y=tsma0)
            if self.show1Box.isChecked():
                if raw:
                    self.curve1.setData(x=s1, y=t1)
                self.curve1_sma.setData(x=simple_moving_average(s1, N), y=tsma1)
        else:
            tsma0 = simple_moving_average(trace.traces[0], N)
            mean0 = tsma0[-1] if len(tsma0) else 0.0
            tsma1 = simple_moving_average(trace.traces[1], N)
            mean1 = tsma1[-1] if len(tsma1) else 0.0
            if self.showtotalBox.isChecked():
                t = trace.traces[0] + trace.traces[1]
                if raw:
                    self.curve_total.setData(t)
                self.curve_total_sma.setData(simple_moving_average(t, N))
            if self.show0Box.isChecked():
                if raw:
                    self.curve0.setData(trace.traces[0])
                self.curve0_sma.setData(tsma0)
            if self.show1Box.isChecked():
                if raw:
                    self.curve1.setData(trace.traces[1])
                self.curve1_sma.setData(tsma1)

        self.labeltotal.setText("Total: {:.2e}".format(mean0 + mean1))
        self.label0.setText("PD0: {:.2e}".format(mean0))
        self.label1.setText("PD1: {:.2e}".format(mean1))
        self.fpsLabel.setText("{:.1f} fps".format(self.fps_counter.tick()))

    def toggle_total(self, show: bool):
        if not show:
            self.curve_total.clear()
            self.curve_total_sma.clear()

    def toggle0(self, show: bool):
        if not show:
            self.curve0.clear()
            self.curve0_sma.clear()

    def toggle1(self, show: bool):
        if not show:
            self.curve1.clear()
            self.curve1_sma.clear()

    def toggle_raw(self, show: bool):
        if not show:
            for c in (self.curve_total, self.curve0, self.curve1):
                c.clear()

    def clear(self):
        self.cli.clear()
        self.fps_counter = FPSCounter()
        self.pi.enableAutoRange()

    def save_data(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "Trace", ".trace")
        if not fn:
            return

        self.param_cli.set_param("work_dir", os.path.split(fn)[0])
        note = self.param_cli.get_param("note", "")
        self.cli.save_trace(fn, note=note)
        image_fn = os.path.splitext(fn)[0] + ".png"
        self.cli.export_trace(image_fn, params={"timestamp": self.timestampBox.isChecked()})


class ConfocalMainWindow(QtWidgets.QMainWindow):
    """MainWindow with ConfocalWidget and traceView."""

    def __init__(self, gconf: dict, name, context, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        lconf = local_conf(gconf, name)
        target = lconf["target"]
        style = lconf.get("style", {})
        invert = lconf.get("invert", (False, False, False))
        move_interval_ms = lconf.get("move_interval_ms", 0)

        self.confocal = ConfocalWidget(
            gconf,
            target["confocal"],
            target["tracker"],
            target["param_server"],
            style,
            invert,
            move_interval_ms,
            context,
            parent=self,
        )
        self.traceView = traceView(
            gconf, target["confocal"], target["param_server"], context, parent=self
        )

        self.setWindowTitle(f"MAHOS.ConfocalGUI ({join_name(target['confocal'])})")
        self.setAnimated(False)
        self.setCentralWidget(self.confocal)
        self.d_traceView = QtWidgets.QDockWidget("Trace", parent=self)
        self.d_traceView.setWidget(self.traceView)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.d_traceView)

        self.file_menu = self.menuBar().addMenu("File")
        self.action_close = QtGui.QAction("Close")
        self.action_shutdown = QtGui.QAction("Shutdown")
        self.action_close.triggered.connect(self.close)
        self.action_shutdown.triggered.connect(self.shutdown)
        self.file_menu.addAction(self.action_close)
        self.file_menu.addAction(self.action_shutdown)

        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction(self.d_traceView.toggleViewAction())

    def shutdown(self):
        self.confocal.cli.shutdown()
        self.close()

    def closeEvent(self, event):
        """Close the clients on closeEvent."""

        self.confocal.close_clients()
        self.traceView.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)


class ConfocalGUI(GUINode):
    """GUINode for Confocal using ConfocalMainWindow."""

    def init_widget(self, gconf: dict, name, context):
        return ConfocalMainWindow(gconf, name, context)
