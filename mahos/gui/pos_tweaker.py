#!/usr/bin/env python3

"""
GUI frontend for PosTweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from functools import partial

from .Qt import QtWidgets, QtGui, question_yn

# from .Qt import QtCore

from ..msgs.pos_tweaker_msgs import PosTweakerStatus
from ..node.global_params import GlobalParamsClient
from .pos_tweaker_client import QPosTweakerClient
from .gui_node import GUINode
from .common_widget import ClientTopWidget
from ..node.node import local_conf, join_name
from .dialog import save_dialog, load_dialog


def set_fontsize(widget, fontsize: int):
    font = QtGui.QFont()
    font.setPointSize(fontsize)
    widget.setFont(font)


class AxisWidgets(object):
    def __init__(
        self,
        pos_label: QtWidgets.QLabel,
        target_box: QtWidgets.QDoubleSpinBox,
        moving_label: QtWidgets.QLabel,
        homed_label: QtWidgets.QLabel,
    ):
        self.pos_label = pos_label
        self.target_box = target_box
        self.moving_label = moving_label
        self.homed_label = homed_label
        self.is_homed = False

    def fmt_pos(self, target: float, pos: float):
        return f"{pos:.3f} ({target:.3f})"

    def fmt_moving(self, moving: bool):
        return "Moving" if moving else "Stopped"

    def fmt_homed(self, homed: bool):
        return "Homed" if homed else "NOT Homed"

    def update(self, state: dict[str, [float, bool]]):
        self.pos_label.setText(self.fmt_pos(state["target"], state["pos"]))
        self.moving_label.setText(self.fmt_moving(state["moving"]))
        self.homed_label.setText(self.fmt_homed(state["homed"]))
        self.is_homed = state["homed"]


class PosTweakerWidget(ClientTopWidget):
    """Top widget for DigitalOutGUI"""

    def __init__(self, gconf: dict, name, gparams_name, verbose, fontsize, context):
        ClientTopWidget.__init__(self)
        self.setWindowTitle(f"MAHOS.PosTweakerGUI ({join_name(name)})")

        self._fontsize = fontsize
        self._widgets = {}
        self._group_name = "__" + name + "__"

        self.cli = QPosTweakerClient(gconf, name, context=context)
        self.cli.statusUpdated.connect(self.init_with_status)

        self.gparams_cli = GlobalParamsClient(gconf, gparams_name, context=context)
        self.add_clients(self.cli, self.gparams_cli)

        self.hl = QtWidgets.QHBoxLayout()
        self.gl = QtWidgets.QGridLayout()
        self.vl = QtWidgets.QVBoxLayout()

        self.stop_all_button = QtWidgets.QPushButton("Stop All")
        self.home_all_button = QtWidgets.QPushButton("Home All")
        self.save_button = QtWidgets.QPushButton("Save")
        self.load_button = QtWidgets.QPushButton("Load")
        self.stop_all_button.pressed.connect(self.request_stop_all)
        self.home_all_button.pressed.connect(self.request_home_all)
        self.save_button.pressed.connect(self.request_save)
        self.load_button.pressed.connect(self.request_load)

        for b in (self.save_button, self.load_button, self.stop_all_button, self.home_all_button):
            self.hl.addWidget(b)
        self.vl.addLayout(self.hl)
        self.vl.addLayout(self.gl)
        self.setLayout(self.vl)

        self.setEnabled(False)

    def init_with_status(self, status: PosTweakerStatus):
        """initialize widget after receiving first status."""

        # only once.
        self.cli.statusUpdated.disconnect(self.init_with_status)

        for i, (ax, state) in enumerate(status.axis_states.items()):
            label = QtWidgets.QLabel(ax)
            pos_label = QtWidgets.QLabel()
            target_box = QtWidgets.QDoubleSpinBox()
            range_min, range_max = state["range"]
            target_box.setValue(state["target"])
            target_box.setMinimum(range_min)
            target_box.setMaximum(range_max)
            target_box.setDecimals(3)
            target_box.lineEdit().returnPressed.connect(partial(self.request_set_target, ax))
            moving_label = QtWidgets.QLabel()
            homed_label = QtWidgets.QLabel()
            set_target_button = QtWidgets.QPushButton("Set")
            set_target_button.pressed.connect(partial(self.request_set_target, ax))
            stop_button = QtWidgets.QPushButton("Stop")
            stop_button.pressed.connect(partial(self.request_stop, ax))
            home_button = QtWidgets.QPushButton("Home")
            home_button.pressed.connect(partial(self.request_home, ax))

            self._widgets[ax] = AxisWidgets(
                pos_label,
                target_box,
                moving_label,
                homed_label,
            )
            self._widgets[ax].update(state)

            for w in (
                label,
                pos_label,
                target_box,
                set_target_button,
                stop_button,
                moving_label,
                homed_label,
                home_button,
            ):
                set_fontsize(w, self._fontsize)
            self.gl.addWidget(label, i, 0)
            self.gl.addWidget(pos_label, i, 1)
            self.gl.addWidget(target_box, i, 2)
            self.gl.addWidget(set_target_button, i, 3)
            self.gl.addWidget(stop_button, i, 4)
            self.gl.addWidget(moving_label, i, 5)
            self.gl.addWidget(homed_label, i, 6)
            self.gl.addWidget(home_button, i, 7)

        self.cli.statusUpdated.connect(self.update)
        self.setEnabled(True)

    # def sizeHint(self):
    #     return QtCore.QSize(500, 1000)

    def update(self, status: PosTweakerStatus):
        for ax, state in status.axis_states.items():
            self._widgets[ax].update(state)

    def request_set_target(self, ax: str):
        v = self._widgets[ax].target_box.value()
        self.cli.set_target({ax: v})

    def request_stop(self, ax: str):
        self.cli.stop(ax)

    def request_stop_all(self):
        self.cli.stop_all()

    def request_home(self, ax: str):
        if self._widgets[ax].is_homed and not question_yn(
            self,
            "Sure to HOME?",
            f"Axis {ax} has already been homed. Are you sure to perform homing again?",
        ):
            return
        self.cli.home(ax)

    def request_home_all(self):
        if question_yn(
            self, "Sure to HOME ALL?", "Are you sure to perform homing for all the axes?"
        ):
            self.cli.home_all()

    def request_save(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = save_dialog(self, default_path, "PosTweaker", ".ptweak")
        if not fn:
            return

        self.cli.save(fn)

    def request_load(self):
        default_path = str(self.gparams_cli.get_param("work_dir"))
        fn = load_dialog(self, default_path, "PosTweaker or measurement", "")
        if not fn:
            return
        if fn.endswith(".ptweak.h5"):
            # data in individual file for PosTweaker
            group = ""
        else:
            # data written within measurement Data
            group = self._group_name

        self.cli.load(fn, group)


class PosTweakerGUI(GUINode):
    def init_widget(self, gconf: dict, name, context):
        lconf = local_conf(gconf, name)
        target = lconf["target"]
        verbose = lconf.get("verbose", True)
        fontsize = lconf.get("fontsize", 26)
        return PosTweakerWidget(
            gconf, target["pos_tweaker"], target["gparams"], verbose, fontsize, context
        )
