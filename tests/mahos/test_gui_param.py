#!/usr/bin/env python3

"""
Tests for mahos.gui.param utilities.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mahos.gui.Qt import QtCore, QtWidgets
from mahos.gui.param import (
    ParamDictComboBoxHandler,
    _infer_adaptive_min_step,
    apply_widget_bounds,
)
from mahos.msgs.param_msgs import FloatParam, IntParam, ParamDict


def test_param_dict_combo_box_handler(qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    combo_box = QtWidgets.QComboBox(parent)
    combo_box.addItems(["first", "second"])

    first = ParamDict(first=IntParam(1))
    second = ParamDict(second=IntParam(2))
    responses = {"first": first, "second": None}
    handler = ParamDictComboBoxHandler(combo_box)
    applied = []

    def update():
        params = handler.get(responses.get)
        if params is not None:
            applied.append(params)

    combo_box.currentIndexChanged.connect(update)

    assert not isinstance(handler, QtCore.QObject)
    update()
    assert applied == [first]

    combo_box.setCurrentIndex(1)
    assert combo_box.currentIndex() == 0
    assert applied == [first]

    responses["second"] = second
    combo_box.setCurrentIndex(1)
    assert applied == [first, second]


def test_param_dict_combo_box_handler_clears_failed_initial_selection(qtbot):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    combo_box = QtWidgets.QComboBox(parent)
    combo_box.addItem("unavailable")

    handler = ParamDictComboBoxHandler(combo_box)

    assert handler.get(lambda label: None) is None
    assert combo_box.currentIndex() == -1


def test_apply_widget_bounds_retains_value(qtbot):
    widget = QtWidgets.QDoubleSpinBox()
    qtbot.addWidget(widget)
    widget.setRange(-100.0, 100.0)
    widget.setValue(20.0)

    params = ParamDict(pulse=FloatParam(10e-9, 1e-9, 1e-3))
    apply_widget_bounds(params, [("pulse", widget, 1e9)])

    assert widget.minimum() == 1.0
    assert widget.maximum() == 1e6
    assert widget.value() == 20.0


def test_infer_adaptive_min_step():
    p = FloatParam(
        0.01,
        0.0,
        100.0,
        digit=6,
        step=0.1,
        adaptive_step=True,
    )
    assert _infer_adaptive_min_step(p) == 1e-7


def test_infer_adaptive_min_step_uses_explicit_value():
    p = FloatParam(
        0.01,
        0.0,
        100.0,
        digit=6,
        step=0.1,
        adaptive_step=True,
        adaptive_min_step=1e-6,
    )
    assert _infer_adaptive_min_step(p) == 1e-6


def test_infer_adaptive_min_step_falls_back_to_bounds():
    p = FloatParam(
        0.0,
        0.0,
        100.0,
        digit=6,
        step=0.1,
        adaptive_step=True,
    )
    assert _infer_adaptive_min_step(p) == 1e-3


def test_infer_adaptive_min_step_none_when_disabled():
    p = FloatParam(0.01, 0.0, 100.0, digit=6, step=0.1, adaptive_step=False)
    assert _infer_adaptive_min_step(p) is None


def test_infer_adaptive_min_step_int_clamped():
    p = IntParam(1000, 0, 100000, digit=6, step=1, adaptive_step=True)
    assert _infer_adaptive_min_step(p) == 1
