#!/usr/bin/env python3

"""
Common widgets/utilities for GUI to deal with Generic Parameters (mahos.msgs.param_msgs).

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import math
from collections.abc import Callable

from mahos.msgs import param_msgs as P
from mahos.gui.Qt import QtWidgets, QtCore
from mahos.gui.common_widget import SpinBox


class ParamDictComboBoxHandler:
    """Fetch a selected ParamDict and roll back the combo box when fetching fails."""

    def __init__(self, combo_box: QtWidgets.QComboBox):
        self._combo_box = combo_box
        self._accepted_index = -1

    def get(
        self, fetch: Callable[[str], P.ParamDict[str, P.PDValue] | None]
    ) -> P.ParamDict[str, P.PDValue] | None:
        """Fetch parameters for the current item and accept or roll back the selection."""

        index = self._combo_box.currentIndex()
        if index < 0:
            return None

        params = fetch(self._combo_box.itemText(index))
        if params is not None:
            self._accepted_index = index
            return params

        was_blocked = self._combo_box.blockSignals(True)
        try:
            self._combo_box.setCurrentIndex(self._accepted_index)
        finally:
            self._combo_box.blockSignals(was_blocked)
        return None


def _is_positive_finite(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def _find_finite_span_or_bound(param: P.NumberParam) -> float | int | None:
    mn, mx = param.bounds()
    if mn is not None and mx is not None:
        span = abs(mx - mn)
        if _is_positive_finite(span):
            return span
    if mn is not None:
        mn_abs = abs(mn)
        if _is_positive_finite(mn_abs):
            return mn_abs
    if mx is not None:
        mx_abs = abs(mx)
        if _is_positive_finite(mx_abs):
            return mx_abs
    return None


def _infer_adaptive_min_step(param: P.NumberParam) -> float | int | None:
    """Infer minStep for adaptive stepping when the user did not specify one."""

    if not param.adaptive_step():
        return None
    if param.adaptive_min_step() is not None:
        return param.adaptive_min_step()

    # Find parameter scale.
    # 1. from current value scale.
    ref = abs(param.some_value())

    # 2. from available bounds.
    if not _is_positive_finite(ref):
        ref = _find_finite_span_or_bound(param)

    # 3. from explicit step scale (fall back, usually not very good).
    if not _is_positive_finite(ref):
        step = abs(param.step())
        if _is_positive_finite(step):
            ref = step

    if not _is_positive_finite(ref):
        return None

    exp = math.floor(math.log10(ref)) - param.digit() + 1
    min_step = 10.0**exp

    if isinstance(param, P.IntParam):
        return max(1, int(round(min_step)))
    return min_step


def _apply_bounds(p: P.NumberParam, widget: QtWidgets.QWidget, coeff: float):
    mn = p.minimum() * coeff if p.minimum() is not None else None
    mx = p.maximum() * coeff if p.maximum() is not None else None
    if isinstance(widget, QtWidgets.QSpinBox) and isinstance(p, P.FloatParam):
        mn = round(mn) if mn is not None else None
        mx = round(mx) if mx is not None else None
    if mn is not None:
        widget.setMinimum(mn)
    elif isinstance(widget, QtWidgets.QSpinBox):
        # unbounded (C++ INT_MIN is the smallest lower bound)
        widget.setMinimum(-2_147_483_648)
    elif isinstance(widget, QtWidgets.QDoubleSpinBox):
        widget.setMinimum(-math.inf)
    if mx is not None:
        widget.setMaximum(mx)
    elif isinstance(widget, QtWidgets.QSpinBox):
        # unbounded (C++ INT_MAX is the largest upper bound)
        widget.setMaximum(2_147_483_647)
    elif isinstance(widget, QtWidgets.QDoubleSpinBox):
        widget.setMaximum(math.inf)


def _apply(p: P.Param, widget: QtWidgets.QWidget, coeff: float):
    if isinstance(p, P.NumberParam):
        _apply_bounds(p, widget, coeff)
        v = p.some_value() * coeff
        if isinstance(widget, QtWidgets.QSpinBox) and isinstance(p, P.FloatParam):
            v = round(v)
        widget.setValue(v)
        if isinstance(widget, SpinBox):
            opts = dict(
                suffix=p.unit(),
                siPrefix=p.SI_prefix(),
                decimals=p.digit(),
                step=p.step(),
                dec=p.adaptive_step(),
            )
            min_step = _infer_adaptive_min_step(p)
            if min_step is not None:
                opts["minStep"] = min_step
            widget.setOpts(**opts)
    if isinstance(p, P.BoolParam):
        widget.setChecked(p.value())


def apply_widgets(
    params: P.ParamDict[str, P.PDValue],
    name_widgets: list[tuple[str, QtWidgets.QWidget | list[QtWidgets.QWidget], float | None]],
):
    for nwc in name_widgets:
        if len(nwc) == 3:
            name, widget, coeff = nwc
            if coeff is None:
                coeff = 1
        elif len(nwc) == 2:
            name, widget = nwc
            coeff = 1
        if name not in params:
            continue
        if isinstance(params[name], (list, tuple)):
            for p, w in zip(params[name], widget):
                _apply(p, w, coeff)
        else:
            _apply(params[name], widget, coeff)


def apply_widget_bounds(
    params: P.ParamDict[str, P.PDValue],
    name_widgets: list[tuple[str, QtWidgets.QWidget | list[QtWidgets.QWidget], float | None]],
):
    """Apply numeric parameter bounds to widgets without changing their values."""

    for nwc in name_widgets:
        if len(nwc) == 3:
            name, widget, coeff = nwc
            if coeff is None:
                coeff = 1
        elif len(nwc) == 2:
            name, widget = nwc
            coeff = 1
        if name not in params:
            continue
        if isinstance(params[name], (list, tuple)):
            for p, w in zip(params[name], widget):
                if isinstance(p, P.NumberParam):
                    _apply_bounds(p, w, coeff)
        elif isinstance(params[name], P.NumberParam):
            _apply_bounds(params[name], widget, coeff)


def set_enabled(params: dict, name_widgets: list[tuple[str, QtWidgets.QWidget]]):
    for name, widget in name_widgets:
        widget.setEnabled(name in params)


class ParamTable(QtWidgets.QTableWidget):
    """QTableWidget for input of ParamDict."""

    paramsChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        QtWidgets.QTableWidget.__init__(self, parent=parent)

        self._params = P.ParamDict()
        self._params_f = P.ParamDict()
        self._to_row = {}
        self.verticalHeader().hide()

    def _add_widget_number(self, param: P.Numberparam):
        if isinstance(param, P.FloatParam):
            opts = dict(
                parent=self,
                value=param.some_value(),
                bounds=param.bounds(),
                suffix=param.unit(),
                siPrefix=param.SI_prefix(),
                decimals=param.digit(),
                step=param.step(),
                dec=param.adaptive_step(),
            )
            min_step = _infer_adaptive_min_step(param)
            if min_step is not None:
                opts["minStep"] = min_step
            widget = SpinBox(**opts)
        elif isinstance(param, P.IntParam):
            opts = dict(
                parent=self,
                int=True,
                value=param.some_value(),
                bounds=param.bounds(),
                suffix=param.unit(),
                siPrefix=param.SI_prefix(),
                decimals=param.digit(),
                step=param.step(),
                dec=param.adaptive_step(),
            )
            min_step = _infer_adaptive_min_step(param)
            if min_step is not None:
                opts["minStep"] = min_step
            widget = SpinBox(**opts)

        def set_(value):
            param.set(value)
            self.paramsChanged.emit()

        widget.valueChanged.connect(set_)

        def restore():
            widget.setValue(param.default())

        return widget, restore

    def _add_widget_str(self, param: P.StrParam):
        widget = QtWidgets.QLineEdit(param.some_value(), parent=self)

        def set_():
            param.set(widget.text())
            self.paramsChanged.emit()

        def restore():
            param.restore_default()
            widget.setText(param.default())
            self.paramsChanged.emit()

        widget.editingFinished.connect(set_)
        return widget, restore

    def _add_widget_uuid(self, param: P.UUIDParam):
        """uuid is readonly label."""

        widget = QtWidgets.QLabel(str(param.some_value()))

        def restore():
            pass

        return widget, restore

    def _add_widget_bool(self, param: P.BoolParam):
        widget = QtWidgets.QCheckBox("True", parent=self)

        def set_(value, emit=True):
            param.set(value)
            widget.setChecked(value)
            widget.setText("True" if value else "False")
            if emit:
                self.paramsChanged.emit()

        def restore():
            widget.setChecked(param.default())

        set_(param.some_value(), emit=False)
        widget.toggled.connect(set_)
        return widget, restore

    def _add_widget_choice(self, param: P.ChoiceParam):
        widget = QtWidgets.QComboBox(parent=self)
        default_i = param.options().index(param.default())

        for o in param.options():
            widget.addItem(f"{o:_d}" if isinstance(param, P.IntChoiceParam) else str(o))

        def set_(i):
            param.set(param.options()[i])
            self.paramsChanged.emit()

        def restore():
            widget.setCurrentIndex(default_i)

        widget.setCurrentIndex(param.options().index(param.some_value()))
        widget.currentIndexChanged.connect(set_)
        return widget, restore

    def _add_checkbox(
        self, param: P.Param, widget: QtWidgets.QWidget, button: QtWidgets.QPushButton
    ):
        checkbox = QtWidgets.QCheckBox("Enabled", parent=self)

        def toggle(checked, emit=True):
            param.set_enable(checked)
            checkbox.setText("Enabled" if checked else "Disabled")
            widget.setEnabled(checked)
            button.setEnabled(checked)
            if emit:
                self.paramsChanged.emit()

        enabled = param.enabled()
        checkbox.setChecked(enabled)
        toggle(enabled, emit=False)
        checkbox.toggled.connect(toggle)
        return checkbox

    def update_contents(self, params: P.ParamDict):
        if not params.validate():
            print("[ERROR] invalid ParamDict. Cannot update contents.")
            return

        self._params = params.copy()
        self._params_f = self._params.flatten()
        self._to_row = {}

        self.clear()
        self.setColumnCount(5)
        self.setRowCount(len(self._params_f))
        self.setHorizontalHeaderLabels(["Name", "Value", "Enable", "Doc", "Default"])

        def add_label_item(row, col, label):
            item = QtWidgets.QTableWidgetItem(label)
            item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            self.setItem(row, col, item)
            return item

        for i, (key, param) in enumerate(self._params_f.items()):
            self._to_row[key] = i

            # name
            add_label_item(i, 0, key)

            btn = QtWidgets.QPushButton("Restore")

            # value, default
            if isinstance(param, P.NumberParam):
                widget, restore = self._add_widget_number(param)
            elif isinstance(param, P.StrParam):
                widget, restore = self._add_widget_str(param)
            elif isinstance(param, P.UUIDParam):
                widget, restore = self._add_widget_uuid(param)
                btn.setEnabled(False)
            elif isinstance(param, P.BoolParam):
                widget, restore = self._add_widget_bool(param)
            elif isinstance(param, (P.EnumParam, P.StrChoiceParam, P.IntChoiceParam)):
                widget, restore = self._add_widget_choice(param)
            else:
                raise TypeError(f"{param} has unsupported Param type")

            btn.clicked.connect(restore)
            self.setCellWidget(i, 1, widget)
            self.setCellWidget(i, 4, btn)

            # enable
            if param.read_only():
                widget.setEnabled(False)
                btn.setEnabled(False)
                add_label_item(i, 2, "Read Only")
            elif param.optional():
                checkbox = self._add_checkbox(param, widget, btn)
                self.setCellWidget(i, 2, checkbox)
            else:
                add_label_item(i, 2, "")

            # doc
            add_label_item(i, 3, param.doc()).setToolTip(param.doc())
        self.resizeColumnsToContents()

    def params(self) -> P.ParamDict:
        return self._params

    def apply_value(self, keyf: str, val):
        if keyf not in self._params_f:
            return
        param = self._params_f[keyf]
        w = self.cellWidget(self._to_row[keyf], 1)

        if isinstance(val, P.Param):
            val = val.value()

        if isinstance(param, P.NumberParam):
            w: SpinBox
            w.setValue(val)
        elif isinstance(param, P.StrParam):
            w: QtWidgets.QLineEdit
            w.setCurrentText(val)
        elif isinstance(param, P.UUIDParam):
            w: QtWidgets.QLabel
            w.setText(str(val))
        elif isinstance(param, P.BoolParam):
            w: QtWidgets.QCheckBox
            w.setChecked(val)
        elif isinstance(param, (P.EnumParam, P.StrChoiceParam, P.IntChoiceParam)):
            w: QtWidgets.QComboBox
            i = w.findText(f"{val:_d}" if isinstance(param, P.IntChoiceParam) else str(val))
            if i >= 0:
                w.setCurrentIndex(i)
        else:
            raise TypeError(f"{param} has unsupported Param type")
