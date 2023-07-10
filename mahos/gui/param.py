#!/usr/bin/env python3

"""
Common widgets/utilities for GUI to deal with Generic Parameters (mahos.msgs.param_msgs).

"""

from __future__ import annotations

from ..msgs import param_msgs as P
from .Qt import QtWidgets, QtCore
from .common_widget import SpinBox


def apply_widgets(
    params: P.ParamDict[str, P.PDValue],
    name_widgets: list[tuple[str, QtWidgets.QWidget, float | None]],
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
        p: P.Param = params[name]
        if isinstance(p, P.NumberParam):
            p: P.NumberParam
            mn = p.minimum() * coeff
            mx = p.maximum() * coeff
            v = p.some_value() * coeff
            if isinstance(widget, QtWidgets.QSpinBox) and isinstance(p, P.FloatParam):
                mn = round(mn)
                mx = round(mx)
                v = round(v)
            widget.setMinimum(mn)
            widget.setMaximum(mx)
            widget.setValue(v)


def set_enabled(params: dict, name_widgets: list[tuple[str, QtWidgets.QWidget]]):
    for name, widget in name_widgets:
        widget.setEnabled(name in params)


class ParamTable(QtWidgets.QTableWidget):
    """QTableWidget for input of ParamDict."""

    def __init__(self, parent=None):
        QtWidgets.QTableWidget.__init__(self, parent=parent)

        self._params = P.ParamDict()
        self._params_f = P.ParamDict()
        self._to_row = {}
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().hide()

    def _add_widget_number(self, param: P.Numberparam):
        if isinstance(param, P.FloatParam):
            widget = SpinBox(
                parent=self,
                value=param.some_value(),
                bounds=param.bounds(),
                suffix=param.unit(),
                siPrefix=param.SI_prefix(),
                decimals=param.digit(),
            )
        elif isinstance(param, P.IntParam):
            widget = SpinBox(
                parent=self,
                int=True,
                value=param.some_value(),
                bounds=param.bounds(),
                suffix=param.unit(),
                siPrefix=param.SI_prefix(),
                decimals=param.digit(),
            )
        widget.valueChanged.connect(param.set)

        def restore():
            widget.setValue(param.default())

        return widget, restore

    def _add_widget_str(self, param: P.StrParam):
        widget = QtWidgets.QLineEdit(param.some_value(), parent=self)

        def set_():
            param.set(widget.text())

        def restore():
            param.restore_default()
            widget.setText(param.default())

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

        def set_(value):
            param.set(value)
            widget.setChecked(value)
            widget.setText("True" if value else "False")

        def restore():
            widget.setChecked(param.default())

        set_(param.some_value())
        widget.toggled.connect(set_)
        return widget, restore

    def _add_widget_choice(self, param: P.ChoiceParam):
        widget = QtWidgets.QComboBox(parent=self)
        default_i = param.options().index(param.default())

        for o in param.options():
            widget.addItem(str(o))

        def set_(i):
            param.set(param.options()[i])

        def restore():
            widget.setCurrentIndex(default_i)

        widget.setCurrentIndex(param.options().index(param.some_value()))
        widget.currentIndexChanged.connect(set_)
        return widget, restore

    def _add_checkbox(
        self, param: P.Param, widget: QtWidgets.QWidget, button: QtWidgets.QPushButton
    ):
        checkbox = QtWidgets.QCheckBox("Enabled", parent=self)

        def toggle(checked):
            param.set_enable(checked)
            checkbox.setText("Enabled" if checked else "Disabled")
            widget.setEnabled(checked)
            button.setEnabled(checked)

        checkbox.toggled.connect(toggle)
        enabled = param.enabled()
        checkbox.setChecked(True)
        checkbox.setChecked(enabled)
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
            if param.optional():
                checkbox = self._add_checkbox(param, widget, btn)
                self.setCellWidget(i, 2, checkbox)
            else:
                add_label_item(i, 2, "")

            # doc
            add_label_item(i, 3, param.doc())

    def params(self) -> P.ParamDict:
        return self._params

    def apply_value(self, keyf: str, val):
        if keyf not in self._params_f:
            return
        param = self._params_f[keyf]
        w = self.cellWidget(self._to_row[keyf], 1)

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
            i = w.findText(str(val))
            if i >= 0:
                w.setCurrentIndex(i)
        else:
            raise TypeError(f"{param} has unsupported Param type")
