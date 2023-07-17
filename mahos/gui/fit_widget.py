#!/usr/bin/env python3

"""
Common widgets/utilities for GUI to deal with Data Buffer/Fitting functions of BasicMeasNode.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
from itertools import cycle

from . import Qt
from .Qt import QtCore, QtWidgets, QtGui

import matplotlib as mpl

from .ui.fitWidget import Ui_FitWidget

from ..msgs.common_meas_msgs import BasicMeasData, Buffer


_colors = [mpl.colors.to_hex(c) for c in mpl.colormaps.get("tab10").colors]


class FitWidget(QtWidgets.QWidget, Ui_FitWidget):
    def __init__(self, cli, param_cli, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.cli = cli
        self.param_cli = param_cli
        self.buffer = Buffer()

        self.init_widgets()
        self.update_buffer_table()

    # application specific method to be overridden
    def colors(self) -> list[str | list[str] | tuple[str]]:
        return _colors

    # application specific method to be overridden
    def load_dialog(self, default_path: str) -> str:
        name = "Any"
        ext = ".pkl"
        return Qt.open_file_dialog(
            self, f"Load {name} Data", default_path, f"{name} data (*{ext})"
        )

    def init_with_status(self):
        self.loadButton.clicked.connect(self.load)
        self.fitButton.clicked.connect(self.request_fit)
        self.clearfitButton.clicked.connect(self.request_clear_fit)
        self.popbufButton.clicked.connect(self.request_pop_buf)
        self.clearbufButton.clicked.connect(self.request_clear_buf)
        self.bufferTable.cellClicked.connect(self.update_index)

        names = self.cli.get_param_dict_names("fit")
        self.methodBox.addItems(names)
        self.methodBox.currentIndexChanged.connect(self.update_param_table)
        if names:
            self.update_param_table()

    def init_widgets(self):
        self.bufferTable.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )

    def update_buffer(self, buffer: Buffer[tuple[str, BasicMeasData]]):
        if buffer.file_names() == self.buffer.file_names():
            self.buffer = buffer
            return

        self.buffer = buffer
        self.update_buffer_table()

    def update_buffer_table(self):
        def make_name_item(name: str):
            item = QtWidgets.QTableWidgetItem(str(name))
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            return item

        def make_show_items():
            show_item = QtWidgets.QTableWidgetItem()
            show_fit_item = QtWidgets.QTableWidgetItem()
            for item in (show_item, show_fit_item):
                item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                    | QtCore.Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(QtCore.Qt.CheckState.Checked)
            return show_item, show_fit_item

        def make_color_item():
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            return item

        def make_items(row: int, name: str, color_num: int):
            show_item, show_fit_item = make_show_items()
            self.bufferTable.setItem(row, 0, make_name_item(name))
            self.bufferTable.setItem(row, 1, show_item)
            self.bufferTable.setItem(row, 2, show_fit_item)
            for i in range(color_num):
                self.bufferTable.setItem(row, 3 + i, make_color_item())

        self.indexBox.setMaximum(len(self.buffer) - 1)

        color0 = self.colors()[0]
        if isinstance(color0, (list, tuple)):
            color_num = len(color0)
            color_labels = [f"Color{i}" for i in range(color_num)]
        else:
            color_num = 1
            color_labels = ["Color"]

        self.bufferTable.clear()
        self.bufferTable.setColumnCount(3 + color_num)
        self.bufferTable.setRowCount(len(self.buffer) + 1)
        self.bufferTable.setHorizontalHeaderLabels(
            ["Filename", "Show Data", "Show Fit Data"] + color_labels
        )
        self.bufferTable.setVerticalHeaderLabels([str(i) for i in range(-1, len(self.buffer))])

        make_items(0, "Current Data", color_num)
        for i, (name, data) in enumerate(self.buffer):
            make_items(i + 1, str(name), color_num)

        for row, color in zip(range(len(self.buffer) + 1), cycle(self.colors())):
            if color_num > 1:
                for i, c in enumerate(color):
                    item = self.bufferTable.item(row, 3 + i)
                    item.setBackground(QtGui.QColor(c))
            else:
                item = self.bufferTable.item(row, 3)
                item.setBackground(QtGui.QColor(color))

    def update_param_table(self):
        method = self.methodBox.currentText()
        d = self.cli.get_param_dict(method, "fit")
        self.paramTable.update_contents(d)

    def update_index(self, row: int, col: int):
        self.indexBox.setValue(row - 1)

    def is_checked(self, row, col) -> bool:
        return self.bufferTable.item(row, col).checkState() == QtCore.Qt.CheckState.Checked

    def get_plottable_data(self, current_data: BasicMeasData) -> list:
        """get plottable data in buffer.

        :returns: list of (BasicMeasData, show_fit: bool, color: content type of colors())

        """

        data_list = []
        for i, (data, colors) in enumerate(
            zip([current_data] + self.buffer.data_list(), cycle(self.colors()))
        ):
            if data.has_data() and self.is_checked(i, 1):
                show_fit = self.is_checked(i, 2)
                data_list.append((data, show_fit, colors))
        return data_list

    def load(self):
        default_path = str(self.param_cli.get_param("work_dir"))
        fn = self.load_dialog(default_path)
        if not fn:
            return

        data = self.cli.load_data(fn, to_buffer=True)
        if data is not None:
            self.apply_widgets(data)

    def request_fit(self):
        m = self.methodBox.currentText()
        params = self.paramTable.params()
        params["method"] = m
        index = self.indexBox.value()

        resp = self.cli.fit(params, index)
        if not resp.success:
            return
        ret = resp.ret

        if "msg" in ret:
            self.resultEdit.setText(ret["msg"])

        if "popt" in ret and self.applyresultBox.isChecked():
            for key, val in ret["popt"].items():
                self.paramTable.apply_value(".".join(("model", key, "value")), val)

        return resp

    def request_clear_fit(self):
        self.cli.clear_fit(data_index=self.indexBox.value())
        self.resultEdit.setText("")

    def request_pop_buf(self):
        i = self.indexBox.value()
        if i == -1:
            # -1 usually means the last element and default for pop operation.
            # however in this class, -1 means current data and pop operation is not allowed.
            return

        self.cli.pop_buffer(i)

    def request_clear_buf(self):
        self.cli.clear_buffer()

    def apply_widgets(self, data: BasicMeasData):
        if (
            data.fit_params is None
            or data.fit_result is None
            or not self.applyloadBox.isChecked()
            or data.fit_params["method"] != self.methodBox.currentText()
            or "popt" not in data.fit_result
        ):
            return

        for key, val in data.fit_result["popt"].items():
            self.paramTable.apply_value(".".join(("model", key, "value")), val)
