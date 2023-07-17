#!/usr/bin/env python3

"""
GUI client of InstrumentServer to operate DigitalOut (On/Off switch) manually.

.. This file is a part of MAHOS project.

"""

from functools import partial
from .Qt import QtCore, QtWidgets, QtGui

from ..inst.daq_interface import DigitalOutInterface
from ..inst.server import MultiInstrumentClient

from .gui_node import GUINode
from .common_widget import ClientTopWidget
from ..node.node import local_conf, join_name


def set_fontsize(widget, fontsize: int):
    font = QtGui.QFont()
    font.setPointSize(fontsize)
    widget.setFont(font)


class DigitalOutWidget(ClientTopWidget):
    """Top widget for DigitalOutGUI"""

    def __init__(self, gconf: dict, name, context):
        ClientTopWidget.__init__(self)

        self.conf = local_conf(gconf, name)
        servers: dict = self.conf["target"]["servers"]
        fontsize = self.conf.get("fontsize", 26)

        self.cli = MultiInstrumentClient(gconf, servers, context=context)
        self.add_client(self.cli)
        self.digital_outs = {}
        self.buttons = {}
        gl = QtWidgets.QGridLayout()
        for i, n in enumerate(servers):
            button = QtWidgets.QPushButton("Unknown")
            button.setCheckable(True)
            button.setChecked(False)
            button.toggled.connect(partial(self.switch, n))
            self.digital_outs[n] = DigitalOutInterface(self.cli, n)
            self.buttons[n] = button
            sizePolicy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
            )
            button.setSizePolicy(sizePolicy)
            label = QtWidgets.QLabel(n)
            set_fontsize(button, fontsize)
            set_fontsize(label, fontsize)
            gl.addWidget(label, i, 0)
            gl.addWidget(button, i, 1)
        self.setLayout(gl)
        self.setWindowTitle(f"MAHOS[F].DigitalOutGUI ({join_name(name)})")

    def sizeHint(self):
        return QtCore.QSize(500, 250 * len(self.buttons))

    def switch(self, name: str, checked: bool):
        if self.request_switch(name, checked):
            if checked:
                self.buttons[name].setText("High")
            else:
                self.buttons[name].setText("Low")
        else:
            self.buttons[name].setText("Failed")

    def request_switch(self, name: str, checked: bool) -> bool:
        if checked:
            return self.digital_outs[name].set_output_high()
        else:
            return self.digital_outs[name].set_output_low()


class DigitalOutGUI(GUINode):
    def init_widget(self, gconf: dict, name, context):
        return DigitalOutWidget(gconf, name, context)
