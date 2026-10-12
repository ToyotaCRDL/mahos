#!/usr/bin/env python3

"""
GUI client of InstrumentServer to operate DigitalOut (On/Off switch) manually.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from functools import partial
from mahos.gui.Qt import QtCore, QtWidgets, QtGui

from mahos.inst.daq_interface import DigitalOutInterface
from mahos.inst.server import MultiInstrumentClient

from mahos.gui.gui_node import GUINode, get_gui_logger
from mahos.gui.common_widget import ClientTopWidget
from mahos.node.node import local_conf, join_name


def set_fontsize(widget, fontsize: int):
    font = QtGui.QFont()
    font.setPointSize(fontsize)
    widget.setFont(font)


class DigitalOutWidget(ClientTopWidget):
    """Top widget for DigitalOutGUI"""

    def __init__(self, gconf: dict, name, context, parent=None, logger=None):
        ClientTopWidget.__init__(self, parent)
        self.logger = get_gui_logger(logger, self.__class__.__name__)

        self.conf = local_conf(gconf, name)
        servers: dict = self.conf["target"]["servers"]
        self.alias: dict = self.conf.get("alias", {})
        fontsize = self.conf.get("fontsize", 26)

        self.cli = MultiInstrumentClient(gconf, servers, context=context)
        self.add_clients(self.cli)
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
        self.setWindowTitle(f"MAHOS.DigitalOutGUI ({join_name(name)})")

    def sizeHint(self):
        return QtCore.QSize(500, 250 * len(self.buttons))

    def switch(self, name: str, checked: bool):
        if self.request_switch(name, checked):
            labels = self.alias.get(name, ("Low", "High"))
            if checked:
                self.buttons[name].setText(labels[1])
            else:
                self.buttons[name].setText(labels[0])
        else:
            self.buttons[name].setText("Failed")

    def request_switch(self, name: str, checked: bool) -> bool:
        if checked:
            return self.digital_outs[name].set_output_high()
        else:
            return self.digital_outs[name].set_output_low()


class DigitalOutGUI(GUINode):
    """GUINode for manual ON/OFF control of DigitalOut instruments.

    This node creates :class:`DigitalOutWidget` and uses the node-local configuration
    to build one toggle button per target server.

    :param target.log: Optional LogBroker target for formal GUI logging.
    :type target.log: tuple[str, str] | str
    :param log_level: (default: ``"INFO"``) Optional logging level.
    :type log_level: str
    :param target.servers: Mapping of display names to target DigitalOut server nodes.
    :type target.servers: dict[str, tuple[str, str] | str]
    :param alias: Optional per-output labels for low/high states.
    :type alias: dict[str, tuple[str, str]]
    :param fontsize: Font size applied to the labels and toggle buttons.
    :type fontsize: int

    """

    def init_widget(self, gconf: dict, name, context):
        return DigitalOutWidget(gconf, name, context, logger=self.logger)
