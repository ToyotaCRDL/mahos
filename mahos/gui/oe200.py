#!/usr/bin/env python3

"""
GUI client of InstrumentServer to operate OE-200 Photo Receiver.

.. This file is a part of MAHOS project.

"""

from .Qt import QtGui

from ..inst.pd_interface import OE200Interface
from ..inst.server import MultiInstrumentClient

from .ui.oe200 import Ui_OE200Widget
from .gui_node import GUINode
from .common_widget import ClientTopWidget
from ..node.node import local_conf, join_name


def set_fontsize(widget, fontsize: int):
    font = QtGui.QFont()
    font.setPointSize(fontsize)
    widget.setFont(font)


class OE200Widget(ClientTopWidget, Ui_OE200Widget):
    """Top widget for OE200"""

    def __init__(self, gconf: dict, name, context):
        ClientTopWidget.__init__(self)
        self.setupUi(self)

        self.conf = local_conf(gconf, name)
        servers: dict = self.conf["target"]["servers"]
        fontsize = self.conf.get("fontsize", 26)

        self.cli = MultiInstrumentClient(gconf, servers, context=context)
        self.add_client(self.cli)

        if len(servers) != 1:
            raise ValueError("supports only one target for now")
        self.pds = []
        for pd in servers:
            self.pds.append(OE200Interface(self.cli, pd))

        for w in (
            self.lowBox,
            self.lowButton,
            self.highBox,
            self.highButton,
            self.acButton,
            self.dcButton,
            self.setButton,
        ):
            set_fontsize(w, fontsize)

        self.lowButton.toggled.connect(self.update_box)
        self.setButton.clicked.connect(self.request_settings)

        self.update_box()
        self.setWindowTitle(f"MAHOS[F].OE200GUI ({join_name(name)})")
        self.get_initialize()

    def get_initialize(self):
        d = self.pds[0].get_gain_coupling()
        if d is None:
            print("[ERROR] Failed to get current setting.")
        self.lowButton.setChecked(d["low_noise"])
        if d["low_noise"]:
            self.lowBox.setCurrentIndex(self.lowBox.findText(str(d["gain_exponent"])))
        else:
            self.highBox.setCurrentIndex(self.highBox.findText(str(d["gain_exponent"])))
        self.dcButton.setChecked(d["DC_coupling"])

    def update_box(self):
        low = self.lowButton.isChecked()
        self.lowBox.setEnabled(low)
        self.highBox.setEnabled(not low)

    def request_settings(self):
        low_noise = self.lowButton.isChecked()
        if low_noise:
            gain_exponent = int(self.lowBox.currentText())
        else:
            gain_exponent = int(self.highBox.currentText())
        DC_coupling = self.dcButton.isChecked()

        self.pds[0].set_gain_coupling(low_noise, gain_exponent, DC_coupling)


class OE200GUI(GUINode):
    def init_widget(self, gconf: dict, name, context):
        return OE200Widget(gconf, name, context)
