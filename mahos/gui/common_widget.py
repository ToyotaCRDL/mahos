#!/usr/bin/env python3

"""
Common GUI widget implementations.

.. This file is a part of MAHOS project.

"""

import pyqtgraph as pg

from .Qt import QtWidgets

from .ui.compressionWidget import Ui_CompressionWidget


class CompressionWidget(QtWidgets.QWidget, Ui_CompressionWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.methodBox.currentIndexChanged.connect(self.update_level_enable)
        self.update_level_enable()

    def update_level_enable(self):
        self.levelBox.setEnabled(self.methodBox.currentText() == "gzip")

    def get_params(self):
        method = self.methodBox.currentText()
        if method == "None":
            method = None
        if method == "gzip":
            level = self.levelBox.value()
        else:
            level = None
        return {"compression": method, "compression_opts": level}


class SpinBox(pg.SpinBox):
    # remove special height update
    def _updateHeight(self):
        pass


class ClientMixin(object):
    """require attr _clients (list). defines add_client(), add_clients(), close_clients()."""

    def add_client(self, cli):
        """Register a Client. Client must have close() method."""

        self._clients.append(cli)

    def add_clients(self, *clis):
        """Register multiple Clients. A Client must have close() method."""

        self._clients.extend(clis)

    # NOTE: QWidget has public slot close(). Thus, we don't name this method close().
    def close_clients(self):
        """Close all the clients."""

        for cli in self._clients:
            if hasattr(cli, "close"):
                cli.close()


class ClientWidget(QtWidgets.QWidget, ClientMixin):
    """QtWidget with client management."""

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self._clients = []


class ClientTopWidget(ClientWidget):
    """QtWidget with client management which calls close_cli on closeEvent."""

    def closeEvent(self, event):
        self.close_clients()
        QtWidgets.QWidget.closeEvent(self, event)


class ClientMainWindow(QtWidgets.QMainWindow, ClientMixin):
    """QMainWindow with client management which calls close_cli on closeEvent.

    MainWindow will be top, right?

    """

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent=parent)

        self._clients = []

    def closeEvent(self, event):
        self.close_clients()
        QtWidgets.QMainWindow.closeEvent(self, event)
