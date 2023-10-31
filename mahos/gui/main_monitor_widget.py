#!/usr/bin/env python3

"""
Custom widgets for the MainMonitor.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from .Qt import QtWidgets, QtCore, QtGui


class NotePlainTextEdit(QtWidgets.QPlainTextEdit):
    commit = QtCore.pyqtSignal()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if (
            e.key() == QtCore.Qt.Key.Key_Return
            and e.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.commit.emit()
        else:
            QtWidgets.QPlainTextEdit.keyPressEvent(self, e)
