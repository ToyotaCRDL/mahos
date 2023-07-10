#!/usr/bin/env python3

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
