#!/usr/bin/env python3

"""
Qt wrapper-wrapper.

.. This file is a part of MAHOS project.
"""

from PyQt6 import QtCore, QtGui, QtWidgets

__all__ = [
    "QtCore",
    "QtGui",
    "QtWidgets",
    "qthread_sleep",
    "save_file_dialog",
    "open_file_dialog",
    "question_yn",
]


def qthread_sleep(sec):
    if sec < 1.0e-3:
        QtCore.QThread.usleep(int(sec * 1.0e6))
    elif sec < 1.0:
        QtCore.QThread.msleep(int(sec * 1.0e3))
    else:
        QtCore.QThread.sleep(int(sec))


def save_file_dialog(parent, title: str, directory: str, initial_filter: str) -> str:
    fn = QtWidgets.QFileDialog.getSaveFileName(parent, title, directory, initial_filter)
    return fn[0]


def open_file_dialog(parent, title: str, directory: str, initial_filter: str) -> str:
    fn = QtWidgets.QFileDialog.getOpenFileName(parent, title, directory, initial_filter)
    return fn[0]


def question_yn(parent, title: str, question: str, default_yes=False) -> bool:
    buttons = QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
    if default_yes:
        default = QtWidgets.QMessageBox.StandardButton.Yes
    else:
        default = QtWidgets.QMessageBox.StandardButton.No
    ret = QtWidgets.QMessageBox.question(parent, title, question, buttons, default)
    return ret == QtWidgets.QMessageBox.StandardButton.Yes
