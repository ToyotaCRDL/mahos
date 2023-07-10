#!/usr/bin/env python3

import typing as T
import sys
import multiprocessing as mp
import threading as mt
import importlib.resources

from ..node.node import NodeBase, join_name
from ..node.comm import Context
from .Qt import QtWidgets, QtCore
from ..util.typing import NodeName
from . import breeze_resources
from .breeze_resources import dark


class GUINode(NodeBase):
    """GUINode is a variant of Nodes for Qt-based GUI frontends."""

    def __init__(self, gconf: dict, name: NodeName, context: T.Optional[Context] = None):
        NodeBase.__init__(self, gconf, name)

        self.app = QtWidgets.QApplication(sys.argv)
        self.load_stylesheet()

        self.widget = self.init_widget(gconf, name, context)

    def init_widget(self, gconf: dict, name: NodeName, context: T.Optional[Context]):
        """Initialize top widget and return it. Every GUINode subclass must implement this."""

        raise NotImplementedError("Implement init_widget")

    def load_stylesheet(self):
        """Load the stylesheet. Currently using BreezeStyleSheet."""

        if sys.version_info.minor == 8:
            # no importlib.resources.files in Python 3.8
            # remove this after dropping Python 3.8 support
            with importlib.resources.path(breeze_resources, "dark") as p:
                path = p
        else:
            path = importlib.resources.files(dark)
        QtCore.QDir.addSearchPath("dark", str(path))
        file = QtCore.QFile("dark:stylesheet.qss")
        file.open(QtCore.QFile.OpenModeFlag.ReadOnly | QtCore.QFile.OpenModeFlag.Text)
        stream = QtCore.QTextStream(file)
        self.app.setStyleSheet(stream.readAll())

    def main(self):
        """Start the QApplication."""

        self.widget.show()
        self.app.setActiveWindow(self.widget)
        sys.exit(self.app.exec())


def run_gui_node_proc(NodeClass, gconf: dict, name: NodeName):
    c: GUINode = NodeClass(gconf, name)
    c.main()


def run_gui_node_thread(NodeClass, gconf: dict, name: NodeName, context: Context):
    c: GUINode = NodeClass(gconf, name, context=context)
    c.main()


def start_gui_node_proc(
    ctx: mp.context.BaseContext, NodeClass, gconf: dict, name: NodeName
) -> mp.Process:
    proc = ctx.Process(
        target=run_gui_node_proc, args=(NodeClass, gconf, name), name=join_name(name)
    )
    proc.start()
    return proc


def start_gui_node_thread(ctx: Context, NodeClass, gconf: dict, name: NodeName) -> mt.Thread:
    thread = mt.Thread(
        target=run_gui_node_thread, args=(NodeClass, gconf, name, ctx), name=join_name(name)
    )
    thread.start()
    return thread
