#!/usr/bin/env python3

"""
Base class GUINode and the runners.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import sys
import multiprocessing as mp
import threading as mt
import importlib.resources

from mahos.node.node import NodeBase, get_value, init_logger, join_name
from mahos.node.comm import Context
from mahos.node.log import DummyLogger
from mahos.gui.Qt import QtWidgets, QtCore
from mahos.util.typing import NodeName
from mahos.gui import breeze_resources
from mahos.gui.breeze_resources import dark


def get_gui_logger(logger, name: str):
    """Return logger or a class-named :class:`DummyLogger` when it is ``None``."""

    return DummyLogger(name) if logger is None else logger


class GUINode(NodeBase):
    """GUINode is a variant of Node for Qt-based GUI frontends.

    :param target.log: Optional LogBroker target. A :class:`DummyLogger` is used when omitted.
    :type target.log: tuple[str, str] | str
    :param log_level: Optional logging level. The default is ``"INFO"``.
    :type log_level: str

    :ivar ctx: Optional communication context used by the logger.
    :ivar logger: Configured logger or :class:`DummyLogger`.

    """

    def __init__(self, gconf: dict, name: NodeName, context: Context | None = None):
        self._closed = False
        self.ctx = None
        self._log_handler = None

        NodeBase.__init__(self, gconf, name)

        if "target" in self.conf and "log" in self.conf["target"]:
            self.ctx = Context(
                context=context, poll_timeout_ms=get_value(gconf, self.conf, "poll_timeout_ms")
            )
            self.logger, self._log_handler = init_logger(
                gconf, name, self.conf["target"]["log"], self.ctx
            )
        else:
            self.logger = DummyLogger(join_name(name))

        self.app = QtWidgets.QApplication(sys.argv)
        self.load_stylesheet()

        self.widget = self.init_widget(gconf, name, context)

    def __del__(self):
        self.close()

    def close(self):
        """Close communication resources owned by this GUINode."""

        if self._closed:
            return
        self._closed = True
        if self._log_handler is not None:
            self.logger.removeHandler(self._log_handler)
            self._log_handler = None
        if self.ctx is not None:
            self.ctx.close()

    def init_widget(self, gconf: dict, name: NodeName, context: Context | None):
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
        try:
            ret = self.app.exec()
        finally:
            self.close()
        sys.exit(ret)


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
