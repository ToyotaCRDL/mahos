#!/usr/bin/env python3

"""
Qt signal-based client of LogBroker.

.. This file is a part of MAHOS project.

"""

from .Qt import QtCore

from ..node.log_broker import parse_log, LogEntry
from .client import QSubWorker, QNodeClient


class QLogSubWorker(QSubWorker):
    """Worker object for QLogSubscriber."""

    logArrived = QtCore.pyqtSignal(tuple)

    def __init__(self, lconf: dict, context, parent=None):
        QSubWorker.__init__(self, lconf, context, parent=parent)
        self.ctx.add_sub(lconf["xpub_endpoint"], handler=self.handle_log, deserial=False)

    def handle_log(self, msg):
        log = parse_log(msg)
        if log is not None:
            self.logArrived.emit(log)


class QLogSubscriber(QNodeClient):
    """Subscribe to LogBroker."""

    logArrived = QtCore.pyqtSignal(LogEntry)

    def __init__(self, gconf: dict, name, context=None, parent=None):
        QNodeClient.__init__(self, gconf, name, context=context, parent=parent)

        self.sub = QLogSubWorker(self.conf, self.ctx)
        # do signal connections here
        self.sub.logArrived.connect(self.logArrived)
        self.add_sub(self.sub)
