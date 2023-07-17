#!/usr/bin/env python3

"""
Qt signal-based client of StateManager.

.. This file is a part of MAHOS project.

"""

from .Qt import QtCore

from ..msgs.state_manager_msgs import ManagerStatus
from .client import QNodeClient, QStatusSubWorker


class QManagerSubscriber(QNodeClient):
    """Subscribe to StateManager."""

    statesUpdated = QtCore.pyqtSignal(dict)

    def __init__(self, gconf: dict, name, context=None, parent=None):
        QNodeClient.__init__(self, gconf, name, context=context, parent=parent)

        self.sub = QStatusSubWorker(self.conf, self.ctx)
        # do signal connections here
        self.sub.statusUpdated.connect(self.check_states)
        self.add_sub(self.sub)

    def check_states(self, status: ManagerStatus):
        if isinstance(status.states, dict):
            self.statesUpdated.emit(status.states)
