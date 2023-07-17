#!/usr/bin/env python3

"""
Qt signal-based clients of Pulse-based meas nodes.

.. This file is a part of MAHOS project.

"""

import typing as T

from .Qt import QtCore

from ..msgs.pulse_msgs import PulsePattern
from .client import QSubWorker, QNodeClient


class QPulseSubWorker(QSubWorker):
    pulseUpdated = QtCore.pyqtSignal(PulsePattern)

    def __init__(self, lconf: dict, context, parent: QtCore.QObject = None):
        QSubWorker.__init__(self, lconf, context, parent=parent)
        self.add_handler(lconf, b"pulse", self.handle_pulse)

    def handle_pulse(self, msg):
        if isinstance(msg, PulsePattern):
            self.pulseUpdated.emit(msg)


class QPulseClient(QNodeClient):
    """QNodeClient for Pulse-based meas nodes."""

    #: subscribed topic pulse is updated.
    pulseUpdated = QtCore.pyqtSignal(PulsePattern)

    def __init__(self, gconf: dict, name, context=None, parent=None):
        QNodeClient.__init__(self, gconf, name, context=context, parent=parent)

        self._pulse = None

        self.sub = QPulseSubWorker(self.conf, self.ctx)
        # do signal connections here
        self.sub.pulseUpdated.connect(self.check_pulse)

        self.add_sub(self.sub)

    def check_pulse(self, pulse: PulsePattern):
        if self._pulse is not None and self._pulse.ident == pulse.ident:
            return  # same pulse received

        self._pulse = pulse
        self.pulseUpdated.emit(pulse)

    def get_pulse(self) -> T.Optional[PulsePattern]:
        return self._pulse
