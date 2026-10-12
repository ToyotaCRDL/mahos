#!/usr/bin/env python3

"""Qt signal-based client for AWG waveform previews."""

from __future__ import annotations

from mahos.gui.Qt import QtCore
from mahos.gui.client import QSubWorker, QNodeClient
from mahos.msgs.inst.awg_msgs import AWGWaveform


class QAWGSubWorker(QSubWorker):
    awgUpdated = QtCore.pyqtSignal(AWGWaveform)

    def __init__(self, lconf: dict, context, parent: QtCore.QObject = None):
        QSubWorker.__init__(self, lconf, context, parent=parent)
        self.add_handler(lconf, b"wave", self.handle_awg)

    def handle_awg(self, msg):
        if isinstance(msg, AWGWaveform):
            self.awgUpdated.emit(msg)


class QAWGClient(QNodeClient):
    """QNodeClient for nodes publishing bounded AWG waveform previews."""

    awgUpdated = QtCore.pyqtSignal(AWGWaveform)

    def __init__(self, gconf: dict, name, context=None, parent=None):
        QNodeClient.__init__(self, gconf, name, context=context, parent=parent)
        self._awg = None
        self.sub = QAWGSubWorker(self.conf, self.ctx)
        self.sub.awgUpdated.connect(self.check_awg)
        self.add_sub(self.sub)

    def check_awg(self, awg: AWGWaveform):
        if self._awg is not None and self._awg.ident == awg.ident:
            return
        self._awg = awg
        self.awgUpdated.emit(awg)

    def get_awg(self) -> AWGWaveform | None:
        return self._awg
