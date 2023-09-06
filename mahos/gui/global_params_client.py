#!/usr/bin/env python3

"""
Qt signal-based client of GlobalParams.

.. This file is a part of MAHOS project.

"""

from .Qt import QtCore

from ..msgs.global_params_msgs import GlobalParamsStatus, SetParamReq
from .client import QStatusSubWorker, QReqClient


class QGlobalParamsClient(QReqClient):
    """Qt-based client of GlobalParams."""

    statusUpdated = QtCore.pyqtSignal(GlobalParamsStatus)

    def __init__(self, gconf: dict, name, context=None, parent=None):
        QReqClient.__init__(self, gconf, name, context=context, parent=parent)

        self._params = {}

        self.sub = QStatusSubWorker(self.conf, self.ctx)
        # do signal connections here
        self.sub.statusUpdated.connect(self.statusUpdated)
        self.sub.statusUpdated.connect(self.store_params)

        self.add_sub(self.sub)

    def store_params(self, msg: GlobalParamsStatus):
        if not isinstance(msg, GlobalParamsStatus):
            return
        self._params = msg.params

    def get_param(self, name: str):
        """Get a parameter. return None if parameter is unknown."""

        return self._params.get(name)

    def set_param(self, name: str, value):
        """Set a parameter. Value can be any pickle-able Python object."""

        resp = self.req.request(SetParamReq(name, value))
        return resp.success
