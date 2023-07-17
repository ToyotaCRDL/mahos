#!/usr/bin/env python3

"""
Base implementation for Qt client for Nodes.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from .Qt import QtCore

from ..node.comm import Context
from ..node.client import init_node_client
from ..node.node import join_name, get_value
from ..msgs.common_msgs import Status, State, BinaryStatus, BinaryState, StateReq
from ..msgs.data_msgs import Data
from ..msgs.common_meas_msgs import BasicMeasData, Buffer
from ..meas.common_meas import ParamDictReqMixin, BasicMeasReqMixin
from ..util.typing import NodeName


class QSubWorker(QtCore.QObject):
    def __init__(self, lconf: dict, context: Context, parent: QtCore.QObject = None):
        QtCore.QObject.__init__(self, parent=parent)
        self.ctx = context
        self._closed = False

    def add_handler(self, lconf: dict, topic: bytes, handler, endpoint: str = "pub_endpoint"):
        self.ctx.add_sub(lconf[endpoint], topic, handler)

    def close(self):
        self._closed = True

    def main(self):
        while not self._closed:
            self.ctx.poll()


class QNodeClient(QtCore.QObject):
    """Qt-based client to use Node's function."""

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        parent: QtCore.QObject = None,
    ):
        QtCore.QObject.__init__(self, parent=parent)
        self._host, self._name, self.conf, self.ctx = init_node_client(
            gconf, name, context=context
        )

        self._subscribers = []
        self._closed = False

    def __del__(self):
        self.close()

    def close(self, close_ctx=True):
        if self._closed:
            return
        self._closed = True

        for sub, thread in self._subscribers:
            sub.close()
            thread.quit()
            thread.wait()
        if close_ctx:
            self.ctx.close()

    def add_sub(self, sub):
        thread = QtCore.QThread()
        sub.moveToThread(thread)
        thread.started.connect(sub.main)
        thread.finished.connect(sub.deleteLater)  # for safety?
        thread.start()  # QtCore.QTimer.singleShot(0, self.sub_thread.start)
        self._subscribers.append((sub, thread))

    def name(self) -> tuple[str, str]:
        return (self._host, self._name)

    def joined_name(self) -> str:
        return join_name((self._host, self._name))


class QReqClient(QNodeClient):
    """Qt-based node client with req."""

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        parent: QtCore.QObject = None,
        rep_endpoint="rep_endpoint",
    ):
        QNodeClient.__init__(self, gconf, name, context=context, parent=parent)
        self.req = self.ctx.add_req(
            self.conf[rep_endpoint],
            timeout_ms=get_value(gconf, self.conf, "req_timeout_ms"),
            logger=self.__class__.__name__,
        )


class QStateReqClient(QReqClient):
    """Qt-based node client with req and change_state()."""

    def change_state(self, state, params=None) -> bool:
        resp = self.req.request(StateReq(state, params=params))
        return resp.success


class QStatusSubWorker(QSubWorker):
    """Worker object for subscriber to Node Status."""

    statusUpdated = QtCore.pyqtSignal(Status)

    def __init__(self, lconf: dict, context, parent: QtCore.QObject = None):
        QSubWorker.__init__(self, lconf, context, parent=parent)
        self.add_handler(lconf, b"status", self.handle_status)

    def handle_status(self, msg):
        if isinstance(msg, Status):
            self.statusUpdated.emit(msg)


class QStatusDataSubWorker(QStatusSubWorker):
    """Worker object for subscriber to Node Status and Data."""

    dataUpdated = QtCore.pyqtSignal(Data)

    def __init__(self, lconf: dict, context, parent: QtCore.QObject = None):
        QStatusSubWorker.__init__(self, lconf, context, parent=parent)
        self.add_handler(lconf, b"data", self.handle_data)

    def handle_data(self, msg):
        if isinstance(msg, Data):
            self.dataUpdated.emit(msg)


class QStatusDataBufferSubWorker(QStatusDataSubWorker):
    """Worker object for subscriber to Node Status, Data, and Buffer."""

    bufferUpdated = QtCore.pyqtSignal(Buffer)

    def __init__(self, lconf: dict, context, parent: QtCore.QObject = None):
        QStatusDataSubWorker.__init__(self, lconf, context, parent=parent)
        self.add_handler(lconf, b"buffer", self.handle_buffer)

    def handle_buffer(self, msg):
        if isinstance(msg, Buffer):
            self.bufferUpdated.emit(msg)


class QStatusSubscriber(QNodeClient):
    """QNodeClient that subscribes to Node Status."""

    statusUpdated = QtCore.pyqtSignal(Status)

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        parent: QtCore.QObject = None,
    ):
        QNodeClient.__init__(self, gconf, name, context=context, parent=parent)

        self.sub = QStatusSubWorker(self.conf, self.ctx)
        # do signal connections here
        self.sub.statusUpdated.connect(self.statusUpdated)

        self.add_sub(self.sub)


class QStateClient(QStateReqClient):
    """QNodeClient for stateful node."""

    statusUpdated = QtCore.pyqtSignal(Status)
    stateUpdated = QtCore.pyqtSignal(State, State)

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        parent: QtCore.QObject = None,
        rep_endpoint="rep_endpoint",
    ):
        QStateReqClient.__init__(
            self, gconf, name, context=context, parent=parent, rep_endpoint=rep_endpoint
        )

        self._state = None

        self.sub = QStatusSubWorker(self.conf, self.ctx)
        # do signal connections here
        self.sub.statusUpdated.connect(self.statusUpdated)
        self.sub.statusUpdated.connect(self.check_state)

        self.add_sub(self.sub)

    def check_state(self, status: Status):
        if self._state is not None:
            self.stateUpdated.emit(status.state, self._state)
        self._state = status.state

    def get_state(self) -> State:
        return self._state


class QBasicMeasClient(QStateReqClient, BasicMeasReqMixin, ParamDictReqMixin):
    """QNodeClient for Basic Meas Nodes (Node with the BinaryState and a data)."""

    #: subscribed topic status is updated.
    statusUpdated = QtCore.pyqtSignal(BinaryStatus)

    #: state is updated: (current_state, last_state).
    stateUpdated = QtCore.pyqtSignal(BinaryState, BinaryState)

    #: data is updated.
    dataUpdated = QtCore.pyqtSignal(BasicMeasData)

    #: buffer is updated.
    bufferUpdated = QtCore.pyqtSignal(Buffer)

    #: measurement has been stopped (running became True to False).
    stopped = QtCore.pyqtSignal(Data)

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        parent: QtCore.QObject = None,
        rep_endpoint="rep_endpoint",
    ):
        QStateReqClient.__init__(
            self, gconf, name, context=context, parent=parent, rep_endpoint=rep_endpoint
        )

        self._state = self._data = self._buffer = None

        self.sub = QStatusDataBufferSubWorker(self.conf, self.ctx)

        # do signal connections here
        self.sub.statusUpdated.connect(self.statusUpdated)
        self.sub.statusUpdated.connect(self.check_state)
        self.sub.dataUpdated.connect(self.check_data)
        self.sub.bufferUpdated.connect(self.check_buffer)

        self.add_sub(self.sub)

    def get_state(self) -> State:
        return self._state

    def get_data(self) -> BasicMeasData:
        return self._data

    def get_buffer(self) -> Buffer:
        return self._buffer

    def check_state(self, status: Status):
        if self._state is not None:
            self.stateUpdated.emit(status.state, self._state)
        self._state = status.state

    def check_data(self, data: Data):
        if self._data == data:  # TODO: this guard has no meaning without Data.__eq__ ?
            return

        self.dataUpdated.emit(data)

        if self._data is not None and data is not None and self._data.running and not data.running:
            self.stopped.emit(data)

        self._data = data

    def check_buffer(self, buffer: Buffer):
        self.bufferUpdated.emit(buffer)
        self._buffer = buffer
