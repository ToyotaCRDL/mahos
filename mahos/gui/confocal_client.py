#!/usr/bin/env python3

"""
Qt signal-based clients of Confocal.

.. This file is a part of MAHOS project.

"""

import typing as T

from .Qt import QtCore

from ..msgs.common_msgs import ShutdownReq, BinaryState, BinaryStatus
from ..msgs.param_msgs import GetParamDictReq
from ..msgs.confocal_msgs import Axis, ConfocalStatus, ConfocalState, ScanDirection, Trace
from ..msgs.confocal_msgs import (
    MoveReq,
    SaveImageReq,
    ExportImageReq,
    ExportViewReq,
    LoadImageReq,
    Image,
)
from ..msgs.confocal_msgs import BufferCommand, CommandBufferReq
from ..msgs.confocal_msgs import (
    SaveTraceReq,
    ExportTraceReq,
    LoadTraceReq,
    CommandTraceReq,
    TraceCommand,
)
from ..msgs.confocal_tracker_msgs import SaveParamsReq, LoadParamsReq, TrackNowReq
from .client import QStatusSubWorker, QStateReqClient, QReqClient, QStateClient


class QConfocalSubWorker(QStatusSubWorker):
    """Worker object for Subscriber part of QConfocalClient."""

    statusUpdated = QtCore.pyqtSignal(ConfocalStatus)
    imageUpdated = QtCore.pyqtSignal(Image)

    def __init__(self, lconf: dict, context, parent=None):
        QStatusSubWorker.__init__(self, lconf, context, parent=parent)
        self.add_handler(lconf, b"image", self.handle_image)

    def handle_image(self, msg):
        self.imageUpdated.emit(msg)


class QConfocalClient(QStateReqClient):
    """Qt-based client for Confocal.

    This class operates on piezo and scanner parts of Confocal.
    Tracer part is not here and done by QTracerClient.

    """

    statusUpdated = QtCore.pyqtSignal(ConfocalStatus)
    stateUpdated = QtCore.pyqtSignal(ConfocalState, ConfocalState)
    xposChanged = QtCore.pyqtSignal(float, bool)
    yposChanged = QtCore.pyqtSignal(float, bool)
    zposChanged = QtCore.pyqtSignal(float, bool)
    xtgtChanged = QtCore.pyqtSignal(float)
    ytgtChanged = QtCore.pyqtSignal(float)
    ztgtChanged = QtCore.pyqtSignal(float)
    xyImageUpdated = QtCore.pyqtSignal(Image)
    xzImageUpdated = QtCore.pyqtSignal(Image)
    yzImageUpdated = QtCore.pyqtSignal(Image)
    scanFinished = QtCore.pyqtSignal(Image)

    def __init__(self, gconf: dict, name, context=None, parent=None):
        QStateReqClient.__init__(self, gconf, name, context=context, parent=parent)

        self._pos = None
        self._state = None
        self._image = None

        self.sub = QConfocalSubWorker(self.conf, self.ctx)
        # do signal connections here
        self.sub.statusUpdated.connect(self.statusUpdated)
        self.sub.statusUpdated.connect(self.check_pos)
        self.sub.statusUpdated.connect(self.check_state)
        self.sub.imageUpdated.connect(self.check_image)

        self.add_sub(self.sub)

    def shutdown(self) -> bool:
        resp = self.req.request(ShutdownReq())
        return resp.success

    def move(self, ax: T.Union[Axis, T.List[Axis]], pos: T.Union[float, T.List[float]]) -> bool:
        resp = self.req.request(MoveReq(ax, pos))
        return resp.success

    def get_param_dict(self, name: str):
        resp = self.req.request(GetParamDictReq(name))
        if resp.success:
            return resp.ret
        else:
            return None

    def save_image(
        self, file_name, direction: T.Optional[ScanDirection] = None, note: str = ""
    ) -> bool:
        resp = self.req.request(SaveImageReq(file_name, direction=direction, note=note))
        return resp.success

    def export_image(
        self, file_name, direction: T.Optional[ScanDirection] = None, params=None
    ) -> bool:
        resp = self.req.request(ExportImageReq(file_name, direction, params))
        return resp.success

    def export_view(self, file_name, params=None) -> bool:
        resp = self.req.request(ExportViewReq(file_name, params))
        return resp.success

    def load_image(self, file_name) -> Image:
        resp = self.req.request(LoadImageReq(file_name))
        if resp.success:
            return resp.ret
        else:
            return None

    def _command_buffer(self, direction: ScanDirection, command: BufferCommand):
        resp = self.req.request(CommandBufferReq(direction, command))
        return resp.success

    def pop_buffer(self, direction: ScanDirection):
        return self._command_buffer(direction, BufferCommand.POP)

    def clear_buffer(self, direction: ScanDirection):
        return self._command_buffer(direction, BufferCommand.CLEAR)

    def get_all_buffer(self, direction: ScanDirection) -> T.List[Image]:
        resp = self.req.request(CommandBufferReq(direction, BufferCommand.GET_ALL))
        if resp.success:
            return resp.ret
        else:
            return []

    def check_state(self, status: ConfocalStatus):
        # if self._state is None or self._state != status.state:
        # if self._state is not None and self._state != status.state:
        if self._state is not None:
            self.stateUpdated.emit(status.state, self._state)
        self._state = status.state

    def check_pos(self, status: ConfocalStatus):
        if self._pos is None:  # first status update
            self._pos = status.pos
            return

        prev, new = self._pos, status.pos
        if prev.x != new.x:
            self.xposChanged.emit(new.x, new.x_ont)
        if prev.y != new.y:
            self.yposChanged.emit(new.y, new.y_ont)
        if prev.z != new.z:
            self.zposChanged.emit(new.z, new.z_ont)
        if prev.x_tgt != new.x_tgt:
            self.xtgtChanged.emit(new.x_tgt)
        if prev.y_tgt != new.y_tgt:
            self.ytgtChanged.emit(new.y_tgt)
        if prev.z_tgt != new.z_tgt:
            self.ztgtChanged.emit(new.z_tgt)

        self._pos = new

    def get_target_pos(self):
        if self._pos is None:
            return None
        else:
            return self._pos.x_tgt, self._pos.y_tgt, self._pos.z_tgt

    def get_state(self):
        return self._state

    def get_direction(self):
        if self._image is None:
            return None
        else:
            return self._image.direction

    def check_image(self, image: Image):
        if not image.has_params():
            return
        if image.direction == ScanDirection.XY:
            self.xyImageUpdated.emit(image)
        if image.direction == ScanDirection.XZ:
            self.xzImageUpdated.emit(image)
        if image.direction == ScanDirection.YZ:
            self.yzImageUpdated.emit(image)

        if (
            self._image is not None
            and image is not None
            and self._image.running
            and not image.running
        ):
            self.scanFinished.emit(image)

        self._image = image


class QTracerSubWorker(QStatusSubWorker):
    """Worker object for Subscriber part of QTracerClient."""

    statusUpdated = QtCore.pyqtSignal(ConfocalStatus)
    traceUpdated = QtCore.pyqtSignal(Trace)

    def __init__(self, lconf: dict, context, parent=None):
        QStatusSubWorker.__init__(self, lconf, context, parent=parent)
        self.add_handler(lconf, b"trace", self.handle_trace)

    def handle_trace(self, msg):
        self.traceUpdated.emit(msg)


class QTracerClient(QReqClient):
    """Qt-based client for Tracer part of Confocal."""

    paused = QtCore.pyqtSignal(bool)
    traceUpdated = QtCore.pyqtSignal(Trace)

    def __init__(self, gconf: dict, name, context=None, parent=None):
        QReqClient.__init__(self, gconf, name, context=context, parent=parent)

        self._trace = None

        self.sub = QTracerSubWorker(self.conf, self.ctx)
        # do signal connections here
        self.sub.traceUpdated.connect(self.traceUpdated)
        self.sub.statusUpdated.connect(self.update_paused)

        self.add_sub(self.sub)

    def update_paused(self, status: ConfocalStatus):
        self.paused.emit(status.tracer_paused)

    def save_trace(self, file_name, note: str = "") -> bool:
        resp = self.req.request(SaveTraceReq(file_name, note=note))
        return resp.success

    def export_trace(self, file_name, params=None) -> bool:
        resp = self.req.request(ExportTraceReq(file_name, params=params))
        return resp.success

    def load_trace(self, file_name) -> Trace:
        resp = self.req.request(LoadTraceReq(file_name))
        if resp.success:
            return resp.ret
        else:
            return None

    def _command(self, command: TraceCommand):
        resp = self.req.request(CommandTraceReq(command))
        return resp.success

    def pause(self):
        return self._command(TraceCommand.PAUSE)

    def resume(self):
        return self._command(TraceCommand.RESUME)

    def clear(self):
        return self._command(TraceCommand.CLEAR)


class QConfocalTrackerClient(QStateClient):
    """Qt-based client for ConfocalTracker."""

    statusUpdated = QtCore.pyqtSignal(BinaryStatus)
    stateUpdated = QtCore.pyqtSignal(BinaryState, BinaryState)

    def save_params(self, conf, file_name: T.Optional[str] = None) -> bool:
        resp = self.req.request(SaveParamsReq(conf, file_name=file_name))
        return resp.success

    def load_params(self, file_name: T.Optional[str] = None) -> T.Optional[dict]:
        resp = self.req.request(LoadParamsReq(file_name))
        if resp.success:
            return resp.ret
        else:
            return None

    def track_now(self) -> bool:
        resp = self.req.request(TrackNowReq())
        return resp.success

    def start(self, params=None) -> bool:
        """Start tracking, i.e., change state to ACTIVE."""

        return self.change_state(BinaryState.ACTIVE, params=params)

    def stop(self, params=None) -> bool:
        """Stop tracking, i.e., change state to IDLE."""

        return self.change_state(BinaryState.IDLE, params=params)
