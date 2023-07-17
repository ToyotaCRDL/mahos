#!/usr/bin/env python3

"""
Logic and instrument control part of confocal microscope.

.. This file is a part of MAHOS project.

"""

import typing as T

from ..msgs.common_msgs import Resp, StateReq, ShutdownReq
from ..msgs import confocal_msgs
from ..msgs.confocal_msgs import Axis, ConfocalState, MoveReq
from ..msgs.confocal_msgs import SaveImageReq, ExportImageReq, ExportViewReq, LoadImageReq
from ..msgs.confocal_msgs import SaveTraceReq, ExportTraceReq, LoadTraceReq
from ..msgs.confocal_msgs import TraceCommand, CommandTraceReq, BufferCommand, CommandBufferReq
from ..msgs.confocal_msgs import ConfocalStatus, Image, Trace, ScanDirection
from ..msgs.param_msgs import GetParamDictReq
from ..node.node import Node
from ..node.client import NodeClient
from ..inst.server import MultiInstrumentClient
from .common_meas import BaseMeasClientMixin
from .common_worker import DummyWorker, PulseGen_CW, Switch
from .confocal_worker import Piezo, Tracer, Scanner
from .confocal_io import ConfocalIO


class ConfocalClient(NodeClient, BaseMeasClientMixin):
    """Simple Confocal Client.

    Simple client API for measurement services provided by Confocal.
    Only latest subscribed messages are hold.
    If you need message-driven things, use QConfocalClient instead.

    """

    M = confocal_msgs

    def __init__(
        self,
        gconf: dict,
        name,
        context=None,
        prefix=None,
        status_handler=None,
        image_handler=None,
        trace_handler=None,
    ):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        getters = self.add_sub(
            [(b"status", status_handler), (b"image", image_handler), (b"trace", trace_handler)]
        )

        self.get_status: T.Callable[[], ConfocalStatus] = getters[0]
        self.get_image: T.Callable[[], Image] = getters[1]
        self.get_trace: T.Callable[[], Trace] = getters[2]

        self.req = self.add_req(gconf)

    def shutdown(self) -> bool:
        resp = self.req.request(ShutdownReq())
        return resp.success

    def move(self, ax: T.Union[Axis, T.List[Axis]], pos: T.Union[float, T.List[float]]) -> bool:
        resp = self.req.request(MoveReq(ax, pos))
        return resp.success

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

    def load_image(self, file_name) -> T.Optional[Image]:
        resp = self.req.request(LoadImageReq(file_name))
        if resp.success:
            return resp.ret
        else:
            return None

    def save_trace(self, file_name) -> bool:
        resp = self.req.request(SaveTraceReq(file_name))
        return resp.success

    def export_trace(self, file_name, params=None) -> bool:
        resp = self.req.request(ExportTraceReq(file_name, params=params))
        return resp.success

    def load_trace(self, file_name) -> T.Optional[Trace]:
        resp = self.req.request(LoadTraceReq(file_name))
        if resp.success:
            return resp.ret
        else:
            return None

    def _command_trace(self, command: TraceCommand):
        resp = self.req.request(CommandTraceReq(command))
        return resp.success

    def pause_trace(self):
        return self._command_trace(TraceCommand.PAUSE)

    def resume_trace(self):
        return self._command_trace(TraceCommand.RESUME)

    def clear_trace(self):
        return self._command_trace(TraceCommand.CLEAR)

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


class ConfocalIORequester(NodeClient):
    """ConfocalClient with limited capability of IO requests.

    It is safer to use this when you only need IO requests.

    """

    M = confocal_msgs

    def __init__(self, gconf: dict, name, context=None, prefix=None):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)
        self.req = self.add_req(gconf)

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

    def load_image(self, file_name) -> T.Optional[Image]:
        resp = self.req.request(LoadImageReq(file_name))
        if resp.success:
            return resp.ret
        else:
            return None

    def save_trace(self, file_name) -> bool:
        resp = self.req.request(SaveTraceReq(file_name))
        return resp.success

    def export_trace(self, file_name, params=None) -> bool:
        resp = self.req.request(ExportTraceReq(file_name, params=params))
        return resp.success

    def load_trace(self, file_name) -> T.Optional[Trace]:
        resp = self.req.request(LoadTraceReq(file_name))
        if resp.success:
            return resp.ret
        else:
            return None


class ImageBuffer(object):
    def __init__(self):
        self.XY = []
        self.XZ = []
        self.YZ = []

    def _buf(self, direction: ScanDirection):
        if direction == ScanDirection.XY:
            return self.XY
        elif direction == ScanDirection.XZ:
            return self.XZ
        elif direction == ScanDirection.YZ:
            return self.YZ

    def append(self, img: Image):
        self._buf(img.direction).append(img)

    def pop(self, direction: ScanDirection) -> T.Optional[Image]:
        try:
            return self._buf(direction).pop()
        except IndexError:
            return None

    def clear(self, direction: ScanDirection):
        self._buf(direction).clear()

    def get_all(self, direction: ScanDirection) -> T.List[Image]:
        return self._buf(direction)

    def latest(self, direction: ScanDirection) -> T.Optional[Image]:
        try:
            return self._buf(direction)[-1]
        except IndexError:
            return None

    def latest_all(self) -> T.List[T.Optional[Image]]:
        return [self.latest(d) for d in (ScanDirection.XY, ScanDirection.XZ, ScanDirection.YZ)]


class Confocal(Node):
    CLIENT = ConfocalClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.state = ConfocalState.IDLE

        self.cli = MultiInstrumentClient(
            gconf, self.conf["target"]["servers"], context=self.ctx, prefix=self.joined_name()
        )
        self.add_client(self.cli)
        self.add_rep()
        self.status_pub = self.add_pub(b"status")
        self.image_pub = self.add_pub(b"image")
        self.trace_pub = self.add_pub(b"trace")

        if "switch" in self.conf["target"]["servers"]:
            self.switch = Switch(self.cli, self.logger, "confocal")
        else:
            self.switch = DummyWorker()
        if "pg" in self.conf["target"]["servers"]:
            self.pg = PulseGen_CW(self.cli, self.logger)
        else:
            self.pg = DummyWorker()

        self.scanner = Scanner(self.cli, self.logger)
        self.piezo = Piezo(self.cli, self.logger, self.conf.get("piezo", {}))
        self.tracer = Tracer(self.cli, self.logger, self.conf.get("tracer", {}))

        self.io = ConfocalIO(self.logger)
        self.image_buf = ImageBuffer()

    def close_resources(self):
        if hasattr(self, "switch") and self._switch_active(self.state):
            self.switch.stop()
        if hasattr(self, "pg") and self._pg_active(self.state):
            self.pg.stop()
        if hasattr(self, "scanner") and self._scanner_active(self.state):
            self.scanner.stop(aborting=True)
        if hasattr(self, "piezo") and self._piezo_active(self.state):
            self.piezo.stop()
        if hasattr(self, "tracer") and self._tracer_active(self.state):
            self.tracer.stop()

    def shutdown(self, msg: ShutdownReq) -> Resp:
        self.change_state(StateReq(ConfocalState.IDLE))
        self.cli.shutdown("piezo")
        self.set_shutdown()
        return Resp(True)

    def change_state(self, msg: StateReq) -> Resp:
        if self.state == msg.state:
            return Resp(True, "Already in that state")

        self.logger.info(f"Changing state from {self.state} to {msg.state}")
        # stop unnecessary modules
        success = True
        if self._to_stop(self._switch_active, msg.state):
            success &= self.switch.stop()
        if self._to_stop(self._pg_active, msg.state):
            success &= self.pg.stop()
        if self._to_stop(self._piezo_active, msg.state):
            success &= self.piezo.stop()
        if self._to_stop(self._tracer_active, msg.state):
            success &= self.tracer.stop()
        if self._to_stop(self._scanner_active, msg.state):
            success &= self.scanner.stop(aborting=True)
            if success:
                self.image_buf.append(self.scanner.image_msg())
        if not success:
            self.restore_state()
            return self.fail_with("Failed to stop internal worker.")

        # start modules
        success = True
        if self._to_start(self._switch_active, msg.state):
            success &= self.switch.start()
        if self._to_start(self._pg_active, msg.state):
            success &= self.pg.start()
        if self._to_start(self._piezo_active, msg.state):
            success &= self.piezo.start()
        if self._to_start(self._tracer_active, msg.state):
            success &= self.tracer.start()
        if self._to_start(self._scanner_active, msg.state):
            success &= self.scanner.start(msg.params)
        if not success:
            self.restore_state()
            return self.fail_with("Failed to start internal worker.")

        self.state = msg.state
        return Resp(True)

    def restore_state(self):
        def restore_idle():
            self.logger.error("Entered in undefined state. Trying to restore IDLE.")
            self.switch.stop()
            self.pg.stop()
            self.piezo.stop()
            self.tracer.stop()
            self.scanner.stop(aborting=True)
            self.state = ConfocalState.IDLE

        # Try to restore PIEZO state regardless of switch or pg's status.
        r = (self.piezo.running, self.tracer.running, self.scanner.running())
        if r == (True, False, False):
            if self.switch.stop() and self.pg.stop():
                self.state = ConfocalState.PIEZO
            else:
                restore_idle()
            self.logger.warn("restored state {}".format(self.state))
            return

        r = (
            self.switch.running,
            self.pg.running,
            self.piezo.running,
            self.tracer.running,
            self.scanner.running(),
        )
        if r == (False, False, False, False, False):
            self.state = ConfocalState.IDLE
        elif r == (True, True, True, True, False):
            self.state = ConfocalState.INTERACT
        elif r == (True, True, False, False, True):
            self.state = ConfocalState.SCAN
        else:  # e.g. only tracer is running (going to INTERACT but failed to start piezo.)
            restore_idle()
        self.logger.warn("restored state {}".format(self.state))

    def _to_stop(self, is_active, new_state):
        return is_active(self.state) and not is_active(new_state)

    def _to_start(self, is_active, new_state):
        return not is_active(self.state) and is_active(new_state)

    def _switch_active(self, state):
        return state in (ConfocalState.INTERACT, ConfocalState.SCAN)

    def _pg_active(self, state):
        return state in (ConfocalState.INTERACT, ConfocalState.SCAN)

    def _piezo_active(self, state):
        return state in (ConfocalState.INTERACT, ConfocalState.PIEZO)

    def _tracer_active(self, state):
        return state == ConfocalState.INTERACT

    def _scanner_active(self, state):
        return state == ConfocalState.SCAN

    def move(self, msg: MoveReq) -> Resp:
        if not self._piezo_active(self.state):
            return Resp(False, f"Piezo is inactive in current state {self.state}")

        self.piezo.move(msg.ax, msg.pos)

        return Resp(True)

    def _get_scan_param_dict(self) -> Resp:
        params = self.scanner.get_param_dict()
        if params is None:
            return self.fail_with("Cannot generate param dict.")
        else:
            return Resp(True, ret=params)

    def get_param_dict(self, msg: GetParamDictReq) -> Resp:
        n = msg.name.lower()
        if n == "scan":
            return self._get_scan_param_dict()
        else:
            return Resp(False, "Unknown param dict name " + n)

    def direction_to_target_pos(self, direction: ScanDirection):
        p = self.piezo.pos
        if direction == ScanDirection.XY:
            return p.x_tgt, p.y_tgt
        elif direction == ScanDirection.XZ:
            return p.x_tgt, p.z_tgt
        elif direction == ScanDirection.YZ:
            return p.y_tgt, p.z_tgt
        else:
            raise ValueError("invalid direction {} is given.".format(direction))

    def save_image(self, msg: SaveImageReq) -> Resp:
        if msg.direction is None:
            img = self.scanner.image_msg()
            if not img.has_params():
                return self.fail_with("No latest image")
        else:
            img = self.image_buf.latest(msg.direction)
            if img is None:
                return self.fail_with(f"{msg.direction.name} buffer is empty")

        success = self.io.save_image(msg.file_name, img, msg.note)
        return Resp(success)

    def export_image(self, msg: ExportImageReq) -> Resp:
        if msg.direction is None:
            img = self.scanner.image_msg()
            if not img.has_params():
                return self.fail_with("No latest image")
        else:
            img = self.image_buf.latest(msg.direction)
            if img is None:
                return self.fail_with(f"{msg.direction.name} buffer is empty")

        if isinstance(msg.params, dict) and msg.params.get("set_pos"):
            msg.params["pos"] = self.direction_to_target_pos(img.direction)
        success = self.io.export_image(msg.file_name, img, msg.params)
        return Resp(success)

    def export_view(self, msg: ExportViewReq) -> Resp:
        images = self.image_buf.latest_all()
        p = self.piezo.pos
        pos = p.x_tgt, p.y_tgt, p.z_tgt

        success = self.io.export_view(msg.file_name, images, pos, msg.params)
        return Resp(success)

    def load_image(self, msg: LoadImageReq) -> Resp:
        image = self.io.load_image(msg.file_name)
        if image is None:
            return Resp(False)
        else:
            self.image_buf.append(image)
            return Resp(True, ret=image)

    def save_trace(self, msg: SaveTraceReq) -> Resp:
        success = self.io.save_trace(msg.file_name, self.tracer.trace_msg(), msg.note)
        return Resp(success)

    def export_trace(self, msg: ExportTraceReq) -> Resp:
        success = self.io.export_trace(msg.file_name, self.tracer.trace_msg(), params=msg.params)
        return Resp(success)

    def load_trace(self, msg: LoadTraceReq) -> Resp:
        trace = self.io.load_trace(msg.file_name)
        if trace is None:
            return Resp(False)
        else:
            return Resp(True, ret=trace)

    def command_trace(self, msg: CommandTraceReq) -> Resp:
        if msg.command == TraceCommand.CLEAR:
            self.tracer.clear_buf()
            return Resp(True)
        elif msg.command == TraceCommand.PAUSE:
            self.tracer.pause_msg()
            return Resp(True)
        elif msg.command == TraceCommand.RESUME:
            self.tracer.resume_msg()
            return Resp(True)
        else:
            return Resp(False, "Unknown trace command: " + str(msg.command))

    def command_buffer(self, msg: CommandBufferReq) -> Resp:
        if msg.command == BufferCommand.POP:
            i = self.image_buf.pop(msg.direction)
            if i is not None:
                self.logger.debug(f"Pop {msg.direction.name} buffer: {i.ident}")
            return Resp(True)
        elif msg.command == BufferCommand.CLEAR:
            self.image_buf.clear(msg.direction)
            self.logger.debug(f"Clear {msg.direction.name} buffer")
            return Resp(True)
        elif msg.command == BufferCommand.GET_ALL:
            self.logger.debug(f"Sending all contents of {msg.direction.name} buffer")
            return Resp(True, ret=self.image_buf.get_all(msg.direction))
        else:
            return Resp(False, "Unknown buffer command: " + str(msg.command))

    def handle_req(self, msg):
        if isinstance(msg, StateReq):
            return self.change_state(msg)
        elif isinstance(msg, ShutdownReq):
            return self.shutdown(msg)
        elif isinstance(msg, MoveReq):
            return self.move(msg)
        elif isinstance(msg, GetParamDictReq):
            return self.get_param_dict(msg)
        elif isinstance(msg, SaveImageReq):
            return self.save_image(msg)
        elif isinstance(msg, ExportImageReq):
            return self.export_image(msg)
        elif isinstance(msg, ExportViewReq):
            return self.export_view(msg)
        elif isinstance(msg, LoadImageReq):
            return self.load_image(msg)
        elif isinstance(msg, SaveTraceReq):
            return self.save_trace(msg)
        elif isinstance(msg, ExportTraceReq):
            return self.export_trace(msg)
        elif isinstance(msg, LoadTraceReq):
            return self.load_trace(msg)
        elif isinstance(msg, CommandTraceReq):
            return self.command_trace(msg)
        elif isinstance(msg, CommandBufferReq):
            return self.command_buffer(msg)
        else:
            return Resp(False, "Invalid message type")

    def _work(self):
        if self._scanner_active(self.state):
            self.scanner.work()
        if self._piezo_active(self.state):
            self.piezo.work()
        if self._tracer_active(self.state):
            self.tracer.work()

    def _publish(self):
        status = ConfocalStatus(
            state=self.state, pos=self.piezo.get_pos(), tracer_paused=self.tracer.is_paused()
        )
        self.status_pub.publish(status)
        if self._scanner_active(self.state):
            self.image_pub.publish(self.scanner.image_msg())
        if self._tracer_active(self.state):
            self.trace_pub.publish(self.tracer.trace_msg())

    def wait(self):
        self.logger.info("Waiting for instrument server...")
        self.cli.wait("scanner")
        self.logger.info("Server is up!")

    def main(self):
        self.poll()
        self._work()
        self._publish()
