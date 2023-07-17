#!/usr/bin/env python3

"""
Instrument RPC.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import importlib
import typing as T
from inspect import signature, getdoc, getfile

from ..msgs.common_msgs import Request, Resp
from ..msgs import inst_server_msgs
from ..msgs.inst_server_msgs import Ident, ServerStatus, LockReq, ReleaseReq, CheckLockReq, CallReq
from ..msgs.inst_server_msgs import ShutdownReq, StartReq, StopReq, PauseReq, ResumeReq
from ..msgs.inst_server_msgs import ResetReq, ConfigureReq, SetReq, GetReq, HelpReq
from ..node.node import Node, split_name
from ..node.client import StatusClient
from .instrument import Instrument
from .overlay.overlay import InstrumentOverlay


class OverlayConf(object):
    def __init__(self, conf: dict):
        self.conf = conf

    def is_ref(self, value):
        return isinstance(value, str) and value.startswith("$")

    def resolve_ref(self, insts: dict) -> dict:
        def conv(value):
            if not self.is_ref(value):
                return value

            if value[1:] not in insts:
                raise ValueError("instrument is referenced but not found: " + value)
            return insts[value[1:]]

        return {k: conv(av) for k, av in self.conf.items()}

    def inst_names(self) -> tuple:
        return tuple((v[1:] for v in self.conf.values() if self.is_ref(v)))


class Locks(object):
    def __init__(self, insts):
        self.insts = insts
        self.locks = {}
        self.overlay_deps = {}

        for name in insts:
            self.locks[name] = None

    def add_overlay(self, name: str, inst_names: tuple):
        self.overlay_deps[name] = inst_names

    def is_locked(self, name: str, ident=None, reduce_=any) -> bool:
        def inst_is_locked(n):
            if ident is not None:
                return self.locks[n] is not None and self.locks[n] != ident
            else:
                return self.locks[n] is not None

        if name in self.insts:
            return inst_is_locked(name)
        else:  # assert name in self.overlay
            return reduce_([inst_is_locked(n) for n in self.overlay_deps[name]])

    def locked_by(self, name):
        if name in self.insts:
            return self.locks[name]
        else:  # assert name in self.overlay
            return tuple((self.locks[n] for n in self.overlay_deps[name]))

    def lock(self, name: str, ident: Ident):
        if name in self.insts:
            self.locks[name] = ident
        else:  # assert name in self.overlay
            for n in self.overlay_deps[name]:
                self.locks[n] = ident

    def release(self, name: str, ident: Ident):
        if name in self.insts:
            self.locks[name] = None
        else:  # assert name in self.overlay
            for n in self.overlay_deps[name]:
                self.locks[n] = None


class InstrumentClient(StatusClient):
    """Instrument RPC Client.

    Client API for RPC services provided by InstrumentServer.

    """

    M = inst_server_msgs

    def __init__(self, gconf: dict, name, context=None, prefix=None, status_handler=None):
        """Instrument RPC Client.

        :param handlers: list of (inst_name, topic_name, handler)

        """

        StatusClient.__init__(
            self, gconf, name, context=context, prefix=prefix, status_handler=status_handler
        )

        self.ident = Ident(self.full_name())

        inst = self.conf["instrument"] if "instrument" in self.conf else {}
        lay = self.conf["instrument_overlay"] if "instrument_overlay" in self.conf else {}
        self.inst_names = tuple(inst.keys()) + tuple(lay.keys())

    def lock(self, name: str) -> bool:
        """Acquire lock of an instrument."""

        resp = self.req.request(LockReq(self.ident, name))

        if not resp.success:
            self.logger.error(resp.message)
        return resp.success

    def is_locked(self, name: str) -> T.Optional[bool]:
        """Check if an instrument is locked."""

        resp = self.req.request(CheckLockReq(self.ident, name))

        if not resp.success:
            self.logger.error(resp.message)
            return None
        return resp.ret

    def release(self, name: str) -> bool:
        """Release lock of an instrument."""

        resp = self.req.request(ReleaseReq(self.ident, name))

        if not resp.success:
            self.logger.error(resp.message)
        return resp.success

    def call(self, name: str, func: str, **args) -> Resp:
        """Call arbitrary function of an instrument.

        Note that this is not quite a safe API (left for debug purpose).
        Consider using one of following standardized APIs:
            start, stop, pause, resume, reset, configure, set, get

        """

        resp = self.req.request(CallReq(self.ident, name, func, args))
        return resp

    def __call__(self, name: str, func: str, **args) -> Resp:
        return self.call(name, func, **args)

    def _noarg_call(self, name: str, Req_T):
        resp = self.req.request(Req_T(self.ident, name))
        return resp.success

    def shutdown(self, name: str) -> bool:
        """Shutdown the instrument and get ready to power-off. Returns True on success."""

        return self._noarg_call(name, ShutdownReq)

    def start(self, name: str) -> bool:
        """Start the instrument operation. Returns True on success."""

        return self._noarg_call(name, StartReq)

    def stop(self, name: str) -> bool:
        """Stop the instrument operation. Returns True on success."""

        return self._noarg_call(name, StopReq)

    def pause(self, name: str) -> bool:
        """Pause the instrument operation. Returns True on success."""

        return self._noarg_call(name, PauseReq)

    def resume(self, name: str) -> bool:
        """Resume the instrument operation. Returns True on success."""

        return self._noarg_call(name, ResumeReq)

    def reset(self, name: str) -> bool:
        """Reset the instrument settings. Returns True on success."""

        return self._noarg_call(name, ResetReq)

    def configure(self, name: str, params: dict) -> bool:
        """Configure the instrument settings. Returns True on success."""

        resp = self.req.request(ConfigureReq(self.ident, name, params))
        return resp.success

    def set(self, name: str, key: str, value=None) -> bool:
        """Set an instrument setting or commanding value. Returns True on success."""

        resp = self.req.request(SetReq(self.ident, name, key, value))
        return resp.success

    def get(self, name: str, key: str, args=None):
        """Get an instrument setting or commanding value."""

        resp = self.req.request(GetReq(self.ident, name, key, args=args))
        if resp.success:
            return resp.ret
        else:
            return None

    def help(self, name: str, func: T.Optional[str] = None) -> str:
        """Get help of instrument `name`.

        If function name `func` is given, get docstring of that function.
        Otherwise, get docstring of the class.

        """

        resp = self.req.request(HelpReq(name, func))
        return resp.message


class MultiInstrumentClient(object):
    """Proxy-interface to multiple InstrumentClients."""

    def __init__(self, gconf: dict, inst_to_name: dict, context=None, prefix=None):
        self.inst_to_name = {k: split_name(v) for k, v in inst_to_name.items()}
        self.name_to_client = {}
        for name in self.inst_to_name.values():
            if name not in self.name_to_client:
                self.name_to_client[name] = InstrumentClient(
                    gconf, name, context=context, prefix=prefix
                )

    def close(self, close_ctx=True):
        for cli in self.name_to_client.values():
            cli.close(close_ctx=close_ctx)

    def get_client(self, name: str) -> InstrumentClient:
        return self.name_to_client[self.inst_to_name[name]]

    def is_locked(self, name: str) -> bool:
        """Check if an instrument is locked."""

        return self.get_client(name).is_locked(name)

    def wait(self, name: str) -> bool:
        """wait until the server is ready."""

        return self.get_client(name).wait()

    def lock(self, name: str) -> bool:
        """Acquire lock of an instrument."""

        return self.get_client(name).lock(name)

    def release(self, name: str) -> bool:
        """Release lock of an instrument."""

        return self.get_client(name).release(name)

    def call(self, name: str, func: str, **args) -> Resp:
        """Call arbitrary function of an instrument.

        Note that this is not quite a safe API (left for debug purpose).
        Consider using one of following standardized APIs:
            shutdown, start, stop, pause, resume, reset, configure, set, get

        """

        return self.get_client(name).call(name, func, **args)

    def __call__(self, name: str, func: str, **args) -> Resp:
        return self.call(name, func, **args)

    def shutdown(self, name: str) -> bool:
        """Shutdown the instrument and get ready to power-off. Returns True on success."""

        return self.get_client(name).shutdown(name)

    def start(self, name: str) -> bool:
        """Start the instrument operation. Returns True on success."""

        return self.get_client(name).start(name)

    def stop(self, name: str) -> bool:
        """Stop the instrument operation. Returns True on success."""

        return self.get_client(name).stop(name)

    def pause(self, name: str) -> bool:
        """Pause the instrument operation. Returns True on success."""

        return self.get_client(name).pause(name)

    def resume(self, name: str) -> bool:
        """Resume the instrument operation. Returns True on success."""

        return self.get_client(name).resume(name)

    def reset(self, name: str) -> bool:
        """Reset the instrument settings. Returns True on success."""

        return self.get_client(name).reset(name)

    def configure(self, name: str, params: dict) -> bool:
        """Configure the instrument settings. Returns True on success."""

        return self.get_client(name).configure(name, params)

    def set(self, name: str, key: str, value=None) -> bool:
        """Set an instrument setting or commanding value. Returns True on success."""

        return self.get_client(name).set(name, key, value)

    def get(self, name: str, key: str, args=None):
        """Get an instrument setting or commanding value."""

        return self.get_client(name).get(name, key, args)


class InstrumentServer(Node):
    """Instrument RPC Server.

    Provides RPC services for instruments.
    Communication is done by REQ/REP pattern.
    Multiple clients can use the resource.
    Each client can lock some instruments for exclusive procedure execution.

    """

    _noarg_calls = (StartReq, StopReq, ShutdownReq, PauseReq, ResumeReq, ResetReq)
    _noarg_func_names = {
        "StartReq": "start",
        "StopReq": "stop",
        "ShutdownReq": "shutdown",
        "PauseReq": "pause",
        "ResumeReq": "resume",
        "ResetReq": "reset",
    }
    _std_funcs = (
        "start",
        "stop",
        "shutdown",
        "pause",
        "resume",
        "reset",
        "configure",
        "set",
        "get",
    )
    _bool_funcs = (
        "start",
        "stop",
        "shutdown",
        "pause",
        "resume",
        "reset",
        "configure",
        "set",
    )

    CLIENT = InstrumentClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self._insts: dict[str, Instrument | None] = {}
        self._overlays: dict[str, InstrumentOverlay] = {}
        self.locks = Locks(self.conf["instrument"])

        for name, inst in self.conf["instrument"].items():
            C = self._get_class(
                ["mahos.inst." + inst["module"], inst["module"]], inst["class"], Instrument
            )
            prefix = self.joined_name()

            try:
                if "conf" in inst:
                    self._insts[name]: Instrument = C(name, conf=inst["conf"], prefix=prefix)
                else:
                    self._insts[name]: Instrument = C(name, prefix=prefix)
            except Exception:
                self.logger.exception(f"Failed to initialize {name}.")
                self._insts[name] = None

        if "instrument_overlay" in self.conf:
            for name, lay in self.conf["instrument_overlay"].items():
                C = self._get_class(
                    ["mahos.inst.overlay." + lay["module"], lay["module"]],
                    lay["class"],
                    InstrumentOverlay,
                )
                prefix = self.joined_name()
                conf = OverlayConf(lay.get("conf", {}))
                self._overlays[name] = C(name, conf=conf.resolve_ref(self._insts), prefix=prefix)
                self.locks.add_overlay(name, conf.inst_names())

        self.add_rep()
        self.status_pub = self.add_pub(b"status")

    def _get_class(self, module_names, class_name, Class_T):
        for name in module_names:
            try:
                m = importlib.import_module(name)
                break
            except ModuleNotFoundError:
                pass
        else:
            msg = "Failed to import all modules: {}".format(module_names)
            self.logger.error(msg)
            raise ModuleNotFoundError(msg)

        C = getattr(m, class_name)
        if not issubclass(C, Class_T):
            msg = f"{C} is not subclass of {Class_T}"
            self.logger.error(msg)
            raise TypeError(msg)

        return C

    def _publish(self):
        msg = ServerStatus(
            host=self._host,
            name=self._name,
            locks=self.locks.locks,
            inst_num=len(self._insts),
            overlay_num=len(self._overlays),
        )
        self.status_pub.publish(msg)

    def main(self):
        self.poll()
        self._publish()

    def close_resources(self):
        for inst in self._insts.values():
            if inst is not None:
                inst.close_once()

    def _get(self, name) -> Instrument | InstrumentOverlay:
        if name in self._insts:
            return self._insts[name]
        else:
            return self._overlays[name]

    def handle_req(self, msg: Request) -> Resp:
        if not (msg.name in self._insts or msg.name in self._overlays):
            return Resp(False, "Unknown instrument {}".format(msg.name))

        if isinstance(msg, CallReq):
            return self._handle_call(msg)
        elif isinstance(msg, self._noarg_calls):
            return self._handle_noarg_calls(msg)
        elif isinstance(msg, ConfigureReq):
            return self._handle_configure(msg)
        elif isinstance(msg, SetReq):
            return self._handle_set(msg)
        elif isinstance(msg, GetReq):
            return self._handle_get(msg)
        elif isinstance(msg, HelpReq):
            return self._handle_help(msg)
        elif isinstance(msg, LockReq):
            return self._handle_lock(msg)
        elif isinstance(msg, ReleaseReq):
            return self._handle_release(msg)
        elif isinstance(msg, CheckLockReq):
            return self._handle_check_lock(msg)
        else:
            return Resp(False, "Unknown message type")

    def _call(self, name, ident, func, args):
        if self.locks.is_locked(name, ident):
            return Resp(
                False, "Instrument {} is locked by {}".format(name, self.locks.locked_by(name))
            )
        inst = self._get(name)
        if not hasattr(inst, func):
            return Resp(False, f"Unknown function name {func} for instrument {name}")
        if func in self._std_funcs and inst.is_closed():
            return Resp(False, f"Instrument {name} is already closed")

        f = getattr(inst, func)
        try:
            r = f() if args is None else f(**args)
        except Exception:
            msg = f"Error calling function {func}."
            self.logger.exception(msg)
            return Resp(False, msg)
        if func in self._bool_funcs:
            # these functions returns success status in bool
            return Resp(r)
        else:
            return Resp(True, ret=r)

    def _handle_call(self, msg: CallReq) -> Resp:
        return self._call(msg.name, msg.ident, msg.func, msg.args)

    def _handle_noarg_calls(self, msg) -> Resp:
        func = self._noarg_func_names[msg.__class__.__name__]
        return self._call(msg.name, msg.ident, func, None)

    def _handle_configure(self, msg: ConfigureReq) -> Resp:
        return self._call(msg.name, msg.ident, "configure", {"params": msg.params})

    def _handle_set(self, msg: SetReq) -> Resp:
        return self._call(msg.name, msg.ident, "set", {"key": msg.key, "value": msg.value})

    def _handle_get(self, msg: GetReq) -> Resp:
        if msg.args is None:
            args = {"key": msg.key}
        else:
            args = {"key": msg.key, "args": msg.args}
        return self._call(msg.name, msg.ident, "get", args)

    def _handle_help(self, msg: HelpReq) -> Resp:
        inst = self._get(msg.name)
        if msg.func is None:
            sig = inst.__class__.__name__ + str(signature(inst.__class__))
            doc = getdoc(inst) or "<No docstring>"
            file = f"(file: {getfile(inst.__class__)})"
            return Resp(True, "\n".join((sig, doc, file)).strip())

        # Fetch document of the function.
        if not hasattr(inst, msg.func):
            return Resp(False, f"No function '{msg.func}' defined for {inst.__class__.__name__}.")

        func = getattr(inst, msg.func)
        sig = f"{inst.__class__.__name__}.{msg.func}{str(signature(func))}"
        doc = getdoc(func) or "<No docstring>"
        file = f"(file: {getfile(func)})"
        return Resp(True, "\n".join((sig, doc, file)).strip())

    def _handle_lock(self, msg: LockReq) -> Resp:
        if self.locks.is_locked(msg.name, msg.ident, any):
            return Resp(False, "Already locked by: {}".format(self.locks.locked_by(msg.name)))

        self.locks.lock(msg.name, msg.ident)
        return Resp(True)

    def _handle_release(self, msg: ReleaseReq) -> Resp:
        if not self.locks.is_locked(msg.name):
            # Do not report double-release as error
            return Resp(True, "Already released")
        if self.locks.is_locked(msg.name, msg.ident, all):
            return Resp(
                False,
                "Lock ident is not matching: {} != {}".format(
                    self.locks.locked_by(msg.name), msg.ident
                ),
            )

        self.locks.release(msg.name, msg.ident)
        return Resp(True)

    def _handle_check_lock(self, msg: CheckLockReq) -> Resp:
        locked = self.locks.is_locked(msg.name)
        return Resp(True, ret=locked)
