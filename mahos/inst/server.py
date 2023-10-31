#!/usr/bin/env python3

"""
Instrument RPC.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import importlib
from collections import ChainMap
from inspect import signature, getdoc, getfile
from functools import wraps

from ..msgs.common_msgs import Request, Resp
from ..msgs import inst_server_msgs
from ..msgs import param_msgs as P
from ..msgs.inst_server_msgs import Ident, ServerStatus, LockReq, ReleaseReq, CheckLockReq, CallReq
from ..msgs.inst_server_msgs import ShutdownReq, StartReq, StopReq, PauseReq, ResumeReq
from ..msgs.inst_server_msgs import ResetReq, ConfigureReq, SetReq, GetReq, HelpReq
from ..msgs.inst_server_msgs import GetParamDictReq, GetParamDictLabelsReq
from ..node.node import Node, NodeName, split_name
from ..node.client import StatusClient
from ..util.graph import sort_dependency
from .instrument import Instrument
from .overlay.overlay import InstrumentOverlay


class OverlayConf(object):
    def __init__(self, conf: dict, insts: dict[str, Instrument]):
        self.conf = conf
        self._insts = insts
        self._lays = {}

        dep_dict = {lay: self.ref_names(lay) for lay in self.conf}
        self.sorted_lays = [lay for lay in sort_dependency(dep_dict) if lay not in insts.keys()]

    def get(self, lay: str) -> dict:
        return self.conf[lay]

    def is_ref(self, value):
        return isinstance(value, str) and value.startswith("$")

    def resolved_conf(self, lay: str) -> dict:
        def conv(value):
            if not self.is_ref(value):
                return value

            inst_or_lay = ChainMap(self._insts, self._lays)
            if value[1:] not in inst_or_lay:
                raise ValueError("instrument / overlay is referenced but not found: " + value)
            return inst_or_lay[value[1:]]

        return {k: conv(av) for k, av in self.conf[lay]["conf"].items()}

    def ref_names(self, lay: str) -> list[str]:
        """Make list of dependent instrument / overlay names."""

        return [v[1:] for v in self.conf[lay]["conf"].values() if self.is_ref(v)]

    def add_overlay(self, lay: str, overlay: InstrumentOverlay):
        self._lays[lay] = overlay

    def inst_names(self, lay: str) -> list[str]:
        """Make list of dependent instrument names."""

        inst_names = []
        for n in self.ref_names(lay):
            if n in self._insts.keys():
                inst_names.append(n)
            else:  # n is lay name
                inst_names.extend(self.inst_names(n))
        return list(set(inst_names))


class Locks(object):
    def __init__(self, insts: list[str]):
        self.insts = insts
        self.locks = {}
        self.overlay_deps = {}

        for inst in insts:
            self.locks[inst] = None

    def add_overlay(self, lay: str, inst_names: list[str]):
        """Add InstrumentOverlay named `lay` that's dependent on `inst_names`."""

        self.overlay_deps[lay] = inst_names

    def is_locked(self, inst: str, ident=None, reduce_=any) -> bool:
        def inst_is_locked(n):
            if ident is not None:
                return self.locks[n] is not None and self.locks[n] != ident
            else:
                return self.locks[n] is not None

        if inst in self.insts:
            return inst_is_locked(inst)
        else:  # assert inst in self.overlay
            return reduce_([inst_is_locked(n) for n in self.overlay_deps[inst]])

    def locked_by(self, inst: str):
        if inst in self.insts:
            return self.locks[inst]
        else:  # assert inst in self.overlay
            return tuple((self.locks[n] for n in self.overlay_deps[inst]))

    def lock(self, inst: str, ident: Ident):
        if inst in self.insts:
            self.locks[inst] = ident
        else:  # assert inst in self.overlay
            for n in self.overlay_deps[inst]:
                self.locks[n] = ident

    def release(self, inst: str, ident: Ident):
        if inst in self.insts:
            self.locks[inst] = None
        else:  # assert inst in self.overlay
            for n in self.overlay_deps[inst]:
                self.locks[n] = None


class InstrumentClient(StatusClient):
    """Instrument RPC Client.

    Client API for RPC services provided by InstrumentServer.

    """

    M = inst_server_msgs

    def __init__(self, gconf: dict, name, context=None, prefix=None, status_handler=None):
        """Instrument RPC Client."""

        StatusClient.__init__(
            self, gconf, name, context=context, prefix=prefix, status_handler=status_handler
        )

        self.ident = Ident(self.full_name())

        self._mod_classes = {}
        for inst, idict in self.conf.get("instrument", {}).items():
            self._mod_classes[inst] = (idict["module"], idict["class"])
        for lay, ldict in self.conf.get("instrument_overlay", {}).items():
            self._mod_classes[lay] = (ldict["module"], ldict["class"])

    def module_class_names(self, inst: str) -> tuple[str, str] | None:
        """Get tuple of (module name, class name) of instrument `inst`."""

        if inst not in self._mod_classes:
            return None
        return self._mod_classes[inst]

    def module_name(self, inst: str) -> str | None:
        """Get module name of instrument `inst`."""

        module_class = self.module_class_names(inst)
        return module_class[0] if module_class is not None else None

    def class_name(self, inst: str) -> str | None:
        """Get class name of instrument `inst`."""

        module_class = self.module_class_names(inst)
        return module_class[1] if module_class is not None else None

    def lock(self, inst: str) -> bool:
        """Acquire lock of an instrument."""

        resp = self.req.request(LockReq(self.ident, inst))

        if not resp.success:
            self.logger.error(resp.message)
        return resp.success

    def is_locked(self, inst: str) -> bool | None:
        """Check if an instrument is locked."""

        resp = self.req.request(CheckLockReq(self.ident, inst))

        if not resp.success:
            self.logger.error(resp.message)
            return None
        return resp.ret

    def release(self, inst: str) -> bool:
        """Release lock of an instrument."""

        resp = self.req.request(ReleaseReq(self.ident, inst))

        if not resp.success:
            self.logger.error(resp.message)
        return resp.success

    def call(self, inst: str, func: str, **args) -> Resp:
        """Call arbitrary function of an instrument.

        Note that this is not quite a safe API (left for debug purpose).
        Consider using one of following standardized APIs:
            start, stop, pause, resume, reset, configure, set, get

        """

        resp = self.req.request(CallReq(self.ident, inst, func, args))
        return resp

    def __call__(self, inst: str, func: str, **args) -> Resp:
        return self.call(inst, func, **args)

    def _noarg_call(self, inst: str, Req_T):
        resp = self.req.request(Req_T(self.ident, inst))
        return resp.success

    def shutdown(self, inst: str) -> bool:
        """Shutdown the instrument and get ready to power-off. Returns True on success."""

        return self._noarg_call(inst, ShutdownReq)

    def start(self, inst: str) -> bool:
        """Start the instrument operation. Returns True on success."""

        return self._noarg_call(inst, StartReq)

    def stop(self, inst: str) -> bool:
        """Stop the instrument operation. Returns True on success."""

        return self._noarg_call(inst, StopReq)

    def pause(self, inst: str) -> bool:
        """Pause the instrument operation. Returns True on success."""

        return self._noarg_call(inst, PauseReq)

    def resume(self, inst: str) -> bool:
        """Resume the instrument operation. Returns True on success."""

        return self._noarg_call(inst, ResumeReq)

    def reset(self, inst: str) -> bool:
        """Reset the instrument settings. Returns True on success."""

        return self._noarg_call(inst, ResetReq)

    def configure(self, inst: str, params: dict, label: str = "", group: str = "") -> bool:
        """Configure the instrument settings. Returns True on success."""

        resp = self.req.request(ConfigureReq(self.ident, inst, params, label, group))
        return resp.success

    def set(self, inst: str, key: str, value=None) -> bool:
        """Set an instrument setting or commanding value. Returns True on success."""

        resp = self.req.request(SetReq(self.ident, inst, key, value))
        return resp.success

    def get(self, inst: str, key: str, args=None):
        """Get an instrument setting or commanding value."""

        resp = self.req.request(GetReq(self.ident, inst, key, args=args))
        if resp.success:
            return resp.ret
        else:
            return None

    def help(self, inst: str, func: str | None = None) -> str:
        """Get help of instrument `inst`.

        If function inst `func` is given, get docstring of that function.
        Otherwise, get docstring of the class.

        """

        resp = self.req.request(HelpReq(inst, func))
        return resp.message

    def get_param_dict(
        self, inst: str, label: str = "", group: str = ""
    ) -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label` in `group`.

        :param inst: instrument name.
        :param label: param dict label.
                      can be empty if target inst provides only one ParamDict.
        :param group: param dict group.
                      can be empty if target inst provides only one group.

        """

        resp = self.req.request(GetParamDictReq(self.ident, inst, label, group))
        if resp.success:
            return resp.ret
        else:
            return None

    def get_param_dict_labels(self, inst: str, group: str = "") -> list[str]:
        """Get list of available ParamDict labels pertaining to `group`.

        :param inst: instrument name.
        :param group: ParamDict group name.
                      can be empty if target inst provides only one group.


        """

        resp = self.req.request(GetParamDictLabelsReq(self.ident, inst, group))
        if resp.success:
            return resp.ret
        else:
            return []


def remap_inst(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        remap = args[0].inst_remap
        inst = args[1]
        if inst in remap:
            args = (args[0], remap[inst]) + args[2:]
        return f(*args, **kwargs)

    return wrapper


class MultiInstrumentClient(object):
    """Proxy-interface to multiple InstrumentClients."""

    def __init__(
        self,
        gconf: dict,
        inst_to_nodename: dict[str, NodeName],
        inst_remap: dict[str, str] | None = None,
        context=None,
        prefix=None,
    ):
        self.inst_to_nodename = {k: split_name(v) for k, v in inst_to_nodename.items()}
        self.inst_remap = inst_remap or {}
        self.nodename_to_client = {}
        for name in self.inst_to_nodename.values():
            if name not in self.nodename_to_client:
                self.nodename_to_client[name] = InstrumentClient(
                    gconf, name, context=context, prefix=prefix
                )

    def close(self, close_ctx=True):
        for cli in self.nodename_to_client.values():
            cli.close(close_ctx=close_ctx)

    @remap_inst
    def __contains__(self, inst: str) -> bool:
        return inst in self.inst_to_nodename

    @remap_inst
    def module_class_names(self, inst: str) -> tuple[str, str] | None:
        """Get tuple of (module name, class name) of instrument `inst`."""

        return self.get_client(inst).module_class_names(inst)

    @remap_inst
    def module_name(self, inst: str) -> str | None:
        """Get module name of instrument `inst`."""

        return self.get_client(inst).module_name(inst)

    @remap_inst
    def class_name(self, inst: str) -> str | None:
        """Get class name of instrument `inst`."""

        return self.get_client(inst).class_name(inst)

    @remap_inst
    def get_client(self, inst: str) -> InstrumentClient:
        return self.nodename_to_client[self.inst_to_nodename[inst]]

    @remap_inst
    def is_locked(self, inst: str) -> bool:
        """Check if an instrument is locked."""

        return self.get_client(inst).is_locked(inst)

    @remap_inst
    def wait(self, inst: str) -> bool:
        """wait until the server is ready."""

        return self.get_client(inst).wait()

    @remap_inst
    def lock(self, inst: str) -> bool:
        """Acquire lock of an instrument."""

        return self.get_client(inst).lock(inst)

    @remap_inst
    def release(self, inst: str) -> bool:
        """Release lock of an instrument."""

        return self.get_client(inst).release(inst)

    @remap_inst
    def call(self, inst: str, func: str, **args) -> Resp:
        """Call arbitrary function of an instrument.

        Note that this is not quite a safe API (left for debug purpose).
        Consider using one of following standardized APIs:
            shutdown, start, stop, pause, resume, reset, configure, set, get

        """

        return self.get_client(inst).call(inst, func, **args)

    def __call__(self, inst: str, func: str, **args) -> Resp:
        return self.call(inst, func, **args)

    @remap_inst
    def shutdown(self, inst: str) -> bool:
        """Shutdown the instrument and get ready to power-off. Returns True on success."""

        return self.get_client(inst).shutdown(inst)

    @remap_inst
    def start(self, inst: str) -> bool:
        """Start the instrument operation. Returns True on success."""

        return self.get_client(inst).start(inst)

    @remap_inst
    def stop(self, inst: str) -> bool:
        """Stop the instrument operation. Returns True on success."""

        return self.get_client(inst).stop(inst)

    @remap_inst
    def pause(self, inst: str) -> bool:
        """Pause the instrument operation. Returns True on success."""

        return self.get_client(inst).pause(inst)

    @remap_inst
    def resume(self, inst: str) -> bool:
        """Resume the instrument operation. Returns True on success."""

        return self.get_client(inst).resume(inst)

    @remap_inst
    def reset(self, inst: str) -> bool:
        """Reset the instrument settings. Returns True on success."""

        return self.get_client(inst).reset(inst)

    @remap_inst
    def configure(self, inst: str, params: dict, label: str = "", group: str = "") -> bool:
        """Configure the instrument settings. Returns True on success."""

        return self.get_client(inst).configure(inst, params, label, group)

    @remap_inst
    def set(self, inst: str, key: str, value=None) -> bool:
        """Set an instrument setting or commanding value. Returns True on success."""

        return self.get_client(inst).set(inst, key, value)

    @remap_inst
    def get(self, inst: str, key: str, args=None):
        """Get an instrument setting or commanding value."""

        return self.get_client(inst).get(inst, key, args)

    @remap_inst
    def help(self, inst: str, func: str | None = None) -> str:
        """Get help of instrument `inst`.

        If function name `func` is given, get docstring of that function.
        Otherwise, get docstring of the class.

        """

        return self.get_client(inst).help(inst, func)

    @remap_inst
    def get_param_dict(
        self, inst: str, label: str = "", group: str = ""
    ) -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label` in `group` of instrument `inst`.

        :param inst: instrument name.
        :param label: param dict label.
                      can be empty if target inst provides only one ParamDict.
        :param group: param dict group.
                      can be empty if target inst provides only one group.

        """

        return self.get_client(inst).get_param_dict(inst, label, group)

    @remap_inst
    def get_param_dict_labels(self, inst: str, group: str = "") -> list[str]:
        """Get list of available ParamDict labels pertaining to `group`.

        :param group: ParamDict group name.
                      can be empty if target inst provides only one group.

        """

        return self.get_client(inst).get_param_dict_labels(inst, group)


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
        "get_param_dict",
        "get_param_dict_labels",
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
        self.locks = Locks(list(self.conf["instrument"].keys()))

        for inst, idict in self.conf["instrument"].items():
            C = self._get_class(
                ["mahos.inst." + idict["module"], idict["module"]], idict["class"], Instrument
            )
            prefix = self.joined_name()

            try:
                if "conf" in idict:
                    self._insts[inst]: Instrument = C(inst, conf=idict["conf"], prefix=prefix)
                else:
                    self._insts[inst]: Instrument = C(inst, prefix=prefix)
            except Exception:
                self.logger.exception(f"Failed to initialize {inst}.")
                self._insts[inst] = None

        if "instrument_overlay" in self.conf:
            conf = OverlayConf(self.conf["instrument_overlay"], self._insts)
            self.logger.debug(f"sorted overlay names: {conf.sorted_lays}")

            for lay in conf.sorted_lays:
                ldict = conf.get(lay)
                C = self._get_class(
                    ["mahos.inst.overlay." + ldict["module"], ldict["module"]],
                    ldict["class"],
                    InstrumentOverlay,
                )
                prefix = self.joined_name()
                self._overlays[lay] = C(lay, conf=conf.resolved_conf(lay), prefix=prefix)
                conf.add_overlay(lay, self._overlays[lay])
                self.locks.add_overlay(lay, conf.inst_names(lay))

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

    def _get(self, inst: str) -> Instrument | InstrumentOverlay:
        if inst in self._insts:
            return self._insts[inst]
        else:
            return self._overlays[inst]

    def handle_req(self, msg: Request) -> Resp:
        if not (msg.inst in self._insts or msg.inst in self._overlays):
            return Resp(False, "Unknown instrument {}".format(msg.inst))

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
        elif isinstance(msg, GetParamDictReq):
            return self._handle_get_param_dict(msg)
        elif isinstance(msg, GetParamDictLabelsReq):
            return self._handle_get_param_dict_labels(msg)
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

    def _call(self, inst, ident, func, args):
        if self.locks.is_locked(inst, ident):
            return Resp(
                False, "Instrument {} is locked by {}".format(inst, self.locks.locked_by(inst))
            )
        inst = self._get(inst)
        if not hasattr(inst, func):
            return Resp(False, f"Unknown function name {func} for instrument {inst}")
        if func in self._std_funcs and inst.is_closed():
            return Resp(False, f"Instrument {inst} is already closed")

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
        return self._call(msg.inst, msg.ident, msg.func, msg.args)

    def _handle_noarg_calls(self, msg) -> Resp:
        func = self._noarg_func_names[msg.__class__.__name__]
        return self._call(msg.inst, msg.ident, func, None)

    def _handle_configure(self, msg: ConfigureReq) -> Resp:
        args = {"params": msg.params, "label": msg.label, "group": msg.group}
        return self._call(msg.inst, msg.ident, "configure", args)

    def _handle_set(self, msg: SetReq) -> Resp:
        return self._call(msg.inst, msg.ident, "set", {"key": msg.key, "value": msg.value})

    def _handle_get(self, msg: GetReq) -> Resp:
        if msg.args is None:
            args = {"key": msg.key}
        else:
            args = {"key": msg.key, "args": msg.args}
        return self._call(msg.inst, msg.ident, "get", args)

    def _handle_get_param_dict(self, msg: GetParamDictReq) -> Resp:
        args = {"label": msg.label, "group": msg.group}
        return self._call(msg.inst, msg.ident, "get_param_dict", args)

    def _handle_get_param_dict_labels(self, msg: GetParamDictLabelsReq) -> Resp:
        args = {"group": msg.group}
        return self._call(msg.inst, msg.ident, "get_param_dict_labels", args)

    def _handle_help(self, msg: HelpReq) -> Resp:
        inst = self._get(msg.inst)
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
        if self.locks.is_locked(msg.inst, msg.ident, any):
            return Resp(False, "Already locked by: {}".format(self.locks.locked_by(msg.inst)))

        self.locks.lock(msg.inst, msg.ident)
        return Resp(True)

    def _handle_release(self, msg: ReleaseReq) -> Resp:
        if not self.locks.is_locked(msg.inst):
            # Do not report double-release as error
            return Resp(True, "Already released")
        if self.locks.is_locked(msg.inst, msg.ident, all):
            return Resp(
                False,
                "Lock ident is not matching: {} != {}".format(
                    self.locks.locked_by(msg.inst), msg.ident
                ),
            )

        self.locks.release(msg.inst, msg.ident)
        return Resp(True)

    def _handle_check_lock(self, msg: CheckLockReq) -> Resp:
        locked = self.locks.is_locked(msg.inst)
        return Resp(True, ret=locked)
