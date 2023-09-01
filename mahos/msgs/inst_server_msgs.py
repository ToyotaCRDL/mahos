#!/usr/bin/env python3

"""
Message Types for Instrument RPC.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import copy
import uuid
from pprint import pformat

from .common_msgs import Message, Request, Status


class ServerStatus(Status):
    def __init__(self, host, name, locks, inst_num, overlay_num):
        self.host = host
        self.name = name
        self.locks = locks
        self.inst_num = inst_num
        self.overlay_num = overlay_num

    def __repr__(self):
        return (
            f"ServerStatus({self.host}, {self.name}, {self.locks},"
            + f" {self.inst_num}, {self.overlay_num})"
        )

    def __str__(self):
        return f"Server({self.host}::{self.name})->locks:\n" + pformat(self.locks)


class Ident(Message):
    def __init__(self, name: str):
        self.name = name
        self.uuid = uuid.uuid4()

    def __eq__(self, o):
        return self.name == o.name and self.uuid == o.uuid

    def __repr__(self):
        return f"Ident({self.name}, {self.uuid})"

    def __str__(self):
        return f"Id({self.name})"


class NoArgReq(Request):
    def __init__(self, ident: Ident, inst: str):
        self.ident = ident
        self.inst = inst


class LockReq(NoArgReq):
    """acquire the lock of instrument `inst`."""

    pass


class ReleaseReq(NoArgReq):
    """release the lock of instrument `inst`."""

    pass


class CheckLockReq(NoArgReq):
    """check if instrument `inst` is locked."""

    pass


class CallReq(Request):
    """call method `func` of instrument `inst` with `args`."""

    def __init__(self, ident: Ident, inst: str, func: str, args=None):
        self.ident = ident
        self.inst = inst
        self.func = func
        self.args = copy.copy(args)


class ShutdownReq(NoArgReq):
    """call shutdown() of instrument `inst`."""

    pass


class StartReq(NoArgReq):
    """call start() of instrument `inst`."""

    pass


class StopReq(NoArgReq):
    """call stop() of instrument `inst`."""

    pass


class PauseReq(NoArgReq):
    """call pause() of instrument `inst`."""

    pass


class ResumeReq(NoArgReq):
    """call resume() of instrument `inst`."""

    pass


class ResetReq(NoArgReq):
    """call reset() of instrument `inst`."""

    pass


class ConfigureReq(Request):
    """call configure() of instrument `inst`."""

    def __init__(self, ident: Ident, inst: str, params: dict, name: str, group: str):
        self.ident = ident
        self.inst = inst
        self.params = copy.copy(params)
        self.name = name
        self.group = group


class SetReq(Request):
    """call set() of instrument `inst`."""

    def __init__(self, ident: Ident, inst: str, key: str, value=None):
        self.ident = ident
        self.inst = inst
        self.key = key
        self.value = value


class GetReq(Request):
    """call get() of instrument `inst`."""

    def __init__(self, ident: Ident, inst: str, key: str, args=None):
        self.ident = ident
        self.inst = inst
        self.key = key
        self.args = args


class HelpReq(Request):
    """get help of instrument `inst`."""

    def __init__(self, inst: str, func: str | None = None):
        self.inst = inst
        self.func = func


class GetParamDictReq(Request):
    """get ParamDict `name` in `group` for instrument `inst`."""

    def __init__(self, ident: Ident, inst: str, name: str = "", group: str = ""):
        self.ident = Ident
        self.inst = inst
        self.name = name
        self.group = group


class GetParamDictNamesReq(Request):
    """Request to get list of param dict names for instrument `inst`."""

    def __init__(self, ident: Ident, inst: str, group: str = ""):
        self.ident = Ident
        self.inst = inst
        self.group = group
