#!/usr/bin/env python3

"""
Message Types for Instrument RPC.

.. This file is a part of MAHOS project.

"""

import copy
import uuid
from pprint import pformat
from typing import Optional

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
    def __init__(self, name):
        self.name = name
        self.uuid = uuid.uuid4()

    def __eq__(self, o):
        return self.name == o.name and self.uuid == o.uuid

    def __repr__(self):
        return f"Ident({self.name}, {self.uuid})"

    def __str__(self):
        return f"Id({self.name})"


class NoArgReq(Request):
    def __init__(self, ident: Ident, name: str):
        self.ident = ident
        self.name = name


class LockReq(NoArgReq):
    """acquire the lock of instrument `name`."""

    pass


class ReleaseReq(NoArgReq):
    """release the lock of instrument `name`."""

    pass


class CheckLockReq(NoArgReq):
    """check if instrument `name` is locked."""

    pass


class CallReq(Request):
    """call method `func` of instrument `name` with `args`."""

    def __init__(self, ident: Ident, name: str, func: str, args=None):
        self.ident = ident
        self.name = name
        self.func = func
        self.args = copy.copy(args)


class ShutdownReq(NoArgReq):
    """call shutdown() of instrument `name`."""

    pass


class StartReq(NoArgReq):
    """call start() of instrument `name`."""

    pass


class StopReq(NoArgReq):
    """call stop() of instrument `name`."""

    pass


class PauseReq(NoArgReq):
    """call pause() of instrument `name`."""

    pass


class ResumeReq(NoArgReq):
    """call resume() of instrument `name`."""

    pass


class ResetReq(NoArgReq):
    """call reset() of instrument `name`."""

    pass


class ConfigureReq(Request):
    """call configure() of instrument `name`."""

    def __init__(self, ident: Ident, name: str, params: dict):
        self.ident = ident
        self.name = name
        self.params = copy.copy(params)


class SetReq(Request):
    """call set() of instrument `name`."""

    def __init__(self, ident: Ident, name: str, key: str, value=None):
        self.ident = ident
        self.name = name
        self.key = key
        self.value = value


class GetReq(Request):
    """call get() of instrument `name`."""

    def __init__(self, ident: Ident, name: str, key: str, args=None):
        self.ident = ident
        self.name = name
        self.key = key
        self.args = args


class HelpReq(Request):
    """get help of instrument `name`."""

    def __init__(self, name: str, func: Optional[str] = None):
        self.name = name
        self.func = func
