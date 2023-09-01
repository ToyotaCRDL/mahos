#!/usr/bin/env python3

from __future__ import annotations
from pprint import pformat

from .common_msgs import Request, Status
from . import param_msgs as P


class TweakerStatus(Status):
    def __init__(self, param_dict_names: list[str]):
        self.param_dict_names = param_dict_names

    def __repr__(self):
        return f"TweakerStatus({self.param_dict_names})"

    def __str__(self):
        return "Tweaker->param_dict_names:\n" + pformat(self.param_dict_names)


class ReadAllReq(Request):
    """Read current parameters for all the targets."""

    pass


class ReadReq(Request):
    """Read current parameters of a target."""

    def __init__(self, pd_name: str):
        self.pd_name = pd_name


class WriteReq(Request):
    """Write parameter to a target."""

    def __init__(self, pd_name: str, params: P.ParamDict[str, P.PDValue]):
        self.pd_name = pd_name
        self.params = params


class SaveReq(Request):
    """Save current parameters to a file"""

    def __init__(self, file_name: str):
        self.file_name = file_name


class LoadReq(Request):
    """Load parameters from a file"""

    def __init__(self, file_name: str):
        self.file_name = file_name
