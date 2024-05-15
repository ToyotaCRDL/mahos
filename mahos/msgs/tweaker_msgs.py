#!/usr/bin/env python3

"""
Message Types for the Tweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from pprint import pformat

from .common_msgs import Request, Status
from . import param_msgs as P


class TweakerStatus(Status):
    def __init__(self, param_dict_ids: list[str]):
        self.param_dict_ids = param_dict_ids

    def __repr__(self):
        return f"TweakerStatus({self.param_dict_ids})"

    def __str__(self):
        return "Tweaker->param_dict_ids:\n" + pformat(self.param_dict_ids)


class ReadAllReq(Request):
    """Read current parameters for all the registered ParamDicts."""

    pass


class ReadReq(Request):
    """Read current parameters of a ParamDict."""

    def __init__(self, param_dict_id: str):
        self.param_dict_id = param_dict_id


class WriteReq(Request):
    """Write parameter to a ParamDict."""

    def __init__(self, param_dict_id: str, params: P.ParamDict[str, P.PDValue]):
        self.param_dict_id = param_dict_id
        self.params = params


class WriteAllReq(Request):
    """Write all the parameters to the ParamDicts."""

    def __init__(self, param_dicts: dict[str, P.ParamDict[str, P.PDValue]]):
        self.param_dicts = param_dicts


class StartReq(Request):
    """Start instrument operation pertaining to a ParamDict.

    Typical application is turning on the output.

    """

    def __init__(self, param_dict_id: str):
        self.param_dict_id = param_dict_id


class StopReq(Request):
    """Stop instrument operation pertaining to a ParamDict.

    Typical application is turning off the output.

    """

    def __init__(self, param_dict_id: str):
        self.param_dict_id = param_dict_id


class SaveReq(Request):
    """Save current parameters to a file"""

    def __init__(self, filename: str, group: str = ""):
        self.filename = filename
        self.group = group


class LoadReq(Request):
    """Load parameters from a file"""

    def __init__(self, filename: str, group: str = ""):
        self.filename = filename
        self.group = group
