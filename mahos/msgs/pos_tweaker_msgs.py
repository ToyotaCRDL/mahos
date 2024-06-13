#!/usr/bin/env python3

"""
Message Types for the PosTweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from pprint import pformat

from .common_msgs import Request, Status


class PosTweakerStatus(Status):
    def __init__(self, axis_states: dict[str, dict[str, [float, bool]]]):
        self.axis_states = axis_states

    def __repr__(self):
        return f"PosTweakerStatus({self.axis_states})"

    def __str__(self):
        return "PosTweaker->axis_states:\n" + pformat(self.axis_states)


class SetTargetReq(Request):
    """Set target of axis."""

    def __init__(self, axis_pos: dict[str, float]):
        self.axis_pos = axis_pos


class HomeReq(Request):
    """Perform homing of single axis."""

    def __init__(self, axis: str):
        self.axis = axis


class HomeAllReq(Request):
    """Perform homing of all the axes."""

    pass


class StopReq(Request):
    """Stop motion of single axis."""

    def __init__(self, axis: str):
        self.axis = axis


class StopAllReq(Request):
    """Stop motion of all the axes."""

    pass


class SaveReq(Request):
    """Save current states to a file"""

    def __init__(self, filename: str, group: str = ""):
        self.filename = filename
        self.group = group


class LoadReq(Request):
    """Load target positions from a file"""

    def __init__(self, filename: str, group: str = ""):
        self.filename = filename
        self.group = group
