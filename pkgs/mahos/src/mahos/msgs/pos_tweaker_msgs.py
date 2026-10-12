#!/usr/bin/env python3

"""
Message Types for the PosTweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from pprint import pformat

from mahos.msgs.common_msgs import Message, Request, Status
from mahos.msgs.tweaker_msgs import SaveReq as SaveReq
from mahos.msgs.tweaker_msgs import LoadReq as LoadReq


class PosTweakerStatus(Status):
    """Status snapshot of per-axis positioner states.

    :ivar axis_states: Mapping from axis name to status dictionary
        (position/target/range/homing flags).

    """

    def __init__(self, axis_states: dict[str, dict[str, [float, bool]]]):
        self.axis_states = axis_states

    def __repr__(self):
        return f"PosTweakerStatus({self.axis_states})"

    def __str__(self):
        return "PosTweaker->axis_states:\n" + pformat(self.axis_states)


class PosTweakerState(Message):
    """Serialized PosTweaker state.

    :ivar axis_states: Current positioner states keyed by axis name.

    """

    def __init__(self, axis_states: dict):
        self.axis_states = axis_states


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
