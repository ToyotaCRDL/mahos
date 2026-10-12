#!/usr/bin/env python3

"""
Message Types for the Tweaker.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
from pprint import pformat

from mahos.msgs.common_msgs import FileArtifact, Message, Request, Status
from mahos.msgs import param_msgs as P


class TweakerStatus(Status):
    """Status message listing available parameter dictionaries.

    :ivar param_dict_ids: Registered ParamDict identifiers exposed by Tweaker.

    """

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


class ResetReq(Request):
    """Reset instrument setting (pertaining to a ParamDict)."""

    def __init__(self, param_dict_id: str):
        self.param_dict_id = param_dict_id


class GetStateReq(Request):
    """Request a serialized snapshot of the current Tweaker state."""

    pass


class TweakerState(Message):
    """Serialized Tweaker state.

    :ivar param_dicts: Current parameter dictionaries keyed by identifier.
    :ivar start_stop_states: Current start/stop states keyed by identifier.

    """

    def __init__(self, param_dicts: dict, start_stop_states: dict):
        self.param_dicts = param_dicts
        self.start_stop_states = start_stop_states


class SaveReq(Request):
    """Save current parameters to a file"""

    def __init__(self, file_name: str):
        self.file_name = file_name


class LoadReq(Request):
    """Load parameters from a file"""

    def __init__(self, file_name: str, group: str = ""):
        self.file_name = file_name
        self.group = group


class SaveArtifactReq(Request):
    """Request creation of a Tweaker-state artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.

    """

    def __init__(self, file_name: str):
        self.file_name = file_name


class LoadArtifactReq(Request):
    """Request loading Tweaker state from a shared artifact.

    :ivar artifact: Metadata identifying the staged file.
    :ivar group: HDF5 group from which to load the state.

    """

    def __init__(self, artifact: FileArtifact, group: str = ""):
        self.artifact = artifact
        self.group = group
