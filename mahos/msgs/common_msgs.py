#!/usr/bin/env python3

"""
Common and base definitions for mahos messages.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import enum
import pprint
import pickle

import numpy as np


# As of Python 3.8, we can use pickle protocol version 5 (that is not default).
# https://peps.python.org/pep-0574/
pickle_proto = 5


class Message(object):
    """Base class for mahos messages."""

    def __repr__(self):
        if isinstance(self, enum.Enum):
            return enum.Enum.__repr__(self)
        type_name = type(self).__name__
        attrs = []
        with np.printoptions(threshold=10):
            for name, value in self.__dict__.items():
                attrs.append(f"{name}={value}")
            return "{}({})".format(type_name, ", ".join(attrs))

    def pprint(self, array_threshold=10):
        if isinstance(self, enum.Enum):
            print(self)
        else:
            with np.printoptions(threshold=array_threshold):
                pprint.pp(self.__dict__)

    def serialize(self) -> bytes:
        """Serialize this message to bytes.

        Default implementation uses pickle.
        Override this method (and deserialize()) to implement custom serialization.

        """

        return pickle.dumps(self, protocol=pickle_proto)

    @classmethod
    def deserialize(cls, b: bytes):
        """Deserialize given bytes `b` to reconstruct an instance if this class.

        Default implementation uses pickle.
        Override this method (and serialize()) to implement custom serialization.

        """

        return pickle.loads(b)


class Resp(Message):
    """Generic response message for requests.

    :ivar success: requests are successful or not.
    :ivar message: message from server (usually error message).
    :ivar ret: return value.

    """

    def __init__(self, success: bool, message="", ret=None):
        self.success = success
        self.message = message
        self.ret = ret

    def __repr__(self):
        return f"Resp({self.success}, {self.message}, {self.ret})"


class Request(Message):
    """Base class for Request to Node."""

    pass


class Status(Message):
    """Base class for Node Status."""

    pass


class State(Message, enum.Enum):
    """Base class for Node State."""

    pass


class BinaryState(State):
    """Generic Node State with binary states IDLE and ACTIVE."""

    IDLE = 0  # do nothing.
    ACTIVE = 1  # active.


class BinaryStatus(Status):
    """Status only with state: BinaryState."""

    def __init__(self, state: BinaryState):
        self.state = state

    def __repr__(self):
        return f"BinaryStatus({self.state})"

    def __str__(self):
        return f"Binary({self.state.name})"


class StateReq(Request):
    """Generic state change request."""

    def __init__(self, state: State, params=None):
        self.state = state
        self.params = params


class ShutdownReq(Request):
    """Generic shutdown request"""

    pass


class SaveDataReq(Request):
    """Generic Save Data Request"""

    def __init__(self, file_name: str, params=None, note: str = ""):
        self.file_name = file_name
        self.params = params
        self.note = note


class ExportDataReq(Request):
    """Generic Export Data Request"""

    def __init__(self, file_name: str, data=None, params=None):
        self.file_name = file_name
        self.data = data
        self.params = params


class LoadDataReq(Request):
    """Generic Load Data Request"""

    def __init__(self, file_name: str, to_buffer: bool = False):
        self.file_name = file_name
        self.to_buffer = to_buffer
