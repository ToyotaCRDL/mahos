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


class Reply(Message):
    """Generic reply message for requests.

    :ivar success: Whether the request succeeded.
    :ivar message: Message from the server (usually an error message).
    :ivar ret: Return value.

    """

    def __init__(self, success: bool, message="", ret=None):
        self.success = success
        self.message = message
        self.ret = ret

    def __repr__(self):
        return f"Reply({self.success}, {self.message}, {self.ret})"


class Request(Message):
    """Base class for requests to a Node."""

    pass


class Status(Message):
    """Base class for node status."""

    pass


class State(Message, enum.Enum):
    """Base class for node state."""

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

    def __init__(self, state: State, params=None, label: str = ""):
        self.state = state
        self.params = params
        self.label = label


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
    """Generic request to load measurement data.

    :ivar file_name: Input file path.
    :ivar to_buffer: Whether to append loaded data to the node-side buffer.
    :ivar buffer_name: Name associated with buffered data; defaults to ``file_name``.

    """

    def __init__(self, file_name: str, to_buffer: bool = False, buffer_name: str | None = None):
        self.file_name = file_name
        self.to_buffer = to_buffer
        self.buffer_name = file_name if buffer_name is None else buffer_name


class FileArtifact(Message):
    """Metadata for an artifact in a configured shared transport directory.

    :ivar basename: Safe basename of the artifact; never a path.
    :ivar size: Artifact size in bytes.

    """

    def __init__(self, basename: str, size: int):
        self.basename = basename
        self.size = size


class FileArtifactBundle(Message):
    """Metadata for a primary output artifact and its derived sidecar files.

    :ivar primary: The artifact copied to the client-requested destination.
    :ivar sidecars: Pairs of destination basename and artifact metadata for files copied beside
        the primary output.

    """

    def __init__(self, primary: FileArtifact, sidecars: list[tuple[str, FileArtifact]]):
        self.primary = primary
        self.sidecars = sidecars


class SaveArtifactReq(Request):
    """Request creation of a saved-data artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.
    :ivar params: Optional save parameters.
    :ivar note: Optional note stored with the dataset.

    """

    def __init__(self, file_name: str, params=None, note: str = ""):
        self.file_name = file_name
        self.params = params
        self.note = note


class ExportArtifactReq(Request):
    """Request creation of an exported-data artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.
    :ivar data: Optional explicit data payload for exporting.
    :ivar params: Optional export parameters.

    """

    def __init__(self, file_name: str, data=None, params=None):
        self.file_name = file_name
        self.data = data
        self.params = params


class LoadArtifactReq(Request):
    """Request loading data from a shared artifact.

    :ivar artifact: Metadata identifying the staged input file.
    :ivar file_name: Original input basename, used as the buffer name.
    :ivar to_buffer: Whether to append loaded data to the node-side buffer.

    """

    def __init__(self, artifact: FileArtifact, file_name: str, to_buffer: bool = False):
        self.artifact = artifact
        self.file_name = file_name
        self.to_buffer = to_buffer


class CleanupArtifactReq(Request):
    """Acknowledge receipt and request removal of a producer-owned artifact.

    :ivar artifact: Metadata identifying the artifact or artifact bundle to remove.

    """

    def __init__(self, artifact: FileArtifact | FileArtifactBundle):
        self.artifact = artifact
