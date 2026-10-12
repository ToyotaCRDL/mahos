#!/usr/bin/env python3

"""
Message Types for Confocal.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import enum
import uuid
import time

import numpy as np
import pandas as pd
import msgpack

from mahos.msgs.common_msgs import FileArtifact, Message, Request, State, Status, BinaryState
from mahos.msgs.data_msgs import Data, ComplexDataMixin, FormatTimeMixin

from mahos.msgs.common_msgs import SaveDataReq, ExportDataReq, LoadDataReq
from mahos.msgs.inst.piezo_msgs import Axis


class ConfocalState(State):
    """State machine values used by the confocal node."""

    IDLE = 0  # do nothing.
    PIEZO = 1  # move piezo position (piezo)
    INTERACT = 2  # move piezo position and trace PD outputs. (piezo, daq)
    SCAN = 3  # do scan. (piezo, daq)


class ScanDirection(Message, enum.Enum):
    """Plane selection for confocal image scans."""

    XY = 0
    XZ = 1
    YZ = 2


def _direction_to_x(d: ScanDirection, rets):
    return rets[d.value]


def _x_to_direction(x, xs) -> ScanDirection:
    return ScanDirection(xs.index(x))


def direction_to_str(d: ScanDirection):
    return d.name


def str_to_direction(s: str) -> ScanDirection:
    s = s.upper()
    return _x_to_direction(s, ("XY", "XZ", "YZ"))


def direction_to_labels(d: ScanDirection):
    s = direction_to_str(d)
    return s[0], s[1]


def direction_to_axes(d: ScanDirection):
    return _direction_to_x(d, ((Axis.X, Axis.Y), (Axis.X, Axis.Z), (Axis.Y, Axis.Z)))


class ScanMode(Message, enum.Enum):
    """Acquisition backend selection for confocal scanner execution."""

    COM_NOWAIT = 0  # Command scan without explicit wait
    COM_COMMAND = 1  # Command scan with command wait
    COM_DIPOLL = 2  # Command scan with digital input wait
    ANALOG = 3  # Analog scan


def mode_to_str(m: ScanMode):
    return m.name


def _x_to_mode(x, xs) -> ScanMode:
    return ScanMode(xs.index(x))


def str_to_mode(s: str) -> ScanMode:
    return _x_to_mode(s, ("COM_NOWAIT", "COM_COMMAND", "COM_DIPOLL", "ANALOG"))


class LineMode(Message, enum.Enum):
    """Line traversal order used within each scan frame."""

    ASCEND = 0
    DESCEND = 1
    ZIGZAG = 2


def line_mode_to_str(m: LineMode):
    return m.name


def _x_to_line_mode(x, xs) -> LineMode:
    return LineMode(xs.index(x))


def str_to_line_mode(s: str) -> LineMode:
    return _x_to_line_mode(s, ("ASCEND", "DESCEND", "ZIGZAG"))


class TraceCommand(Message, enum.Enum):
    """Control command for confocal trace buffering."""

    PAUSE = 0
    RESUME = 1
    CLEAR = 2


class BufferCommand(Message, enum.Enum):
    """Control command for image buffer operations."""

    POP = 0
    CLEAR = 1
    GET_ALL = 2


class PiezoPos(Message):
    """Piezo position snapshot with current, target, and range values.

    :ivar x: Current X position.
    :ivar y: Current Y position.
    :ivar z: Current Z position.
    :ivar x_range: Configured ``(min, max)`` range for X.
    :ivar y_range: Configured ``(min, max)`` range for Y.
    :ivar z_range: Configured ``(min, max)`` range for Z.
    :ivar x_ont: Whether the X axis is on target.
    :ivar y_ont: Whether the Y axis is on target.
    :ivar z_ont: Whether the Z axis is on target.
    :ivar x_tgt: Current X target position.
    :ivar y_tgt: Current Y target position.
    :ivar z_tgt: Current Z target position.

    """

    def __init__(
        self,
        x=None,
        y=None,
        z=None,
        x_range=None,
        y_range=None,
        z_range=None,
        x_ont=False,
        y_ont=False,
        z_ont=False,
        x_tgt=None,
        y_tgt=None,
        z_tgt=None,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.x_ont = x_ont
        self.y_ont = y_ont
        self.z_ont = z_ont
        self.x_tgt = x_tgt
        self.y_tgt = y_tgt
        self.z_tgt = z_tgt

    def __repr__(self):
        return (
            f"PiezoPos({self.x}, {self.y}, {self.z},"
            + f" {self.x_range}, {self.y_range}, {self.z_range},"
            + f" {self.x_ont}, {self.y_ont}, {self.z_ont},"
            + f" {self.x_tgt}, {self.y_tgt}, {self.z_tgt})"
        )

    def _fmt(self, v):
        if v is None:
            return "None"
        else:
            return f"{v:.4f}"

    def __str__(self):
        x, x_tgt, y, y_tgt, z, z_tgt = [
            self._fmt(v) for v in (self.x, self.x_tgt, self.y, self.y_tgt, self.z, self.z_tgt)
        ]
        return f"P({x} ({x_tgt}), {y} ({y_tgt}), {z} ({z_tgt}))"

    def has_pos(self) -> bool:
        return all([v is not None for v in (self.x, self.y, self.z)])

    def has_range(self) -> bool:
        return all([v is not None for v in (self.x_range, self.y_range, self.z_range)])

    def has_target(self) -> bool:
        return all([v is not None for v in (self.x_tgt, self.y_tgt, self.z_tgt)])

    def has_range_and_target(self) -> bool:
        return self.has_range() and self.has_target()


class Image(Data, ComplexDataMixin, FormatTimeMixin):
    """Confocal image frame and acquisition state snapshot.

    :ivar direction: Scan plane for this image (XY/XZ/YZ).
    :ivar image: 2D image array, possibly complex-valued.
    :ivar running: True while scan acquisition is active.
    :ivar aborted: True when acquisition was aborted before normal completion.
    :ivar start_time: Acquisition start timestamp.
    :ivar finish_time: Acquisition finish timestamp, or ``None`` while running.
    :ivar clabel: Colorbar label string for visualization.
    :ivar cunit: Colorbar unit string for visualization.

    """

    def __init__(self, params: dict | None = None):
        self.set_version(1)
        self.init_params(params)

        if self.has_params():
            self.direction = self.params.get("direction")
        else:
            self.direction = None

        self.image = None
        self.running: bool = False
        self.aborted: bool = False
        self.start_time: float = time.time()
        self.finish_time: float | None = None

        self.clabel: str = "Intensity"
        self.cunit: str = ""

    def has_data(self) -> bool:
        """return True if data is ready and valid data could be read out."""

        return self.image is not None

    def is_complex(self) -> bool:
        return self.has_data() and np.issubdtype(self.image.dtype, np.complexfloating)

    def get_image(self, complex_conv: str = "real"):
        return self.conv_complex(self.image, complex_conv)

    def finalize(self, aborted: bool) -> float:
        """set attributes to finalize the data.

        set `running` (to False), `aborted`, and `finish_time`.

        :returns: elapsed time (self.finish_time - self.start_time)

        """

        self.running = False
        self.aborted = aborted
        self.finish_time = time.time()
        return self.finish_time - self.start_time

    def is_finalized(self):
        return not self.running and self.finish_time is not None

    def _h5_write_direction(self, val):
        return val.value

    def _h5_read_direction(self, val):
        return ScanDirection(val)

    def _h5_write_params(self, val):
        d = {}
        for k, v in val.items():
            if k == "ident" and isinstance(v, uuid.UUID):
                d[k] = v.hex
            elif k in ("mode", "line_mode", "direction"):
                # unwrap enum values
                d[k] = v.value
            else:
                d[k] = v
        return np.void(msgpack.dumps(d))

    def _h5_read_params(self, val):
        d = msgpack.loads(val.tobytes())
        if "ident" in d and isinstance(d["ident"], str):
            d["ident"] = uuid.UUID(hex=d["ident"])
        d["mode"] = ScanMode(d["mode"])
        d["line_mode"] = LineMode(d["line_mode"])
        d["direction"] = ScanDirection(d["direction"])
        return d

    def _h5_attr_writers(self) -> dict:
        return {"direction": self._h5_write_direction, "params": self._h5_write_params}

    def _h5_readers(self) -> dict:
        return {"direction": self._h5_read_direction, "params": self._h5_read_params}


def update_image(image: Image):
    """update image to latest format"""

    if image.version() <= 0:
        # version 0 to 1
        ## add missing attributes
        image.clabel: str = "Intensity"
        image.cunit: str = "cps"  # old data were in cps
        image.set_version(1)

    return image


class Trace(Data, ComplexDataMixin):
    """Multi-channel confocal trace buffer with timestamped samples.

    :ivar _complex: True when trace storage uses complex dtype.
    :ivar traces: Per-channel trace arrays.
    :ivar stamps: Per-channel ``datetime64[ns]`` timestamp arrays.
    :ivar ylabel: Y-axis label used by viewers.
    :ivar yunit: Y-axis unit used by viewers.

    """

    def __init__(self, size: int = 0, channels: int = 2, _complex: bool = False):
        self.set_version(1)

        self._complex = _complex

        # trace data
        self.traces = [np.zeros(size, self.dtype()) for _ in range(channels)]
        # time stamps
        self.stamps = [np.zeros(size, dtype="datetime64[ns]") for _ in range(channels)]

        self.ylabel: str = "Intensity"
        self.yunit: str = ""

    def channels(self) -> int:
        return len(self.traces)

    def size(self) -> int:
        return len(self.traces[0])

    def dtype(self):
        return np.cdouble if self._complex else np.double

    def clear(self):
        channels = self.channels()
        size = self.size()
        for ch in range(channels):
            self.traces[ch] = np.zeros(size, self.dtype())
            self.stamps[ch] = np.zeros(size, dtype="datetime64[ns]")

    def is_complex(self) -> bool:
        return np.issubdtype(self.traces[0].dtype, np.complexfloating)

    def get_trace(self, ch=0, complex_conv: str = "real"):
        return self.conv_complex(self.traces[ch], complex_conv)

    def valid_trace(self, ch=0, complex_conv: str = "real"):
        idx = self.stamps[ch].view(np.int64) > 0.0
        return self.stamps[ch][idx], self.conv_complex(self.traces[ch][idx], complex_conv)

    def as_dataframe(self, ch=0):
        s, t = self.valid_trace(ch)
        return pd.DataFrame(t, index=pd.DatetimeIndex(s))

    def as_dataframes(self):
        return [self.as_dataframe(ch) for ch in range(len(self.traces))]

    def _h5_write_stamps(self, val):
        return np.array(val).astype(np.int64)

    def _h5_read_stamps(self, val):
        return list(np.array(val).astype("datetime64[ns]"))

    def _h5_write_traces(self, val):
        return np.array(val)

    def _h5_read_traces(self, val):
        return list(val)

    def _h5_dataset_writers(self) -> dict:
        return {"stamps": self._h5_write_stamps, "traces": self._h5_write_traces}

    def _h5_readers(self) -> dict:
        return {"stamps": self._h5_read_stamps, "traces": self._h5_read_traces}


def update_trace(trace: Trace):
    """update trace to latest format"""

    if trace.version() <= 0:
        # version 0 to 1
        ## add missing attributes
        trace.ylabel: str = "Intensity"
        trace.yunit: str = "cps"  # old data were in cps
        trace.set_version(1)

    return trace


class ConfocalStatus(Status):
    """Status message published by the confocal node.

    :ivar state: Current confocal state.
    :ivar pos: Current piezo position snapshot.
    :ivar tracer_paused: True when trace acquisition is paused.

    """

    def __init__(self, state: ConfocalState, pos: PiezoPos, tracer_paused: bool):
        self.state = state
        self.pos = pos
        self.tracer_paused = tracer_paused

    def __repr__(self):
        return f"ConfocalStatus({self.state}, {self.pos}, {self.tracer_paused})"

    def __str__(self):
        return f"Confocal({self.state.name}, {self.pos}, {self.tracer_paused})"


class TraceStatus(Status):
    """Status message published by the TraceNode (trace-only node).

    :ivar state: Current binary state of the trace node.
    :ivar tracer_paused: True when trace acquisition is paused.

    """

    def __init__(self, state: BinaryState, tracer_paused: bool):
        self.state = state
        self.tracer_paused = tracer_paused

    def __repr__(self):
        return f"TraceStatus({self.state}, {self.tracer_paused})"

    def __str__(self):
        return f"Trace({self.state.name}, {self.tracer_paused})"


class MoveReq(Request):
    """Request to move one or more piezo axes.

    :ivar ax: Target axis (or axis list) to move.
    :ivar pos: Target position (or position list) paired with ``ax``.

    """

    def __init__(self, ax: Axis | list[Axis], pos: float | list[float]):
        self.ax = ax
        self.pos = pos


class SaveImageReq(SaveDataReq):
    """Request to save the latest or buffered confocal image.

    :ivar file_name: Output file path.
    :ivar direction: Optional buffered direction; ``None`` uses current scan image.
    :ivar note: Optional note string stored with the dataset.

    """

    def __init__(self, file_name: str, direction: ScanDirection | None = None, note: str = ""):
        self.file_name = file_name
        self.direction = direction
        self.note = note


class ExportImageReq(ExportDataReq):
    """Request to export an image using visualization/export parameters.

    :ivar file_name: Output file path.
    :ivar direction: Optional buffered direction; ``None`` uses current scan image.
    :ivar params: Export parameter dictionary.

    """

    def __init__(self, file_name: str, direction: ScanDirection | None = None, params=None):
        self.file_name = file_name
        self.direction = direction
        self.params = params


class ExportViewReq(ExportDataReq):
    """Request to export a combined multi-plane confocal view.

    :ivar file_name: Output file path.
    :ivar params: Export parameter dictionary.

    """

    def __init__(self, file_name: str, params=None):
        self.file_name = file_name
        self.params = params


class LoadImageReq(LoadDataReq):
    """Request to load a confocal image from disk.

    :ivar file_name: Input file path.
    :ivar to_buffer: If True, append loaded data to node-side buffer.

    """

    pass


class SaveTraceReq(SaveDataReq):
    """Request to save the latest confocal trace dataset.

    :ivar file_name: Output file path.
    :ivar params: Optional save parameters.
    :ivar note: Optional note string stored with the dataset.

    """

    pass


class ExportTraceReq(ExportDataReq):
    """Request to export trace data with visualization/export parameters.

    :ivar file_name: Output file path.
    :ivar data: Optional explicit data payload for exporting.
    :ivar params: Export parameter dictionary.

    """

    pass


class LoadTraceReq(LoadDataReq):
    """Request to load trace data from disk.

    :ivar file_name: Input file path.
    :ivar to_buffer: If True, append loaded data to node-side buffer.

    """

    pass


class SaveImageArtifactReq(Request):
    """Request creation of a saved confocal-image artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.
    :ivar direction: Optional buffered scan direction; ``None`` uses the current image.
    :ivar note: Optional note stored with the image.

    """

    def __init__(self, file_name: str, direction: ScanDirection | None = None, note: str = ""):
        self.file_name = file_name
        self.direction = direction
        self.note = note


class ExportImageArtifactReq(Request):
    """Request creation of an exported confocal-image artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.
    :ivar direction: Optional buffered scan direction; ``None`` uses the current image.
    :ivar params: Export parameter dictionary.

    """

    def __init__(self, file_name: str, direction: ScanDirection | None = None, params=None):
        self.file_name = file_name
        self.direction = direction
        self.params = params


class ExportViewArtifactReq(Request):
    """Request creation of an exported combined-view artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.
    :ivar params: Export parameter dictionary.

    """

    def __init__(self, file_name: str, params=None):
        self.file_name = file_name
        self.params = params


class LoadImageArtifactReq(Request):
    """Request loading a confocal image from a shared artifact.

    :ivar artifact: Metadata identifying the staged image file.

    """

    def __init__(self, artifact: FileArtifact):
        self.artifact = artifact


class SaveTraceArtifactReq(Request):
    """Request creation of a saved trace artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.
    :ivar note: Optional note stored with the trace.

    """

    def __init__(self, file_name: str, note: str = ""):
        self.file_name = file_name
        self.note = note


class ExportTraceArtifactReq(Request):
    """Request creation of an exported trace artifact.

    :ivar file_name: Original destination basename, used to preserve its suffix.
    :ivar params: Export parameter dictionary.

    """

    def __init__(self, file_name: str, params=None):
        self.file_name = file_name
        self.params = params


class LoadTraceArtifactReq(Request):
    """Request loading a trace from a shared artifact.

    :ivar artifact: Metadata identifying the staged trace file.

    """

    def __init__(self, artifact: FileArtifact):
        self.artifact = artifact


class CommandTraceReq(Request):
    """Request to execute a trace command.

    :ivar command: Trace command enum value.

    """

    def __init__(self, command: TraceCommand):
        self.command = command


class CommandBufferReq(Request):
    """Request to execute an image-buffer command for one scan direction.

    :ivar direction: Buffer direction (XY/XZ/YZ) to operate on.
    :ivar command: Buffer command enum value.

    """

    def __init__(self, direction: ScanDirection, command: BufferCommand):
        self.direction = direction
        self.command = command
