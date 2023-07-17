#!/usr/bin/env python3

"""
Message Types for Confocal.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import enum
import uuid
import time
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import msgpack

from .common_msgs import Message, Request, State, Status
from .data_msgs import Data

from .common_msgs import SaveDataReq, ExportDataReq, LoadDataReq


class ConfocalState(State):
    IDLE = 0  # do nothing.
    PIEZO = 1  # move piezo position (piezo)
    INTERACT = 2  # move piezo position and trace PD outputs. (piezo, daq)
    SCAN = 3  # do scan. (piezo, daq)


class Axis(Message, enum.Enum):
    X = 0
    Y = 1
    Z = 2


class ScanDirection(Message, enum.Enum):
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
    PAUSE = 0
    RESUME = 1
    CLEAR = 2


class BufferCommand(Message, enum.Enum):
    POP = 0
    CLEAR = 1
    GET_ALL = 2


class PiezoPos(Message):
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


class Image(Data):
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


class Trace(Data):
    def __init__(self, size=0, channels=2):
        self.set_version(1)

        # trace data
        self.traces = [np.zeros(size) for _ in range(channels)]
        # time stamps
        self.stamps = [np.zeros(size, dtype="datetime64[ns]") for _ in range(channels)]

        self.ylabel: str = "Intensity"
        self.yunit: str = ""

    def channels(self) -> int:
        return len(self.traces)

    def size(self) -> int:
        return len(self.traces[0])

    def clear(self):
        channels = self.channels()
        size = self.size()
        for ch in range(channels):
            self.traces[ch] = np.zeros(size)
            self.stamps[ch] = np.zeros(size, dtype="datetime64[ns]")

    def valid_trace(self, ch=0):
        idx = self.stamps[ch].view(np.int64) > 0.0
        return self.stamps[ch][idx], self.traces[ch][idx]

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
    def __init__(self, state: ConfocalState, pos: PiezoPos, tracer_paused: bool):
        self.state = state
        self.pos = pos
        self.tracer_paused = tracer_paused

    def __repr__(self):
        return f"ConfocalStatus({self.state}, {self.pos}, {self.tracer_paused})"

    def __str__(self):
        return f"Confocal({self.state.name}, {self.pos}, {self.tracer_paused})"


class MoveReq(Request):
    def __init__(self, ax: Union[Axis, List[Axis]], pos: Union[float, List[float]]):
        self.ax = ax
        self.pos = pos


class SaveImageReq(SaveDataReq):
    def __init__(self, file_name: str, direction: Optional[ScanDirection] = None, note: str = ""):
        self.file_name = file_name
        self.direction = direction
        self.note = note


class ExportImageReq(ExportDataReq):
    def __init__(self, file_name: str, direction: Optional[ScanDirection] = None, params=None):
        self.file_name = file_name
        self.direction = direction
        self.params = params


class ExportViewReq(ExportDataReq):
    def __init__(self, file_name: str, params=None):
        self.file_name = file_name
        self.params = params


class LoadImageReq(LoadDataReq):
    pass


class SaveTraceReq(SaveDataReq):
    pass


class ExportTraceReq(ExportDataReq):
    pass


class LoadTraceReq(LoadDataReq):
    pass


class CommandTraceReq(Request):
    def __init__(self, command: TraceCommand):
        self.command = command


class CommandBufferReq(Request):
    def __init__(self, direction: ScanDirection, command: BufferCommand):
        self.direction = direction
        self.command = command
