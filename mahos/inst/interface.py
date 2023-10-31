#!/usr/bin/env python3

"""
Typed Interface for InstrumentClient.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from ..msgs.common_msgs import Resp
from ..msgs import param_msgs as P
from .server import InstrumentClient, MultiInstrumentClient


class InstrumentInterface(object):
    """Base class for Instrument Interface."""

    def __init__(self, client: InstrumentClient | MultiInstrumentClient, inst: str):
        self.cli = client
        self.inst = inst

    def lock(self) -> bool:
        """Acquire lock of this instrument. Returns True on success."""

        return self.cli.lock(self.inst)

    def is_locked(self) -> bool | None:
        """Check if this instrument is locked.

        if None is returned, status is unknown due to server error.

        """

        return self.cli.is_locked(self.inst)

    def release(self) -> bool:
        """Release lock of this instrument. Returns True on success."""

        return self.cli.release(self.inst)

    def call(self, func: str, **args) -> Resp:
        """Call arbitrary function of the instrument.

        Note that this is not quite a safe API (left for debug purpose).
        Consider using one of following standardized APIs:
            start, stop, pause, resume, reset, configure, set, get

        """

        return self.cli.call(self.inst, func, **args)

    def __call__(self, func: str, **args) -> Resp:
        return self.call(func, **args)

    def shutdown(self) -> bool:
        """Shutdown the instrument and get ready to power-off. Returns True on success."""

        return self.cli.shutdown(self.inst)

    def start(self) -> bool:
        """Start the instrument operation. Returns True on success."""

        return self.cli.start(self.inst)

    def stop(self) -> bool:
        """Stop the instrument operation. Returns True on success."""

        return self.cli.stop(self.inst)

    def pause(self) -> bool:
        """Pause the instrument operation. Returns True on success."""

        return self.cli.pause(self.inst)

    def resume(self) -> bool:
        """Resume the instrument operation. Returns True on success."""

        return self.cli.resume(self.inst)

    def reset(self) -> bool:
        """Reset the instrument settings. Returns True on success."""

        return self.cli.reset(self.inst)

    def configure(self, params: dict, label: str = "", group: str = "") -> bool:
        """Configure the instrument settings. Returns True on success."""

        return self.cli.configure(self.inst, params, label, group)

    def set(self, key: str, value=None) -> bool:
        """Set an instrument setting or commanding value. Returns True on success."""

        return self.cli.set(self.inst, key, value)

    def get(self, key: str, args=None):
        """Get an instrument setting or commanding value."""

        return self.cli.get(self.inst, key, args)

    def help(self, func: str | None = None) -> str:
        """Get help of instrument `name`.

        If function name `func` is given, get docstring of that function.
        Otherwise, get docstring of the class.

        """

        return self.cli.help(self.inst, func)

    def get_param_dict(
        self, label: str = "", group: str = ""
    ) -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label` in `group`."""

        return self.cli.get_param_dict(self.inst, label, group)

    def get_param_dict_labels(self, group: str) -> list[str]:
        """Get list of available ParamDict labels pertaining to `group`."""

        return self.cli.get_param_dict_labels(self.inst, group)
