#!/usr/bin/env python3

"""
Typed Interface for InstrumentClient.

.. This file is a part of MAHOS project.

"""

from typing import Union, Optional

from ..msgs.common_msgs import Resp
from .server import InstrumentClient, MultiInstrumentClient


class InstrumentInterface(object):
    """Base class for Instrument Interface."""

    def __init__(self, client: Union[InstrumentClient, MultiInstrumentClient], name: str):
        self.cli = client
        self.name = name

    def lock(self) -> bool:
        """Acquire lock of this instrument. Returns True on success."""

        return self.cli.lock(self.name)

    def is_locked(self) -> Optional[bool]:
        """Check if this instrument is locked.

        if None is returned, status is unknown due to server error.

        """

        return self.cli.is_locked(self.name)

    def release(self) -> bool:
        """Release lock of this instrument. Returns True on success."""

        return self.cli.release(self.name)

    def call(self, func: str, **args) -> Resp:
        """Call arbitrary function of the instrument.

        Note that this is not quite a safe API (left for debug purpose).
        Consider using one of following standardized APIs:
            start, stop, pause, resume, reset, configure, set, get

        """

        return self.cli.call(self.name, func, **args)

    def __call__(self, func: str, **args) -> Resp:
        return self.call(func, **args)

    def shutdown(self) -> bool:
        """Shutdown the instrument and get ready to power-off. Returns True on success."""

        return self.cli.shutdown(self.name)

    def start(self) -> bool:
        """Start the instrument operation. Returns True on success."""

        return self.cli.start(self.name)

    def stop(self) -> bool:
        """Stop the instrument operation. Returns True on success."""

        return self.cli.stop(self.name)

    def pause(self) -> bool:
        """Pause the instrument operation. Returns True on success."""

        return self.cli.pause(self.name)

    def resume(self) -> bool:
        """Resume the instrument operation. Returns True on success."""

        return self.cli.resume(self.name)

    def reset(self) -> bool:
        """Reset the instrument settings. Returns True on success."""

        return self.cli.reset(self.name)

    def configure(self, params: dict) -> bool:
        """Configure the instrument settings. Returns True on success."""

        return self.cli.configure(self.name, params)

    def set(self, key: str, value=None) -> bool:
        """Set an instrument setting or commanding value. Returns True on success."""

        return self.cli.set(self.name, key, value)

    def get(self, key: str, args=None):
        """Get an instrument setting or commanding value."""

        return self.cli.get(self.name, key, args)

    def help(self, func: Optional[str] = None) -> str:
        """Get help of instrument `name`.

        If function name `func` is given, get docstring of that function.
        Otherwise, get docstring of the class.

        """

        return self.cli.help(self.name, func)
