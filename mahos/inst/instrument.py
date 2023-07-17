#!/usr/bin/env python3

"""
Base class for instrument.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import typing as T

from ..node.log import init_topic_logger


class Instrument(object):
    """Base class for Instrument.

    :param conf: instrument's local config. if None, empty dict is set.

    :ivar conf: instrument's local config.
    :ivar logger: configured logger.

    """

    def __init__(self, name: str, conf: dict | None = None, prefix: str | None = None):
        self.conf = {} if conf is None else conf
        topic = "{}({})".format(self.__class__.__name__, name)
        self.logger, self._full_name = init_topic_logger(topic, prefix=prefix)
        self._closed = False

    def check_required_conf(self, req_keys: T.Iterable[str]):
        """Check if required keys are defined in self.conf. raises ValueError if undefined.

        :raises KeyError: any key of req_keys is undefined in self.conf.

        """

        if any((k not in self.conf for k in req_keys)):
            raise KeyError(f"These configs must be given: {req_keys}")

    def check_required_params(self, params: T.Container[str], req_keys: T.Iterable[str]) -> bool:
        """Check if required keys (`req_keys`) are defined in `params`.

        :returns: False if undefined.

        """

        if any((k not in params for k in req_keys)):
            self.logger.error(f"These params must be given: {req_keys}")
            return False
        else:
            return True

    def full_name(self) -> str:
        """Get full name: (prefix->)ClassName(name)"""

        return self._full_name

    def __repr__(self):
        return self.full_name()

    def __del__(self):
        self.close_once()

    def fail_with(self, msg: str) -> bool:
        """Log error and return False."""

        self.logger.error(msg)
        return False

    def close_once(self):
        """Close instrument resources.

        - This function checks self._closed and avoid double-close.
        - This function is called when InstrumentServer is exiting.

        """

        if self._closed:
            return
        self.close()
        self._closed = True

    def close(self):
        """Close instrument resources.

        Implement custom close behaviour overriding this method.

        """

        pass

    def is_closed(self) -> bool:
        if self._closed:
            self.logger.error("Already closed. Skipping operation.")
        return self._closed

    # Standard API

    def start(self) -> bool:
        """Start the instrument operation. Returns True on success."""

        self.logger.error("start() is called but not implemented.")
        return False

    def stop(self) -> bool:
        """Stop the instrument operation. Returns True on success."""

        self.logger.error("stop() is called but not implemented.")
        return False

    def shutdown(self) -> bool:
        """Shutdown the instrument and get ready to hard power-off. Returns True on success.

        Although default implementation is fall-back to close(),
        a different behaviour can be implemented.
        This function is meant to be a bit harder operation than close() and
        called by user request (rather than implicit call on InstrumentServer's exit).

        """

        self.logger.warn("shutdown() is called but not implemented. falling back to _close().")
        self.close_once()
        return True

    def pause(self) -> bool:
        """Pause the instrument operation. Returns True on success."""

        self.logger.warn("pause() is called but not implemented. falling back to stop().")
        return self.stop()

    def resume(self) -> bool:
        """Resume the instrument operation from pause. Returns True on success."""

        self.logger.warn("resume() is called but not implemented. falling back to start().")
        return self.start()

    def reset(self) -> bool:
        """Reset the instrument settings. Returns True on success."""

        self.logger.error("reset() is called but not implemented.")
        return False

    def configure(self, params: dict) -> bool:
        """Configure the instrument settings. Returns True on success."""

        self.logger.error("configure() is called but not implemented.")
        return False

    def set(self, key: str, value=None) -> bool:
        """Set an instrument setting or commanding value. Returns True on success."""

        self.logger.error("set() is called but not implemented.")
        return False

    def get(self, key: str, args=None):
        """Get an instrument setting or measurement data."""

        self.logger.error("get() is called but not implemented.")
        return None
