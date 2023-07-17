#!/usr/bin/env python3

"""
ZMQ-based loggers for mahos Nodes.

.. This file is a part of MAHOS project.

"""

import sys
import io
import typing as T
import logging
import traceback

import zmq
from zmq.utils.strtypes import cast_bytes

TOPIC_DELIM = "->"
TIME_DELIM = ";"


class PUBHandler(logging.Handler):
    """A basic logging handler that emits log messages through a PUB socket.

    :param socket: a PUB socket already bound to interfaces.

    Log messages handled by this handler are broadcast with a ZMQ topics
    `(self.root_topic->)loglevel(->additional_topic)`.
    Here, loglevel is like DEBUG, INFO, WARN, etc.
    The additional_topic can be included by submitting the log message like:
    log.info("additional_topic->real message")

    """

    def __init__(self, socket: zmq.Socket, root_topic: str = ""):
        logging.Handler.__init__(self)
        self._root_topic = root_topic
        self.formatters = {
            logging.DEBUG: logging.Formatter(
                f"%(asctime)s{TIME_DELIM}%(message)s (%(filename)s:%(lineno)d)\n"
            ),
            logging.INFO: logging.Formatter(f"%(asctime)s{TIME_DELIM}%(message)s\n"),
            logging.WARN: logging.Formatter(
                f"%(asctime)s{TIME_DELIM}%(message)s (%(filename)s:%(lineno)d)\n"
            ),
            logging.ERROR: logging.Formatter(
                f"%(asctime)s{TIME_DELIM}%(message)s (%(filename)s:%(lineno)d)\n"
            ),
            logging.CRITICAL: logging.Formatter(
                f"%(asctime)s{TIME_DELIM}%(message)s (%(filename)s:%(lineno)d)\n"
            ),
        }
        if not isinstance(socket, zmq.Socket):
            raise ValueError("zmq.Socket must be given")

        self.socket = socket

    @property
    def root_topic(self):
        return self._root_topic

    @root_topic.setter
    def root_topic(self, value):
        self.setRootTopic(value)

    def setRootTopic(self, root_topic):
        """Set the root topic for this handler.

        This value is prepended to all messages published by this handler, and it
        defaults to the empty string ''. When you subscribe to this socket, you must
        set your subscription to an empty string, or to at least the first letter of
        the binary representation of this string to ensure you receive any messages
        from this handler.

        If you use the default empty string root topic, messages will begin with
        the binary representation of the log level string (INFO, WARN, etc.).
        Note that ZMQ SUB sockets can have multiple subscriptions.
        """

        self._root_topic = root_topic

    def setFormatter(self, fmt, level=logging.NOTSET):
        """Set the Formatter for this handler.

        If no level is provided, the same format is used for all levels. This
        will overwrite all selective formatters set in the object constructor.
        """

        if level == logging.NOTSET:
            for fmt_level in self.formatters.keys():
                self.formatters[fmt_level] = fmt
        else:
            self.formatters[level] = fmt

    def format(self, record):
        """Format a record."""

        return self.formatters[record.levelno].format(record)

    def emit(self, record):
        """Emit a log message on my socket."""

        try:
            topic, record.msg = record.msg.split(TOPIC_DELIM, 1)
        except Exception:
            topic = ""
        try:
            bmsg = cast_bytes(self.format(record))
        except Exception:
            self.handleError(record)
            return

        topic_list = []

        if self.root_topic:
            topic_list.append(self.root_topic)

        topic_list.append(record.levelname)

        if topic:
            topic_list.append(topic)

        btopic = cast_bytes(TOPIC_DELIM).join(cast_bytes(t) for t in topic_list)

        self.socket.send_multipart([btopic, bmsg])


class TopicLogger(logging.Logger):
    def __init__(self, name):
        logging.Logger.__init__(self, name)
        self._topic = None

    def set_topic(self, topic):
        self._topic = topic

    def _fix_msg(self, msg):
        if self._topic:
            return TOPIC_DELIM.join((self._topic, msg))
        else:
            return msg

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        msg = self._fix_msg(msg)
        logging.Logger._log(
            self,
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel,
        )


class DummyLogger(object):
    """Dummy for logger. Just print to report the message."""

    def __init__(self, name: str = ""):
        self.name = name

    def debug(self, msg, *args, **kwargs):
        print(f"[DEBUG] {self.name} {msg}")

    def info(self, msg, *args, **kwargs):
        print(f"[INFO] {self.name} {msg}")

    def warn(self, msg, *args, **kwargs):
        print(f"[WARN] {self.name} {msg}")

    def warning(self, msg, *args, **kwargs):
        print(f"[WARN] {self.name} {msg}")

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def format_exc(self, ei):
        """Taken from logging.Formatter.formatException()."""

        sio = io.StringIO()
        tb = ei[2]
        traceback.print_exception(ei[0], ei[1], tb, None, sio)
        s = sio.getvalue()
        sio.close()
        if s[-1:] == "\n":
            s = s[:-1]
        return s

    def print_exc(self):
        print(self.format_exc(sys.exc_info()))

    def error(self, msg, *args, **kwargs):
        print(f"[ERROR] {self.name} {msg}")
        if kwargs.get("exc_info"):
            self.print_exc()


def get_topic_logger(prefix, name):
    if prefix:
        full_name = ".".join((prefix, name))
    else:
        full_name = name
    logging.setLoggerClass(TopicLogger)
    logger = logging.getLogger(full_name)
    logger.set_topic(name)
    return logger


def init_topic_logger(
    topic: str, prefix: T.Optional[str] = None
) -> T.Tuple[T.Union[TopicLogger, DummyLogger], str]:
    """Initialize TopicLogger and return (TopicLogger, full_name).

    If prefix is None, return (DummyLogger, topic).

    """

    if prefix is None:
        return DummyLogger(topic), topic

    return get_topic_logger(prefix, topic), TOPIC_DELIM.join((prefix, topic))
