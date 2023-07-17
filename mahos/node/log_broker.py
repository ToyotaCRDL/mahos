#!/usr/bin/env python3

"""
Global log broker.

.. This file is a part of MAHOS project.

"""

import typing as T
from os import path
import copy
import datetime
import threading
import logging

from .node import Node, NAME_DELIM, split_name
from .client import NodeClient, SubWorker
from .log import TOPIC_DELIM, TIME_DELIM
from ..util.term_color import col
from .. import log_dir


class LogEntry(T.NamedTuple):
    timestamp: str
    level: str
    host: str
    name: str
    topic: str
    body: str


def parse_log(msg) -> T.Optional[LogEntry]:
    def report_parse_err():
        print("Received invalid log: ", msg)
        return None

    if len(msg) != 2:
        return report_parse_err()
    try:
        header = msg[0].decode()
        message = msg[1].decode()
        headers = header.split(TOPIC_DELIM)
        if len(headers) < 2:
            return report_parse_err()
        joined_name, level = headers[:2]
        host, name = split_name(joined_name)
        topic = TOPIC_DELIM.join(headers[2:])  # in case topic contains TOPIC_DELIM
        i = message.index(TIME_DELIM)
        timestamp = message[:i]
        body = message[i + 1 :]
        return LogEntry(timestamp, level, host, name, topic, body)
    except (UnicodeDecodeError, ValueError):
        return report_parse_err()


def _format(log: LogEntry) -> str:
    timestamp, level, host, name, topic, body = log
    if topic:
        m = "{} [{}]\t{}{}{}{}{}\t{}".format(
            timestamp, level, host, NAME_DELIM, name, TOPIC_DELIM, topic, body
        )
    else:
        m = "{} [{}]\t{}{}{}\t{}".format(timestamp, level, host, NAME_DELIM, name, body)
    return m


def format_log(log: LogEntry) -> str:
    """Format the log entry. Returns \n-terminated string."""

    m = _format(log)
    if not m.endswith("\n"):
        m += "\n"
    return m


def _level_to_int(levelname: str) -> int:
    if levelname in logging._nameToLevel:
        return logging._nameToLevel[levelname]
    print(f"[ERROR] invalid log level name {levelname}")
    return logging.NOTSET


def should_show(filter_levelname: str, log: T.Optional[LogEntry]) -> bool:
    """Return True if `log` should be shown with `filter_level_name`."""

    if log is None:
        return False
    return _level_to_int(log.level) >= _level_to_int(filter_levelname)


_LEVEL_TO_TERM_COLOR = {
    logging.NOTSET: None,
    logging.DEBUG: None,
    logging.INFO: "g",
    logging.WARNING: "y",
    logging.ERROR: "r",
    logging.CRITICAL: "R",
}


def _level_to_term_color(levelname: str) -> T.Optional[str]:
    return _LEVEL_TO_TERM_COLOR[_level_to_int(levelname)]


_LEVEL_TO_HTML_COLOR = {
    logging.NOTSET: None,
    logging.DEBUG: None,
    logging.INFO: "#8fc029",
    logging.WARNING: "#d4c96e",
    logging.ERROR: "#dc2566",
    logging.CRITICAL: "#fa2772",
}


def _level_to_html_color(levelname: str) -> T.Optional[str]:
    return _LEVEL_TO_HTML_COLOR[_level_to_int(levelname)]


def format_log_term_color(log: LogEntry) -> str:
    """Fancy colored format for terminals. Returns \n-terminated string."""

    m = _format(log)
    if not m.endswith("\n"):
        m += "\n"
    color = _level_to_term_color(log.level)
    if color:
        m = col(m, fg=color)

    return m


def format_log_html_color(log: LogEntry) -> str:
    """Fancy colored format as HTML (for Qt etc.). Returns rstripped (NON-\n-terminated) string."""

    m = _format(log)
    m = m.rstrip()
    color = _level_to_html_color(log.level)
    if color:
        m = f'<font color="{color}">{m}</font>'

    return m


class FileLogger(object):
    def __init__(self, filename, mode="a"):
        self.stream = open(filename, mode=mode)
        self._closed = False

    def write(self, s: str):
        self.stream.write(s)

    def write_log(self, msg):
        log = parse_log(msg)
        if log is None:
            return
        m = format_log(log)
        self.write(m)

    def __del__(self):
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True

        if self.stream:
            self.stream.flush()
            self.stream.close()


class LogClient(NodeClient):
    """Simple Log Client."""

    def __init__(
        self, gconf: dict, name, log_num=100, context=None, prefix=None, log_handler=None
    ):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self.log_num = log_num
        self._logbuf = []
        self.logbuf_lock = threading.Lock()

        subw = SubWorker(self.ctx)
        if log_handler is None:
            handler = self.handle_log
        elif callable(log_handler):
            handler = [self.handle_log, log_handler]
        elif isinstance(log_handler, (tuple, list)):
            handler = [self.handle_log] + list(log_handler)
        subw.add_sub(self.conf["xpub_endpoint"], handler=handler, deserial=False)
        self.start_subworker(subw)

    def handle_log(self, msg):
        m = parse_log(msg)

        # if log list is too long, just discard new log.
        if m is not None and len(self._logbuf) < self.log_num:
            with self.logbuf_lock:
                self._logbuf.append(m)

    def get_logs(self):
        with self.logbuf_lock:
            return copy.copy(self._logbuf)

    def pop_log(self):
        with self.logbuf_lock:
            if self._logbuf:
                return self._logbuf.pop(0)
            else:
                return None


class LogBroker(Node):
    """Broker Node for logs."""

    CLIENT = LogClient

    def __init__(self, gconf: dict, name, context=None):
        Node.__init__(self, gconf, name, context=context)

        self.ctx.add_broker(
            self.conf["xpub_endpoint"], self.conf["xsub_endpoint"], xsub_handler=self.xsub_handler
        )

        if self.conf.get("file", False):
            if "file_name" in self.conf:
                n = self.conf["file_name"]
            else:
                dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
                n = path.join(log_dir, "{}_{}.log".format(self.joined_name(), dt))
            self.file_logger = FileLogger(n)
            self.file_logger.write("# LogBroker started at {}\n".format(dt))
        else:
            self.file_logger = None

    def close_resources(self):
        if hasattr(self, "file_logger") and self.file_logger is not None:
            self.file_logger.close()

    def xsub_handler(self, msg):
        if self.file_logger is not None:
            self.file_logger.write_log(msg)
