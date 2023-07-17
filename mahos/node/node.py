#!/usr/bin/env python3

"""
Base implementation for mahos Node.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import multiprocessing as mp
import threading as mt
import logging

import toml

from .comm import Context, Publisher, Request
from .log import DummyLogger
from ..util.typing import NodeName, RepHandler
from ..msgs.common_msgs import Resp


# name

NAME_DELIM = "::"


def join_name(names_or_joined: NodeName) -> str:
    n = names_or_joined
    if isinstance(n, str):
        return n
    elif isinstance(n, (tuple, list)) and len(n) == 2:
        return NAME_DELIM.join(n)
    else:
        raise ValueError("names must be string or 2-tuple of string.")


def _split(joined_name: str) -> tuple:
    t = tuple(joined_name.split(NAME_DELIM))
    if len(t) != 2:
        raise ValueError(f"name must contain single '{NAME_DELIM}'")
    return t


def split_name(names_or_joined: NodeName) -> tuple[str, str]:
    n = names_or_joined
    if isinstance(n, str):
        return _split(n)
    elif isinstance(n, (tuple, list)) and len(n) == 2:
        return n
    else:
        raise ValueError("name must be string or 2-tuple of string.")


def infer_name(name_or_nodename: str, default_host="localhost") -> tuple[str, str]:
    if isinstance(name_or_nodename, str) and NAME_DELIM not in name_or_nodename:
        name = (default_host, name_or_nodename)
    else:
        name = name_or_nodename
    return split_name(name)


# config


class PickleableTomlDecoder(toml.TomlDecoder):
    """TomlDecoder to fix issues of toml. https://github.com/uiri/toml/issues/362"""

    def get_empty_inline_table(self):
        return self.get_empty_table()


def load_gconf(fn: str) -> dict:
    return toml.load(fn, decoder=PickleableTomlDecoder())


def local_conf(gconf: dict, name: NodeName):
    names = split_name(name)
    return gconf[names[0]][names[1]]


GLOBAL_KEY = "global"
THREAD_KEY = "thread"


class HostIterator(object):
    def __init__(self, gconf: dict):
        self.it = iter(gconf)

    def __iter__(self):
        return self

    def __next__(self):
        key = next(self.it)
        while key in (GLOBAL_KEY, THREAD_KEY):
            key = next(self.it)
        return key


class HostNodeIterator(object):
    def __init__(self, gconf: dict):
        self.gconf = gconf
        self.host_it = HostIterator(gconf)
        self.node_it = iter([])
        self.host = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            node = next(self.node_it)
        except StopIteration:
            self.host = next(self.host_it)
            self.node_it = iter(self.gconf[self.host])
            node = next(self.node_it)
        return self.host, node


def hosts(gconf: dict) -> HostIterator:
    """return an iterator over host names in gconf."""

    return HostIterator(gconf)


def host_nodes(gconf: dict) -> HostNodeIterator:
    """return an iterator over (host, node) names in gconf."""

    return HostNodeIterator(gconf)


def threaded_nodes(gconf: dict, host: str) -> dict[str, list[str]]:
    """return dictionary of threaded nodes (threaded_nodes_name, nodes) for host in gconf."""

    try:
        return gconf[THREAD_KEY][host]
    except KeyError:
        return {}


def get_value(gconf: dict, lconf: dict, key: str, default=None):
    """get a config value from lconf or gconf. return default if there's no config."""

    if key in lconf:
        return lconf[key]
    elif GLOBAL_KEY in gconf and key in gconf[GLOBAL_KEY]:
        return gconf[GLOBAL_KEY][key]
    else:
        return default


# Node classes


class NodeBase(object):
    """The base class for Nodes. Implements basics: configuration and name.

    :param gconf: global config dict.
    :param name: name of this node.

    :ivar conf: node's local config.

    """

    def __init__(self, gconf: dict, name: NodeName):
        self._host, self._name = split_name(name)
        self.conf = local_conf(gconf, name)

    def names(self) -> tuple[str, str]:
        """Get node name as a tuple (hostname, nodename)."""

        return (self._host, self._name)

    def joined_name(self) -> str:
        """Get node name as single string 'hostname::nodename'."""

        return join_name((self._host, self._name))

    def main(self):
        pass


_str_to_log_level = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def init_logger(gconf, my_name, log_name, context: Context):
    # avoid invoking logging.basicConfig() through functions such as logging.info etc.
    # 3rd party libraries could call these functions.
    logging.root.addHandler(logging.NullHandler())

    my_conf = local_conf(gconf, my_name)
    log_conf = local_conf(gconf, log_name)

    if "log_level" in my_conf:
        level = _str_to_log_level[my_conf["log_level"].upper()]
    else:
        level = logging.INFO

    joined_name = join_name(my_name)
    handler = context.add_pub_handler(log_conf["xsub_endpoint"], root_topic=joined_name)
    logger = logging.getLogger(joined_name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


class Node(NodeBase):
    """Node is the main type of Nodes that works in the loop and communicates each other.

    Node implements main() function that is iterated until shutdown.
    If loop should be at constant rate (i.e. not busy loop),
    the rate management should be done inside main() function.

    :param gconf: global config dict.
    :param name: name of this node.

    :ivar conf: node's local config.
    :ivar ctx: communication context.
    :ivar logger: configured logger.

    """

    #: The class for standard client.
    CLIENT = None

    def __init__(self, gconf: dict, name: NodeName, context: Context | None = None):
        NodeBase.__init__(self, gconf, name)
        self.ctx = Context(
            context=context, poll_timeout_ms=get_value(gconf, self.conf, "poll_timeout_ms")
        )
        if "target" in self.conf and "log" in self.conf["target"]:
            self.logger = init_logger(gconf, name, self.conf["target"]["log"], self.ctx)
        else:
            self.logger = DummyLogger(join_name(name))
        self._clients = []
        self._closed = False
        self._shutdown = False

    def __del__(self):
        self.close()

    def close(self):
        """Close this node."""

        if self._closed:
            return
        self._closed = True

        self.close_resources()
        self._close_comm()

    def close_resources(self):
        """Close custom resources."""

        pass

    def _close_comm(self, close_ctx=True):
        """Close inter-node communication (clients and context)."""

        for cli in self._clients:
            cli.close(close_ctx=False)

        if close_ctx:
            self.ctx.close()

    def set_shutdown(self):
        """Shutdown this node, quitting from the main loop."""

        self._shutdown = True

    def wait(self):
        """Wait until required resources are ready."""

        pass

    def add_client(self, cli):
        """Register an internal NodeClient."""

        self._clients.append(cli)

    def add_clients(self, *clis):
        """Register multiple internal NodeClients."""

        self._clients.extend(clis)

    def add_rep(self, handler: RepHandler | None = None, endpoint: str = "rep_endpoint"):
        """Add the reply handler at `endpoint`."""

        if handler is None:
            handler = self._handle_req
        self.ctx.add_rep(self.conf[endpoint], handler)

    def add_pub(self, topic: bytes | str, endpoint: str = "pub_endpoint") -> Publisher:
        """Add and return a Publisher for `topic` at `endpoint`."""

        return self.ctx.add_pub(self.conf[endpoint], topic, self.logger)

    def _handle_req(self, msg: Request) -> Resp:
        """The default RepHandler to wrap handle_req()."""

        try:
            return self.handle_req(msg)
        except Exception:
            msg = "Exception raised while handling request."
            self.logger.exception(msg)
            return Resp(False, msg)

    def handle_req(self, msg: Request) -> Resp:
        return Resp(False, "Handler is not implemented.")

    def poll(self):
        """Poll inbound messages and call corresponding handlers.

        this function can consume poll_timeout_ms if any inbound communication is registered.

        """

        self.ctx.poll()

    def main(self):
        """Main procedure that will be looped.

        Default implementation calls poll().

        """

        self.poll()

    def main_event(self, shutdown_ev: mp.Event | mt.Event):
        """Main loop that can be stopped by shutdown_ev."""

        try:
            self.wait()
            while not (shutdown_ev.is_set() or self._shutdown):
                self.main()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exitting {}.".format(self.joined_name()))
        finally:
            self.close()

    def main_interrupt(self):
        """Main loop that can be stopped by KeyboardInterrupt."""

        try:
            self.wait()
            while not self._shutdown:
                self.main()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exitting {}.".format(self.joined_name()))
        finally:
            self.close()

    def fail_with(self, msg) -> Resp:
        """log error and return failing response with same message."""

        self.logger.error(msg)
        return Resp(False, msg)


def run_node_proc(NodeClass, gconf: dict, name: NodeName, shutdown_ev: mp.Event):
    c: Node = NodeClass(gconf, name)
    c.main_event(shutdown_ev)


def run_node_thread(
    NodeClass, gconf: dict, name: NodeName, context: Context, shutdown_ev: mt.Event
):
    c: Node = NodeClass(gconf, name, context=context)
    c.main_event(shutdown_ev)


def start_node_proc(
    ctx: mp.context.BaseContext, NodeClass, gconf: dict, name: NodeName
) -> (mp.Process, mp.Event):
    shutdown_ev = mp.Event()
    proc = ctx.Process(
        target=run_node_proc, args=(NodeClass, gconf, name, shutdown_ev), name=join_name(name)
    )
    proc.start()
    return proc, shutdown_ev


def start_node_thread(
    ctx: Context, NodeClass, gconf: dict, name: NodeName
) -> (mt.Thread, mt.Event):
    shutdown_ev = mt.Event()
    thread = mt.Thread(
        target=run_node_thread,
        args=(NodeClass, gconf, name, ctx, shutdown_ev),
        name=join_name(name),
    )
    thread.start()
    return thread, shutdown_ev
