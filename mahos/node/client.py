#!/usr/bin/env python3

"""
Clients for mahos Node.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import time
import threading
import importlib

from .comm import Context, Requester
from .node import join_name, split_name, infer_name, local_conf, load_gconf, get_value
from .log import init_topic_logger
from ..msgs.common_msgs import Message, StateReq, State, Status
from ..msgs.data_msgs import Data
from ..util.typing import NodeName, SubHandler, MessageGetter


def init_node_client(
    gconf: dict, name: NodeName, context: Context | None = None
) -> tuple[str, str, dict, Context]:
    """initialize NodeClient-like class without inheriting it.

    We must inherit QObject for GUI clients and we want avoid multiple inheritance.
    (i.e. class QSomeClient(NodeClient, QtCore.QObject): )

    Use this function at the constructor of such class like below:
    self._host, self._name, self.conf, self.ctx = init_node_client(gconf, name, context)

    """

    host, name = split_name(name)
    conf = gconf[host][name]
    ctx = Context(context=context, poll_timeout_ms=get_value(gconf, conf, "poll_timeout_ms"))

    return host, name, conf, ctx


class HandlerWrapper(object):
    """Wrap custom handler along with the default handler.

    Default handler just stores the latest message and provide get_latest_msg() function.
    Other handler(s) can be added to custom the callback behaviour.

    """

    def __init__(self, handler: SubHandler | list[SubHandler] | None = None):
        self.lock = threading.Lock()
        self.latest_msg = None

        if handler is None:
            self.handlers = []
        elif callable(handler):
            self.handlers = [handler]
        elif isinstance(handler, (list, tuple)):
            self.handlers = handler
        else:
            raise TypeError("type of handler is SubHandler | list[SubHandler] | None")

    def handler(self, msg):
        with self.lock:
            self.latest_msg = msg
        for handler in self.handlers:
            handler(self.latest_msg)

    def get_latest_msg(self) -> Message:
        with self.lock:
            return self.latest_msg


class SubWorker(object):
    def __init__(self, context: Context):
        self.ctx = context
        self.shutdown_ev = threading.Event()
        self.handler_wrappers = []

    def add_sub(
        self, endpoint: str, topic: bytes | str = b"", handler=None, deserial=True
    ) -> MessageGetter:
        hw = HandlerWrapper(handler=handler)
        self.ctx.add_sub(endpoint, topic=topic, handler=hw.handler, deserial=deserial)
        self.handler_wrappers.append(hw)
        return hw.get_latest_msg

    def close(self):
        self.shutdown_ev.set()

    def main(self):
        while not self.shutdown_ev.is_set():
            self.ctx.poll()


class NodeClient(object):
    """Client to use Node's function.

    :param gconf: global config dict.
    :param name: name of target node.
    :param context: communication context.
    :param prefix: name prefix for logger.

    :ivar conf: target node's local config.
    :ivar ctx: communication context.
    :ivar logger: configured logger.

    """

    #: The module that defines message types for target node.
    M = None

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        prefix: str | None = None,
    ):
        self._host, self._name, self.conf, self.ctx = init_node_client(
            gconf, name, context=context
        )

        topic = "{}({})".format(self.__class__.__name__, self.joined_name())
        self.logger, self._full_name = init_topic_logger(topic, prefix)

        self._subscribers = []
        self._closed = False

    def __repr__(self):
        return self._full_name

    def __del__(self):
        self.close()

    def close(self, close_ctx=True):
        """Close this client."""

        if self._closed:
            return
        self._closed = True

        for sub, thread in self._subscribers:
            sub.close()
            thread.join()
        if close_ctx:
            self.ctx.close()

    def add_sub(
        self,
        topic_handlers: list[tuple[bytes | str, SubHandler | None]],
        endpoint: str = "pub_endpoint",
    ) -> list[MessageGetter]:
        """Subscribe to topics from `endpoint` registering the handlers.

        To do so, add and start a SubWorker for this client.

        :param topic_handlers: List of Tuple[topic, handler] or topic. latter case,
                               handler is considered None.
        :returns: List of MessageGetters. Length is same as topic_handlers.

        """

        subw = SubWorker(self.ctx)
        getters = []
        for topic_handler in topic_handlers:
            if isinstance(topic_handler, (bytes, str)):
                topic, handler = topic_handler, None
            else:
                topic, handler = topic_handler
            getters.append(subw.add_sub(self.conf[endpoint], topic, handler))
        self.start_subworker(subw)
        return getters

    def start_subworker(self, subw: SubWorker):
        """Start a thread for a SubWorker."""

        thread = threading.Thread(target=subw.main)
        thread.start()
        self._subscribers.append((subw, thread))

    def name(self) -> tuple[str, str]:
        """Get target node name as a tuple (hostname, nodename)."""

        return (self._host, self._name)

    def joined_name(self) -> str:
        """Get target node name as single string 'hostname::nodename'."""

        return join_name((self._host, self._name))

    def full_name(self) -> str:
        """Get full name: (prefix->)ClientClassName(target_node.joined_name())"""

        return self._full_name

    def get_status(self):
        """Get status of node: default implementation is void and raises NotImplementedError."""

        raise NotImplementedError("get_status() is not defined")

    def wait(self):
        """Wait until the target is ready.

        get_status() must be implemented to use this.

        """

        while self.get_status() is None:
            time.sleep(1e-3 * self.ctx.poll_timeout_ms)

    def is_up(self) -> bool:
        """Check if the target is up.

        get_status() must be implemented to use this.

        """

        return self.get_status() is not None

    def add_req(self, gconf: dict, endpoint: str = "rep_endpoint") -> Requester:
        """Add and return a Requester for `endpoint`.

        Request timeout (req_timeout_ms) is searched from self.conf or `gconf`.

        """

        return self.ctx.add_req(
            self.conf[endpoint],
            timeout_ms=get_value(gconf, self.conf, "req_timeout_ms"),
            logger=self.logger,
        )


class StatusSubscriber(NodeClient):
    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        prefix: str | None = None,
        status_handler=None,
    ):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self._get_status = self.add_sub([(b"status", status_handler)])[0]

    def get_status(self) -> Status:
        return self._get_status()


class StatusDataSubscriber(NodeClient):
    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        prefix: str | None = None,
        status_handler=None,
        data_handler=None,
    ):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self._get_status, self._get_data = self.add_sub(
            [(b"status", status_handler), (b"data", data_handler)]
        )

    def get_status(self) -> Status:
        return self._get_status()

    def get_data(self) -> Data:
        return self._get_data()


class StatusClient(StatusSubscriber):
    """Client with a subscriber to status and a requester."""

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        prefix: str | None = None,
        status_handler=None,
    ):
        StatusSubscriber.__init__(
            self, gconf, name, context=context, prefix=prefix, status_handler=status_handler
        )
        self.req = self.add_req(gconf)


class StatusDataClient(StatusDataSubscriber):
    """Client with a subscriber to status & data, and a requester."""

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        prefix: str | None = None,
        status_handler=None,
        data_handler=None,
    ):
        StatusDataSubscriber.__init__(
            self,
            gconf,
            name,
            context=context,
            prefix=prefix,
            status_handler=status_handler,
            data_handler=data_handler,
        )
        self.req = self.add_req(gconf)


class StateClientMixin(object):
    """Implements change_state() and get_state()."""

    def change_state(self, state: State, params=None) -> bool:
        resp = self.req.request(StateReq(state, params=params))
        return resp.success

    def get_state(self) -> State:
        s = self.get_status()
        return s.state if s is not None else None


class StateClient(StatusClient, StateClientMixin):
    """Client for Stateful node.

    Client with a subscriber to status (with state inside) and a requester.

    """

    pass


class EchoSubscriber(NodeClient):
    """ "Subscriber to print incoming messages."""

    def __init__(
        self,
        gconf: dict,
        name: NodeName,
        context: Context | None = None,
        prefix: str | None = None,
        topic=b"status",
        rate=False,
    ):
        NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

        self.msg_count = -1
        self.add_sub([(topic, self.print_rate if rate else lambda msg: print(msg))])

    def print_rate(self, msg):
        self.msg_count += 1
        if self.msg_count == 0:
            self.start_time = time.perf_counter()
            return
        elapsed = time.perf_counter() - self.start_time
        rate = self.msg_count / elapsed
        print(f"{self.msg_count} msgs in {elapsed:.1f} sec.: {rate:.2f} Hz")


def get_client(
    name_or_nodename: str, conf_path: str = "conf.toml", host: str = "localhost", **kwargs
) -> NodeClient:
    gconf = load_gconf(conf_path)
    name = infer_name(name_or_nodename, default_host=host)
    lconf = local_conf(gconf, name)
    module = importlib.import_module(lconf["module"])
    NodeClass = getattr(module, lconf["class"])
    if NodeClass.CLIENT is None:
        print("[ERROR] {} doesn't define CLIENT.".format(lconf["class"]))
        return None

    ClientClass = NodeClass.CLIENT
    return ClientClass(gconf, name, **kwargs)
