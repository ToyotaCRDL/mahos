#!/usr/bin/env python3

"""
Communication library for mahos Node.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import pickle
import logging
import typing as T

import zmq

from ..msgs.common_msgs import pickle_proto, Message, Request, Resp
from ..util.typing import SubHandler, RepHandler

from .log import PUBHandler, DummyLogger


def serialize(msg: Message | T.Any) -> bytes:
    """Serialize a Message or any object to bytes.

    Default serialization method is pickle.
    The Message type can implement custom serialize() method
    (e.g. for data compression or foreign language interface).
    When custom serialization is used, deserializer must know how to deserialize it.
    For that, you must pass the Message type in add_sub(), add_req() or add_rep().

    """

    if isinstance(msg, Message):
        return msg.serialize()
    else:
        return pickle.dumps(msg, protocol=pickle_proto)


def deserialize(b: bytes, msg_type: T.Type[Message] | None) -> Message | T.Any:
    """Deserialize a bytes to reconstruct Message or any object.

    msg_type is None (unspecified), deserialize is done assuming default serialization
    method, pickle.
    Otherwise, classmethod msg_type.deserialize() is invoked.

    """

    if msg_type is not None:
        return msg_type.deserialize(b)
    else:
        return pickle.loads(b)


def get_logger(logger):
    if isinstance(logger, logging.Logger):
        return logger
    elif isinstance(logger, str):
        return DummyLogger(logger)
    else:
        return DummyLogger()


class Requester(object):
    """Class providing request() for REQ-REP pattern communication."""

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        linger_ms: int,
        timeout_ms: int | None = None,
        resp_type: T.Type[Resp] | None = None,
        logger=None,
    ):
        self.ctx = context
        self.endpoint = endpoint
        self.linger_ms = linger_ms
        self.timeout_ms = timeout_ms
        self.resp_type = resp_type
        self.logger = get_logger(logger)

        self.create_socket()

    def create_socket(self):
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, self.linger_ms)
        if self.timeout_ms is not None:
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.connect(self.endpoint)

        self._socket = sock
        return self._socket

    def request(self, msg: Request) -> Resp:
        """Send request and return response."""

        try:
            self._socket.send(serialize(msg))
            resp = deserialize(self._socket.recv(), self.resp_type)
        except zmq.ZMQError:
            # Comes here when timed-out for example.
            self.logger.exception("ZMQError in Requester.request().")
            self.logger.info("Creating new socket.")
            self.create_socket()
            return Resp(False, message="ZMQError in request()")
        return resp

    def close(self):
        self._socket.close()
        # Don't close self.ctx because Context will do that.


class Publisher(object):
    """Class providing publish() for PUB-SUB pattern communication."""

    def __init__(self, socket: zmq.Socket, topic: bytes, logger=None):
        self._socket = socket
        self.topic = bytes(topic)
        self.logger = get_logger(logger)

    def socket(self) -> zmq.Socket:
        return self._socket

    def publish(self, msg):
        """Publish the given message."""

        try:
            self._socket.send_multipart([self.topic, serialize(msg)])
        except zmq.ZMQError:
            self.logger.exception("ZMQError in Publisher.publish().")

    def close(self):
        self._socket.close()


def null_handler(msg):
    pass


class Context(object):
    """The communication context for mahos Node.

    :param context: if passed, internal ZMQ context is shared.
    :param poll_timeout_ms: polling timeout in milliseconds.

    Current implementation is based on ZMQ.

    The communications are divided into two classes:
    1. Inbound (rep, sub, broker)
    2. Outbound (req, pub)

    On adding inbound communication, register a function `handler`.
    The `handler` is called upon inbound message reception.
    The poll() method does polling and handling of all registered inbound messages.
    (So, users don't have to care about it.)

    By adding outbound communication, a sender object (Requester or Publisher) is returned.
    Through the object, users can send the message (and get reply for Requester) on their timing.

    Note that context itself is thread-safe, but derived objects (ZMQ sockets) are not.
    Let's consider a use case:
    You start a sub-thread for inbound message polling (in a loop),
    and use main-thread for outbound message sending
    (and provide API to get inbound message, etc.).
    In this case, only sub-thread should add inbound messages and use poll() method;
    and only main-thread should add outbound messages and use sender objects.
    See mahos.node.client for example implementations.

    """

    def __init__(
        self,
        context: "Context" | zmq.Context | None = None,
        poll_timeout_ms: int | None = None,
        linger_ms: int | None = None,
    ):
        if isinstance(context, Context):
            self.ctx = context.zmq_context()
        elif isinstance(context, zmq.Context):
            self.ctx = context
        else:
            self.ctx = zmq.Context()

        if poll_timeout_ms is None:
            self.poll_timeout_ms = 100
        else:
            self.poll_timeout_ms = poll_timeout_ms
        if linger_ms is None:
            self.linger_ms = 0
        else:
            self.linger_ms = linger_ms

        ## endpoint: socket
        self.requesters: dict[str, zmq.Socket] = {}
        self.publishers: dict[str, zmq.Socket] = {}
        ## socket: handler
        self.rep_handlers: dict[zmq.Socket, tuple[RepHandler, T.Type[Message] | None]] = {}
        self.sub_handlers: dict[zmq.Socket, tuple[SubHandler, T.Type[Message] | None, bool]] = {}
        self.broker_handlers = []

        self.poller = zmq.Poller()
        self._closed = False

    def close(self, close_zmq_ctx=False):
        """Close this context."""

        if self._closed:
            return
        self._closed = True

        # Close the sockets explicitly.
        for s in tuple(self.requesters.values()) + tuple(self.publishers.values()):
            s.close()
        for s in tuple(self.rep_handlers.keys()) + tuple(self.sub_handlers.keys()):
            s.close()
        for xpub, xsub, _, _ in self.broker_handlers:
            xpub.close()
            xsub.close()

        # Since close_zmq_ctx is False by default, we don't terminate zmq context here.
        # But this will be done in zmq context's destructor.
        if close_zmq_ctx:
            self.ctx.term()

    def zmq_context(self):
        """Get internal ZMQ context."""

        return self.ctx

    def add_req(
        self,
        endpoint: str,
        timeout_ms: int | None = None,
        resp_type: T.Type[Resp] | None = None,
        logger=None,
    ) -> Requester:
        """Add and return a Requester."""

        if endpoint in self.requesters:
            print(f"[WARN] Requester at {endpoint} has already been added.")
            return self.requesters[endpoint]

        r = Requester(
            self.ctx,
            endpoint,
            self.linger_ms,
            timeout_ms=timeout_ms,
            resp_type=resp_type,
            logger=logger,
        )
        self.requesters[endpoint] = r
        return r

    def add_rep(
        self, endpoint: str, handler=null_handler, req_type: T.Type[Message] | None = None
    ):
        """Add rep handler."""

        sock = self.ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, self.linger_ms)
        sock.bind(endpoint)
        self.poller.register(sock, zmq.POLLIN)
        self.rep_handlers[sock] = (handler, req_type)

    def add_pub(self, endpoint: str, topic: bytes | str, logger=None) -> Publisher:
        """Add and return a Publisher."""

        if endpoint in self.publishers:
            sock = self.publishers[endpoint].socket()
        else:
            sock = self.ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.LINGER, self.linger_ms)
            sock.bind(endpoint)
        p = Publisher(sock, self._topic_to_bytes(topic), logger=logger)
        self.publishers[endpoint] = p
        return p

    def add_sub(
        self,
        endpoint: str,
        topic: bytes | str = b"",
        handler=null_handler,
        msg_type: T.Type[Message] | None = None,
        deserial=True,
    ):
        """Add sub handler.

        :param msg_type: Type (class object) of expected message.
            Some value must be passed if custom-serialization will be received.
            Otherwise, it can be omitted.

        """

        sock = self.ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, self.linger_ms)
        sock.setsockopt(zmq.SUBSCRIBE, self._topic_to_bytes(topic))
        sock.connect(endpoint)
        self.poller.register(sock, zmq.POLLIN)
        self.sub_handlers[sock] = (handler, msg_type, deserial)

    def add_pub_handler(self, endpoint: str, root_topic: str = ""):
        """Add PUBHandler for logging."""

        sock = self.ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, self.linger_ms)
        sock.connect(endpoint)  # pub but connect because targeting xpub/xsub broker.
        return PUBHandler(sock, root_topic=root_topic)

    def add_broker(
        self,
        xpub_endpoint: str,
        xsub_endpoint: str,
        xpub_handler=null_handler,
        xsub_handler=null_handler,
    ):
        """Add broker (xpub/xsub) handlers."""

        xpub = self.ctx.socket(zmq.XPUB)
        xsub = self.ctx.socket(zmq.XSUB)
        xpub.setsockopt(zmq.LINGER, self.linger_ms)
        xsub.setsockopt(zmq.LINGER, self.linger_ms)
        xpub.bind(xpub_endpoint)
        xsub.bind(xsub_endpoint)
        self.poller.register(xpub, zmq.POLLIN)
        self.poller.register(xsub, zmq.POLLIN)
        self.broker_handlers.append((xpub, xsub, xpub_handler, xsub_handler))

    def _topic_to_bytes(self, topic: bytes | str) -> bytes:
        if isinstance(topic, bytes):
            return topic
        elif isinstance(topic, str):
            if not topic.isascii():
                raise ValueError("topic (in str) must be ASCII.")
            return topic.encode()
        else:
            raise TypeError("topic must be bytes or str")

    def _handle_rep(self, socks):
        for sock, (handler, req_type) in self.rep_handlers.items():
            if sock in socks:
                msg = deserialize(sock.recv(), req_type)
                resp = handler(msg)
                sock.send(serialize(resp))

    def _handle_sub(self, socks):
        for sock, (handler, msg_type, deserial) in self.sub_handlers.items():
            if sock in socks:
                if not deserial:
                    handler(sock.recv_multipart())
                    continue

                frames = sock.recv_multipart()
                if len(frames) == 2:
                    topic, msg = frames
                    handler(deserialize(msg, msg_type))
                else:
                    # this should not happen as ZMQ assures multipart message to be atomic.
                    print(f"[ERROR] {len(frames)} parts received instead of 2.")
                    for f in frames:
                        print(f[:10], end="")
                    print()

    def _handle_broker(self, socks):
        for xpub, xsub, xpub_handler, xsub_handler in self.broker_handlers:
            if xpub in socks:
                msg = xpub.recv_multipart()
                xsub.send_multipart(msg)
                xpub_handler(msg)
            if xsub in socks:
                msg = xsub.recv_multipart()
                xpub.send_multipart(msg)
                xsub_handler(msg)

    def poll(self):
        """Poll inbound sockets and call corresponding handlers."""

        socks = dict(self.poller.poll(self.poll_timeout_ms))
        self._handle_rep(socks)
        self._handle_sub(socks)
        self._handle_broker(socks)
