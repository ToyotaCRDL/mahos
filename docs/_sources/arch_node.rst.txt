Node, Client, and Communication
===============================

MAHOS helps to build `distributed systems`, which consists of multiple programs (processes) communicating each other.
We call these programs :term:`nodes <node>` (:class:`Node <mahos.node.node.Node>`).
A node usually provides some services for other nodes.
The :term:`node client` (:class:`NodeClient <mahos.node.client.NodeClient>`) provides the standard way to access node's functions.
The core functionalities of node systems are implemented in :ref:`mahos.node` package.

The node :class:`communication <mahos.node.comm.Context>` is currently based on `ZeroMQ <https://zeromq.org/>`_ library.
The transport layer is abstracted by ZeroMQ, and we can choose different transports by changing a configuration (the endpoint) only.
However, we assume TCP unless otherwise specified.
By choosing TCP, multi-host (multi-computer) configuration is quite straightforward.
ZeroMQ doesn't specify the data serialization format; it provides methods to send / receive byte array of arbitrary length.
We adopt Python standard pickle for serialization.

Communication patterns
----------------------

We use two different communication patterns.

Req-Rep
^^^^^^^

:term:`Req-Rep` (Request-Reply) pattern is synchronous :term:`RPC`.
First, the client sends a request to the server (node).
The server node does something accordingly, and send back a reply to the client.

Pub-Sub
^^^^^^^

:term:`Pub-Sub` (Publish-Subscribe) pattern is asynchronous, one-to-many data distribution.
The message regarding a topic is published by single node (Publisher).
Here, the topic is a label to distinguish the data type.
Publisher can publish multiple topics from single endpoint.
The subscriber which `subscribes to` a topic receives the messages.
The number of subscribers can be changed dynamically,
and zero subscriber is a valid state too (message is just not sent to anywhere in this case).

Client usage
------------

It can be said that a node `provides` functions via Rep and Pub;
corresponding client uses them via Req and Sub.
Thus, :class:`Node <mahos.node.node.Node>` has ways to :meth:`add_rep <mahos.node.node.Node.add_rep>`
and :meth:`add_pub <mahos.node.node.Node.add_pub>`.
:class:`NodeClient <mahos.node.client.NodeClient>` has :meth:`add_req <mahos.node.client.NodeClient.add_req>`
and :meth:`add_sub <mahos.node.client.NodeClient.add_sub>`.

A nodes can internally use the clients to access the others.
The custom programs can utilize them as well.
This relationships are visualized in the figure below.

.. figure:: ./img/mahos-node-server-client.svg
   :alt: Connection between Nodes and Clients
   :width: 85%

   Connection between Nodes and Clients
