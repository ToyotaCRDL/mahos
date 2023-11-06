Node, Client, and Communication
===============================

MAHOS helps to build `distributed` measurement automation systems, which consist of multiple programs (processes) communicating with each other.
We call these programs :term:`nodes <node>` (:class:`Node <mahos.node.node.Node>`).
A node usually provides some services for other nodes.
The :term:`node client` (:class:`NodeClient <mahos.node.client.NodeClient>`) provides the standard way to access the node's functions.
The core functionalities of node systems are implemented in the :ref:`mahos.node` package.

Data transport and serialization
--------------------------------

The node :class:`communication <mahos.node.comm.Context>` is currently based on the `ZeroMQ <https://zeromq.org/>`_ library.
The transport layer is abstracted by ZeroMQ, and we can choose different transports by changing a configuration (the endpoint) only.
We recommend using TCP as the default.
By choosing TCP, multi-host (multi-computer) configuration is quite straightforward.
However, the transportation overhead can become an issue when one deals with very large data
such as high-resolution images produced at a high rate.
In such a case, the overhead can be reduced significantly by running the relevant nodes as threads
in a single process and using intra-process transportation.

ZeroMQ doesn't specify the data serialization format; it provides methods to send/receive byte arrays of arbitrary length.
By default, we define the message types (classes) in the :ref:`mahos.msgs` package and serialize the instances
using Python standard pickle.
This approach is adopted due to pickle's high compatibility with Python objects and moderate performance.

However, the pickle-based serialization practically limits the messaging within Python only.
The other serialization can also be utilized to support different programming languages.
For that, serializer methods must be overridden in your :class:`Message <mahos.msgs.common_msgs.Message>` classes
and they must be passed when :meth:`add_sub <mahos.node.client.NodeClient.add_sub>`,
:meth:`add_req<mahos.node.client.NodeClient.add_req>`, or :meth:`add_rep <mahos.node.node.Node.add_rep>`.
See ``examples/custom_serializer`` for the examples of this approach.

Communication patterns
----------------------

We use two different communication patterns.

Req-Rep
^^^^^^^

:term:`Req-Rep` (Request-Reply) pattern is synchronous :term:`RPC`.
First, the client sends a request to the server (node).
The server node does something accordingly and sends back a reply to the client.

Pub-Sub
^^^^^^^

:term:`Pub-Sub` (Publish-Subscribe) pattern is asynchronous, one-to-many data distribution.
The message regarding a topic is published by a single node (Publisher).
Here, the topic is a label to distinguish the data type.
A publisher can publish multiple topics from a single endpoint.
The subscriber which `subscribes to` a topic receives the messages.
The number of subscribers can be changed dynamically,
and zero subscribers is a valid state too (the message is just not sent anywhere in this case).

Client usage
------------

It can be said that a node `provides` functions via Rep and Pub;
the corresponding client uses them via Req and Sub.
Thus, :class:`Node <mahos.node.node.Node>` has ways to :meth:`add_rep <mahos.node.node.Node.add_rep>`
and :meth:`add_pub <mahos.node.node.Node.add_pub>`.
:class:`NodeClient <mahos.node.client.NodeClient>` has :meth:`add_req <mahos.node.client.NodeClient.add_req>`
and :meth:`add_sub <mahos.node.client.NodeClient.add_sub>`.

Nodes can internally use the clients to access the others.
The custom programs can utilize them as well.
This relationships are visualized in the figure below.

.. figure:: ./img/mahos-node-server-client.svg
   :alt: Connection between Nodes and Clients
   :width: 85%

   Connection between Nodes and Clients
