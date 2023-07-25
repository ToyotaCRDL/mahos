Tutorial 1: Communication
=========================

Preparation
-----------

Before starting this, :doc:`install <installation>` the library and dependencies.

Then, find the ``examples/comm`` directory in the mahos repository.
We will use four files (``conf.toml``, ``nodes.py``, ``print_clock.py``, and ``multipy_numbers.py``)  in it for this tutorial.
You can copy this directory to somewhere in your computer if you want.

Pub-Sub
-------

We will learn two types of comminucation methods.
First one here is Pub-Sub (Publish-Subscribe).
This is asynchronous, one-to-many data distribution.

Let's run a data-publishing node named clock by a command ``mahos run clock``.
And in another terminal, subscribe to and print the published message by a command ``mahos echo -t time clock``.
It is successful if you see something like below in the second terminal.

.. code-block::

   Subscribing to time of localhost::clock (config file: conf.toml).
   2023-02-16 16:30:49
   2023-02-16 16:30:50
   2023-02-16 16:30:51

You can test one-to-many communication by repeating ``mahos echo -t time clock`` in another terminal,
where you'd see the same output.
Hit Ctrl-C in the terminals to stop them.

Lets's see what was happening here.
The first command ``mahos run clock`` was abbreviated form of ``mahos run -c conf.toml -H localhost clock`` or ``mahos run -c conf.toml localhost::clock``.
``localhost`` is a hostname (computer name).
This will be important for multi-computer configuration,
but let's think it is always localhost because we are on single-computer during this tutorial.
``conf.toml`` is the name of configuration file (``conf.toml`` is the default name and can be omitted in the command),
which defines what the node ``clock`` is, like below.

.. code-block:: toml
   :linenos:
   :lineno-start: 1
   :caption: conf.toml

   [localhost.clock]
   module = "nodes"
   class = "Clock"
   pub_endpoint = "tcp://127.0.0.1:5566"

Line 1 defines the group in form ``[<hostname>.<nodename>]``; we are defining node named `clock` on host `localhost`.
The `module` and `class` (Line 2 and 3) assigns Python module (any importable module) and class names that defines the node.
The `pub_endpoint` is the endpoint for Pub-Sub communication.

Let's find the node implementation: ``Clock`` class defined in module nodes (file ``nodes.py``).

.. code-block:: python
   :linenos:
   :lineno-start: 13
   :caption: nodes.py

   class ClockClient(NodeClient):
       def __init__(self, gconf: dict, name: NodeName, context=None, prefix=None):
           NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

           self.get_time = self.add_sub(["time"])[0]


   class Clock(Node):
       CLIENT = ClockClient

       def __init__(self, gconf: dict, name: NodeName, context=None):
           Node.__init__(self, gconf, name, context=context)

           self.time_pub = self.add_pub("time")
           self.interval = IntervalSleeper(1.0)

       def main(self):
           self.interval.sleep()
           dt = datetime.datetime.now()
           self.time_pub.publish(dt.strftime("%F %T"))

As we can see, Clock is a subclass of Node and a few things are implemented.
Work is done by a :class:`Publisher <mahos.node.comm.Publisher>` initialized in Line 26, which is used to publish the formatted time in the main loop (Line 32).
The publisher is initialized with a topic (label for the messages) `time`.
Note that the type of the topic must be ASCII str (or bytes).

"You said main loop but I don't see any `for` or `while`" ?
The loop is implemented elsewhere and automatically incorporated when using ``mahos run`` command.
We always define `the contents of main loop` as the ``main()`` function.
IntervalSleeper regulates the loop interval to 1.0 secs (rate of 1 Hz).

Although :class:`Publisher <mahos.node.comm.Publisher>` can send any picklable Python object (we are sending str now),
the dedicated types can be used for serious applications (see :mod:`mahos.msgs.common_msgs`).

Corresponding NodeClient ``ClockClient`` is  defined above,
and this is referenced as a class variable ``CLIENT`` in Clock.
This is used to look up the client class from config file.
Implementation of ClockClient is even simpler; it just registers a subscriber for `time`.

See file ``print_clock.py`` to see how to use the client from a custom script.

Req-Rep
-------

Req-Rep (Request-Reply) is the second way of the node communication.
This is synchronous :term:`RPC`.

Let's run a node `multiplier`, and start a IPython shell by a command ``mahos shell multiplier``.
In the IPython shell, use ``cli.multiply(2, 3)`` to multiply the numbers.
It is successful if you get correct answer 6, and see a log message in the terminal running the node.

``cli`` is a MultiplierClient (defined as below) in the IPython shell.
Obviously, ``MultiplierClient`` registers a :class:`Requester <mahos.node.comm.Requester>` (Line 39) and
using in the ``multiply()`` method (Line 42).
By calling :meth:`Requester.request <mahos.node.comm.Requester.request>`, a request is sent from the client to serving node, and the response is returned.
The ``Multiplier`` node defines how the request is handled.
A handler method ``handle_multiply`` is registered in Line 51.
This method does the calculation, send a log message, and return the answer (Line 55-57).

.. code-block:: python
   :linenos:
   :lineno-start: 35
   :caption: nodes.py

   class MultiplierClient(NodeClient):
       def __init__(self, gconf: dict, name: NodeName, context=None, prefix=None):
           NodeClient.__init__(self, gconf, name, context=context, prefix=prefix)

           self.req = self.add_req(gconf)

       def multiply(self, a: int, b: int) -> int:
           return self.req.request((a, b))


   class Multiplier(Node):
       CLIENT = MultiplierClient

       def __init__(self, gconf: dict, name: NodeName, context=None):
           Node.__init__(self, gconf, name, context=context)

           self.add_rep(self.handle_multiply)

       def handle_multiply(self, req: T.Tuple[int, int]) -> int:
           a, b = req
           rep = a * b
           self.logger.info(f"{a} * {b} = {rep}")
           return rep

See file ``multiply_numbers.py`` to see how to use the client from a custom script.

Exercise
--------

The Node classes ``A`` and ``B`` are defined in ``nodes.py``.
See the definitions, run both nodes, and interact with them.
What happens by calling ``cli.set_data()``?

Further Reading
---------------

* :doc:`arch_node`
* `zguide <https://zguide.zeromq.org/>`_ : The Guide for ZeroMQ library (used for node communication).
