Tutorial 2: Basic Measurement
=============================

Preparation
-----------

Before starting this, :doc:`install <installation>` the library and dependencies.

It is recommended to finish :doc:`tutorial_comm` before this.

Find the ``examples/ivcurve`` directory in the mahos repository.
We will use the files in it for this tutorial.
You can copy this directory to somewhere in your computer if you want.

Running the mock
----------------

Run all the nodes defined in our ``conf.toml`` by command ``mahos launch``.
You will see a GUI window popping out.
If you click the Start button, some noisy line (IV curve) is plotted and updated.
This is our mock for IV curve measurement.


The IV curve measurement is illustrated by the figure below.
We have two instruments `source` (Voltage Source) and `meter` (Current meter) to
measure IV characteristics of DUT (Device Under Test).

.. figure:: ./img/ivcurve-sch.svg
   :alt: Schematic of IV curve measurement
   :width: 50%

   Schematic of IV curve measurement

Visualizing config
------------------

Let's analyze below what's happening behind the scenes.
First, try visualizing our nodes by ``mahos graph``, and you will see a graph below.

.. figure:: ./img/ivcurve-nodes.svg
   :alt: Node graph for IV curve
   :width: 30%

   Node graph for IV curve

As visualized, the config file defines four nodes.
We will go through these from bottom to top (from top to bottom in ``conf.toml``).

The topmost group in ``conf.toml`` is named `global`.
This is special group to define the default value for all the nodes.
If the same key is defined for a node, the global value is just ignored.
Otherwise, the global value is used.
(The behaviour is similar to local and global variable in some programming languages.)

log
---

The second group is named `localhost.log`.
It is important to observe the log messages for the debugging or monitoring.
Since mahos adopted a distributed system,
the sources of logs (i.e., :term:`nodes <node>`) are running on multiple processes.
In order to sort out the distributed logs, it seems good to gather the logs to single node,
and then redistribute.
The :class:`LogBroker <mahos.node.log_broker.LogBroker>` is implemented for this purpose.

It is highly recommended to define a `log` in ``conf.toml``, as in the file for this tutorial.
You can see arrows labeled `log` are coming from `server` and `ivcurve` to the `log` node in the graph.
These arrows are corresponding to Line 15 and Line 37-38 in ``conf.toml``.

(In :doc:`tutorial_comm`, we have omitted this and used dummy loggers.)

server
------

The `server` (:class:`InstrumentServer <mahos.inst.server.InstrumentServer>`) is defined as below.

.. code-block:: toml
   :linenos:
   :lineno-start: 12
   :caption: conf.toml

   [localhost.server]
   module = "mahos.inst.server"
   class = "InstrumentServer"
   target = { log = "localhost::log" }
   log_level = "DEBUG"
   rep_endpoint = "tcp://127.0.0.1:5559"
   pub_endpoint = "tcp://127.0.0.1:5560"

   [localhost.server.instrument.source]
   module = "instruments"
   class = "VoltageSource_mock"
   [localhost.server.instrument.source.conf]
   resource = "VISA::DUMMY0"

   [localhost.server.instrument.meter]
   module = "instruments"
   class = "Multimeter_mock"
   [localhost.server.instrument.meter.conf]
   resource = "VISA::DUMMY1"

:class:`InstrumentServer <mahos.inst.server.InstrumentServer>` is the node for :doc:`arch_inst` to provide :term:`RPC` for instrument drivers.
Thus, you don't need to write a :term:`node` for this purpose; you write instrument driver classes (:class:`Instrument <mahos.inst.instrument.Instrument>`) instead.
The second group above ``[localhost.server.instrument.source]`` defines
an instrument `source` inside the `server`.
The `VoltageSource_mock` is an example of :class:`Instrument <mahos.inst.instrument.Instrument>` class here.

.. code-block:: python
   :linenos:
   :lineno-start: 9
   :caption: instruments.py

   class VoltageSource_mock(Instrument):
       def __init__(self, name, conf, prefix=None):
           Instrument.__init__(self, name, conf=conf, prefix=prefix)

           self.check_required_conf(["resource"])
           resource = self.conf["resource"]
           self.logger.info(f"Open VoltageSource at {resource}.")

       def set_output(self, on: bool) -> bool:
           self.logger.info("Set output " + ("on" if on else "off"))
           return True

       def set_voltage(self, volt: float) -> bool:
           self.logger.debug(f"Dummy voltage {volt:.3f} V")
           return True

       # Standard API

       def start(self, label: str = "") -> bool:
           return self.set_output(True)

       def stop(self, label: str = "") -> bool:
           return self.set_output(False)

       def set(self, key: str, value=None, label: str = "") -> bool:
           if key == "volt":
               return self.set_voltage(value)
           else:
               self.logger.error(f"Unknown set() key: {key}")
               return False

As the name suggests, this class is just a mock and doesn't consume any external resources.
However, a real instrument usually requires a resource identifier for communication (VISA resource, IP Address, DLL path, etc.), and we have included how to pass such a configuration to an Instrument.
We define a configuration dictionary (:term:`conf`) as Line 23-24 in ``conf.toml``.
This is passed to Instrument and referred by ``self.conf`` (Line 14).
Line 13 uses a utility method to check existence of required key.

Only two functions of voltage source are implemented: ``set_output()`` and ``set_voltage()``.
Meanings of these may be obvious.
We assume an output relay for voltage source, that is turned on/off by ``set_output()``.
The output voltage can be set by ``set_voltage()``.

Line 27 and below makes these adapted to the :ref:`instrument-api`.
The ``set_output()`` is wrapped by :meth:`start <mahos.inst.instrument.Instrument.start>` and :meth:`stop <mahos.inst.instrument.Instrument.stop>`.
And ``set_voltage()`` is by :meth:`set <mahos.inst.instrument.Instrument.set>`.
Note that most of the :ref:`instrument-api` (excepting :meth:`get <mahos.inst.instrument.Instrument.get>`) must return bool (True on success).

In :ref:`instrument-api`, :meth:`set <mahos.inst.instrument.Instrument.set>`, :meth:`get <mahos.inst.instrument.Instrument.get>`, and :meth:`configure <mahos.inst.instrument.Instrument.configure>` accept some arguments and the type information of the arguments are lost (function signature of e.g. ``set_voltage()`` cannot be seen from the client).
We can define :class:`InstrumentInterface <mahos.inst.interface.InstrumentInterface>` to recover this, as below.
This procedure looks like a duplication of effort, but the positive side is that
we can define an explicit interface (which method is exported and which is not, as in static programming languages).

.. code-block:: python
   :linenos:
   :lineno-start: 79
   :caption: instruments.py

   class VoltageSourceInterface(InstrumentInterface):
       def set_voltage(self, volt: float) -> bool:
           """Set the output voltage."""

           return self.set("volt", volt)

Let's interact with the server.
Launch `server` and `log` with ``mahos launch log server``.
In the second terminal, ``mahos log`` to print the logs.
And ``mahos shell server`` to start IPython shell for server.

There are two ways to call the functions:

.. code-block:: python

   # Method1: raw client calls
   cli.start("source")
   cli.set("source", "volt", 12.3)

   # Method2: call through interface
   from instruments import VoltageSourceInterface
   source = VoltageSouraceInterface(cli, "source")
   source.start()
   source.set_voltage(12.3)

ivcurve
-------

``ivcurve`` is in the second layer (:doc:`arch_meas`): core measurement logic.
We have defined ``ivcurve`` in the config as below.

.. code-block:: toml
   :linenos:
   :lineno-start: 32
   :caption: conf.toml

   [localhost.ivcurve]
   module = "ivcurve"
   class = "IVCurve"
   rep_endpoint = "tcp://127.0.0.1:5561"
   pub_endpoint = "tcp://127.0.0.1:5562"
   [localhost.ivcurve.target]
   log = "localhost::log"
   [localhost.ivcurve.target.servers]
   source = "localhost::server"
   meter = "localhost::server"

Line 40-41 tells us that we need instruments `source` and `meter`
(both on `localhost::server`) for this measurement.

Operating from shell or script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before looking into the code, let's run and interact with the ivcurve.
Launch nodes with ``mahos launch log server ivcurve``.
In the second terminal, ``mahos log`` to print the logs.
And ``mahos shell ivcurve`` to start IPython shell for ivcurve.
The ivcurve measurement can be performed by following snippet.

.. code-block:: python

   params = cli.get_param_dict()
   cli.start(params)
   data = cli.get_data()
   cli.stop()

Here, ``get_data()`` returns ``IVCurveData`` defined in ``ivcurve_msgs.py``,
and ``data.data`` is the measurement result: a 2D numpy array of shape `(number of voltage points (params["num"]), number of sweeps)`.

For a bit more meaningful application, try executing file ``measure_and_plot.py`` and understanding it.
``cli.get_param_dict()`` returns a :ref:`param-dict`, str-keyed dict of :class:`Param <mahos.msgs.param_msgs.Param>`.
:class:`Param <mahos.msgs.param_msgs.Param>` is wrapper of basic (mostly builtin) types with default value, bounds (for int or float), etc.
You can set values of parameters by :meth:`set() <mahos.msgs.param_msgs.Param.set>` and pass it to ``cli.start()`` as in ``measure_and_plot.py``.

Reading IVCurve node
^^^^^^^^^^^^^^^^^^^^

What happens at ivcurve node side?
Look at implementation of ``IVCurve`` node in ``ivcurve.py``.
``IVCurve`` is subclass of :class:`BasicMeasNode <mahos.meas.common_meas.BasicMeasNode>`,
which is a convenient Node implementation for simple measurement nodes.
We explain how this node works by following ``main()`` method line by line.

.. code-block:: python
   :linenos:
   :lineno-start: 172
   :caption: ivcurve.py

   def main(self):
       self.poll()
       publish_data = self._work()
       self._check_finished()
       self._publish(publish_data)

First line of ``main()`` (Line 173) calls ``poll()``.
Here, this node checks incoming requests, and if there is a request, the handler is called.
The handler is implemented in :class:`BasicMeasNode <mahos.meas.common_meas.BasicMeasNode>` (read the implementation if you are interested in) and it calls ``change_state()`` or ``get_param_dict()`` [#f1]_ according to the request.

When ``cli.get_param_dict()`` is called, request is sent to ivcurve and the result of ``IVCurve.get_param_dict()`` is returned.
The result of this method is hard-coded here; however, the parameter bounds may be determined by instruments for real application.

By observing ``change_state()``, you will see that this node has explicit state: ``BinaryState.IDLE`` or ``BinaryState.ACTIVE``.
All measurement nodes are advised to have explicit state like this, and BinaryState is the most simplest case.
``cli.start()`` is a shorthand of `change_state(ACTIVE)`, and ``cli.stop()`` is `change_state(IDLE)`.
When state is changing from IDLE to ACTIVE, ``self.sweeper.start()`` is called.
``self.sweeper`` is an instance of ``Sweeper`` class, that communicates with the server and do real jobs.

At the second line of ``main()`` (Line 174), through ``_work()``, ``self.sweeper.work()`` is called.
A sweep measurement for IV curve is done there; `source` is used to apply voltage and the current is read by `meter`.

The third line of ``main()`` (Line 175) checks if we can finish the measurement.
Measurement is finished when ``params["sweeps"]`` is positive and the sweeps have already been repeated ``params["sweeps"]`` times (see ``Sweeper.is_finished()``).

By final line of ``main()`` (Line 176), the node status and data are published.

ivcurve_gui
-----------

The ivcurve_gui, a GUI frontend of ivcurve, is defined at the last group in ``conf.toml``.
The class ``IVCurveGUI`` is in ``ivcurve_gui.py``.
This is what we were operating in `Running the mock`_.

Let's launch all the nodes by ``mahos launch`` and confirm GUI is working.
Then, start the IPython shell with ``mahos shell ivcurve`` and send start or stop requests.
Furthermore, try running ``measure_and_plot.py`` script (stop the measurement before running).
It is quite important that we can operate the measurement from both the GUI and programs (shell, or a custom script).
This extensibility is one of the advantages of the distributed systems.

If you have experience in Qt (PyQt) programming, let's take a look at ``ivcurve_gui.py``.
The GUI component (IVCurveWidget) is composed quite simply by virtue of ``QBasicMeasClient``.
This class is `Qt-version` of `BasicMeasClient` and emits Qt signal on reception of subscribed messages.
In other words, it translates MAHOS communication into Qt communication (signal-slot).
All we have to do for widget implementation is connecting the signals to slots updating the GUI state (Line 102-103) and sending requests (Line 124-136).

There is a bit special custom on initialization (``init_with_status()``).
We cannot initialize the GUI completely without the target node (``ivcurve``) because we have to know the target's status (by ``get_param_dict()`` for example).
But we are not sure if the target node is up when GUI starts.
To assure this point, we first disable the widget (Line 29) and connect `statusUpdate` event to ``init_with_status()`` (Line 26).
When the first status message arrives, this method is fired and remaining initializations are done.
The widget is enabled finally at Line 105.
This method is called only once because the signal is disconnect at Line 79.

overlay
-------

TODO: explain overlay case ``conf_overlay.toml``, ``overlay.py``, and ``ivcurve_overlay.py``.

threading
---------

TODO: explain threading case ``conf_thread.toml`` and ``conf_thread_partial.toml``.

.. rubric:: Footnotes

.. [#f1] also calls ``save_data()`` etc. but omitted in this tutorial.
