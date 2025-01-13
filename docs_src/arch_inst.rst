inst layer
==========

The inst layer is lowest-lying layer providing instrument drivers, implementated in :ref:`mahos.inst` package.

InstrumentServer
----------------

The core of inst layer is :class:`InstrumentServer <mahos.inst.server.InstrumentServer>` node.
As this node is designed to serve :term:`RPC` of arbitrary the instrument drivers,
one doesn't need to implement a :term:`node` to add a new driver.

Instrument
----------

An instrument driver is implemented as a subclass of :class:`Instrument <mahos.inst.instrument.Instrument>`.
The instrument should provide :ref:`instrument-api` by overriding methods like ``start()``.

InstrumentOverlay
-----------------

:class:`InstrumentOverlay <mahos.inst.overlay.overlay.InstrumentOverlay>` is a mechanism to define `virtual instrument` using multiple instruments.
The role is similar to :doc:`arch_meas`; however, overlay works on the same process / thread with each instrument and has direct reference to the :class:`Instrument <mahos.inst.instrument.Instrument>` instances.
Thus, core procedures should be implemented as overlay when it needs strictly synchronous behaviour or the best performance (in terms of latency).
It is also useful to define application-specific parametrization or fundamental modification to single instrument.

Interfacing
-----------

The :ref:`instrument-api` is a primary interface between Instrument and the user code.
However, one cannot understand the usage of :meth:`set <mahos.inst.instrument.Instrument.set>`,
:meth:`get <mahos.inst.instrument.Instrument.get>`, and :meth:`configure <mahos.inst.instrument.Instrument.configure>` from their signatures.
Though this may be overcomed by the documentations, following two methods are provided for better interfacing.

.. _inst-instrument-interface:

Static interface using InstrumentInterface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A "static" or "functional" interface, i.e., type-annotated signature, can be provided by defining additional methods in a subclass of
:class:`InstrumentInterface <mahos.inst.interface.InstrumentInterface>`.
This is useful for interfacing with the codes (measurement logics in :doc:`arch_meas`), and configuration with small number of parameters.

.. _inst-params-interface:

Dynamic interface using ParamDict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another interface is "dynamic" or "object-oriented" one.
Instrument can report available parameters by returning a :class:`ParamDict <mahos.msgs.param_msgs.ParamDict>`
from :meth:`get_param_dict <mahos.inst.instrument.Instrument.get_param_dict>`.
The users can modify the parameters in it, and send back with :meth:`configure <mahos.inst.instrument.Instrument.configure>`.
This is useful if the number of parameters is large (so that providing as a method signature is cumbersome).
Another benefit is that instrument can tell the bounds (minimum or maximum) of numeric parameters.
:ref:`meas-tweaker` node assumes this interface.

Lock mechanism
--------------

It is dangerous if one client can operate an instrument while another client is using it.
InstrumentServer has a lock mechanism to achieve exclusive :term:`RPC`;
one client can prevent the other clients from operating an instrument by acquiring lock.
The lock can be used for `InstrumentOverlay`_ as well.
The lock for overlay is nearly equivalent to the locks for all the instruments referred by the overlay.
An example is shown in the figure below.

.. figure:: ./img/mahos-instrument-lock.svg
   :alt: Example of lock operations on InstrumentServer
   :width: 90%

   Example of lock operations on InstrumentServer

As in (a), server1 has inst1, inst2, and inst3.
The overlay1 refers to inst1 and inst2.
The lock states are changed by request as follows.

* (a): Nothing is locked in initial state.
* (b): client1 sends a lock request for overlay1 (Lock(overlay1)), which succeeds as both inst1 and inst2 are free.
* (c): client2 sends Lock(inst3), which succeeds too.
* (d): client2 sends Lock(inst1), which fails because inst1 has been locked by client1 since (b).
* (e): client2 sends release request for overlay1 (Release(overlay1)); inst1 and inst2 are released.
* (f): client2 sends Lock(inst1) again, which succeeds this time.
