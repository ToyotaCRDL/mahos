Overview
========

.. currentmodule:: mahos

What is this?
-------------

MAHOS is a framework for building distributed measurement automation systems.
The experimental researchers often face challenges building up systems
consisting of multiple instruments or computers to automate their measurement procedures
in physical science and engineering, or related areas.
We believe that a distributed system would be a better solution for complex measurement automation
than the monolithic one.
MAHOS is created to provide a flexible framework for constructing distributed systems,
and to help write debuggable, testable, and maintainable measurement automation codes.

Nodes and clients
^^^^^^^^^^^^^^^^^

By `distributed system`, we mean that the system consists of multiple programs (processes) communicating each other.
We call these programs :term:`nodes <node>`.
A node usually provides some services for other nodes.
The :term:`node client` provides the standard way to access node's functions.
Nodes internally use the clients to access the others, and custom programs can utilize them as well.

Layers
^^^^^^

Because there exists mostly one-way data flow on measurement automation,
the layer-structure is introduced; the nodes can be categorized into three layers.
The inst layer is lowest-lying layer (source of data flow), consisting of low-level drivers for the instruments.
The meas layer is middle layer providing the measurement logics.
The gui layer is apparently topmost layer for GUI frontends, which are optional but usually important for the users.

For more detailed description of the architecture, see :doc:`arch`.

How to use?
-----------

First of all, :doc:`install <installation>` the library.

To use the existing nodes to construct the system,
you have to write (or customize existing one) a :doc:`configuration file <conf>`.
Once configuration file is done, you can run the system by using :ref:`command line interface<mahos.cli>` (:ref:`mahos run` or :ref:`mahos launch`).

:doc:`tutorial` will help you to get used to these concepts,
as well as how to write your own programs.
