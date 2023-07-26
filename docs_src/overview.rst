Overview
========

.. currentmodule:: mahos

What is this?
-------------

MAHOS is a library to create distributed measurement automation systems.
The experimental researchers often face challenges building up the systems
to automate their measurement procedures in physical science & engineering, or related areas.
This library tries to help ease these processes by providing simple framework.
The distributed architecture is adopted to help writing debuggable, testable, and maintainable programs.

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
