Overview
========

.. currentmodule:: mahos

What is this?
-------------

MAHOS is a framework for building `modular` and `distributed` measurement automation systems.
Experimental researchers often face challenges in building systems
consisting of multiple instruments or computers to automate their measurement procedures
in physical science and engineering, or related areas.

By `modular`, we mean the pieces of code are organized within high-level abstraction mechanisms (module, class, etc.), and the interfaces are clearly defined.
Such modularity is necessary to write maintainable (debuggable and testable) programs.

We also believe that `distributed` messaging could help create flexible and accessible systems,
that has not been widely recognized and tested in this field.
Before explaining why the distributed messaging approach could be valuable,
let us introduce basic concepts and terms.

Nodes and clients
^^^^^^^^^^^^^^^^^

By `distributed` system, we mean that the system consists of multiple programs (processes) communicating with each other.
We call these programs :term:`nodes <node>`.
A node usually provides some services for other nodes.
The :term:`node client` provides the standard way to access the node's functions.
Nodes internally use the clients to access the others, and custom programs can utilize them as well.

Layers
^^^^^^

Because there exists mostly one-way data flow on measurement automation,
the layer structure is introduced; the nodes can be categorized into three layers.
The inst layer is the lowest-lying layer (source of data flow), consisting of low-level drivers for the instruments.
The meas layer is the middle layer providing the measurement logic.
The gui layer is the topmost layer for GUI frontends, which are optional but usually important for the users.

For a more detailed description of the architecture, see :doc:`arch`.

System configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^

The MAHOS-based system is considered `static` reflecting the purpose of measurement automation,
i.e., we assume to know all the possible nodes and messages beforehand.
This assumption enables the transparent configuration of the whole system;
it can be written in a single :doc:`TOML configuration file <conf>`.

Why distributed?
----------------

Flexible multi-host
^^^^^^^^^^^^^^^^^^^

Since the messages can be delivered across computers even with different platforms (operating systems),
we can build up a multi-computer (multi-host) system flexibly.
For example, if an instrument has a driver only for Windows but we have to use Linux for the rest of the system,
we can run a driver node on the Windows host and access it from the other nodes on the Linux.
All it takes to move a node from one host to another is to change a few lines in the configuration file.

Runtime accessibility
^^^^^^^^^^^^^^^^^^^^^

High runtime data / service accessibility are the best benefits brought to MAHOS by adopting
the distributed messaging approach.
Notable examples can be listed below.

- The data published by a measurement logic (meas layer) node can be simultaneously A) visualized in a GUI node
  and B) processed and analyzed in an interactive console such as IPython or Jupyter.
- It is straightforward to conduct an initial measurement using a GUI node,
  and then write an ad-hoc automation script to run many measurements with different parameters
  because the GUI node and the script are seen as equivalent clients from the measurement logic node.
  The GUI can visualize the data even when the measurement is started by the script.
- We can inspect the instrument's status or perform ad-hoc operations on the instrument at run time
  without shutting down any nodes.
  This is because the operation requests from measurement logic nodes and ad-hoc ones are equivalent
  from the perspective of the instrument driver (inst layer) node.

How to use it?
--------------

First of all, :doc:`install <installation>` the library.

To use the existing nodes to construct the system,
you have to write (or customize existing ones) a :doc:`configuration file <conf>`.
Once the configuration file is done, you can run the system by using the :ref:`command line interface<mahos.cli>` (:ref:`mahos run` or :ref:`mahos launch`).

:doc:`tutorial` will help you to get used to these concepts,
as well as how to write your own programs.
