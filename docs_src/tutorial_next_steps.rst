Tutorial: Next Steps
====================

There are several ways to go forward after finishing the tutorial.

The most `practical` way may be try :ref:`to build your own system <next-step-build>`.
If you have enough time or interest, try :ref:`to understand mahos further <next-step-understand>`.

.. _next-step-build:

Building your own system
------------------------

Writing Instrument
^^^^^^^^^^^^^^^^^^

If you want to build your own system, first step might be preparing the :class:`Instrument <mahos.inst.instrument.Instrument>` class for your instrument.
You are very lucky if you can find corresponding class in :ref:`mahos.inst` package.
Even this is not the case, writing your own :class:`Instrument <mahos.inst.instrument.Instrument>` is not a very hard way; you can use existing classes in :ref:`mahos.inst` as the examples.
Following subsections explain several methods to write :class:`Instrument <mahos.inst.instrument.Instrument>`.
It depends on the real instrument's interface / driver provided by the manufacturer, but most of the instruments will fit into (at least) one of these.

VISA
....

VISA (Virtual Instrument Software Architecture) is a standardized communication library for measurement instruments.
VISA is an abstraction layer above GPIB, RS232 (Serial), Ethernet or USB.
We can use `PyVISA <https://pyvisa.readthedocs.io/>`_ library to communicate with the instruments following this standard.
:class:`VisaInstrument <mahos.inst.visa_instrument.VisaInstrument>` serves as a base class which implements a few convenient methods.

Wrapping Python library
.......................

Some (rather modern) instruments provide the libraries for Python.
You are lucky! This is usually the easiest case.

Wrapping C library
..................

We can use Python standard `ctypes <https://docs.python.org/3/library/ctypes.html>`_ library to wrap the library for C.

DAQ
...

We can use `PyDAQmx <https://pythonhosted.org/PyDAQmx/>`_ library to implement :class:`Instrument <mahos.inst.instrument.Instrument>` based on National Instruments DAQ.
There are several classes in ``mahos.inst.daq`` module.

Writing meas Node
^^^^^^^^^^^^^^^^^

If your measurement is not very complex,
the easiest way to write meas node is sub-classing :class:`BasicMeasNode <mahos.meas.common_meas.BasicMeasNode>`.
We have already seen an example in :doc:`tutorial_ivcurve`.

Writing gui Node
^^^^^^^^^^^^^^^^

If GUI is necessary, you can try to write a :class:`GUINode <mahos.gui.gui_node.GUINode>`.
Since the existing implementations are based on `PyQt <https://riverbankcomputing.com/software/pyqt/>`_,
you will need some experiences or practices to use it.

If you are familiar with other UI (either native / web-based) technologies,
it will be also possible to implement the gui node with that.
In that case, your node will not be based on :class:`GUINode <mahos.gui.gui_node.GUINode>` (because this is for PyQt).

.. _next-step-understand:

Understanding mahos further
---------------------------

* :doc:`arch` explains the architecture of mahos system.
* :doc:`conf` provides a bit detailed specification of configuration files.
* :doc:`cli` is a list of cli commands; but ``mahos [subcommand] -h`` may provide more information. You can also try to read the code in ``mahos.cli`` package, that is not quite big.

Reading the codes
^^^^^^^^^^^^^^^^^

:doc:`api` is not perfect for the time being: listed are not all the stuffs / there are some undocumented classes.
But we are working toward better documentation.
It could be used at least to discover which module or class is important.

You will eventually be required to read the codes to fully understand the mahos internals.
We are very happy if you could report some bugs / add missing documentation / give us any feedbacks.
