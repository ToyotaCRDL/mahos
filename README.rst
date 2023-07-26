#####
MAHOS
#####

.. image:: https://github.com/ToyotaCRDL/mahos/actions/workflows/build.yaml/badge.svg

MAHOS: Measurement Automation Handling and Orchestration System.

This package currently includes the following.

- Base system for distributed measurement automation.
- Implementations of microscopy / optically detected magnetic resonance (ODMR) system
  for solid-state color center research, based on above.

Documentation
=============

`Documentation is browsable here <https://toyotacrdl.github.io/mahos/>`_.

You can also browse the documentation locally by ``make browse`` or
opening ``docs`` directory with a web browser.

Install
=======

Read the `Installation guide <https://toyotacrdl.github.io/mahos/installation.html>`_.

In short, we recommend editable installation with cloned repository
because this library is under development:

#. Clone this repo somewhere.
#. Install the requirements: ``pip install -r requirements.txt``
#. Install the ``mahos`` package: ``pip install -e .``

Run
===

To use the mahos-based system, you have to write a toml `configuration file <https://toyotacrdl.github.io/mahos/conf.html>`_ first.
With your config, use the `command line interface <https://toyotacrdl.github.io/mahos/cli.html>`_ to start the nodes and interact with them.

- The `tutorial <https://toyotacrdl.github.io/mahos/tutorial.html>`_ and corresponding `examples <https://github.com/ToyotaCRDL/mahos/tree/main/examples>`_ are provided to get used to these concepts.
- There is an `example config <https://github.com/ToyotaCRDL/mahos/blob/main/tests/conf.toml>`_ for the unit test too.
  Here you can observe main built-in measurement logics and GUIs with mock instruments
  (microscopy / ODMR system for color centers, for the time being).

Test
====

``pytest``

License
=======

The mahos project is licensed under the `3-Clause BSD License <https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE>`_.

Redistribution
--------------

The `GUI theme <https://github.com/ToyotaCRDL/mahos/tree/main/mahos/gui/breeze_resources>`_ is taken from `BreezeStyleSheets <https://github.com/Alexhuszagh/BreezeStyleSheets>`_ project,
which is licensed under the `MIT license: Copyright 2013-2014 Colin Duquesnoy and 2015-2016 Alex Huszagh <https://github.com/Alexhuszagh/BreezeStyleSheets/blob/main/LICENSE.md>`_.

A `file <https://github.com/ToyotaCRDL/mahos/blob/main/mahos/util/unit.py>`_ includes a function from the `pyqtgraph <https://github.com/pyqtgraph/pyqtgraph>`_ project,
which is licensed under the `MIT license: Copyright 2012 Luke Campagnola, University of North Carolina at Chapel Hill <https://github.com/pyqtgraph/pyqtgraph/blob/master/LICENSE.txt>`_.

Contributing
============

Please check out `Contribution Guidelines <https://toyotacrdl.github.io/mahos/contributing.html>`_.
