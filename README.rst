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

`Documentation is browsable here <https://toyotacrdl.github.io/mahos/html/>`_.

You can also make the documentation locally with ``make docs`` and browse it with ``make browse``.

Install
=======

#. You need Python >= 3.8, < 3.11.
#. ``virtualenv`` is the recommended tool for virtual environment management.
#. Clone this repo somewhere.
#. Install `Requirements`_.
#. Install `The mahos package`_.

Requirements
------------

#. Manual installations: there are some requirements installation of which is not quite straightforward.
   Follow instructions in the subsections below.
#. See `requirements.txt <requirements.txt>`_ for (remaining) required python packages.
   You can install them with ``pip install -r requirements.txt``

Graphviz
^^^^^^^^

You need `Graphviz <https://graphviz.org/download/>`_ for ``mahos graph`` command.
You can skip this if you don't need ``mahos graph`` command.

Windows
.......

You have to install C++ compiler `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/>`_ and
graphviz binary (version 2).
And then, use following command to install pygraphviz.

.. code-block:: bash

  pip install --global-option build_ext --global-option "-IC:\Program Files\Graphviz\include" --global-option "-LC:\Program Files\Graphviz\lib" pygraphviz

Linux
.....

You can install the graphviz with package manager (e.g. ``sudo apt install graphviz libgraphviz-dev`` for Ubuntu/Debian),
and then ``pip install pygraphviz``.

OpenCV
^^^^^^

To use full-features of image analysis modules, install OpenCV (>= 3.0.0) with Python binding.
There are several methods to install this, and the easiest are following.

- Windows: ``pip install opencv-python`` to install CPU-only binary
- Linux: the pre-compiled package (e.g. ``sudo apt install python3-opencv`` for Ubuntu/Debian)

The mahos package
-----------------

Clone this repo and ``pip install -e .``

Here, -e (editable) is optional but recommended.

Run
===

To use the mahos system, you have to write a toml configuration file first.
There is an `example file <tests/conf.toml>`_ for local testing.

With your config, use ``mahos`` command to start the nodes and interact with them.

- ``mahos run [nodename]``: run a node.
- ``mahos launch [nodenames]``: start all the nodes (or specified nodes) pertaining to single host in the config file.
- ``mahos log``: subscribe to a LogBroker to print logs on console.
- ``mahos ls``: print the list of defined nodes in the config file.
- ``mahos graph``: visualize the config file as graph.
- ``mahos echo [nodename] -t [topicname]``: subscribe to a topic and print published messages.
- ``mahos shell [nodenames]``: start an IPython shell with the clients for the nodes.
- ``mahos data [subcommand]``: list/print attributes in data files or plot data files.
- ``mahos plot``: shorthand of ``mahos data plot``.

Check ``mahos [subcommand] -h`` for more details.

Test
====

``pytest``

License
=======

The mahos project is licensed under the `3-Clause BSD License <LICENSE>`_.

Redistribution
--------------

The `GUI theme <mahos/gui/breeze_resources>`_ is taken from `BreezeStyleSheets <https://github.com/Alexhuszagh/BreezeStyleSheets>`_ project,
which is licensed under the `MIT license <https://github.com/Alexhuszagh/BreezeStyleSheets/blob/main/LICENSE.md>`_:
Copyright 2013-2014 Colin Duquesnoy and 2015-2016 Alex Huszagh.

A `file <mahos/util/unit.py>`_ includes a function from the `pyqtgraph <https://github.com/pyqtgraph/pyqtgraph>`_ project,
which is licensed under the `MIT license <https://github.com/pyqtgraph/pyqtgraph/blob/master/LICENSE.txt>`_:
Copyright 2012 Luke Campagnola, University of North Carolina at Chapel Hill.

Contributing
============

Please check out `Contribution Guidelines <https://toyotacrdl.github.io/mahos/html/contributing.html>`_.
