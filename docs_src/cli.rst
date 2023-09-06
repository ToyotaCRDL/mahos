.. _mahos.cli:

Command Line Interface
======================

.. currentmodule:: mahos

After installing mahos package, a command ``mahos`` is installed.
This command serves as a command line interface (CLI) to use mahos.
The ``mahos`` CLI provides several sub-commands described below.

Node operations
---------------

Common arguments
^^^^^^^^^^^^^^^^

There are common arguments for node operation commands:

-c [config file]
................

To operate on MAHOS nodes, you have to prepare toml-based :doc:`configuration file <conf>`.
The ``-c`` option specifies the path to config file.
Defaults to ``conf.toml``.

-H [hostname]
.............

CLI command usually targets a node.
The full-name of the node consists of hostname and nodename.
The ``-H`` option specifies the hostname part.
Defaults to real hostname (``platform.uname().node``) if it is in the config file, or ``localhost``.

.. _mahos run:

mahos run
^^^^^^^^^

``mahos run [nodename]`` starts a mahos node.

.. _mahos launch:

mahos launch
^^^^^^^^^^^^

``mahos launch [nodenames]`` starts all the nodes (or specified nodes) pertaining to single host.

``mahos launch -e [nodenames-to-exclude]`` starts all the nodes excluding specified names.

mahos log
^^^^^^^^^

``mahos log [nodename=log]`` subscribes to a :class:`LogBroker <node.log_broker.LogBroker>` to print logs on console.

mahos ls
^^^^^^^^

``mahos ls`` prints the list of defined nodes in the config file.

.. _mahos graph:

mahos graph
^^^^^^^^^^^

``mahos graph`` visualizes the config file as graph.

Add option ``-o [filename]`` to save to a file.

mahos echo
^^^^^^^^^^

``mahos echo -t [topicname] [nodename]`` subscribes to a topic and print published messages.

mahos shell
^^^^^^^^^^^

``mahos shell [nodenames]`` starts an IPython shell with clients for the nodes.

A bound variable ``cli`` holds instances of the clients.
For each client, ``client.M`` is the message module that may be required to invoke client APIs.

Data operations
---------------

mahos data ls
^^^^^^^^^^^^^

``mahos data ls [filenames]`` prints the list of attributes and types in data files.

mahos data note
^^^^^^^^^^^^^^^

``mahos data note [filenames]`` prints (or amends) the note attribute in data files.

Add option ``-a [new note string]`` to amend the note in the files.

mahos data print
^^^^^^^^^^^^^^^^

``mahos data print [filenames] -k [keys=[params]]`` prints the attributes in data files.

mahos data plot
^^^^^^^^^^^^^^^

``mahos data plot [subcommand]`` plots the data inside data files.

``mahos plot`` is a shorthand of ``mahos data plot``.
