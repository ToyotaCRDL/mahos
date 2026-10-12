Installation
============

Steps to install mahos:

#. Check the `System requirements`_.
#. `Create virtual environment`_ (this is optional, but recommended).
#. `Clone the repository`_.
#. Install `Optional requirements`_.
#. Install `The mahos packages`_ (and requirements).

If you are upgrading from MAHOS 0.3.x, see :doc:`migration`.

System requirements
-------------------

You need Python (>= 3.11, < 3.14) on Windows or Linux.

Create virtual environment
--------------------------

`venv <https://docs.python.org/3/library/venv.html>`_ is the recommended tool for virtual environment management.
Skip this section if you prefer one of the following alternatives.

- You could install the requirements and the mahos package with your system Python.
- You could use ``virtualenv``, ``conda`` or other virtual environment management tools too.

``venv`` comes with Python; you do not have to install additional tools.

.. code-block:: bash

  python -m venv mahos

If you want to use packages installed on system Python outside virtual environment, add ``--system-site-packages`` option.

Small notes on venv usage:

- activate: ``source <mahos env directory>/bin/activate`` or ``source <mahos env directory>/Scripts/activate``
- deactivate: ``deactivate``

The following content assumes that the virtual environment has already been activated.

Clone the repository
--------------------

Use git to clone the mahos repository.
The following commands clone it to a local directory, ``<preferred directory>/mahos``.
We call this directory the `repository directory`.

- ``cd <preferred directory>`` [#f1]_
- ``git clone https://github.com/ToyotaCRDL/mahos``

Optional requirements
---------------------

There are some optional requirements whose installation is not straightforward.
If you want to install them, follow the instructions in the subsections below.

C++ compiler
^^^^^^^^^^^^

If you wish to install the C extensions (`mahos-dq-ext`_ package),
you need to install a C++ compiler.

On Windows, you have to download and install `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/>`_ in order to build the extensions.
You can skip this if you have already installed Visual Studio on your computer.

On Linux, install ``gcc`` and the dev-package for Python3.
The ``gcc`` is installed by default on many distros.
For the latter, try ``sudo apt install python3-dev`` on Debian-based distros
or ``sudo yum install python3-devel`` on RHEL-like distros.

Graphviz
^^^^^^^^

You need `Graphviz <https://graphviz.org/download/>`_ for :ref:`mahos graph` command.
You can skip this if you don't need :ref:`mahos graph` command.
See `pygraphviz documentation <https://pygraphviz.github.io/documentation/stable/install.html>`_ for details.

Windows
.......

You have to install the `C++ compiler`_ and the `graphviz binary <https://graphviz.org/download/#windows>`_ (version 2).
Then use the following command to install pygraphviz.

.. code-block:: bash

  pip install --use-pep517 --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz

Linux
.....

You can install Graphviz with a package manager (e.g. ``sudo apt install graphviz libgraphviz-dev`` for Ubuntu/Debian),
and then ``pip install pygraphviz``.

OpenCV
^^^^^^

To use the full features of image analysis modules, install OpenCV (>= 3.0.0) with Python bindings.
There are several methods to install this, and the easiest are the following.

- Windows: ``pip install opencv-python`` to install CPU-only binary
- Linux: ``pip install opencv-python``, or the pre-compiled package by OS when ``--system-site-packages`` option is enabled for virtual environment. (e.g. ``sudo apt install python3-opencv`` for Ubuntu/Debian)

The mahos packages
------------------

mahos
^^^^^

``pip install -e ./pkgs/mahos``

Here, ``-e`` (editable) is optional but recommended.
Runtime requirements in ``requirements.txt`` are installed by this command.
This includes ``argcomplete``, which is used for :ref:`bash/zsh CLI completion <mahos shell completion>`.

Dev requirements
................

Development tools (pytest, Ruff, Sphinx, etc.) are in
``dev-requirements.txt``.

- You can install all of them by ``pip install -e './pkgs/mahos[dev]'`` or
  ``pip install -r ./pkgs/mahos/dev-requirements.txt``

Inst requirements
.................

There are additional requirements for instrument drivers in ``inst-requirements.txt``.

- You can install all of them by ``pip install -e './pkgs/mahos[inst]'`` or ``pip install -r ./pkgs/mahos/inst-requirements.txt``
- If you don't want to install unnecessary packages, manually pick and install what you need.

mahos-dq
^^^^^^^^

``pip install -e ./pkgs/mahos-dq``

Here, ``-e`` (editable) is optional but recommended.

mahos-dq-ext
^^^^^^^^^^^^

The `C++ compiler`_ is required to install this.

``pip install -e ./pkgs/mahos-dq-ext``

Here, ``-e`` (editable) is optional but recommended.

Test
----

To check whether installation is successful, run tests with ``pytest``.

Notes
-----

PyQt6
^^^^^

This note applies only if ``--system-site-packages`` option is enabled for the virtual environment.
PyQt6 can fail if you have the `PyQt6` package inside the virtual environment, but dependencies (`PyQt6-sip` and `PyQt6-Qt6`) outside.
Check the locations of these three packages if PyQt6 is not working correctly.
To resolve the situation, try one of the following.

To install things `outside` the virtual environment:

.. code-block:: bash

  # assuming you are inside the virtual environment here
  pip uninstall PyQt6 PyQt6-sip PyQt6-Qt6
  deactivate
  pip install PyQt6

To install things `inside` the virtual environment:

.. code-block:: bash

  # assuming you are inside the virtual environment here
  deactivate
  pip uninstall PyQt6 PyQt6-sip PyQt6-Qt6
  source <mahos env directory>/bin/activate # or source <mahos env directory>/Scripts/activate
  pip install PyQt6

Qt on WSL
^^^^^^^^^

Under WSL, Qt may select the Wayland platform plugin automatically.
If a mahos GUI intermittently crashes during interaction,
try using the XCB platform plugin instead. ::

  QT_QPA_PLATFORM=xcb mahos launch

This selects WSLg's XWayland path, which seems more stable with mahos.
To enable it for every mahos session, the environment variable can be exported
when activating the mahos virtual environment.

If Qt reports that the XCB platform plugin cannot be initialized,
additional system XCB libraries may be required.
On Ubuntu, install ``libxcb-cursor0`` first.

Qt tools on Ubuntu
^^^^^^^^^^^^^^^^^^

Install Qt6 tools for development on Ubuntu 24.04. ::

   sudo apt install qt6-tools-dev assistant-qt6 designer-qt6

.. rubric:: Footnotes

.. [#f1] substitute <preferred directory> with your preference.
