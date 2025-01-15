Installation
============

Steps to install mahos:

#. Check the `System requirements`_.
#. `Create virtual environment`_ (this is optional, but recommended).
#. `Clone the repository`_.
#. Install `Optional requirements`_.
#. Install `The mahos package`_ (and requirements).

System requirements
-------------------

You need Python (>= 3.10, < 3.13) on Windows or Linux, and prerequisites for C extensions (see below).

For C extensions
^^^^^^^^^^^^^^^^

On Windows, you have to download and install the C++ compiler `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/>`_ in order to build the C extensions.
You can skip this if you have already installed Visual Studio on your computer.

On Linux, install ``gcc`` and the dev-package for Python3.
The ``gcc`` is installed by default on many distros.
For the latter, try ``sudo apt install python3-dev`` on Debian-based distros
or ``sudo yum install python3-devel`` on RHEL-like distros.

Create virtual environment
--------------------------

The `venv <https://docs.python.org/3/library/venv.html>`_ is the recommended tool for virtual environment management.
Skip this section if you prefer one of the following alternatives.

- You could install the requirements and the mahos package with your system Python.
- You could use ``virtualenv``, ``conda`` or other virtual environment management tools too.

``venv`` comes with Python; you don't have to install additional stuffs.

For some packages which are not quite straightforward to install via pip (OpenCV etc.),
we recommend to enable system-site-packages (remove ``--system-site-packages`` if you don't want it).

.. code-block:: bash

  python -m venv mahos --system-site-packages

Small notes on venv usage:

- activate: ``source <mahos env directory>/bin/activate`` or ``source <mahos env directory>/Scripts/activate``
- deactivate: ``deactivate``

Following contents assume that the virtual environment has already been activated.

Clone the repository
--------------------

Use git to clone the mahos repository.
Following commands clone it to local directory `<preferred directory>/mahos`.
We call this directory the `repository directory`.

- ``cd <preferred directory>`` [#f1]_
- ``git clone https://github.com/ToyotaCRDL/mahos``

Optional requirements
---------------------

There are some optional requirements installation of which is not quite straightforward.
If you want to install them, follow instructions in the subsections below.

Graphviz
^^^^^^^^

You need `Graphviz <https://graphviz.org/download/>`_ for :ref:`mahos graph` command.
You can skip this if you don't need :ref:`mahos graph` command.
See `pygraphviz documentation <https://pygraphviz.github.io/documentation/stable/install.html>`_ for details.

Windows
.......

You have to install the `C++ compiler <For C extensions_>`_ and the `graphviz binary <https://graphviz.org/download/#windows>`_ (version 2).
And then, use following command to install pygraphviz.

.. code-block:: bash

  pip install --use-pep517 --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz

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

In the mahos `repository directory`, ``pip install -e .``

Here, -e (editable) is optional but recommended.
Basic requirements in ``requirements.txt`` are installed by this command.

Inst requirements
^^^^^^^^^^^^^^^^^

There are additional requirements for instrument drivers in ``inst-requirements.txt``.

- You can install all of them by ``pip install -e .[inst]`` or ``pip install -r inst-requirements.txt``
- If you don't want to install unnecessary packages, manually pick and install what you need.

Test
----

To check if installation is successful, run test with ``pytest``.

Notes
-----

PyQt6
^^^^^

The PyQt6 fails if you have `PyQt6` package inside the virtual environment, but dependencies (`PyQt6-sip` and `PyQt6-Qt6`) outside.
Check the locations of these three packages if PyQt6 is going wrong.
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

Qt tools on Ubuntu
^^^^^^^^^^^^^^^^^^

Install Qt6 tools for development on Ubuntu 22.04. ::

   sudo apt install qt6-tools-dev assistant-qt6 designer-qt6

.. rubric:: Footnotes

.. [#f1] substitute <preferred directory> with your preference.
