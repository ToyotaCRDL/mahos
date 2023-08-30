Installation
============

Steps to install mahos:

#. Check the `System requirements`_.
#. `Create virtualenv`_ (this is optional, but recommended).
#. `Clone the repository`_.
#. Install `Requirements`_.
#. Install `The mahos package`_.

System requirements
-------------------

You need Python (>= 3.8, < 3.11) on Windows or Linux, and a C++ compiler (see below).

C++ compiler
^^^^^^^^^^^^

On Windows, you have to download and install the C++ compiler `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/>`_ in order to build the C extensions.
You can skip this if you have already installed Visual Studio on your computer.

On most of Linux, there's nothing to do here because you should have ``gcc`` already.

Create virtualenv
-----------------

`Virtualenv <https://virtualenv.pypa.io/en/latest/>`_ is the recommended tool for virtual environment management.
Skip this section if you prefer one of the following alternatives.

- You could install the requirements and the mahos package with your system Python.
- You could use ``conda`` or other virtual environment management tools too.

``virtualenv`` is installable via ``pip``. ::

  pip install virtualenv

For some packages which are not quite straightforward to install via pip (OpenCV etc.),
we recommend to enable system-site-packages.

.. code-block:: bash

  virtualenv mahos -p <python to use> --system-site-packages

Small notes on virtualenv usage:

- activate: ``source <mahos env directory>/bin/activate`` or ``source <mahos env directory>/Scripts/activate``
- deactivate: ``deactivate``

Following contents assume that the virtualenv has already been activated.

Clone the repository
--------------------

Use git to clone the mahos repository.
Following commands clone it to local directory `<preferred directory>/mahos`.
We call this directory the `repository directory`.

- ``cd <preferred directory>`` [#f1]_
- ``git clone https://github.com/ToyotaCRDL/mahos``

Requirements
------------

#. Manual installations: there are some requirements installation of which is not quite straightforward.
   Follow instructions in the subsections below.
#. See ``requirements.txt`` in the `repository directory` for (remaining) required python packages.
   You can install them with ``pip install -r requirements.txt``

Graphviz
^^^^^^^^

You need `Graphviz <https://graphviz.org/download/>`_ for :ref:`mahos graph` command.
You can skip this if you don't need :ref:`mahos graph` command.

Windows
.......

You have to install the `C++ compiler`_ and the `graphviz binary <https://graphviz.org/download/#windows>`_ (version 2).
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

In the mahos `repository directory`, ``pip install -e .``

Here, -e (editable) is optional but recommended.

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
