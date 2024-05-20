Tutorial 3: Manual Operation
============================

.. note::
   This tutorial document is incomplete, however, the example code works fine.

In this tutorial chapter, we will learn basic usage and construction of manually-operated instruments.

Preparation
-----------

Before starting this, :doc:`install <installation>` the library and dependencies.

It is recommended to finish :doc:`tutorial_comm` and :doc:`tutorial_ivcurve` before this.

Find the ``examples/manual_op`` directory in the mahos repository.
We will use the files in it for this tutorial.
You can copy this directory to somewhere in your computer if you want.

Running the mock
----------------

In a terminal, ``mahos log`` to show log messages.
In another terminal, run all the nodes defined in our ``conf.toml`` by command ``mahos launch``.
You will see two GUI windows popping out.
These windows correspond to generic measurement nodes.
One is :class:`Tweaker <mahos.meas.tweaker.Tweaker>` for manual-tuning of instrument parameters.
Another is :class:`Recorder <mahos.meas.recorder.Recorder>` for simple time-series data collection.

Tweaker
-------

TODO: explain :class:`Tweaker <mahos.meas.tweaker.Tweaker>`.

To use Tweaker, instrument must provide :ref:`inst-params-interface`,
which is the counterpart to :ref:`inst-instrument-interface` explaned in :doc:`tutorial_ivcurve`.

Recorder
--------

TODO: explain :class:`Recorder <mahos.meas.recorder.Recorder>`.
