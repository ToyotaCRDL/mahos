Tutorial 3: Manual Operation
============================

In this tutorial chapter, we will learn basic usage and configuration of
generic nodes for manually-operated instruments.

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
You will see several GUI windows popping out.
These windows correspond to generic measurement nodes explained below.

Tweaker
-------

:class:`Tweaker <mahos.meas.tweaker.Tweaker>` is a generic node for manual-tuning of instrument parameters.

To use the Tweaker, instrument must provide :ref:`inst-params-interface`,
which is the counterpart to :ref:`inst-instrument-interface` explained in :doc:`tutorial_ivcurve`.
In this example, the ``VoltageSource_mock`` class is implemented to be used via Tweaker.
The ``get_param_dict()`` method is called when Tweaker starts and when ``Read`` request is issued (``Read`` button is pressed).
The instrument class can tell the available parameters, current values, bounds, etc. to the Tweaker.
Then ``Write`` request is issued (``Write`` button is pressed), the ``configure()`` method of the instrument is invoked with the modified parameter.
The ParamDict labels for these requests are defined inside ``param_dicts`` of Tweaker configuration
in the form ``<instrument name>::<ParamDict label name>``.
The successful write operation is confirmed in the log message (Dummy voltage at chX ...).

In addition to this configuration feature, Tweaker provides interfaces
to ``start()``, ``stop()`` and ``reset()`` APIs for each instrument.
The successful start / stop operations are confirmed in the log message (Set output chX on/off).
The reset operation fails (but it is not a problem)
because ``reset()`` method is not implemented in ``VoltageSource_mock``.

PosTweaker
----------

:class:`PosTweaker <mahos.meas.pos_tweaker.PosTweaker>` is a generic node for manual-tuning of positioners.
Its role is similar to :class:`Tweaker <mahos.meas.tweaker.Tweaker>`,
but PosTweaker has dedicated interface and GUI for positioners.

In this example, two ``Positioner_mock`` instances, ``pos_x`` and ``pos_y``, are configured.
You can operate PosTweaker GUI and check if it works in the log message (Dummy move / homing etc.).

Recorder
--------

:class:`Recorder <mahos.meas.recorder.Recorder>` is a generic node for recording of time-series data from instruments.

To use Recorder, instrument must implement following APIs: ``get_param_dict_labels()``, ``get_param_dict()``, ``configure()``, ``start()``, ``stop()``, ``get("unit")``, and ``get("data")``.
In this example, the ``Multimeter_mock`` class is implemented to demonstrate Recorder's feature.
The mock instrument provides two different configurations labeled ``voltage`` and ``current``.
The ``mode`` block of the Recorder configuration defines which label (mode of instrument) is used, as below.

.. code-block:: toml

   [localhost.recorder.mode.voltage]
   meter_voltage = ["meter", "voltage"]
   [localhost.recorder.mode.current]
   meter_current = ["meter", "current"]

The first two lines define a recorder mode "voltage" with single measurement value named "meter_voltage",
which corresponds to instrument "meter" and label "voltage".
The last two lines define the other mode using "current" label instead.

After sending ``Start`` request (pushing ``Start`` button), you can see the measurement values
(random values) are collected.
