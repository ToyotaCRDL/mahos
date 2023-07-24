mahos.inst.instrument.Instrument
================================

.. currentmodule:: mahos.inst.instrument

.. autoclass:: Instrument

.. _instrument-api:

Standard Instrument APIs
------------------------

Below is the list of standard instrument APIs.
Override these methods (with the same signature) to expose the instrument's functionality.

   .. automethod:: Instrument.start
   .. automethod:: Instrument.stop
   .. automethod:: Instrument.shutdown
   .. automethod:: Instrument.pause
   .. automethod:: Instrument.resume
   .. automethod:: Instrument.reset
   .. automethod:: Instrument.configure
   .. automethod:: Instrument.set
   .. automethod:: Instrument.get

.. rubric:: Other Methods

.. autosummary::

   ~Instrument.__init__
   ~Instrument.check_required_conf
   ~Instrument.check_required_params
   ~Instrument.close
   ~Instrument.fail_with
   ~Instrument.full_name
