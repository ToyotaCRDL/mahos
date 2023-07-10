gui layer
=========

The gui layer provides GUI frontend for :doc:`arch_meas` (or :doc:`arch_inst` sometimes).
This is implementated as :ref:`mahos.gui` package.
We use `Qt <https://www.qt.io/>`_ as the GUI toolkit, and frequently use `PyQtGraph <https://www.pyqtgraph.org/>`_ for data visualization.

Qt has its own (inter-or-intra thread) communication mechanism called `signal` and `slot`.
The GUI event (such as this button is clicked) `emits` the `signal`.
If signal is connected to `slots` (or methods), the methods are called.

Basic structure
---------------

Because GUI is basically a :term:`client` of the nodes, we have to implement two functions: Req (Requester) and Sub (Subscriber).
The Req is relatively simple, because :term:`Req-Rep` can be implemented as single method call (it's a `slot` in Qt terminology).
For Sub, it becomes quite tidy if we `emit` the `signal` on subscribed message's arrival.
Let's see this point by observing the structure of an example ``IVCurveWidget`` from :doc:`tutorial_ivcurve`.

.. figure:: ./img/ivcurve-gui.svg
   :alt: Class relationships of IVCurve GUI
   :width: 85%

   Class relationships of IVCurve GUI

The :class:`QBasicMeasClient <mahos.gui.client.QBasicMeasClient>` can be used as `Qt-version` client of :class:`BasicMeasNode <mahos.meas.common_meas.BasicMeasNode>`.
As noted, Req-Rep looks quite simple. When startButton is clicked, the `click` signal invokes ``request_start()`` method (`slot`), which subsequently calls ``QBasicMeasClient.change_state()`` to send the request.
The `data` topic published by ``IVCurve`` is subscribed by ``QStatusDataSubWorker``, which is working in a dedicated thread (different from main thread running GUI main loop).
When `data` arrives, ``QStatusDataSubWorker`` emits `dataUpdated` signal and received by the ``QBasicMeasClient.check_data()`` (here, inter-thread communication is done).
``check_data()`` again emits `dataUpdated` signal, which eventually updates the data visualized by `plot_item`.

It is important that :class:`QBasicMeasClient <mahos.gui.client.QBasicMeasClient>` does all the mahos communication stuffs (:term:`Req-Rep` and :term:`Pub-Sub`); it `converts Qt-communication to mahos-communication` in other words.
As a result, ``IVCurveWidget`` don't have to care about mahos communications and can focus on using `signals` and `slots` provided by :class:`QBasicMeasClient <mahos.gui.client.QBasicMeasClient>`.
