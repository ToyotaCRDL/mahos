ivcurve
=======

See Tutorial 2 in the document for detailed explanation.

Speedtest
---------
This is an advanced topic that is not in the tutorial.
You can compare the dependency of speed on the implementation method.

Start
^^^^^

Start `server` and `ivcurve`.

* Raw: ``mahos launch -i server ivcurve``
* Overlay: ``mahos launch -c conf_overlay.toml -i server ivcurve``
* Inproc: ``mahos run -c conf_thread_partial.toml -t server_ivcurve``

Measure
^^^^^^^

``mahos echo -r -t data ivcurve``
