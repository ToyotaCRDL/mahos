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

- Raw: ``mahos launch server ivcurve``
- Overlay: ``mahos launch -c conf_overlay.toml server ivcurve``
- Inproc: ``mahos run -c conf_thread_partial.toml -t server_ivcurve``

Run measurement using shell.

- ``mahos shell -c conf.toml``
- ``cli.start(cli.get_param_dict())``

Measure
^^^^^^^

``mahos echo -r -t data ivcurve``
