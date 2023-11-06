Custom serializer
=================

This is an example to define messages with custom serializer.

The ``clock`` and ``calc`` are similar to examples in `comm`,
but uses message types defined in ``msgs.py``, that uses msgpack as serializer.

JavaScript
----------

We can write clients in other programming language due to msgpack.
To test example client scripts in JavaScript:

.. code-block:: bash

  npm install
  mahos launch
  node print_clock.js
  node calc_numbers.js
