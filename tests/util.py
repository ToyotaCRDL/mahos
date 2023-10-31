#!/usr/bin/env python3

"""
Common utilities for tests.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import time
from io import BytesIO

from mahos.util.comp import dict_equal_inspect


def get_some(get, poll_timeout_ms, trials=500):
    """Try to get some value from asyncronous status update."""

    for i in range(trials):
        s = get()
        if s is not None:
            return s
        time.sleep(1e-3 * poll_timeout_ms)
    else:
        return None


def expect_value(get, value, poll_timeout_ms, trials=500, verbose=False):
    """Repeatedly expect value from asyncronous status update."""

    for i in range(trials):
        s = get()
        if verbose:
            print(s, value)
        if s == value:
            return True
        time.sleep(1e-3 * poll_timeout_ms)
    else:
        return False


def save_load_test(io, data):
    """Test save data to default (hdf5) format and load. Compare the data before and after."""

    f = BytesIO()
    assert io.save_data(f, data)
    loaded = io.load_data(f)
    assert loaded is not None
    assert dict_equal_inspect(data.__dict__, loaded.__dict__)
