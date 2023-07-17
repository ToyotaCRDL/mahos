#!/usr/bin/env python3

"""
Unit conversion module.

.. This file is a part of MAHOS project.

The SI_scale function is taken from the pyqtgraph project (https://github.com/pyqtgraph/pyqtgraph),
which is licensed under the MIT License:
Copyright 2012 Luke Campagnola, University of North Carolina at Chapel Hill.

The function is redefined here because we sometimes want to avoid importing pyqtgraph
(from non-GUI codes) only for this function.

"""

import typing as T
import math
import decimal

SI_PREFIXES = "yzafpnµm kMGTPEZY"
SI_PREFIXES_ASCII = "yzafpnum kMGTPEZY"


def _clip(val, vmin, vmax):
    return vmin if val < vmin else vmax if val > vmax else val


def SI_scale(x: T.Union[float, int, decimal.Decimal], min_val=1e-25, use_unicode=True):
    """
    Return the recommended scale factor and SI prefix string for x.

    Example::

        siScale(0.0001)   # returns (1e6, 'μ')
        # This indicates that the number 0.0001 is best represented as 0.0001 * 1e6 = 100 μUnits

    """

    if isinstance(x, decimal.Decimal):
        x = float(x)
    try:
        if not math.isfinite(x):
            return (1, "")
    except Exception:
        raise
    if abs(x) < min_val:
        m = 0
    else:
        m = int(_clip(math.floor(math.log(abs(x)) / math.log(1000)), -9.0, 9.0))
    if m == 0:
        pref = ""
    elif m < -8 or m > 8:
        pref = "e%d" % (m * 3)
    else:
        if use_unicode:
            pref = SI_PREFIXES[m + 8]
        else:
            pref = SI_PREFIXES_ASCII[m + 8]
    m1 = -3 * m
    p = 10.0**m1
    return (p, pref)
