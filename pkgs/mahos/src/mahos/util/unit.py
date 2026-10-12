#!/usr/bin/env python3

"""
Unit conversion module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

The SI_scale and SI_format functions are taken from the pyqtgraph project
(https://github.com/pyqtgraph/pyqtgraph), which is licensed under the MIT License:
Copyright 2012 Luke Campagnola, University of North Carolina at Chapel Hill.

The functions are redefined here because we sometimes want to avoid importing pyqtgraph
(from non-GUI codes) only for these functions.

"""

from __future__ import annotations
import math
import decimal

import mahos.util.validation as V

SI_PREFIXES = "yzafpnµm kMGTPEZY"
SI_PREFIXES_ASCII = "yzafpnum kMGTPEZY"


def _clip(val, vmin, vmax):
    return vmin if val < vmin else vmax if val > vmax else val


def SI_scale(x: float | int | decimal.Decimal, min_val=1e-25, use_unicode=True):
    """
    Return the recommended scale factor and SI prefix string for x.

    Example::

        SI_scale(0.0001)   # returns (1e6, 'μ')
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


def SI_format(x, precision=3, suffix="", space=True, error=None, min_val=1e-25, use_unicode=True):
    """
    Return the number x formatted in engineering notation with SI prefix.

    Example::
        SI_format(0.0001, suffix='V')  # returns "100 μV"

    """

    if space:
        space = " "
    else:
        space = ""

    (p, pref) = SI_scale(x, min_val, use_unicode)
    if not (len(pref) > 0 and pref[0] == "e"):
        pref = space + pref

    if error is None:
        fmt = "%." + str(precision) + "g%s%s"
        return fmt % (x * p, pref, suffix)
    else:
        if use_unicode:
            plusminus = space + "±" + space
        else:
            plusminus = " +/- "
        fmt = "%." + str(precision) + "g%s%s%s%s"
        return fmt % (
            x * p,
            pref,
            suffix,
            plusminus,
            SI_format(
                error,
                precision=precision,
                suffix=suffix,
                space=space,
                min_val=min_val,
                use_unicode=use_unicode,
            ),
        )


def dBm_to_Vpeak(power_dBm: float, impedance: float = 50.0) -> float:
    """Convert sine power in dBm (into `impedance`) to zero-to-peak voltage.

    ``V_pk = sqrt(2 * Z * P)`` with ``P = 1 mW * 10**(power_dbm / 10)``.
    Examples (50 Ohm): 0 dBm -> 0.3162 V peak; -10 dBm -> 0.100 V peak.

    """

    V.check_num(power_dBm, "power_dBm")
    V.check_pos_num(impedance, "impedance")
    return math.sqrt(2.0 * impedance * 1e-3 * 10.0 ** (power_dBm / 10.0))


def Vpeak_to_dBm(Vpeak: float, impedance: float = 50.0) -> float:
    """Inverse of :func:`dBm_to_Vpeak`: zero-to-peak sine voltage -> dBm."""

    V.check_pos_num(Vpeak, "volt_peak")
    V.check_pos_num(impedance, "impedance")
    return 10.0 * math.log10(Vpeak**2 / (2.0 * impedance) / 1e-3)
