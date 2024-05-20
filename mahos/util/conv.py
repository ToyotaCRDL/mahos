#!/usr/bin/env python3

"""
Conversion module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

Defines generic conversion functions.

"""

from __future__ import annotations
import typing as T

import numpy as np
from scipy import fft


def step_to_num(start, stop, step, decimals=10):
    return int(round(abs((stop - start) / float(step)), decimals)) + 1


def num_to_step(start, stop, num):
    try:
        return abs(stop - start) / float(abs(num) - 1)
    except Exception:
        return 0.0


K = T.TypeVar("K")
V = T.TypeVar("V")


def args_to_list(args: tuple[list[V] | tuple[V, ...]] | list[V] | tuple[V, ...] | V) -> list[V]:
    """Convert possibly nested (dim = 2) / raw (dim = 0) arguments to list of values (dim = 1).

    Useful for functions accepting variable-length args (*args).
    Conversion examples:
    - [[v0, v1, ...]] -> [v0, v1, ...]
    - [v0, v1, ...] -> [v0, v1, ...]
    - v0 -> [v0]

    """

    if len(args) == 1:
        if isinstance(args[0], (list, tuple)):
            return list(args[0])
        else:
            return [args[0]]
    return args


def invert_mapping(d: T.Mapping[K, V]) -> dict[V, list[K]]:
    """Invert given mapping.

    New values (original keys) are stored as lists
    because the keys (original values) can be same for multiple values
    (Mapping is not always injective).

    """

    ret = {}
    for key, value in d.items():
        if value in ret:
            ret[value].append(key)
        else:
            ret[value] = [key]
    return ret


def real_fft(xdata, ydata, remove_dc=True):
    N = len(xdata)
    if N != len(ydata):
        raise ValueError("xdata and ydata has wrong length")

    freq = fft.fftfreq(N, abs(xdata[1] - xdata[0]))
    spec = fft.rfft(ydata)
    if remove_dc:
        spec[0] = 0

    return freq[: N // 2], np.abs(spec[: N // 2]) / N


def real_fftfreq(xdata):
    N = len(xdata)
    return fft.fftfreq(len(xdata), abs(xdata[1] - xdata[0]))[: N // 2]
