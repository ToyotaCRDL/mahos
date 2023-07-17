#!/usr/bin/env python3

"""
Conversion module.

.. This file is a part of MAHOS project.

Defines generic conversion functions.

"""

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


def invert_mapping(d: T.Mapping[K, V]) -> T.Dict[V, T.List[K]]:
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
