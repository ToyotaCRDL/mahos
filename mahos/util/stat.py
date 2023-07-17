#!/usr/bin/env python3

"""
Statistical Data Processing Utilities.

.. This file is a part of MAHOS project.

"""

import typing as T

import numpy as np


def filter_outlier_2d(
    data: np.ndarray, n: float = 4.0, axis: int = 0, both=False
) -> T.Tuple[np.ndarray, np.ndarray]:
    """filter outlier in 2D array data.

    This function look for outliers along one axis.
    Returns (not_outliers, outliers)

    :param n: data is considered outlier if (datapoint - mean) exceeds (n * sigma)
    :param axis: (0 or 1) look for outliers (take mean and std) along this axis.
    :param both: If True, look for both sides (include too-small values)

    """

    if axis == 0:
        d = data
    elif axis == 1:
        d = data.T
    else:
        raise ValueError("axis must be 0 or 1")

    std = np.std(data, axis=axis)
    if np.any(std == 0.0):
        # std has exact zero. abort.
        if axis == 0:
            return data, np.empty((0, data.shape[1]))
        else:
            return data, np.empty((data.shape[0], 0))

    k = (d - np.mean(data, axis=axis)) / std
    if both:
        k = np.abs(k)

    outlier = np.any(k > n, axis=1)
    not_ = np.logical_not(outlier)
    if axis == 0:
        return data[not_, :], data[outlier, :]
    else:
        return data[:, not_], data[:, outlier]


def simple_moving_average(a: np.ndarray, n) -> np.ndarray:
    """Simple (non-weighted) moving average of given array `a`.

    :param a: 1D-array to be moving-averaged.
    :param n: sampling width.
    :returns: the length is shortened by n - 1.

    """

    r = np.cumsum(a)
    r[n:] = r[n:] - r[:-n]
    return r[n - 1 :] / n
