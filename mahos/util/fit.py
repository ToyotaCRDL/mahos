#!/usr/bin/env python3

"""
Fitting module.

.. This file is a part of MAHOS project.

"""

import numpy as np
from scipy import optimize


def gaussian2d(height, center_x, center_y, width_x, width_y, background=0.0):
    """Returns a gaussian function with the given parameters"""

    width_x = float(width_x)
    width_y = float(width_y)
    return (
        lambda x, y: height
        * np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)
        + background
    )


def gaussian2d_corr(height, x0, y0, sx, sy, rho, bg):
    """Returns a gaussian function with the given parameters"""
    sx = float(sx)
    sy = float(sy)
    return (
        lambda x, y: height
        * np.exp(
            -1
            / 2
            / (1 - rho**2)
            * (
                ((x - x0) / sx) ** 2
                - 2 * rho * (x - x0) / sx * (y - y0) / sy
                + ((y - y0) / sy) ** 2
            )
        )
        + bg
    )


def moments(data, with_bg=False):
    """Returns (height, x, y, width_x, width_y[, background])
    the gaussian parameters of a 2D distribution by calculating its moments
    """

    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    background = 0.0

    if with_bg:
        return height, x, y, width_x, width_y, background
    else:
        return height, x, y, width_x, width_y


def fit_gaussian2d(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    """

    params = moments(data)

    def errorfunction(p):
        return np.ravel(gaussian2d(*p)(*np.indices(data.shape)) - data)

    p, ier = optimize.leastsq(errorfunction, params)
    # print(p, ier)
    return p, ier
