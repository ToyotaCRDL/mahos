#!/usr/bin/env python3

"""
Plot / Convertion functions for 2D image data.

.. This file is a part of MAHOS project.

"""


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import cui


def to_map(df, xl, yl, cl):
    """Convert parallel DataFrame to mapped DataFrame.
    col is used to designate which col (of input DF) to map.
    """

    ind = pd.MultiIndex.from_arrays(np.array(df.loc[:, [yl, xl]]).T)
    df.index = ind
    s = df.loc[:, cl]

    return s.unstack()


def derive_map(df, xl, yl, cl, units=None, axis="x"):
    """Take derivative of the map DataFrame."""

    df = df.sort(axis=0).sort(axis=1)

    if axis == "x":
        step = df.columns[1] - df.columns[0]
        ar = np.array([np.gradient(ar, step) for ar in df.values])

        _cl = cl
        cl = "d%s_d%s" % (cl, xl)
        if units is not None:
            units[cl] = "%s/%s" % (units[_cl], units[xl])

    elif axis == "y":
        step = df.index[1] - df.index[0]
        ar = np.array([np.gradient(ar, step) for ar in df.values.T]).T
        _cl = cl
        cl = "d%s_d%s" % (cl, yl)
        if units is not None:
            units[cl] = "%s/%s" % (units[_cl], units[xl])

    new_df = pd.DataFrame(ar, index=df.index, columns=df.columns)

    return new_df, xl, yl, cl


def plot_map(
    fpath,
    df,
    xl,
    yl,
    cl,
    units=None,
    fig=None,
    ax=None,
    vmin=None,
    vmax=None,
    invX=False,
    invY=False,
    aspect="auto",
    figsize=(8, 6),
    fontsize=None,
    dpi=None,
    suffix="",
    exts=(".png",),
    cmap="inferno",
    mapped=False,
    nocheck=False,
    derivative=None,
    plot_pos=None,
    pos_color="#2bb32b",
    tight=True,
    cunit_only=True,
):
    """Plot DataFrame as pseudo-color image"""

    if not mapped:
        df2 = to_map(df, xl, yl, cl)
        if derivative is not None:
            df2, xl, yl, cl = derive_map(df2, xl, yl, cl, units=units, axis=derivative)
    else:
        df2 = df

    x, y = np.array(df2.columns), np.array(df2.index)
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    X, Y = np.meshgrid(x, y)

    if fontsize:
        plt.rcParams["font.size"] = fontsize
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)
    ax.set_aspect(aspect)
    plt.tick_params(axis="both", direction="out")
    map_ = ax.pcolormesh(X, Y, np.array(df2), cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(map_, ax=ax)
    if invX:
        ax.set_xlim(xmax, xmin)
    else:
        ax.set_xlim(xmin, xmax)
    if invY:
        ax.set_ylim(ymax, ymin)
    else:
        ax.set_ylim(ymin, ymax)

    if units is not None:
        ax.set_xlabel(f"{xl} ({units[xl]})")
        ax.set_ylabel(f"{yl} ({units[yl]})")
        if cunit_only:
            cbar.set_label(units[cl])
        else:
            cbar.set_label(f"{cl} ({units[cl]})")
    else:
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        cbar.set_label(cl)

    if plot_pos is not None:
        x, y = plot_pos
        ax.plot(x, y, color=pos_color, marker="x", markeredgewidth=1.0, markersize=8)

    if tight:
        plt.tight_layout()

    for ext in exts:
        fn = fpath + ext
        if nocheck or cui.check_existance(fn):
            plt.savefig(fn, dpi=dpi)
    if exts:
        plt.close()


def plot_map_only(
    fpath,
    df,
    xl,
    yl,
    cl,
    units=None,
    fig=None,
    ax=None,
    vmin=None,
    vmax=None,
    invX=False,
    invY=False,
    aspect="auto",
    figsize=(8, 6),
    dpi=None,
    suffix="",
    exts=(".png",),
    cmap="inferno",
    mapped=False,
    nocheck=False,
    derivative=None,
    plot_pos=None,
    pos_color="#2bb32b",
):
    """Plot DataFrame as pseudo-color image.

    Difference with plot_map(): This function saves the image only
    (No axis, colorbar, labels, etc.).

    """

    if not mapped:
        df2 = to_map(df, xl, yl, cl)
        if derivative is not None:
            df2, xl, yl, cl = derive_map(df2, xl, yl, cl, units=units, axis=derivative)
    else:
        df2 = df

    x, y = np.array(df2.columns), np.array(df2.index)
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    X, Y = np.meshgrid(x, y)

    if fig is None:
        fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    fig.add_axes(ax)
    ax.set_axis_off()
    ax.set_aspect(aspect)
    ax.pcolormesh(X, Y, np.array(df2), cmap=cmap, vmin=vmin, vmax=vmax)
    if invX:
        ax.set_xlim(xmax, xmin)
    else:
        ax.set_xlim(xmin, xmax)
    if invY:
        ax.set_ylim(ymax, ymin)
    else:
        ax.set_ylim(ymin, ymax)

    if plot_pos is not None:
        x, y = plot_pos
        ax.plot(x, y, color=pos_color, marker="x", markeredgewidth=1.0, markersize=8)

    for ext in exts:
        fn = fpath + ext
        if nocheck or cui.check_existance(fn):
            plt.savefig(fn, dpi=dpi)
    if exts:
        plt.close()


def save_map(
    fpath,
    df,
    xl,
    yl,
    cl,
    ext=".txt",
    suffix="",
    delimiter=",",
    mapped=False,
    nocheck=False,
    derivative=None,
    rename=True,
):
    if not mapped:
        df2 = to_map(df, xl, yl, cl)

        if derivative is not None:
            df2, xl, yl, cl = derive_map(df2, xl, yl, cl, axis=derivative)
    else:
        df2 = df

    if rename:
        save_fpath = os.path.splitext(fpath)[0] + "_%s%s%s%s%s" % (xl, yl, cl, suffix, ext)
    else:
        save_fpath = fpath + ext

    if nocheck or cui.check_existance(save_fpath):
        df2.to_csv(save_fpath, sep=delimiter, index_label="%s\\%s/%s" % (yl, cl, xl))


def save_image(
    fpath,
    img: np.ndarray,
    fig=None,
    ax=None,
    xlabel="",
    ylabel="",
    title="",
    aspect="equal",
    figsize=(8, 6),
    vmin=None,
    vmax=None,
    cmap="inferno",
    patches=(),
    lines=(),
    show=False,
    dpi=None,
    fontsize=None,
    tight=True,
):
    """Save 2D monochrome image data as pseudo-color image.

    The image is without spatial unit, like camera image.

    """

    if fontsize:
        plt.rcParams["font.size"] = fontsize
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)
    ax.set_aspect(aspect)

    # assume row-major order (Height, Width)
    hlen, wlen = img.shape
    h = np.linspace(0.5, hlen - 0.5, hlen)
    w = np.linspace(0.5, wlen - 0.5, wlen)
    W, H = np.meshgrid(w, h)

    plt.pcolormesh(W, H, img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.xlim(0, wlen)
    plt.ylim(hlen, 0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    for p in patches:
        plt.gca().add_patch(p)
    for l in lines:
        plt.gca().add_line(l)

    if tight:
        plt.tight_layout()

    if fpath is not None:
        plt.savefig(fpath, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def apply_binning(data, binning=1):
    if data.shape[0] % binning != 0 or data.shape[1] % binning != 0:
        print("the data shape cannot be devided by binning")
        return

    new_data = np.zeros((data.shape[0] // binning, data.shape[1] // binning))
    for i in range(new_data.shape[0]):
        for j in range(new_data.shape[1]):
            new_data[i, j] = np.mean(
                data[i * binning : (i + 1) * binning, j * binning : (j + 1) * binning]
            )

    return new_data
