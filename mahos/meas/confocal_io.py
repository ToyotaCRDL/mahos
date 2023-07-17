#!/usr/bin/env python3

"""
File I/O for confocal microscope.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
from os import path
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..msgs.confocal_msgs import Image, Trace, ScanDirection, update_image, update_trace
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5
from ..util.image import save_map, plot_map, plot_map_only


class ConfocalIO(object):
    """IO class for Confocal."""

    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_image(self, filename: str, image: Image, note: str = "") -> bool:
        """Save image to filename. return True on success."""

        return save_pickle_or_h5(filename, image, Image, self.logger, note=note)

    def load_image(self, filename: str) -> Image | None:
        """Load image from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, Image, self.logger)
        if d is not None:
            return update_image(d)

    def export_image(self, filename: str, image: Image, params: dict | None = None) -> bool:
        """Export the image to text or image files.

        :param params.only: set True to use image only mode
        :type params.only: bool
        :param params.pos: if given, position crosshair is plotted
        :type params.pos: tuple[float,float]|None
        :param params.vmin: lower limit of intensity value
        :type params.vmin: float
        :param params.vmax: upper limit of intensity value
        :type params.vmax: float
        :param params.invX: set True to invert x-axis
        :type params.invX: bool
        :param params.invY: set True to invert y-axis
        :type params.invY: bool
        :param params.cmap: matplotlib colormap name
        :type params.cmap: str
        :param params.aspect: matplotlib aspect
        :type params.aspect: str|float
        :param params.figsize: matplotlib figure size (w, h)
        :type params.figsize: tuple[float, float]
        :param params.fontsize: matplotlib fontsize
        :type params.fontsize: float
        :param params.dpi: matplotlib dpi
        :type params.dpi: float

        """

        if params is None:
            params = {}
        if not isinstance(image, Image):
            self.logger.error(f"Given object ({image}) is not an Image.")
            return False

        fn_head, ext = path.splitext(filename)
        if ext in (".txt", ".csv"):
            return self._export_image_text(filename, image)
        elif ext in (".png", ".pdf", ".eps"):
            return self._export_image_image(filename, image, params)
        else:
            self.logger.error(f"Unknown extension to export image: {filename}")
            return False

    def export_view(
        self,
        filename: str,
        images: tuple[Image, Image, Image],
        pos_xyz: tuple[float, float, float],
        params: tuple[dict, dict, dict] | None = None,
    ) -> bool:
        """Export the current view to image file.

        :param images: images in (XY, XZ, YZ).
        :param pos_xyz: positon in (x, y, z).
        :param params.vmin: lower limit of intensity value
        :type params.vmin: float
        :param params.vmax: upper limit of intensity value
        :type params.vmax: float
        :param params.invX: set True to invert x-axis
        :type params.invX: bool
        :param params.invY: set True to invert y-axis
        :type params.invY: bool
        :param params.cmap: matplotlib colormap name
        :type params.cmap: str
        :param params.pos_color: matplotlib color for position crosshair
        :type params.pos_color: str

        """

        def plot(image, param, pos, fig, ax):
            _x, _y = self.get_xylabel(image)
            df = self.create_dataframe(image)
            fn_head, ext = path.splitext(filename)
            units = {_x: "um", _y: "um", "I": image.cunit}
            plot_map(
                filename,
                df,
                _x,
                _y,
                "I",
                units=units,
                fig=fig,
                ax=ax,
                vmin=param.get("vmin"),
                vmax=param.get("vmax"),
                invX=param.get("invX", False),
                invY=param.get("invY", False),
                aspect="equal",
                exts=(),
                cmap=param.get("cmap", "inferno"),
                plot_pos=pos,
                pos_color=param.get("pos_color", "#2bb32b"),
                mapped=True,
                nocheck=True,
            )

        if params is None:
            params = [{}, {}, {}]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        positions = ((pos_xyz[0], pos_xyz[1]), (pos_xyz[0], pos_xyz[2]), (pos_xyz[1], pos_xyz[2]))
        axes_ = axes[0, 0], axes[0, 1], axes[1, 0]
        for img, param, pos, ax in zip(images, params, positions, axes_):
            if img is not None:
                plot(img, param, pos, fig, ax)

        plt.savefig(filename)
        plt.close()
        self.logger.info(f"Exported View to {filename}.")
        return True

    def save_trace(self, filename: str, trace: Trace, note: str = ""):
        """Save trace to filename. return True on success."""

        return save_pickle_or_h5(filename, trace, Trace, self.logger, note=note)

    def load_trace(self, filename: str) -> Trace | None:
        """Load trace from filename. return None if load is failed."""

        d = load_pickle_or_h5(filename, Trace, self.logger)
        if d is not None:
            return update_trace(d)

    def export_trace(self, filename: str, trace: Trace, params: dict | None = None) -> bool:
        """Export the trace to text or image files."""

        if not isinstance(trace, Trace):
            self.logger.error(f"Given object ({trace}) is not a Trace.")
            return False

        if params is None:
            params = {}

        fn_head, ext = path.splitext(filename)
        if ext == ".txt":
            return self._export_trace_text(filename, trace)
        elif ext in (".png", ".pdf", ".eps"):
            return self._export_trace_image(filename, trace, params)
        else:
            self.logger.error(f"Unknown extension to export trace: {filename}")
            return False

    def get_xarray(self, image: Image):
        return np.linspace(image.params["xmin"], image.params["xmax"], image.params["xnum"])

    def get_yarray(self, image: Image):
        return np.linspace(image.params["ymin"], image.params["ymax"], image.params["ynum"])

    def get_arrays(self, image: Image):
        return self.get_xarray(image), self.get_yarray(image)

    def get_xylabel(self, image: Image):
        if image.direction == ScanDirection.XY:
            return "x", "y"
        elif image.direction == ScanDirection.XZ:
            return "x", "z"
        elif image.direction == ScanDirection.YZ:
            return "y", "z"
        else:
            raise ValueError("invalid direction {} is given.".format(image.direction))

    def create_dataframe(self, image: Image):
        xs, ys = image.image.shape
        xar = self.get_xarray(image)[:xs]
        yar = self.get_yarray(image)[:ys]
        df = pd.DataFrame(data=image.image.T, index=yar, columns=xar)

        return df

    def _export_image_text(self, fn, image: Image) -> bool:
        fn_head, ext = path.splitext(fn)
        _x, _y = self.get_xylabel(image)
        df = self.create_dataframe(image)
        save_map(
            fn_head,
            df,
            _x,
            _y,
            "I",
            ext=ext,
            delimiter=",",
            mapped=True,
            nocheck=True,
            rename=False,
        )
        self.logger.info(f"Exported Image to {fn}.")
        return True

    def _export_image_image(self, fn, image: Image, params: dict) -> bool:
        _x, _y = self.get_xylabel(image)
        df = self.create_dataframe(image)

        fn_head, ext = path.splitext(fn)
        units = {_x: "um", _y: "um", "I": image.cunit}
        if params.get("only", False):
            plot_map_only(
                fn_head,
                df,
                _x,
                _y,
                "I",
                units=units,
                vmin=params.get("vmin"),
                vmax=params.get("vmax"),
                invX=params.get("invX", False),
                invY=params.get("invY", False),
                aspect=params.get("aspect", "auto"),
                figsize=params.get("figsize", (8, 6)),
                dpi=params.get("dpi", 100),
                exts=(ext,),
                cmap=params.get("cmap", "inferno"),
                plot_pos=params.get("pos"),
                mapped=True,
                nocheck=True,
            )
        else:
            plot_map(
                fn_head,
                df,
                _x,
                _y,
                "I",
                units=units,
                vmin=params.get("vmin"),
                vmax=params.get("vmax"),
                invX=params.get("invX", False),
                invY=params.get("invY", False),
                aspect=params.get("aspect", "equal"),
                figsize=params.get("figsize", (8, 6)),
                fontsize=params.get("fontsize", 20),
                dpi=params.get("dpi", 100),
                exts=(ext,),
                cmap=params.get("cmap", "inferno"),
                plot_pos=params.get("pos"),
                mapped=True,
                nocheck=True,
            )
        self.logger.info(f"Exported Image to {fn}.")
        return True

    def _export_trace_text(self, fn, trace: Trace):
        tim = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        with open(fn, "w", encoding="utf-8", newline="\n") as fo:
            fo.write("# PD trace data taken by mahos.meas.confocal. Saved at: %s\n" % (tim,))
            fo.write("# {}\n".format(" ".join([f"PD{i}" for i in range(trace.channels())])))
            fo.write("# {}\n".format(" ".join([trace.yunit] * trace.channels())))
        with open(fn, "ab") as fo:
            np.savetxt(fo, np.column_stack(trace.traces))
        self.logger.info(f"Exported Trace to {fn}.")
        return True

    def _export_trace_image(self, fn, trace: Trace, params: dict):
        plt.figure()
        if params.get("timestamp", False):
            for ch in range(len(trace.traces)):
                s, t = trace.valid_trace(ch)
                x = [datetime.fromtimestamp(s) for s in s.astype(np.float64) * 1e-9]
                plt.plot(x, t, label=f"PD{ch}")
        else:
            for ch in range(len(trace.traces)):
                plt.plot(trace.traces[ch], label=f"PD{ch}")
            plt.xlabel("Data points")

        plt.legend()
        plt.ylabel(f"Intensity ({trace.yunit})")
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        self.logger.info(f"Exported Trace to {fn}.")
        return True
