#!/usr/bin/env python3

"""
File I/O for Camera.

.. This file is a part of MAHOS project.

"""

from os import path
import typing as T

# import numpy as np
import matplotlib.pyplot as plt

from ..msgs.camera_msgs import Image
from ..node.log import DummyLogger
from ..util.io import save_pickle_or_h5, load_pickle_or_h5


class CameraIO(object):
    def __init__(self, logger=None):
        if logger is None:  # use DummyLogger on interactive use
            self.logger = DummyLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def save_data(self, fn, image: Image, note: str = "") -> bool:
        return save_pickle_or_h5(fn, image, Image, self.logger, note=note)

    def load_data(self, fn) -> T.Optional[Image]:
        return load_pickle_or_h5(fn, Image, self.logger)

    def export_data(self, fn, image: Image, params: T.Optional[dict] = None) -> bool:
        if params is None:
            params = {}

        if not isinstance(image, Image):
            self.logger.error(f"Given object ({image}) is not an Image.")
            return False

        ext = path.splitext(fn)[1]
        if ext in (".png", ".pdf", ".eps"):
            return self._export_data_image(fn, image, params)
        else:
            self.logger.error(f"Unknown extension to export image: {fn}")
            return False

    def _export_data_image(self, fn, image: Image, params):
        fig = plt.figure()
        fig.add_subplot(111)

        plt.imshow(image.image, origin="upper")

        plt.savefig(fn)
        plt.close()

        self.logger.info(f"Exported Image to {fn}.")
        return True
