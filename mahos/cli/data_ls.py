#!/usr/bin/env python3

from os import path
import argparse

from ..util.io import list_attrs_h5
from ..node.log import DummyLogger

from ..msgs.confocal_msgs import Image, Trace
from ..msgs.odmr_msgs import ODMRData
from ..msgs.podmr_msgs import PODMRData
from ..msgs.iodmr_msgs import IODMRData
from ..msgs.hbt_msgs import HBTData
from ..msgs.spectroscopy_msgs import SpectroscopyData
from ..msgs.camera_msgs import Image as CameraImage


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog="mahos data print", description="Print list attributes and types of h5 data file(s)."
    )
    parser.add_argument("names", nargs="+", help="file names")
    args = parser.parse_args(args)

    return args


exts_to_data = {
    ".scan.h5": Image,
    ".trace.h5": Trace,
    ".odmr.h5": ODMRData,
    ".podmr.h5": PODMRData,
    ".iodmr.h5": IODMRData,
    ".hbt.h5": HBTData,
    ".spec.h5": SpectroscopyData,
    ".camera.h5": CameraImage,
}

logger = DummyLogger()


def get_ext(fn):
    head, e0 = path.splitext(fn)
    head, e1 = path.splitext(head)
    return e1 + e0


def print_attrs(fn):
    print(f"## {fn} ##")
    ext = get_ext(fn)
    if ext not in exts_to_data:
        logger.error(f"Unknown extension {ext}")
        return

    for key, type_ in list_attrs_h5(fn, exts_to_data[ext], logger):
        print(f"{key}: {type_}")


def main(args=None):
    args = parse_args(args)
    for fn in args.names:
        print_attrs(fn)
