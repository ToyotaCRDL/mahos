#!/usr/bin/env python3

from os import path
import argparse

from pprint import pprint

# from IPython.lib.pretty import pprint
import numpy as np

from ..util.io import get_attrs_h5, list_attrs_h5
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
        prog="mahos data print", description="Print attributes of h5 data file(s)."
    )
    parser.add_argument(
        "-k", "--keys", nargs="*", help="attribute keys to print. default is ['params']"
    )
    parser.add_argument(
        "-t", "--threshold", type=int, default=20, help="threshold for np.printoptions"
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="print all the attributes. if given, -k is ignored",
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


def decompose_keys(keys):
    raw_keys = []
    dict_keys = []
    for k in keys:
        ks = k.split(".")
        raw_keys.append(ks[0])
        dict_keys.append(ks[1:])
    return raw_keys, dict_keys


def print_all_attrs(fn):
    print(f"## {fn} ##")
    ext = get_ext(fn)
    if ext not in exts_to_data:
        logger.error(f"Unknown extension {ext}")
        return

    keys, _ = zip(*list_attrs_h5(fn, exts_to_data[ext], logger))
    values = get_attrs_h5(fn, exts_to_data[ext], keys, logger)
    if values is None:
        return
    for key, value in zip(keys, values):
        print(f"[{key}]")
        pprint(value, compact=True)


def print_attrs(fn, keys):
    print(f"## {fn} ##")
    ext = get_ext(fn)
    if ext not in exts_to_data:
        logger.error(f"Unknown extension {ext}")
        return

    raw_keys, dict_keys = decompose_keys(keys)
    values = get_attrs_h5(fn, exts_to_data[ext], raw_keys, logger)
    if values is None:
        return
    for key, value, dkeys in zip(keys, values, dict_keys):
        print(f"[{key}]")
        try:
            for dk in dkeys:
                value = value[dk]
            pprint(value, compact=True)
        except Exception:
            logger.exception("Error looking for dict values")


def main(args=None):
    args = parse_args(args)
    if not args.keys:
        args.keys = ["params"]
    with np.printoptions(threshold=args.threshold):
        for fn in args.names:
            if args.all:
                print_all_attrs(fn)
            else:
                print_attrs(fn, args.keys)
