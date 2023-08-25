#!/usr/bin/env python3

import argparse

import h5py

from ..node.log import DummyLogger


def parse_args(args):
    parser = argparse.ArgumentParser(
        prog="mahos data note", description="Print or amend the note attribute in h5 data file(s)."
    )
    parser.add_argument("-a", "--amend", type=str, help="amend the note")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print the note even in amend mode"
    )
    parser.add_argument("names", nargs="+", help="file names")
    args = parser.parse_args(args)

    return args


logger = DummyLogger()


def print_note(fn):
    print(f"## {fn} ##")

    with h5py.File(fn, "r") as f:
        if "_note" in f.attrs:
            print(f.attrs["_note"])


def amend_note(fn, new_note: str, verbose: bool):
    def no_print(*args):
        pass

    pr = print if verbose else no_print

    pr(f"## {fn} ##")

    with h5py.File(fn, "r+") as f:
        if "_note" in f.attrs:
            pr(f.attrs["_note"])
            pr("-->")
        f.attrs["_note"] = new_note
        pr(new_note)


def main(args=None):
    args = parse_args(args)
    for fn in args.names:
        if args.amend:
            amend_note(fn, args.amend, args.verbose)
        else:
            print_note(fn)
