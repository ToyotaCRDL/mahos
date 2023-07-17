#!/usr/bin/env python3

"""
File I/O module.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import pickle
import bz2

import h5py

from ..msgs.data_msgs import Data


def save_pickle(
    fn, data: Data, Data_T, logger, note: str = "", compression=None, compression_opts=None
) -> bool:
    if not isinstance(data, Data_T):
        logger.error(f"Given object ({data}) is not {Data_T.__name__}.")
        return False
    try:
        if note:
            data.set_note(note)
        if compression == "bz2":
            with bz2.open(fn, "wb", compresslevel=compression_opts or 9) as f:
                pickle.dump(data, f)
        elif compression is not None:
            logger.error(f"Unknown compression: {compression}")
            return False
        else:
            with open(fn, "wb") as f:
                pickle.dump(data, f)
        logger.info(f"Saved {fn}.")
        return True
    except Exception:
        logger.exception(f"Error saving {fn}.")
        return False


def load_pickle(fn, Data_T, logger, compression=None):  # -> Optional[Data_T]
    try:
        if compression == "bz2":
            with bz2.open(fn, "rb") as f:
                data = pickle.load(f)
        elif compression is not None:
            logger.error(f"Unknown compression: {compression}")
            return None
        else:
            with open(fn, "rb") as f:
                data = pickle.load(f)
        if isinstance(data, Data_T):
            logger.info(f"Loaded {fn}.")
            return data
        else:
            logger.error(f"{fn} is not {Data_T.__name__}.")
            return None
    except Exception:
        logger.exception(f"Error loading {fn}.")
        return None


def save_h5(
    fn, data: Data, Data_T, logger, note: str = "", compression=None, compression_opts=None
) -> bool:
    if not isinstance(data, Data_T):
        logger.error(f"Given object ({data}) is not {Data_T.__name__}.")
        return False
    try:
        if note:
            data.set_note(note)
        with h5py.File(fn, "w") as f:
            data.to_h5(f, compression=compression, compression_opts=compression_opts)
        logger.info(f"Saved {fn}.")
        return True
    except Exception:
        logger.exception(f"Error saving {fn}.")
        return False


def load_h5(fn, Data_T, logger):  # -> Optional[Data_T]
    try:
        with h5py.File(fn, "r") as f:
            data = Data_T.of_h5(f)
        if isinstance(data, Data_T):
            logger.info(f"Loaded {fn}.")
            return data
        else:
            logger.error(f"{fn} is not {Data_T.__name__}.")
            return None
    except Exception:
        logger.exception(f"Error loading {fn}.")
        return None


def get_attrs_h5(fn, Data_T, keys: list[str] | str, logger):
    try:
        with h5py.File(fn, "r") as f:
            c_ver = Data_T().version()
            l_ver = Data_T.get_attr_h5(f, "_version", 0)
            if c_ver != l_ver:
                msg = f"{Data_T.__name__} version is different. current: {c_ver} loaded: {l_ver}"
                logger.warn(msg)
            if isinstance(keys, str):
                return Data_T.get_attr_h5(f, keys)
            else:
                return [Data_T.get_attr_h5(f, k) for k in keys]
    except Exception:
        logger.exception(f"Error getting attributes of {fn}.")
        return None


def list_attrs_h5(fn, Data_T, logger) -> list[tuple[str, str]]:
    try:
        with h5py.File(fn, "r") as f:
            c_ver = Data_T().version()
            l_ver = Data_T.get_attr_h5(f, "_version", 0)
            if c_ver != l_ver:
                msg = f"{Data_T.__name__} version is different. current: {c_ver} loaded: {l_ver}"
                logger.warn(msg)
            return Data_T.list_attr_h5(f)
    except Exception:
        logger.exception(f"Error getting list of attributes of {fn}.")
        return []


def save_pickle_or_h5(
    fn, data: Data, Data_T, logger, note: str = "", compression=None, compression_opts=None
) -> bool:
    if isinstance(fn, str) and fn.endswith(".pkl"):
        return save_pickle(fn, data, Data_T, logger, note)
    elif isinstance(fn, str) and fn.endswith(".pkl.bz2"):
        return save_pickle(
            fn, data, Data_T, logger, note, compression="bz2", compression_opts=compression_opts
        )
    else:
        return save_h5(
            fn,
            data,
            Data_T,
            logger,
            note,
            compression=compression,
            compression_opts=compression_opts,
        )


def load_pickle_or_h5(fn, Data_T, logger):  # -> Optional[Data_T]
    if isinstance(fn, str) and fn.endswith(".pkl"):
        return load_pickle(fn, Data_T, logger)
    elif isinstance(fn, str) and fn.endswith(".pkl.bz2"):
        return load_pickle(fn, Data_T, logger, compression="bz2")
    else:
        return load_h5(fn, Data_T, logger)
