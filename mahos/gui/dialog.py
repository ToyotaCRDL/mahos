#!/usr/bin/env python3

"""
Common dialogs

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

from . import Qt


def save_dialog(
    parent, default_path: str, name: str, pre_ext: str, enable_bz2: bool = False
) -> str:
    h5_ext = pre_ext + ".h5"
    pkl_ext = pre_ext + ".pkl"
    bz2_ext = pre_ext + ".pkl.bz2"
    exts = [h5_ext, pkl_ext]
    filters = [f"{name} HDF5 data (*{h5_ext})", f"{name} pickle data (*{pkl_ext})"]
    if enable_bz2:
        exts.append(bz2_ext)
        filters.append(f"{name} bzip2-compressed pickle data (*{bz2_ext})")
    fn = Qt.save_file_dialog(parent, f"Save {name} data", default_path, ";;".join(filters))

    if not fn:
        return ""
    if not any((fn.endswith(e) for e in exts)):
        fn += exts[0]
        return fn

    # On windows, the ext is unintentionally repeated (maybe Qt bug?).
    # fix the fn to be suffixed with ext only once.
    ext = exts[[fn.endswith(e) for e in exts].index(True)]
    while fn.endswith(ext):
        fn = fn[: fn.rfind(ext)]
    return fn + ext


def load_dialog(
    parent, default_path: str, name: str, pre_ext: str, enable_bz2: bool = False
) -> str:
    h5_ext = pre_ext + ".h5"
    pkl_ext = pre_ext + ".pkl"
    bz2_ext = pre_ext + ".pkl.bz2"
    exts = [h5_ext, pkl_ext]
    if enable_bz2:
        exts.append(bz2_ext)
    exts_s = " ".join(("*" + e for e in exts))
    return Qt.open_file_dialog(
        parent, f"Load {name} data", default_path, f"{name} data ({exts_s})"
    )


def export_dialog(
    parent, default_path: str, name: str, exts: list[str], append_first_ext: bool = True
) -> str:
    """open export dialog and return filename.

    :param append_first_ext: Change behaviour when extension of returned filename is unknown.
                             If True, append the default ext (first element in exts).
                             If False, return empty string (export will be skipped).

    """

    ext_filter = " ".join(f"*{e}" for e in exts)
    fn = Qt.save_file_dialog(
        parent, f"Export {name} data", default_path, f"supported formats ({ext_filter})"
    )
    if not fn:
        return ""
    if not any([fn.endswith(e) for e in exts]):
        if not append_first_ext:
            return ""
        fn += exts[0]
    return fn
