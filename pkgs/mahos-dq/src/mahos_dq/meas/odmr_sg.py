#!/usr/bin/env python3

"""
Shared SG-related logic for ODMR.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

MOD_LABELS = ["iq_ext", "am_ext", "fm_ext", "iq_int", "am_int", "fm_int"]


def configure_modulation(sg, label: str, mod: dict) -> bool:
    """Configure ODMR SG modulation selected by the measurement label."""

    if label == "iq_ext":
        return sg.configure_iq_ext()
    elif label == "iq_int":
        return sg.configure_iq_int()
    elif label == "fm_ext":
        return sg.configure_fm_ext(mod["fm_deviation"])
    elif label == "fm_int":
        return sg.configure_fm_int(mod["fm_deviation"], mod["fm_rate"])
    elif label == "am_ext":
        return sg.configure_am_ext(mod["am_depth"], mod["am_log"])
    elif label == "am_int":
        return sg.configure_am_int(mod["am_depth"], mod["am_log"], mod["am_rate"])
    return True
