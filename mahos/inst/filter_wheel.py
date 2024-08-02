#!/usr/bin/env python3

"""
Filter Wheel module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import os
import ctypes as C

from .instrument import Instrument
from ..msgs import param_msgs as P
from ..util.conv import invert_mapping


class Thorlabs_FW102(Instrument):
    """Thorlabs Filter Wheel FW102.

    You need to place the native (C) DLL somewhere.

    :param dll_dir: The directory path containing DLL.
    :type dll_dir: str
    :param dll_name: (default: FilterWheel102_win64.dll) The file name of DLL.
    :type dll_name: str
    :param serial: (default: "") Serial string to discriminate multiple wheels.
        Blank is fine if only one wheel is connected.
    :type serial: str
    :param pos_filter: Mapping from filter position (int) to name (str).
        Note that this mapping must be injective (every name must be unique).
    :type pos_filter: dict[int, str]
    :param baud_rate: (default: 115200) Baud rate for connection.
    :type baud_rate: int
    :param timeout: (default: 3) Timeout (in sec) for connection.
    :type timeout: int

    """

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        if "dll_dir" in self.conf:
            p = os.path.expanduser(self.conf["dll_dir"])
            os.environ["PATH"] += os.pathsep + p
            os.add_dll_directory(p)

        dll_name = self.conf.get("dll_name", "FilterWheel102_win64.dll")
        baud_rate = self.conf.get("baud_rate", 115200)
        timeout = int(self.conf.get("timeout", 3))

        if "pos_filter" in self.conf:
            self.pos_filter = {int(k): v for k, v in self.conf["pos_filter"].items()}
            self.filter_pos = invert_mapping(self.pos_filter)
            if any([len(v) > 1 for v in self.filter_pos.values()]):
                raise ValueError("conf['pos_filter']: every name must be unique.")
            self.filter_pos = {k: v[0] for k, v in self.filter_pos.items()}
        else:
            self.pos_filter = self.filter_pos = None

        self.dll = C.windll.LoadLibrary(dll_name)
        buf = C.create_string_buffer(1024)
        self.dll.List(buf, 1024)
        serials = buf.value.decode().split(",")[::2]

        if not serials:
            self.logger.error("No FW detected.")
            raise ValueError("No FW detected.")
        if len(serials) == 1:
            if "serial" in self.conf and self.conf["serial"] != serials[0]:
                self.logger.warn(
                    "Given serial {} looks wrong. Opening available one {} anyway.".format(
                        self.conf["serial"], serials[0]
                    )
                )
            self.handler = self.dll.Open(serials[0].encode(), baud_rate, timeout)
            self.logger.info(f"Opened FW ({serials[0]}) handler: {self.handler}")
        else:
            if "serial" not in self.conf:
                msg = "Must specify conf['serial'] as multiple FWs are detected."
                msg += "\nAvailable serials: " + ", ".join(serials)
                self.logger.error(msg)
                raise ValueError(msg)
            if self.conf["serial"] not in serials:
                msg = "Specified serial {} is not available. (not in ({}))".format(
                    self.conf["serial"], ", ".join(serials)
                )
                self.logger.error(msg)
                raise ValueError(msg)
            self.handler = self.dll.Open(self.conf["serial"].encode(), baud_rate, timeout)
            self.logger.info(f"Opened FW ({self.conf['serial']}) handler: {self.handler}")

        if self.handler < 0:
            raise RuntimeError(f"Negative handler received: {self.handler}")

    def get_pos(self) -> int:
        """Get wheel position (1-12). Returns 0 on error."""

        pos = C.c_int()
        ret = self.dll.GetPosition(self.handler, C.byref(pos))
        if ret != 0:
            self.logger.error("Error getting pos.")
            return 0
        return pos.value

    def get_filter(self) -> str:
        if self.pos_filter is None:
            self.logger.error("get_filter(): conf['pos_filter'] is not defined")
            return ""

        return self.pos_filter[self.get_pos()]

    def get_pos_count(self) -> int:
        """Get wheel position (1-12). Returns 0 on error."""

        count = C.c_int()
        ret = self.dll.GetPositionCount(self.handler, C.byref(count))
        if ret != 0:
            self.logger.error("Error getting pos count.")
            return 0
        return count.value

    def set_pos(self, pos: int) -> bool:
        """Set wheel position (1-12). Returns True on sucess."""

        ret = self.dll.SetPosition(self.handler, pos)
        if ret == 0:
            return True
        self.logger.error(f"Error setting pos: ret = {ret}")
        return False

    def close_resources(self):
        if hasattr(self, "dll") and hasattr(self, "handler") and self.handler >= 0:
            ret = self.dll.Close(self.handler)
            self.logger.info(f"Closed FW. ret = {ret}.")

    # Standard API

    def get_param_dict_labels(self) -> list[str]:
        """Get list of available ParamDict labels."""

        return ["pos", "filter"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        if label == "pos":
            return P.ParamDict(
                pos=P.IntChoiceParam(self.get_pos(), list(range(1, 1 + self.get_pos_count())))
            )
        elif label == "filter":
            if self.pos_filter is None:
                self.logger.error("conf['pos_filter'] is not defined")
                return None

            return P.ParamDict(
                filter=P.StrChoiceParam(self.get_filter(), self.pos_filter.values())
            )

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "pos":
            return self.set_pos(params["pos"])
        elif label == "filter":
            return self.set_pos(self.filter_pos[params["filter"]])
        else:
            return self.fail_with(f"unknown label {label}")

    def get(self, key: str, args=None, label: str = ""):
        if key == "pos":
            return self.get_pos()
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None
