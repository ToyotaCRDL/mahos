#!/usr/bin/env python3

"""
Overlay for Photo Detector + Programmable Amplifier.

.. This file is a part of MAHOS project.

"""

import numpy as np

from .overlay import InstrumentOverlay
from ..msgs import param_msgs as P
from ..lockin import LI5640
from ..pd import LUCI_OE200, LockinAnalogPD


class OE200_LI5640(InstrumentOverlay):
    """FEMTO Messtechnik OE-200 Variable Gain Photoreceiver and with LI5640 Lockin & DAQ AnalogIn.

    :param luci: a LUCI_OE200 instance
    :type luci: LUCI_OE200
    :param li5640: a LI5640 instance
    :type li5640: LI5640
    :param pd: a LockinAnalogPD instance. gain should be 1.
    :type pd: LockinAnalogPD

    """

    def __init__(self, name, conf, prefix=None):
        InstrumentOverlay.__init__(self, name, conf=conf, prefix=prefix)
        self.luci: LUCI_OE200 = self.conf.get("luci")
        self.li5640: LI5640 = self.conf.get("li5640")
        self.pd: LockinAnalogPD = self.conf.get("pd")
        self.add_instruments(self.luci, self.li5640, self.pd)

        self.init_lockin()

    def init_lockin(self):
        #  These settings should not be changed afterwards.
        # TODO: lock these value at LI5640 side
        self.li5640.set_data1(LI5640.Data1.X)
        self.li5640.set_data2(LI5640.Data2.Y)
        self.li5650.set_Xoffset_enable(False)
        self.li5650.set_Yoffset_enable(False)
        self.li5640.set_data_normalization(False)

    def gain(self) -> tuple[float, float]:
        v1_gain = self.li5640._data1_expand * 10.0 / self.li5650._volt_sensitivity
        v2_gain = self.li5640._data2_expand * 10.0 / self.li5650._volt_sensitivity
        return v1_gain * self.luci.gain, v2_gain * self.luci.gain

    def _convert_data(self, data: np.ndarray | np.cdouble) -> np.ndarray | np.cdouble:
        gain_r, gain_i = self.gain()

        if isinstance(data, np.cdouble):
            # read_on_demand
            return np.cdouble(data.real / gain_r, data.imag / gain_i)
        else:
            # buffered read
            data.real /= gain_r
            data.imag /= gain_i
            return data

    def _convert_all_data(self, data: list[np.ndarray]):
        return [self._convert_data(d) for d in data]

    def set(self, key: str, value=None) -> bool:
        # no set() key for pd
        key = key.lower()
        if key in ("led", "gain", "coupling"):
            return self.luci.set(key, value)
        else:
            return self.li5640.set(key, value)

    def get(self, key: str, args=None):
        if key == "data":
            return self._convert_data(self.pd.get(key, args))
        elif key == "all_data":
            return self._convert_all_data(self.pd.get(key, args))
        elif key == "unit":
            return "W"
        elif key in ("devices", "id", "pin", "product"):
            return self.luci.get(key, args)
        else:
            return self.li5640.get(key, args)

    def get_param_dict_labels(self, group: str) -> list[str]:
        return ["luci", "li5640"]

    def get_param_dict(
        self, label: str = "", group: str = ""
    ) -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label` in `group`."""

        if label == "luci":
            return self.luci.get_param_dict(label, group)
        elif label == "li5640":
            return self.li5640.get_param_dict(label, group)
        else:
            return self.pd.get_param_dict(label, group)

    def configure(self, params: dict, label: str = "", group: str = "") -> bool:
        if label == "luci":
            return self.luci.configure(params, label, group)
        elif label == "li5640":
            return self.li5640.configure(params, label, group)
        else:
            return self.pd.configure(params, label, group)
