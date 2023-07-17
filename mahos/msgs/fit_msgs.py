#!/usr/bin/env python3

"""
Message Types for Fitting in Meas Node.

.. This file is a part of MAHOS project.

"""

import enum

from .common_msgs import Message


class PeakType(Message, enum.Enum):
    Gaussian = 0
    Lorentzian = 1
    Voigt = 2


def str_to_peak_type(s: str) -> PeakType:
    s = s.lower()
    if s.startswith("v"):
        return PeakType.Voigt
    elif s.startswith("l"):
        return PeakType.Lorentzian
    elif s.startswith("g"):
        return PeakType.Gaussian
    raise ValueError("Unknown peak type")
