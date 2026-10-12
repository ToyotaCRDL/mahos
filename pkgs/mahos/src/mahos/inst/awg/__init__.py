#!/usr/bin/env python3

"""
AWG (Arbitrary Waveform Generator) module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from mahos.inst.awg.spectrum import Spectrum_AWG_Core, Spectrum_AWG

__all__ = [
    "Spectrum_AWG_Core",
    "Spectrum_AWG",
]
