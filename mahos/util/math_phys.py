#!/usr/bin/env python3

"""

Math and Physics module

.. This file is a part of MAHOS project.

"""

import numpy as np

# Physical Constants (normally SI units)
# Values taken from http://physics.nist.gov/cuu/Constants/
eec = 1.602176565e-19
planck_h = 6.62606957e-34
dirac_h = 1.054571726e-34
dirac_h_eVs = 6.58211928e-16  # eV s
ee_h = eec**2 / planck_h
electric_const = 8.854187817e-12
kb = 1.3806488e-23
kb_eV = 8.6173324e-5  # in eV/K


def round_halfint(x):
    """Round float x to nearest integer or half integer.

    round_halfint(1.1) ==> 1.0
    round_halfint(1.2) ==> 1.0
    round_halfint(1.6) ==> 1.5
    round_halfint(1.8) ==> 2.0

    """

    s = np.sign(x)
    a = np.absolute(x)
    y = a - np.floor(a)

    if y < 0.250:
        return s * float(np.floor(a))
    elif y < 0.750:
        return s * (float(np.floor(a)) + 0.5)
    else:
        return s * float(np.ceil(a))
