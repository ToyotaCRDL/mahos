#!/usr/bin/env python3

"""
exceptions.py

.. This file is a part of MAHOS project.
"""


class PlatformError(Exception):
    def __init__(self, description):
        import platform

        Exception.__init__(
            self, description + "\nUnsupported on this platform (%s)." % platform.platform()
        )


class PlatformBitnessError(Exception):
    def __init__(self, description):
        import platform

        Exception.__init__(
            self, description + "\nUnsupported on this bitness (%s)." % platform.architecture()[0]
        )


class InstError(Exception):
    """Exception for instrument's errors"""

    def __init__(self, name, description):
        self.instrument_name = name
        msg = "{}: {}".format(name, description)
        Exception.__init__(self, msg)
