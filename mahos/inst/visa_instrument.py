#!/usr/bin/env python3

"""
Base class for visa instrument.

.. This file is a part of MAHOS project.
"""

import pyvisa

from .instrument import Instrument


class VisaInstrument(Instrument):
    """Base class for visa instrument. Implements common visa set/query commands."""

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("resource",))

        rm = pyvisa.ResourceManager()
        self.inst = rm.open_resource(self.conf.get("resource"))
        self.logger.info("opened {} on {}".format(name, self.inst.resource_name))

        for attr in ("write_termination", "read_termination", "timeout", "baud_rate"):
            if attr in conf:
                setattr(self.inst, attr, conf[attr])

        if self.conf.get("clear", True):
            self.inst.clear()

        if self.conf.get("query_idn", True):
            self.logger.info("*IDN? > " + self.query_idn())

    def __repr__(self):
        return "VisaInstrument({}, {})".format(self.full_name(), self.inst.resource_name)

    def rst(self) -> bool:
        self.inst.write("*RST")
        return True

    def cls(self) -> bool:
        self.inst.write("*CLS")
        return True

    def rst_cls(self) -> bool:
        self.inst.write("*RST;*CLS")
        return True

    def query_opc(self, delay=None) -> bool:
        return self.inst.query("*OPC?", delay=delay) == "1"

    def query_error(self):
        return self.inst.query("SYST:ERR?")

    def check_error(self) -> bool:
        ret = self.query_error()
        try:
            s = ret.split(",")
            code = int(s[0])
            msg = ",".join(s[1:]).strip("'\"")
            if not code:
                return True
            self.logger.error(f"{code}: {msg}")
            return False
        except Exception:
            self.logger.exception(f"Error parsing error message: {ret}")
            return False

    def trg(self) -> bool:
        self.inst.write("*TRG")
        return True

    def query_idn(self):
        return self.inst.query("*IDN?")

    def flush_writebuf(self):
        self.inst.flush(pyvisa.constants.BufferOperation.flush_write_buffer)

    # Standard API

    def close(self):
        """Close visa resource explicitly."""

        if hasattr(self, "inst"):
            self.logger.info("Closing Visa Resource {}.".format(self.inst.resource_name))
            self.inst.close()

    def reset(self) -> bool:
        return self.rst_cls()
