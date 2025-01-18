#!/usr/bin/env python3

"""
Signal Generator module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import enum

from .visa_instrument import VisaInstrument


class Mode(enum.Enum):
    """This is for carrier frequency mode (CW or sweep) and won't indicate modulation."""

    UNCONFIGURED = 0
    CW = 1
    POINT_TRIG_FREQ_SWEEP = 2


class N5182B(VisaInstrument):
    """Keysight N5182B/N5172B Vector Signal Generator.

    :param power_bounds: Power bounds in dBm (min, max).
    :type power_bounds: tuple[float, float]
    :param freq_bounds: Frequency bounds in Hz (min, max).
    :type freq_bounds: tuple[float, float]
    :param point_trig_freq_sweep.trig: point trigger source on freq sweep. one of TRIG_SOURCE.
    :type point_trig_freq_sweep.trig: str
    :param point_trig_freq_sweep.ext_trig: ext trig source on freq sweep. one of EXT_TRIG_SOURCE.
    :type point_trig_freq_sweep.ext_trig: str
    :param point_trig_freq_sweep.sweep_out_ch: sweep output channel on freq sweep. one of (1, 2).
    :type point_trig_freq_sweep.sweep_out_ch: int

    """

    FREQ_MODE = ("FIXED", "FIX", "CW", "LIST")
    POWER_MODE = ("FIXED", "FIX", "LIST")
    TRIG_SOURCE = (
        "IMM",
        "IMMEDIATE",
        "BUS",
        "EXT",
        "EXTERNAL",
        "INT",
        "INTERNAL",
        "KEY",
        "TIM",
        "TIMER",
        "MAN",
        "MANUAL",
    )
    _EXT = ("EXT", "EXTERNAL")
    EXT_TRIG_SOURCE = ("TRIG1", "TRIGGER1", "TRIG2", "TRIGGER2", "PULS", "PULSE")
    TRIG_OUT_ROUTE = (
        "SWE",
        "SWEEP",
        "SETT",
        "SETTLED",
        "PVID",
        "PVIDEO",
        "PSYN",
        "PSYNC",
        "LXI",
        "PULS",
        "PULSE",
        "TRIG",
        "TRIGGER1",
        "TRIGGER2",
        "SFDONE",
        "NONE",
    )

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 20000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self.power_min, self.power_max = self.conf.get("power_bounds", (-144.0, 0.0))
        self.freq_min, self.freq_max = self.conf.get("freq_bounds", (9e3, 6e9))
        self._mode = Mode.UNCONFIGURED

        c = self.conf.get("point_trig_freq_sweep", {})
        self.point_trig_freq_sweep_conf = {
            "trig": c.get("trig", "EXT"),
            "ext_trig": c.get("ext_trig", "TRIGGER1"),
            "sweep_out_ch": c.get("sweep_out_ch", 2),
        }
        self.logger.debug(f"point_trig_freq_sweep conf: {self.point_trig_freq_sweep_conf}")

        c = self.conf.get("fm_ext", {})
        self.fm_ext_conf = {
            "source": c.get("source", "EXT"),
            "DC_coupling": c.get("DC_coupling", True),
            "impedance": c.get("impedance", 50),
        }
        self.logger.debug(f"fm_ext conf: {self.fm_ext_conf}")

        c = self.conf.get("am_ext", {})
        self.am_ext_conf = {
            "source": c.get("source", "EXT"),
            "DC_coupling": c.get("DC_coupling", True),
            "impedance": c.get("impedance", 50),
        }
        self.logger.debug(f"am_ext conf: {self.am_ext_conf}")

    def get_power_bounds(self):
        return self.power_min, self.power_max

    def get_freq_bounds(self):
        return self.freq_min, self.freq_max

    def get_bounds(self):
        return {
            "power": self.get_power_bounds(),
            "freq": self.get_freq_bounds(),
        }

    def query_power_condition(self) -> int:
        ans = int(self.inst.query("STAT:QUES:COND?"))

        if ans == 8:
            self.logger.warning("OUTPUT UNLEVELED")
        elif ans > 0:
            self.logger.warning(f"DATA QUESTIONABLE CONDITION REGISTER IS NON-ZERO: {ans}")

        return ans

    def set_output(self, on: bool, silent: bool = False) -> bool:
        if on:
            self.inst.write("OUTP:STAT ON")
            if not silent:
                self.logger.info("Output ON")
        else:
            self.inst.write("OUTP:STAT OFF")
            if not silent:
                self.logger.info("Output OFF")
        return True

    def set_init_cont(self, on: bool) -> bool:
        if on:
            self.inst.write("INIT:CONT ON")
        else:
            self.inst.write("INIT:CONT OFF")
        return True

    def initiate(self) -> bool:
        self.inst.write("INIT")
        return True

    def abort(self) -> bool:
        self.inst.write("ABOR")
        return True

    def trigger(self) -> bool:
        self.inst.write("TRIG")
        return True

    def set_freq_mode(self, mode: str) -> bool:
        """Set Frequency mode.

        Available values are defined in self.FREQ_MODE.

        """

        if mode.upper() not in self.FREQ_MODE:
            return self.fail_with("invalid frequency mode.")

        self.inst.write("FREQ:MODE " + mode)
        return True

    def set_power_mode(self, mode: str) -> bool:
        """Set Power mode.

        Available values are defined in self.POWER_MODE.

        """

        if mode.upper() not in self.POWER_MODE:
            return self.fail_with("invalid power mode.")

        self.inst.write("POW:MODE " + mode)
        return True

    def _fmt_freq(self, freq) -> str | None:
        """Format frequency as string.

        Frequency may be passed as string (e.g. '100 MHz') or number (1.0E8).
        The value is checked only if a number is passed.

        """

        if isinstance(freq, str):
            return freq
        elif isinstance(freq, float) or isinstance(freq, int):
            if freq < self.freq_min or freq > self.freq_max:
                raise ValueError("Invalid frequency.")
            return f"{freq:.12E}"
        else:
            self.logger.error("Invalid type {} of frequency {}".format(type(freq), freq))
            return None

    def set_freq_CW(self, freq) -> bool:
        f = self._fmt_freq(freq)
        if f is None:
            return False

        self.inst.write("FREQ " + f)
        return True

    def set_freq_range(self, start, stop) -> bool:
        start, stop = self._fmt_freq(start), self._fmt_freq(stop)
        if start is None or stop is None:
            return False

        self.inst.write("FREQ:STAR {};STOP {}".format(start, stop))
        return True

    def set_freq_list(self, freq_list) -> bool:
        fs = [self._fmt_freq(freq) for freq in freq_list]
        if any([f is None for f in fs]):
            return False

        cmd = "LIST:FREQ " + ",".join(fs)
        self.inst.write(cmd)
        return True

    def set_sweep_points(self, num: int) -> bool:
        self.inst.write(f"SWE:POIN {num:d}")
        return True

    def set_power(self, power_dBm) -> bool:
        if power_dBm < self.power_min or power_dBm > self.power_max:
            return self.fail_with("Invalid power.")

        self.inst.write(f"POW {power_dBm:.3f} dBm")
        return True

    def set_list_type(self, stepped=True) -> bool:
        if stepped:
            self.inst.write("LIST:TYPE STEP")
        else:
            self.inst.write("LIST:TYPE LIST")

        return True

    def set_route_trig_out(self, ch: int, route: str) -> bool:
        if ch not in (1, 2):
            return self.fail_with("invalid output channel")
        if route.upper() not in self.TRIG_OUT_ROUTE:
            return self.fail_with("invalid output route")

        self.inst.write(f"ROUT:TRIG{ch:d}:OUTP {route}")
        return True

    def set_trig_source(self, source: str, ext: str = "TRIGGER1") -> bool:
        """Set the sweep trigger source."""

        source = source.upper()
        if source not in self.TRIG_SOURCE:
            return self.fail_with("invalid trigger source.")

        self.inst.write("TRIG:SOUR " + source)

        if source not in self._EXT:
            return True

        if ext.upper() not in self.EXT_TRIG_SOURCE:
            return self.fail_with("invalid external trigger source.")

        self.inst.write("TRIG:EXT:SOUR " + ext)
        return True

    def set_point_trig_source(self, source: str, ext: str = "TRIGGER1") -> bool:
        """Set the stepped sweep point-to-point trigger source.

        This only applies when SWEep:GEN is set to STEPped.

        """

        source = source.upper()
        if source not in self.TRIG_SOURCE:
            return self.fail_with("invalid sweep trigger source.")

        self.inst.write("LIST:TRIG:SOUR " + source)

        if source not in self._EXT:
            return True

        if ext.upper() not in self.EXT_TRIG_SOURCE:
            return self.fail_with("invalid external trigger source")

        self.inst.write("LIST:TRIG:EXT:SOUR " + ext)
        return True

    def set_dm_source(self, source: str) -> bool:
        """Set digital (IQ) moduration source.

        By default (on RST*), INTernal is selected.

        """

        if source.upper() not in ("EXT", "EXTERNAL", "INT", "INTERNAL", "SUM"):
            return self.fail_with("invalid digital modulation source")

        self.inst.write(":DM:SOUR " + source)
        return True

    def set_dm_invert(self, invert: bool) -> bool:
        """Set digital (IQ) modulation polarity.

        If invert is True, Q signal is inverted.
        This will work only for internally generated I/Q signal.
        (NOT for external signal!)

        By default (on RST*), polarity is not inverted.

        """

        if invert:
            self.inst.write(":DM:POL INV")
        else:
            self.inst.write(":DM:POL NORM")
        return True

    def set_dm(self, on: bool) -> bool:
        """If on is True turn on digital modulation."""

        if on:
            self.inst.write(":DM:STAT ON")
            self.logger.info("Digital modulation ON.")
        else:
            self.inst.write(":DM:STAT OFF")
            self.logger.info("Digital modulation OFF.")
        return True

    def _set_analog_ext_opts(self, AM_FM: str, DC_coupling: bool, impedance: int) -> bool:
        cpl = "DC" if DC_coupling else "AC"
        if impedance not in (50, 600, 1000000):
            return self.fail_with(f"invalid impedance {impedance}")
        self.inst.write(f":{AM_FM}:EXT:COUP " + cpl)
        self.inst.write(f":{AM_FM}:EXT:IMP {impedance:d}")
        return True

    def set_fm_ext_opts(self, DC_coupling: bool, impedance: int) -> bool:
        return self._set_analog_ext_opts("FM", DC_coupling, impedance)

    def set_am_ext_opts(self, DC_coupling: bool, impedance: int) -> bool:
        return self._set_analog_ext_opts("AM", DC_coupling, impedance)

    def set_fm_source(self, source: str) -> bool:
        """Set FM source.

        By default (on RST*), FUNCtion[1] is selected.

        """

        # TODO: ignoring several valid values (SWEep, DUAL, ...)
        if source.upper() not in (
            "EXT",
            "EXT1",
            "EXT2",
            "FUNC",
            "FUNC1",
            "FUNCTION1",
            "FUNC2",
            "FUNCTION2",
        ):
            return self.fail_with("invalid FM source")

        self.inst.write(":FM:SOUR " + source)
        return True

    def set_fm_deviation(self, deviation_Hz: float) -> bool:
        """Set FM deviation in Hz."""

        self.inst.write(f":FM {deviation_Hz:.8E}")
        return True

    def set_fm(self, on: bool) -> bool:
        """If on is True turn on FM (Frequency Modulation)."""

        if on:
            self.inst.write(":FM:STAT ON")
            self.logger.info("Frequency modulation ON.")
        else:
            self.inst.write(":FM:STAT OFF")
            self.logger.info("Frequency modulation OFF.")
        return True

    def set_am_source(self, source: str) -> bool:
        """Set AM source.

        By default (on RST*), FUNCtion[1] is selected.

        """

        # TODO: ignoring several valid values (SWEep, DUAL, ...)
        if source.upper() not in (
            "EXT",
            "EXT1",
            "EXT2",
            "FUNC",
            "FUNC1",
            "FUNCTION1",
            "FUNC2",
            "FUNCTION2",
        ):
            return self.fail_with("invalid AM source")

        self.inst.write(":AM:SOUR " + source)
        return True

    def set_am_depth(self, depth: float, log: bool) -> bool:
        """Set AM depth."""

        if log:
            self.inst.write("AM:TYPE LOG")
            self.inst.write(f"AM:EXP {depth:.8f}")
        else:
            self.inst.write("AM:TYPE LIN")
            self.inst.write(f"AM {depth:.8f}")
        return True

    def set_am(self, on: bool) -> bool:
        """If on is True turn on AM (Amplitude Modulation)."""

        if on:
            self.inst.write(":AM:STAT ON")
            self.logger.info("Amplitude modulation ON.")
        else:
            self.inst.write(":AM:STAT OFF")
            self.logger.info("Amplitude modulation OFF.")
        return True

    def set_modulation(self, on: bool) -> bool:
        """If on is True turn on modulation."""

        if on:
            self.inst.write(":OUTP:MOD:STAT ON")
            self.logger.info("Modulation ON.")
        else:
            self.inst.write(":OUTP:MOD:STAT OFF")
            self.logger.info("Modulation OFF.")
        return True

    def configure_cw(self, freq, power, reset: bool = True) -> bool:
        """Setup Continuous Wave output with fixed freq and power."""

        self._mode = Mode.UNCONFIGURED
        success = (
            (self.rst_cls() if reset else True)
            and self.set_freq_mode("CW")
            and self.set_power_mode("FIX")
            and self.set_freq_CW(freq)
            and self.set_power(power)
            and self.check_error()
        )
        if success:
            self._mode = Mode.CW
            self.logger.info("Configured CW output.")
        else:
            self.logger.error("Failed to configure CW output.")
        return success

    def configure_iq_ext(self) -> bool:
        """Setup external IQ modulation mode."""

        success = (
            self.set_modulation(True)
            and self.set_dm_source("EXT")
            and self.set_dm(True)
            and self.check_error()
        )
        if success:
            self.logger.info("Configured external IQ modulation.")
        else:
            self.logger.error("Failed to configure external IQ modulation.")
        return success

    def configure_fm_ext(self, deviation) -> bool:
        """Setup external FM mode."""

        success = (
            self.set_modulation(True)
            and self.set_fm_source(self.fm_ext_conf["source"])
            and self.set_fm_ext_opts(
                self.fm_ext_conf["DC_coupling"], self.fm_ext_conf["impedance"]
            )
            and self.set_fm_deviation(deviation)
            and self.set_fm(True)
            and self.check_error()
        )
        if success:
            self.logger.info("Configured external FM.")
        else:
            self.logger.error("Failed to configure external FM.")
        return success

    def configure_am_ext(self, depth, log) -> bool:
        """Setup external AM mode."""

        success = (
            self.set_modulation(True)
            and self.set_am_source(self.am_ext_conf["source"])
            and self.set_am_ext_opts(
                self.am_ext_conf["DC_coupling"], self.am_ext_conf["impedance"]
            )
            and self.set_am_depth(depth, log)
            and self.set_am(True)
            and self.check_error()
        )
        if success:
            self.logger.info("Configured for external AM.")
        else:
            self.logger.error("Failed to configure external AM.")
        return success

    def configure_point_trig_freq_sweep(
        self,
        start,
        stop,
        num,
        power,
        reset: bool = True,
        trig="",
        ext_trig="",
        sweep_out_ch=0,
    ) -> bool:
        """Convenient function to set up triggered frequency sweep.

        sweep can be initiated by start() after this function.
        set_output(True) should be called before initiation if you want actual RF output.

        """

        self._mode = Mode.UNCONFIGURED
        success = (
            (self.rst_cls() if reset else True)
            and self.set_freq_mode("LIST")
            and self.set_power_mode("FIX")
            and self.set_trig_source("IMM")
            and self.set_point_trig_source(
                source=trig or self.point_trig_freq_sweep_conf["trig"],
                ext=ext_trig or self.point_trig_freq_sweep_conf["ext_trig"],
            )
            and self.set_list_type(stepped=True)
            and self.set_route_trig_out(
                sweep_out_ch or self.point_trig_freq_sweep_conf["sweep_out_ch"], "SETT"
            )
            and self.set_freq_range(start, stop)
            and self.set_sweep_points(num)
            and self.set_power(power)
            and self.check_error()
        )
        if success:
            self._mode = Mode.POINT_TRIG_FREQ_SWEEP
            self.logger.info("Configured point trigger freq sweep.")
        else:
            self.logger.info("Failed to configure point trigger freq sweep.")
        return success

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "bounds":
            return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "output":
            return self.set_output(value)
        elif key == "dm_source":
            return self.set_dm_source(value)
        elif key == "dm":
            return self.set_dm(value)
        elif key == "modulation":
            return self.set_modulation(value)
        elif key == "init_cont":
            return self.set_init_cont(value)
        elif key == "abort":
            return self.abort()
        else:
            self.logger.error("Unknown set() key.")
            return False

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label == "iq_ext":
            return self.configure_iq_ext()
        elif label == "fm_ext":
            if not self.check_required_params(params, "deviation"):
                return False
            return self.configure_fm_ext(params["deviation"])
        elif label == "am_ext":
            if not self.check_required_params(params, ("depth", "log")):
                return False
            return self.configure_am_ext(params["depth"], params["log"])
        elif label == "point_trig_freq_sweep":
            if not self.check_required_params(params, ("start", "stop", "num", "power")):
                return False
            return self.configure_point_trig_freq_sweep(
                params["start"],
                params["stop"],
                params["num"],
                params["power"],
                reset=params.get("reset", True),
                trig=params.get("trig", ""),
                ext_trig=params.get("ext_trig", ""),
                sweep_out_ch=params.get("sweep_out_ch", 0),
            )
        elif label == "cw":
            if not self.check_required_params(params, ("freq", "power")):
                return False
            return self.configure_cw(params["freq"], params["power"], params.get("reset", True))
        else:
            self.logger.error(f"Unknown label {label}")
            return False

    def start(self, label: str = "") -> bool:
        if self._mode != Mode.POINT_TRIG_FREQ_SWEEP:
            msg = f"start() is only for point_trig_freq_sweep (current mode: {self._mode}).\n"
            msg += "use set_output(True) to turn on RF output."
            return self.fail_with(msg)

        success = self.set_init_cont(True)
        self.logger.info("Started point_trig_freq_sweep.")
        return success

    def stop(self, label: str = "") -> bool:
        if self._mode != Mode.POINT_TRIG_FREQ_SWEEP:
            msg = f"stop() is only for point_trig_freq_sweep (current mode: {self._mode}).\n"
            msg += "use set_output(False) to turn off RF output."
            return self.fail_with(msg)

        success = self.abort() and self.set_init_cont(False)
        self.logger.info("Stopped point_trig_freq_sweep.")
        return success


class MG3710E(VisaInstrument):
    """Anritsu MG3710E Vector Signal Generator.

    If second SG (SG2) is available, only CW output from SG2 can be used through
    configure(label="cw2") set(key="output", label="2").

    :param power_bounds: Power bounds in dBm (min, max).
    :type power_bounds: tuple[float, float]
    :param freq_bounds: Frequency bounds in Hz (min, max).
    :type freq_bounds: tuple[float, float]
    :param point_trig_freq_sweep.trig: trigger source on frequency sweep. one of TRIG_SOURCE.
    :type point_trig_freq_sweep.trig: str

    """

    FREQ_MODE = ("FIXED", "FIX", "CW", "LIST")
    POWER_MODE = ("FIXED", "FIX", "LIST")
    TRIG_SOURCE = (
        "IMM",
        "IMMEDIATE",
        "BUS",
        "EXT",
        "EXTERNAL",
        "KEY",
        "TIM",
        "TIMER",
        "MAN",
        "MANUAL",
    )
    MARKER1_ROUTE = (
        "M1A",
        "M1B",
        "M2A",
        "M2B",
        "M3A",
        "M3B",
        "SG2M1A",
        "SG2M1B",
        "SG2M2A",
        "SG2M2B",
        "SG2M3A",
        "SG2M3B",
        "SYNC",
        "PT1",
        "PT2",
        "PT3",
        "PS1",
        "POINT",
        "PSY",
        "SG2PSY",
        "PVID",
        "SG2PVID",
        "SET",
        "SG2SET",
    )

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\r\n"
        if "timeout" not in conf:
            conf["timeout"] = 20000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self.power_min, self.power_max = self.conf.get("power_bounds", (-144.0, 0.0))
        self.freq_min, self.freq_max = self.conf.get("freq_bounds", (9e3, 6e9))
        self.power_min2, self.power_max2 = self.conf.get("power_bounds2", (-144.0, 0.0))
        self.freq_min2, self.freq_max2 = self.conf.get("freq_bounds2", (9e3, 6e9))
        self._mode = Mode.UNCONFIGURED

        c = self.conf.get("point_trig_freq_sweep", {})
        self.point_trig_freq_sweep_conf = {
            "trig": c.get("trig", "EXT"),
        }
        self.logger.debug(f"point_trig_freq_sweep conf: {self.point_trig_freq_sweep_conf}")

        c = self.conf.get("fm_ext", {})
        self.fm_ext_conf = {
            "DC_coupling": c.get("DC_coupling", True),
            "impedance": c.get("impedance", 600),
        }
        self.logger.debug(f"fm_ext conf: {self.fm_ext_conf}")

        c = self.conf.get("am_ext", {})
        self.am_ext_conf = {
            "DC_coupling": c.get("DC_coupling", True),
            "impedance": c.get("impedance", 600),
        }
        self.logger.debug(f"am_ext conf: {self.am_ext_conf}")

    def get_power_bounds(self):
        return self.power_min, self.power_max

    def get_freq_bounds(self):
        return self.freq_min, self.freq_max

    def get_bounds(self):
        return {
            "power": self.get_power_bounds(),
            "freq": self.get_freq_bounds(),
        }

    def get_power_bounds2(self):
        return self.power_min2, self.power_max2

    def get_freq_bounds2(self):
        return self.freq_min2, self.freq_max2

    def get_bounds2(self):
        return {
            "power": self.get_power_bounds2(),
            "freq": self.get_freq_bounds2(),
        }

    def query_power_condition(self):
        ans = int(self.inst.query("STAT:QUES:COND?"))

        if ans == 8:
            self.logger.warning("OUTPUT UNLEVELED")
        elif ans > 0:
            self.logger.warning(f"DATA QUESTIONABLE CONDITION REGISTER IS NON-ZERO: {ans}")

        return ans

    def set_output(self, on: bool, ch: int = 1, silent: bool = False) -> bool:
        if on:
            self.inst.write(f"OUTP{ch}:STAT ON")
            if not silent:
                self.logger.info(f"SG{ch} Output ON")
        else:
            self.inst.write(f"OUTP{ch}:STAT OFF")
            if not silent:
                self.logger.info(f"SG{ch} Output OFF")
        return True

    def set_init_cont(self, on: bool) -> bool:
        if on:
            self.inst.write("INIT:CONT ON")
        else:
            self.inst.write("INIT:CONT OFF")
        return True

    def initiate(self) -> bool:
        self.inst.write("INIT")
        return True

    def abort(self) -> bool:
        self.logger.warning("Abort is not supported for this instrument.")
        return False

    def set_freq_mode(self, mode: str, ch: int = 1) -> bool:
        """Set Frequency mode.

        Available values are defined in self.FREQ_MODE.

        """

        if mode.upper() not in self.FREQ_MODE:
            return self.fail_with("invalid frequency mode.")

        self.inst.write(f"SOUR{ch}:FREQ:MODE " + mode)
        return True

    def set_power_mode(self, mode: str, ch: int = 1) -> bool:
        """Set Power mode.

        Available values are defined in self.POWER_MODE.

        """

        if mode.upper() not in self.POWER_MODE:
            return self.fail_with("invalid power mode.")

        self.inst.write(f"SOUR{ch}:POW:MODE " + mode)
        return True

    def _fmt_freq(self, freq) -> str | None:
        """Format frequency as string.

        Frequency may be passed as string (e.g. '100 MHz') or number (1.0E8).
        The value is checked only if a number is passed.

        """

        if isinstance(freq, str):
            return freq
        elif isinstance(freq, float) or isinstance(freq, int):
            if freq < self.freq_min or freq > self.freq_max:
                raise ValueError("Invalid frequency.")
            return f"{freq:.12E}"
        else:
            self.logger.error("Invalid type {} of frequency {}".format(type(freq), freq))
            return None

    def set_freq_CW(self, freq, ch: int = 1) -> bool:
        f = self._fmt_freq(freq)
        if f is None:
            return False

        self.inst.write(f"SOUR{ch}:FREQ " + f)
        return True

    def set_freq_range(self, start, stop) -> bool:
        start, stop = self._fmt_freq(start), self._fmt_freq(stop)
        if start is None or stop is None:
            return False

        self.inst.write("FREQ:STAR {};FREQ:STOP {}".format(start, stop))
        return True

    def set_sweep_points(self, num: int) -> bool:
        self.inst.write(f"SWE:POIN {num:d}")
        return True

    def set_power(self, power_dBm, ch: int = 1) -> bool:
        if power_dBm < self.power_min or power_dBm > self.power_max:
            return self.fail_with("Invalid power.")

        self.inst.write(f"SOUR{ch}:POW {power_dBm:.3f} dBm")
        return True

    def set_list_type(self, stepped=True) -> bool:
        if stepped:
            self.inst.write("LIST:TYPE STEP")
        else:
            self.inst.write("LIST:TYPE LIST")

        return True

    def set_route_marker1(self, route: str) -> bool:
        if route.upper() not in self.MARKER1_ROUTE:
            return self.fail_with("invalid output route")

        self.inst.write(f"ROUT:OUTP:MARKER1 {route}")
        return True

    def set_trig_source(self, source: str) -> bool:
        """Set the sweep trigger source."""

        source = source.upper()
        if source not in self.TRIG_SOURCE:
            return self.fail_with("invalid trigger source.")

        self.inst.write(":TRIG:SEQ:SOUR " + source)
        return True

    def set_sweep_trigger(self, on: bool) -> bool:
        if on:
            self.inst.write("LIST:TRIG ON")
        else:
            self.inst.write("LIST:TRIG OFF")
        return True

    def set_sweep_trigger_mode(self, point: bool) -> bool:
        if point:
            self.inst.write("LIST:TRIG:MODE POINTS")
        else:
            self.inst.write("LIST:TRIG:MODE START")
        return True

    def set_sweep_trig_source(self, source: str) -> bool:
        """Set the stepped sweep trigger source.

        This only applies when SWEep:GEN is set to STEPped.

        """

        source = source.upper()
        if source not in self.TRIG_SOURCE:
            return self.fail_with("invalid sweep trigger source.")

        self.inst.write("LIST:TRIG:SOUR " + source)
        return True

    def set_sweep_dwell_time(self, time: str) -> bool:
        self.inst.write(f"SWE:DWELL {time:s}")
        return True

    def set_dm_source(self, source: str) -> bool:
        """Set digital (IQ) moduration source.

        By default (on RST*), INTernal is selected.

        """

        if source.upper() not in ("AEXT", "AEXTERNAL", "EXT", "EXTERNAL", "INT", "INTERNAL"):
            return self.fail_with("invalid digital modulation source")
        if source.startswith("EXT"):
            source = "A" + source

        self.inst.write(":DM:SOUR " + source)

        return True

    def set_dm_output(self, external: bool) -> bool:
        if external:
            self.inst.write(":DM:OUTP AEXT")
        else:
            self.inst.write(":DM:OUTP RFO")
        return True

    def set_dm_invert(self, invert: bool) -> bool:
        """Set digital (IQ) modulation polarity.

        If invert is True, Q signal is inverted.
        This will work only for internally generated I/Q signal.
        (NOT for external signal!)

        By default (on RST*), polarity is not inverted.

        """

        if invert:
            self.inst.write(":DM:POL INV")
        else:
            self.inst.write(":DM:POL NORM")
        return True

    def set_dm(self, on: bool) -> bool:
        """If on is True turn on digital modulation."""

        if on:
            self.logger.info("Digital modulation always ON.")
            return True
        else:
            self.logger.info("Digital modulation cannot be turned OFF.")
            return False

    def set_ext_mod_opts(self, DC_coupling: bool, impedance: int | str, ch: int = 1) -> bool:
        cpl = "DC" if DC_coupling else "AC"
        if impedance not in (50, 600, "HIZ"):
            return self.fail_with(f"invalid impedance {impedance}")
        self.inst.write(f"SOUR{ch}:EXTM:COUP " + cpl)
        self.inst.write(f"SOUR{ch}:EXTM:IMP {impedance:d}")
        return True

    def set_fm_source(self, source: str, ch: int = 1) -> bool:
        """Set FM source."""

        if source.upper() not in ("EXT", "INT", "INT1", "INT2"):
            return self.fail_with("invalid FM source")

        self.inst.write(f"SOUR{ch}:FM:SOUR " + source)
        return True

    def set_fm_deviation(self, deviation_Hz: float, ch: int = 1) -> bool:
        """Set FM deviation in Hz."""

        self.inst.write(f"SOUR{ch}:FM {deviation_Hz:.8E}")
        return True

    def set_fm(self, on: bool, ch: int = 1) -> bool:
        """If on is True turn on FM (Frequency Modulation)."""

        if on:
            self.inst.write(f"SOUR{ch}:FM:STAT ON")
            self.logger.info(f"SG{ch} Frequency modulation ON.")
        else:
            self.inst.write(f"SOUR{ch}:FM:STAT OFF")
            self.logger.info(f"SG{ch} Frequency modulation OFF.")
        return True

    def set_am_source(self, source: str, ch: int = 1) -> bool:
        """Set AM source."""

        if source.upper() not in ("EXT", "INT", "INT1", "INT2"):
            return self.fail_with("invalid AM source")

        self.inst.write(f"SOUR{ch}:AM:SOUR " + source)
        return True

    def set_am_depth(self, depth: float, log: bool, ch: int = 1) -> bool:
        """Set AM depth."""

        if log:
            self.inst.write(f"SOUR{ch}:AM:TYPE LOG")
            self.inst.write(f"SOUR{ch}:AM:EXP {depth:.8f}")
        else:
            self.inst.write(f"SOUR{ch}:AM:TYPE LIN")
            self.inst.write(f"SOUR{ch}:AM {depth:.8f}")
        return True

    def set_am(self, on: bool, ch: int = 1) -> bool:
        """If on is True turn on AM (Amplitude Modulation)."""

        if on:
            self.inst.write(f"SOUR{ch}:AM:STAT ON")
            self.logger.info(f"SG{ch} Amplitude modulation ON.")
        else:
            self.inst.write(f"SOUR{ch}:AM:STAT OFF")
            self.logger.info(f"SG{ch} Amplitude modulation OFF.")
        return True

    def set_modulation(self, on: bool, ch: int = 1) -> bool:
        """If on is True turn on modulation."""

        if on:
            self.inst.write(f":OUTP{ch}:MOD:STAT ON")
            self.logger.info(f"SG{ch} Modulation ON.")
        else:
            self.inst.write(f":OUTP{ch}:MOD:STAT OFF")
            self.logger.info(f"SG{ch} Modulation OFF.")
        return True

    def set_arb(self, on: bool, ch: int = 1) -> bool:
        if on:
            self.inst.write(f"SOUR{ch}:RAD:ARB ON")
        else:
            self.inst.write(f"SOUR{ch}:RAD:ARB OFF")
        return True

    def configure_cw(self, freq, power, ch: int = 1, reset: bool = True) -> bool:
        """Setup Continuous Wave output with fixed freq and power."""

        self._mode = Mode.UNCONFIGURED
        success = (
            (self.rst_cls() if reset else True)
            and self.set_freq_mode("CW", ch=ch)
            and self.set_power_mode("FIX", ch=ch)
            and self.set_freq_CW(freq, ch=ch)
            and self.set_power(power, ch=ch)
            and self.set_arb(False, ch=ch)
            and self.set_modulation(False, ch=ch)
            and self.check_error()
        )
        if success:
            self._mode = Mode.CW
            self.logger.info("Configured CW output.")
        else:
            self.logger.info("Failed to configure CW output.")
        return success

    def configure_iq_ext(self) -> bool:
        """Setup external IQ modulation mode."""

        if self._mode != Mode.CW:
            return self.fail_with("external IQ modulation is only for CW.")

        success = (
            self.set_modulation(True, ch=1)
            and self.set_dm_output(True)
            and self.set_dm_source("EXT")
            and self.set_dm(True)
            and self.check_error()
        )
        if success:
            self.logger.info("Configured external IQ modulation.")
        else:
            self.logger.error("Failed to configure external IQ modulation.")
        return success

    def configure_fm_ext(self, deviation, ch: int = 1) -> bool:
        """Setup external FM mode."""

        if self._mode != Mode.CW:
            return self.fail_with("external FM is only for CW.")

        success = (
            self.set_modulation(True, ch=ch)
            and self.set_fm_source("EXT", ch=ch)
            and self.set_ext_mod_opts(
                self.fm_ext_conf["DC_coupling"], self.fm_ext_conf["impedance"], ch=ch
            )
            and self.set_fm_deviation(deviation, ch=ch)
            and self.set_fm(True, ch=ch)
            and self.check_error()
        )
        if success:
            self.logger.info("Configured external FM.")
        else:
            self.logger.error("Failed to configure external FM.")
        return success

    def configure_am_ext(self, depth, log, ch: int = 1) -> bool:
        """Setup external AM mode."""

        if self._mode != Mode.CW:
            return self.fail_with("external AM is only for CW.")

        success = (
            self.set_modulation(True, ch=ch)
            and self.set_am_source("EXT", ch=ch)
            and self.set_ext_mod_opts(
                self.am_ext_conf["DC_coupling"], self.am_ext_conf["impedance"], ch=ch
            )
            and self.set_am_depth(depth, log, ch=ch)
            and self.set_am(True, ch=ch)
            and self.check_error()
        )
        if success:
            self.logger.info("Configured external AM.")
        else:
            self.logger.error("Failed to configure external AM.")
        return success

    def configure_point_trig_freq_sweep(
        self,
        start,
        stop,
        num,
        power,
        trig="",
        reset: bool = True,
    ) -> bool:
        """Convenient function to set up triggered frequency sweep.

        sweep can be initiated by start() after this function.
        set_output(True) should be called before initiation if you want actual RF output.

        """

        self._mode = Mode.UNCONFIGURED
        success = (
            (self.rst_cls() if reset else True)
            and self.set_freq_mode("LIST")
            and self.set_power_mode("FIX")
            and self.set_sweep_trigger(True)
            and self.set_sweep_trigger_mode(True)
            and self.set_sweep_trig_source(
                trig or self.point_trig_freq_sweep_conf["trig"],
            )
            and self.set_sweep_dwell_time("100US")
            and self.set_list_type(stepped=True)
            and self.set_route_marker1("SET")
            and self.set_freq_range(start, stop)
            and self.set_sweep_points(num)
            and self.set_power(power)
            and self.set_init_cont(True)
            and self.check_error()
        )
        if success:
            self._mode = Mode.POINT_TRIG_FREQ_SWEEP
            self.logger.info("Configured point trigger freq sweep.")
        else:
            self.logger.info("Failed to configure point trigger freq sweep.")
        return success

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "bounds":
            if label.endswith("2"):
                return self.get_bounds2()
            else:
                return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "output":
            if label.endswith("2"):
                return self.set_output(value, ch=2)
            else:
                return self.set_output(value, ch=1)
        elif key == "dm_source":
            return self.set_dm_source(value)
        elif key == "dm":
            return self.set_dm(value)
        elif key == "modulation":
            return self.set_modulation(value)
        elif key == "init_cont":
            return self.set_init_cont(value)
        elif key == "abort":
            return self.abort()
        else:
            self.logger.error("Unknown set() key.")
            return False

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label == "iq_ext":
            return self.configure_iq_ext()
        elif label == "iq_ext2":
            return self.fail_with("External IQ modulation isn't available for ch2.")
        elif label.startswith("fm_ext"):
            if not self.check_required_params(params, "deviation"):
                return False
            ch = 2 if label.endswith("2") else 1
            return self.configure_fm_ext(params["deviation"], ch=ch)
        elif label.startswith("am_ext"):
            if not self.check_required_params(params, ("depth", "log")):
                return False
            ch = 2 if label.endswith("2") else 1
            return self.configure_am_ext(params["depth"], params["log"], ch=ch)
        elif label == "point_trig_freq_sweep":
            if not self.check_required_params(params, ("start", "stop", "num", "power")):
                return False
            return self.configure_point_trig_freq_sweep(
                params["start"],
                params["stop"],
                params["num"],
                params["power"],
                trig=params.get("trig", ""),
                reset=params.get("reset", True),
            )
        elif label.startswith("cw"):
            if not self.check_required_params(params, ("freq", "power")):
                return False
            ch = 2 if label.endswith("2") else 1
            return self.configure_cw(
                params["freq"], params["power"], ch=ch, reset=params.get("reset", True)
            )
        else:
            self.logger.error(f"Unknown label {label}")
            return False

    def start(self, label: str = "") -> bool:
        if self._mode != Mode.POINT_TRIG_FREQ_SWEEP:
            msg = f"start() is only for point_trig_freq_sweep (current mode: {self._mode}).\n"
            msg += "use set_output(True) to turn on RF output."
            return self.fail_with(msg)

        success = self.initiate()
        self.logger.info("Started point_trig_freq_sweep.")
        return success

    def stop(self, label: str = "") -> bool:
        if self._mode != Mode.POINT_TRIG_FREQ_SWEEP:
            msg = f"stop() is only for point_trig_freq_sweep (current mode: {self._mode}).\n"
            msg += "use set_output(False) to turn off RF output."
            return self.fail_with(msg)

        success = self.set_init_cont(False)
        self.logger.info("Stopped point_trig_freq_sweep.")
        return success


class DS_SG(VisaInstrument):
    """DS Instruments Signal Generator.

    :param power_bounds: Power bounds in dBm (min, max).
    :type power_bounds: tuple[float, float]
    :param freq_bounds: Frequency bounds in Hz (min, max).
    :type freq_bounds: tuple[float, float]

    """

    def __init__(self, name, conf, prefix=None):
        conf["write_termination"] = "\n"
        conf["read_termination"] = "\r\n"
        conf["baud_rate"] = 115200
        conf["clear"] = False
        if "timeout" not in conf:
            conf["timeout"] = 5000.0

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self.power_min, self.power_max = self.conf.get("power_bounds", (-20.0, 10.0))
        self.freq_min, self.freq_max = self.conf.get("freq_bounds", (25e6, 6e9))
        self.set_buzzer(False)

    def get_power_bounds(self):
        return self.power_min, self.power_max

    def get_freq_bounds(self):
        return self.freq_min, self.freq_max

    def get_bounds(self):
        return {
            "power": self.get_power_bounds(),
            "freq": self.get_freq_bounds(),
        }

    def set_output(self, on: bool, silent: bool = False) -> bool:
        if on:
            self.inst.write("OUTP:STAT ON")
            if not silent:
                self.logger.info("Output ON")
        else:
            self.inst.write("OUTP:STAT OFF")
            if not silent:
                self.logger.info("Output OFF")
        return True

    def set_init_cont(self, on: bool) -> bool:
        if on:
            self.inst.write("INIT:CONT 1")
        else:
            self.inst.write("INIT:CONT 0")
        return True

    def initiate(self) -> bool:
        self.inst.write("INIT:IMM")
        return True

    def abort(self) -> bool:
        self.inst.write("ABORT")
        return True

    def _fmt_freq(self, freq) -> str | None:
        """Format frequency as string.

        Frequency may be passed as string (e.g. '100 MHz') or number (1.0E8).
        The value is checked only if a number is passed.

        """

        if isinstance(freq, str):
            return freq
        elif isinstance(freq, float) or isinstance(freq, int):
            if freq < self.freq_min or freq > self.freq_max:
                raise ValueError("Invalid frequency.")
            f = freq * 1e-6
            return f"{f:.4f}MHZ"
        else:
            self.logger.error("Invalid type {} of frequency {}".format(type(freq), freq))
            return None

    def set_freq_CW(self, freq) -> bool:
        f = self._fmt_freq(freq)
        if f is None:
            return False

        self.inst.write("FREQ:CW " + f)
        return True

    def set_freq_range(self, start, stop) -> bool:
        start, stop = self._fmt_freq(start), self._fmt_freq(stop)
        if start is None or stop is None:
            return False

        self.inst.write(f"FREQ:START {start}")
        self.inst.write(f"FREQ:STOP {stop}")
        return True

    def set_sweep_points(self, num: int) -> bool:
        self.inst.write(f"SWE:POINTS {num:d}")
        return True

    def set_sweep_dwell_time(self, time_ms: int) -> bool:
        self.inst.write(f"SWE:DWELL {time_ms:d}")
        return True

    def set_power(self, power_dBm) -> bool:
        if power_dBm < self.power_min or power_dBm > self.power_max:
            return self.fail_with("Invalid power.")

        self.inst.write(f"POWER {power_dBm:.3f} dBm")
        return True

    def set_trigger_mode(self, step: bool) -> bool:
        """Set the trigger mode: step or sweep."""

        m = "STEP" if step else "SWEEP"
        self.inst.write("TRIG:" + m)
        return True

    def set_list_type(self, stepped=True) -> bool:
        if stepped:
            self.inst.write("SWE:MODE SCAN")
        else:
            self.inst.write("SWE:MODE LIST")

        return True

    def set_buzzer(self, on: bool) -> bool:
        on_off = "ON" if on else "OFF"
        self.inst.write(f"*BUZZER {on_off}")

    def configure_cw(self, freq, power) -> bool:
        """Setup Continuous Wave output with fixed freq and power."""

        success = (
            # self.rst() and
            self.set_freq_CW(freq)
            and self.set_power(power)
        )

        return success

    def configure_point_trig_freq_sweep(self, start, stop, num, power) -> bool:
        """Convenient function to set up triggered frequency sweep.

        sweep can be initiated by calling initiate() or set_init_cont(True)
        after this function. set_output(True) should be called before initiation
        if you want actual RF output.

        if set_init_cont(True):
            Sweep is started automatically.
            On first trigger, the start frequency is out.
            On the trigger after the stop frequency, the start frequency is out.
            this behaviour is similar to N5182B.configure_freq_sweep()
            with start_trig=BUS(EXT), point_trig=BUS(EXT) and set_init_cont(True).

        if set_init_cont(False):
            Sweep is not started automatically.
            On first trigger, the start frequency is out.
            On the trigger after the stop frequency, the stop frequency is still out.
            The next trigger after the stop frequency, the start frequency is out.

        """

        success = (
            # self.rst() and
            self.set_trigger_mode(True)
            and self.set_list_type(stepped=True)
            and self.set_freq_range(start, stop)
            and self.set_sweep_points(num)
            and self.set_power(power)
        )

        self.logger.info("Configured freq sweep")

        return success

    # Standard API

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return self.query_opc(delay=args)
        elif key == "bounds":
            return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "output":
            return self.set_output(value)
        elif key == "freq":
            return self.set_freq_CW(value)
        elif key == "init_cont":
            return self.set_init_cont(value)
        elif key == "abort":
            return self.abort()
        else:
            self.logger.error("Unknown set() key.")
            return False

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label == "point_trig_freq_sweep":
            if not self.check_required_params(params, ("start", "stop", "num", "power")):
                return False
            return self.configure_point_trig_freq_sweep(
                params["start"],
                params["stop"],
                params["num"],
                params["power"],
            )
        elif label == "cw":
            if not self.check_required_params(params, ("freq", "power")):
                return False
            return self.configure_cw(params["freq"], params["power"])
        else:
            self.logger.error(f"Unknown label {label}")
            return False
