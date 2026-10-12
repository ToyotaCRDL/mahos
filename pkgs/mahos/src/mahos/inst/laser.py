#!/usr/bin/env python3

"""
Laser module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from mahos.inst.visa_instrument import VisaInstrument
from mahos.msgs import param_msgs as P
from mahos.util.conv import invert_mapping


class Hubner_Cobolt(VisaInstrument):
    """Hubner Cobolt Laser. This class implements ParamDict interface for Tweaker."""

    _FAULT_CODES = {
        0: "No fault",
        1: "Temperature error",
        3: "Open interlock",
        4: "Constant power fault",
    }

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\r"
        if "read_termination" not in conf:
            conf["read_termination"] = "\r\n"
        if "baud_rate" not in conf:
            conf["baud_rate"] = 115_200

        # Force disable *IDN? because it's not supported
        conf["query_idn"] = False

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        mn = self.get_model_number()
        sn = self.get_serial_number()
        self.logger.info(f"Model: {mn} Serial: {sn}")

    def get_model_number(self) -> str:
        """Get model number string."""

        return self.inst.query("glm?")

    def get_serial_number(self) -> str:
        """Get serial number string."""

        return self.inst.query("sn?")

    def get_hours(self) -> float:
        """Get operation hours."""

        ans = self.inst.query("hrs?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply to hrs?: {ans}")
            return -1.0

    def get_enable(self) -> bool:
        """get if the laser is turned on."""

        return self.inst.query("l?") == "1"

    def set_enable(self, on: bool) -> bool:
        """set laser enable."""

        if on:
            self.logger.info("Turning on the laser.")
            return self.inst.query("@cob1") == "OK"
        else:
            self.logger.info("Turning off the laser.")
            return self.inst.query("l0") == "OK"

    def get_interlock(self) -> bool:
        """get the interlock state. True means interlock is closed and ready to go."""

        return self.inst.query("ilk?") == "0"

    def get_fault(self) -> int:
        """get the fault state."""

        ans = self.inst.query("f?")
        try:
            code = int(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply to f?: {ans}")
            return -1

        if not code:
            # code == 0 means no fault.
            return code

        if code in self._FAULT_CODES:
            msg = self._FAULT_CODES[code]
            self.logger.error(f"Fault {code}: {msg}")
        else:
            self.logger.error(f"Unexpected fault code: {code}")
        return code

    def clear_fault(self) -> bool:
        return self.inst.query("cf") == "OK"

    def get_actual_current(self) -> float:
        ans = self.inst.query("i?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply to i?: {ans}")
            return 0.0

    def get_power(self) -> float:
        """Get power set point in W."""

        ans = self.inst.query("p?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply to p?: {ans}")
            return 0.0

    def set_power(self, power_W: float) -> bool:
        """Set power set point in W."""

        return self.inst.query(f"p {power_W:.12f}") == "OK"

    def get_actual_power(self):
        """Get actual power in W."""

        ans = self.inst.query("pa?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply to p?: {ans}")
            return 0.0

    def set_control_mode(self, const_power: bool) -> bool:
        if const_power:
            self.logger.info("Constant power mode.")
            return self.inst.query("cp") == "OK"
        else:
            self.logger.info("Constant current mode.")
            return self.inst.query("ci") == "OK"

    # Standard API

    def start(self, label: str = "") -> bool:
        return self.set_enable(True)

    def stop(self, label: str = "") -> bool:
        return self.set_enable(False)

    def get_param_dict_labels(self) -> list[str]:
        return [""]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        d = P.ParamDict(
            power=P.FloatParam(
                self.get_power(), minimum=0.0, unit="W", SI_prefix=True, doc="Power setpoint"
            ),
            # read only params
            actual_power=P.FloatParam(
                self.get_actual_power(),
                read_only=True,
                unit="W",
                SI_prefix=True,
                doc="Measured power",
            ),
            actual_current=P.FloatParam(
                self.get_actual_current(),
                read_only=True,
                unit="A",
                SI_prefix=True,
                doc="Measured current",
            ),
            hours=P.FloatParam(self.get_hours(), read_only=True, doc="Operation hours"),
            fault=P.IntParam(self.get_fault(), read_only=True, doc="Fault status"),
            interlock=P.BoolParam(
                self.get_interlock(),
                read_only=True,
                doc="Interlock status (True: closed)",
            ),
        )
        return d

    def configure(self, params: dict, label: str = "") -> bool:
        ps = P.unwrap(params)
        return self.set_power(ps["power"])


class Coherent_OBIS(VisaInstrument):
    """Coherent OBIS Series Laser. This class implements ParamDict interface for Tweaker.

    This class enforces the handshaking feature to be disabled (default is ON).
    To turn this ON after using, send "SYST[device_id]:COMM:HAND ON".
    (see also p 154 of OBIS LX/LS Operators Manual)

    Also, command prompt is assumed to be OFF (default).
    To turn this OFF, send "SYST[device_id]:COMM:PROM OFF".

    :param devices: (default: {"": ""}) Map from a label to device index (1, 2, ...).
        This parameter should be given if the target is the box including multiple lasers.
    :type devices: dict[str, str | int]

    """

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\r\n"
        if "read_termination" not in conf:
            conf["read_termination"] = "\r\n"

        # Force disable *IDN? because it might need device_index
        conf["query_idn"] = False

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        if "devices" in self.conf:
            devs = self.conf["devices"]
            if isinstance(devs, dict):
                self._devices = {}
                for label, i in devs.items():
                    self._devices[label] = str(i)
            else:
                raise TypeError("devices must be dict[str, int | str]: label to index")
        else:
            self._devices = {"": ""}
        self._index_to_label = {k: v[0] for k, v in invert_mapping(self._devices).items()}

        self.model_names = {}
        self.power_lims = {}
        for label, i in self._devices.items():
            if not self._disable_handshake(i):
                raise RuntimeError("Failed to disable handshaking. try to disable it manually.")
            self.clear_error(i)
            self.logger.info(f"{self._header(i)}*IDN{i}? > " + self.inst.query(f"*IDN{i}?"))
            self.model_names[label] = self.get_model_name(i)
            self.power_lims[label] = self.get_power_limit(i)

    def _header(self, i: str) -> str:
        label = self._index_to_label[i]
        if not label:
            return ""
        return f"[{label}] "

    def _disable_handshake(self, i: str) -> bool:
        # first check if it's already OFF, to avoid unnecessary write
        # everytime the class is initialized (saves EEPROM life)
        resp = self.inst.query(f"SYST{i}:COMM:HAND?")
        if resp == "OFF":
            return True
        # when handshake is ON, the answer will be "OK" -> "ON" (as specified in the manual)
        # or "ON" -> "OK" (the message order is flipped for unknown reason)
        elif resp == "OK":
            resp2 = self.inst.read()
            if resp2 == "ON":
                self.inst.write(f"SYST{i}:COMM:HAND OFF")
                self.logger.info(f"{self._header(i)}Disabled handshake of index {i}")
                return True
            else:
                return self.fail_with(f"unexpected response: {resp}, {resp2}")
        elif resp == "ON":
            resp2 = self.inst.read()
            if resp2 == "OK":
                self.inst.write(f"SYST{i}:COMM:HAND OFF")
                self.logger.info(f"{self._header(i)}Disabled handshake of index {i}")
                return True
            else:
                return self.fail_with(f"unexpected response: {resp}, {resp2}")
        else:
            return self.fail_with(f"unexpected response: {resp}")

    def clear_error(self, i: str = ""):
        self.inst.write(f"SYST{i}:ERR:CLE?")

    def query_error(self, i: str = ""):
        return self.inst.query(f"SYST{i}:ERR:NEXT?")

    def check_error(self, i: str = "") -> bool:
        """query error, parse the error message, and log it if there is an error."""

        ret = self.query_error(i)
        try:
            s = ret.split(",")
            code = int(s[0])
            msg = ",".join(s[1:]).strip("'\" ")
            if not code:
                return True
            self.logger.error(f"{self._header(i)}{code}: {msg}")
            return False
        except Exception:
            self.logger.exception(f"{self._header(i)}Error parsing error message: {ret}")
            return False

    def get_model_name(self, i: str = "") -> str:
        """Get model name string."""

        return self.inst.query(f"SYST{i}:INF:MOD?")

    def get_power_limit(self, i: str = "") -> tuple[float, float]:
        try:
            lb = float(self.inst.query(f"SOUR{i}:POW:LIM:LOW?"))
            ub = float(self.inst.query(f"SOUR{i}:POW:LIM:HIGH?"))
            return (lb, ub)
        except (TypeError, ValueError):
            return 0.0

    def get_hours(self, i: str = "") -> float:
        """Get diode operation hours."""

        ans = self.inst.query(f"SYST{i}:DIOD:HOUR?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"{self._header(i)}Unexpected reply to SYST:DIOD:HOUR?: {ans}")
            return -1.0

    def get_enable(self, i: str = "") -> bool:
        """get if the laser is turned on."""

        return self.inst.query(f"SOUR{i}:AM:STAT?") == "ON"

    def set_enable(self, on: bool, i: str = "") -> bool:
        """set laser enable."""

        if on:
            self.logger.info(f"{self._header(i)}Turning on the laser.")
            return self.inst.write(f"SOUR{i}:AM:STAT ON")
        else:
            self.logger.info(f"{self._header(i)}Turning off the laser.")
            return self.inst.write(f"SOUR{i}:AM:STAT OFF")

    def get_interlock(self, i: str = "") -> bool:
        """get the interlock state."""

        return self.inst.query(f"SYST{i}:LOCK?") == "ON"

    def get_status(self, i: str = "") -> str:
        """get the system status (hex-string representation of a 32-bit word)."""

        return self.inst.query(f"SYST{i}:STAT?")

    def get_fault(self, i: str = "") -> str:
        """get the fault status (hex-string representation of a 32-bit word)."""

        return self.inst.query(f"SYST{i}:FAUL?")

    def get_actual_current(self, i: str = "") -> float:
        ans = self.inst.query(f"SOUR{i}:POW:CURR?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"{self._header(i)}Unexpected reply: {ans}")
            return 0.0

    def get_power(self, i: str = "") -> float:
        """Get power set point in W."""

        ans = self.inst.query(f"SOUR{i}:POW:LEV:IMM:AMPL?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"{self._header(i)}Unexpected reply: {ans}")
            return 0.0

    def set_power(self, power_W: float, i: str = "") -> bool:
        """Set power set point in W."""

        return self.inst.write(f"SOUR{i}:POW:LEV:IMM:AMPL {power_W:.12f}")

    def get_actual_power(self, i: str = ""):
        """Get actual power in W."""

        ans = self.inst.query(f"SOUR{i}:POW:LEV?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"{self._header(i)}Unexpected reply: {ans}")
            return 0.0

    def get_mode(self, i: str = "") -> str:
        return self.inst.query(f"SOUR{i}:AM:SOUR?")

    def set_mode(self, mode: str, i: str = "") -> bool:
        hd = self._header(i)
        mode = mode.upper()
        if mode == "CWP":
            self.logger.info(f"{hd}mode: constant power, no external modulation.")
            return self.inst.write(f"SOUR{i}:AM:INT CWP")
        elif mode == "CWC":
            self.logger.info(f"{hd}mode: constant current, no external modulation.")
            return self.inst.write(f"SOUR{i}:AM:INT CWC")
        elif mode in ("DIG", "DIGITAL"):
            self.logger.info(f"{hd}mode: current feedback, digital modulation.")
            return self.inst.write(f"SOUR{i}:AM:EXT DIGITAL")
        elif mode in ("ANAL", "ANALOG"):
            self.logger.info(f"{hd}mode: power feedback, analog modulation.")
            return self.inst.write(f"SOUR{i}:AM:EXT ANALOG")
        elif mode in ("MIX", "MIXED"):
            self.logger.info(f"{hd}mode: current feedback, mixed (digital + analog) modulation.")
            return self.inst.write(f"SOUR{i}:AM:EXT MIXED")
        elif mode == "DIGSO":
            self.logger.info(f"{hd}mode: power feedback, digital modulation.")
            return self.inst.write(f"SOUR{i}:AM:EXT DIGSO")
        elif mode == "MIXSO":
            self.logger.info(f"{hd}mode: power feedback, mixed (digital + analog) modulation.")
            return self.inst.write(f"SOUR{i}:AM:EXT MIXSO")

        return self.fail_with(f"{self._header(i)}Unrecognized operation mode: {mode}")

    # Standard API

    def start(self, label: str = "") -> bool:
        if label not in self._devices:
            return self.fail_with(f"Invalid device label {label}")
        return self.set_enable(True, i=self._devices[label])

    def stop(self, label: str = "") -> bool:
        if label not in self._devices:
            return self.fail_with(f"Invalid device label {label}")
        return self.set_enable(False, i=self._devices[label])

    def get_param_dict_labels(self) -> list[str]:
        return list(self._devices.keys())

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        if label not in self._devices:
            self.logger.error(f"Invalid device label {label}")
            return None

        i = self._devices[label]
        d = P.ParamDict(
            model_name=P.StrParam(self.model_names[label], read_only=True, doc="Model name"),
            mode=P.StrChoiceParam(
                self.get_mode(i),
                ("CWP", "CWC", "DIGITAL", "ANALOG", "MIXED", "DIGSO", "MIXSO"),
                doc="Operation mode",
            ),
            power=P.FloatParam(
                self.get_power(i),
                minimum=self.power_lims[label][0],
                maximum=self.power_lims[label][1],
                unit="W",
                SI_prefix=True,
                doc="Power setpoint",
            ),
            # read only params
            actual_power=P.FloatParam(
                self.get_actual_power(i),
                read_only=True,
                unit="W",
                SI_prefix=True,
                doc="Measured power",
            ),
            actual_current=P.FloatParam(
                self.get_actual_current(i),
                read_only=True,
                unit="A",
                SI_prefix=True,
                doc="Measured current",
            ),
            hours=P.FloatParam(self.get_hours(i), read_only=True, doc="Operation hours"),
            status=P.StrParam(self.get_status(i), read_only=True, doc="Fault status"),
            fault=P.StrParam(self.get_fault(i), read_only=True, doc="Fault status"),
            # don't include interlock here because some model doesn't support this.
            # interlock=P.BoolParam(self.get_interlock(i), read_only=True, doc="Interlock status"),
        )
        # not checking error here because the error queue can be messed up in some devices.
        return d

    def configure(self, params: dict, label: str = "") -> bool:
        if label not in self._devices:
            return self.fail_with(f"Invalid device label {label}")

        ps = P.unwrap(params)
        i = self._devices[label]
        # not checking error here because the error queue can be messed up in some devices.
        return self.set_mode(ps["mode"], i) and self.set_power(ps["power"], i)


class Thorlabs_LD(VisaInstrument):
    """Thorlabs Laser Diode Drivers. This class implements ParamDict interface for Tweaker.

    Supported models are ITC4000 and CLD1000 series integrated LD drivers.

    NOTE: This class currently assumes the device is running in constant current
    (not in constant power). Also, temperature is read-only
    (setting command is omitted) for safety reasons.

    """

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\n"
        if "read_termination" not in conf:
            conf["read_termination"] = "\n"

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        if self.get_control_mode():
            raise ValueError("This class assumes constant current mode.")

    def get_control_mode(self) -> bool:
        """Get control mode. True (False) means constant power (current) mode."""

        ans = self.inst.query("SOUR:FUNC:MODE?").strip().upper()
        if ans in ("POW", "POWER"):
            return True
        elif ans in ("CURR", "CURRENT"):
            return False
        self.logger.error(f"Unexpected reply: {ans}")
        return True

    def get_enable(self) -> bool:
        """get if the laser is turned on."""

        return self.inst.query("OUTP?") == "1"

    def set_enable(self, on: bool) -> bool:
        """set laser enable."""

        if on:
            self.logger.info("Turning on the laser.")
            self.inst.write("OUTP 1")
        else:
            self.logger.info("Turning off the laser.")
            self.inst.write("OUTP 0")
        return True

    def get_current(self) -> float:
        """Get current set point in A."""

        ans = self.inst.query("SOUR:CURR?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply: {ans}")
            return 0.0

    def set_current(self, current_A: float) -> bool:
        """Set power set point in A."""

        self.inst.write(f"SOUR:CURR {current_A:.6e}")
        return True

    def get_temperature(self) -> float:
        """Get TEC temperature set point in degC."""

        ans = self.inst.query("SOUR2:TEMP?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply: {ans}")
            return 0.0

    # Standard API

    def start(self, label: str = "") -> bool:
        return self.set_enable(True)

    def stop(self, label: str = "") -> bool:
        return self.set_enable(False)

    def get_param_dict_labels(self) -> list[str]:
        return [""]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        d = P.ParamDict(
            current=P.FloatParam(
                self.get_current(), minimum=0.0, unit="A", SI_prefix=True, doc="Current setpoint"
            ),
            # read only params
            temperature=P.FloatParam(
                self.get_temperature(),
                read_only=True,
                unit="degC",
                doc="TEC setpoint",
            ),
        )
        return d

    def configure(self, params: dict, label: str = "") -> bool:
        ps = P.unwrap(params)
        self.set_current(ps["current"])
        return self.check_error()
