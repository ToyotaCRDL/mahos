#!/usr/bin/env python3

"""
Power Supply module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import threading
import time

from mahos.inst.visa_instrument import VisaInstrument
from mahos.msgs import param_msgs as P
from mahos.util.conf import ConfAccessorMixin


class Kikusui_PAN_E(VisaInstrument, ConfAccessorMixin):
    """Kikusui PAN-E series DC power supply with software-controlled current ramping.

    Currently implements only the ``current_ramp`` interface for Tweaker/Recorder integration.
    The current setpoint ramps toward the target at a configurable rate, while the voltage
    limit is set directly.

    :param volt_bounds: Minimum and maximum voltage limit in V. Default: ``(0.0, 16.0)``.
    :type volt_bounds: tuple[float, float]
    :param curr_bounds: Minimum and maximum current setpoint in A. Default: ``(0.0, 60.0)``.
    :type curr_bounds: tuple[float, float]
    :param curr_ramp_rate: Current ramp rate in A/s. Default: ``1.0``.
        Setting zero disables the software ramp feature and the target current is set immediately.
    :type curr_ramp_rate: float
    :param init_output_off: Set the current setpoint to zero and attempt to turn off the output
        during initialization. Default: ``True``.
    :type init_output_off: bool
    :param off_current_threshold: Current threshold in A for turning off the output.
        Output-off requests are rejected when the measured current exceeds this value.
        Default: ``0.05``.
    :type off_current_threshold: float

    """

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\n"
        if "read_termination" not in conf:
            conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 5_000.0  # 5 seconds timeout

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        self._lock = threading.Lock()

        # Send SYST:REM to switch to remote control mode
        with self._lock:
            self.inst.write("SYST:REM")
            # Disable external analog control to enforce SCPI command control
            self.inst.write("CURR:EXT:SOUR NONE")
            self.inst.write("VOLT:EXT:SOUR NONE")
            self.inst.write("OUTP:EXT OFF")
            self.check_error()

        # Use bounds from conf if specified,
        # otherwise default to PAN16-30E and PAN16-30E parallel ratings (0-16V, 0-60A)
        self.volt_min, self.volt_max = self._conf_ascending_numbers("volt_bounds", 2, (0.0, 16.0))
        self.curr_min, self.curr_max = self._conf_ascending_numbers("curr_bounds", 2, (0.0, 60.0))

        # ramp rate (default: 1.0 A/sec)
        self.curr_ramp_rate = float(self._conf_nonneg_num("curr_ramp_rate", 1.0))
        if self.curr_ramp_rate > 0.0:
            self.logger.info(f"Current ramp rate set to {self.curr_ramp_rate:.2f} A/s")
        else:
            self.logger.warning("Disabling the current ramp feature, given curr_ramp_rate = 0")

        self._off_curr_threshold = self._conf_pos_num("off_current_threshold", 0.05)

        # maintain internal states
        self.target_current = 0.0  # Ampere
        self._ramp_thread = None
        self._stop_event = threading.Event()  # to stop ramping immediately

        # Turn off output on initialization for safety if requested
        if self._conf_bool("init_output_off", True):
            self.set_current_direct(0.0)
            self.set_output(False)

    # Core Ramping Logic

    def set_current_direct(self, current: float) -> bool:
        """Set current directly to hardware without ramping."""

        with self._lock:
            current = max(self.curr_min, min(self.curr_max, current))
            self.inst.write(f"SOUR:CURR {current:.3f}")
            return self.check_error()

    def _start_ramp_thread(self, target_current: float):
        """Start or restart the background ramping thread."""

        # If some ramping is running, it will be stopped and overwritten by the current ramping.
        self._stop_ramp_thread()

        # change the current value without ramping
        if self.curr_ramp_rate <= 0:
            self.set_current_direct(target_current)
            return

        # run ramping mode
        self._ramp_thread = threading.Thread(
            target=self._ramp_loop, args=(target_current,), daemon=True
        )
        self._ramp_thread.start()

    def _stop_ramp_thread(self):
        if not (self._ramp_thread and self._ramp_thread.is_alive()):
            return

        self._stop_event.set()
        self._ramp_thread.join()
        self._stop_event.clear()

    def _ramp_loop(self, target_current: float):
        """Background loop to step current smoothly."""

        dt = 0.1  # 100 ms period
        step = self.curr_ramp_rate * dt
        current_set = self.get_current()

        while not self._stop_event.is_set():
            diff = target_current - current_set
            if abs(diff) <= step:
                # When the current value is close enough to the target value
                self.set_current_direct(target_current)
                break

            current_set += step if diff > 0 else -step
            self.set_current_direct(current_set)
            time.sleep(dt)

    # Internal helpers

    def set_current(self, target_current: float) -> bool:
        """Set target current. Ramps immediately if Output is ON."""

        self.target_current = max(self.curr_min, min(self.curr_max, target_current))

        # If the output is ON, the ramping will run immediately.
        if self.get_output_state():
            self._start_ramp_thread(self.target_current)
        # If the output is OFF, the target value is just stored and the ramping will not run.
        else:
            self.logger.info(f"Target current set to {self.target_current} A (Output is OFF).")

        return True

    def set_voltage(self, voltage: float) -> bool:
        """Set voltage limit in Volts (Direct, no ramping)."""

        with self._lock:
            voltage = max(self.volt_min, min(self.volt_max, voltage))
            self.inst.write(f"SOUR:VOLT {voltage:.3f}")
            return self.check_error()

    def set_output(self, on: bool) -> bool:
        """Turn output ON or OFF with automatic ramping."""

        if on:
            self.logger.info("Turning ON power supply output.")
            self.inst.write("OUTP ON")
            self.check_error()
            # After turning ON, start ramping from 0.0 A to the target value.
            self._start_ramp_thread(self.target_current)
            return True

        else:
            self._stop_ramp_thread()

            actual_i = self.get_actual_current()
            threshold = self._off_curr_threshold

            # Check whether the current is small enough and the output can be OFF safely.
            if actual_i > threshold:
                return self.fail_with(
                    f"Cannot turn OFF: Actual current is still {actual_i:.3f} A (>{threshold} A). "
                    "Output remains ON, but the current ramp has been stopped. "
                    "Apply output_state=True with current=0 A to ramp down, "
                    "wait until the measured current is at or below the threshold, "
                    "then apply output_state=False again."
                )

            # Then, output OFF
            self.logger.info("Turning OFF power supply output.")
            self.set_current_direct(0.0)  # And set the current value to zero
            self.inst.write("OUTP OFF")  # Finally output OFF

        return self.check_error()

    def get_current(self) -> float:
        """Get setpoint current in Amperes."""

        with self._lock:
            ans = self.inst.query("SOUR:CURR?")
            try:
                return float(ans)
            except ValueError:
                self.logger.exception(f"Unexpected reply: {ans}")
                return 0.0

    def get_actual_current(self) -> float:
        """Measure actual output current in Amperes (MEASure:CURRent)."""

        with self._lock:
            ans = self.inst.query("MEAS:CURR?")
            try:
                return float(ans)
            except ValueError:
                self.logger.exception(f"Unexpected reply: {ans}")
                return 0.0

    def get_voltage(self) -> float:
        """Get setpoint voltage in Volts."""

        with self._lock:
            ans = self.inst.query("SOUR:VOLT?")
            try:
                return float(ans)
            except ValueError:
                self.logger.exception(f"Unexpected reply: {ans}")
                return 0.0

    def get_actual_voltage(self) -> float:
        """Measure actual output voltage in Volts (MEASure:VOLTage)."""

        with self._lock:
            ans = self.inst.query("MEAS:VOLT?")
            try:
                return float(ans)
            except ValueError:
                self.logger.exception(f"Unexpected reply: {ans}")
                return 0.0

    def get_output_state(self) -> bool:
        """Get output state (True: ON, False: OFF)."""

        with self._lock:
            return self.inst.query("OUTP?").strip() == "1"

    # Standard API for Recorder/Tweaker

    def start(self, label: str = "") -> bool:
        """Dummy start for Recorder/Tweaker Start request."""

        return True

    def stop(self, label: str = "") -> bool:
        """Dummy stop for Recorder/Tweaker Stop request."""

        return True

    def get_param_dict_labels(self) -> list[str]:
        return ["current_ramp"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        if label == "current_ramp":
            d = P.ParamDict(
                output_state=P.BoolParam(
                    self.get_output_state(),
                    doc="Current Output State (True: ON, False: OFF)",
                ),
                # target_current differs from hardware set_point determined by the ramp logic
                current=P.FloatParam(
                    self.target_current,
                    minimum=self.curr_min,
                    maximum=self.curr_max,
                    unit="A",
                    SI_prefix=True,
                    doc="Target current setpoint",
                ),
                voltage=P.FloatParam(
                    self.get_voltage(),
                    minimum=self.volt_min,
                    maximum=self.volt_max,
                    unit="V",
                    SI_prefix=True,
                    doc="Voltage limit setpoint",
                ),
                actual_current=P.FloatParam(
                    self.get_actual_current(),
                    read_only=True,
                    unit="A",
                    SI_prefix=True,
                    doc="Measured actual current",
                ),
                actual_voltage=P.FloatParam(
                    self.get_actual_voltage(),
                    read_only=True,
                    unit="V",
                    SI_prefix=True,
                    doc="Measured actual voltage",
                ),
            )
            return d
        else:
            return self.fail_with(f"Unknown label: {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "current_ramp":
            ps = P.unwrap(params)
            if "output_state" in ps:
                target_on = bool(ps["output_state"])
                if target_on != self.get_output_state():
                    if not self.set_output(target_on):
                        return False
            success = True
            if "current" in ps:
                success &= self.set_current(ps["current"])
            if "voltage" in ps:
                success &= self.set_voltage(ps["voltage"])
            return success
        else:
            return self.fail_with(f"unknown label: {label}")

    def get(self, key: str, args=None, label: str = "") -> float | str | dict | None:
        if label == "current_ramp":
            if key in ("data", "current", "actual_current"):
                return self.get_actual_current()
            elif key in ("voltage", "actual_voltage"):
                return self.get_actual_voltage()
            elif key == "unit":
                return "A"  # default unit
            else:
                self.logger.error(f"Unknown key: {key}")
                return None
        else:
            self.logger.error(f"Unknown label: {label}")
            return None
