#!/usr/bin/env python3

"""
Power Supply module for Kikusui PAN-E series.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import threading
import time

from mahos.inst.visa_instrument import VisaInstrument
from mahos.msgs import param_msgs as P


class Kikusui_PAN_E(VisaInstrument):

    """Kikusui PAN-E series DC Power Supply Module with smooth ramping."""

    def __init__(self, name, conf, prefix=None):
        if "write_termination" not in conf:
            conf["write_termination"] = "\n"
        if "read_termination" not in conf:
            conf["read_termination"] = "\n"
        if "timeout" not in conf:
            conf["timeout"] = 5_000.0  # 5 seconds timeout

        VisaInstrument.__init__(self, name, conf, prefix=prefix)

        # Send SYST:REM to switch to remote control mode via LAN
        self.inst.write("SYST:REM")
        # Disable external analog control to enforce SCPI command control
        self.inst.write("CURR:EXT:SOUR NONE")
        self.inst.write("VOLT:EXT:SOUR NONE")
        self.inst.write("OUTP:EXT OFF")
        self.check_error()
        
        # Use bounds from conf if specified, otherwise default to PAN16-30E and PAN16-30E parallel ratings (0-16V, 0-60A)
        self.volt_min, self.volt_max = self.conf.get("volt_bounds", (0.0, 16.0))
        self.curr_min, self.curr_max = self.conf.get("curr_bounds", (0.0, 60.0))

        # sweep rate (default: 1.0 A/sec)
        self.sweep_rate_min, self.sweep_rate_max = self.conf.get("sweep_rate_bounds", (0.1, 3.0))

        raw_rate = float(self.conf.get("sweep_rate", 1.0))
        self.sweep_rate = max(self.sweep_rate_min, min(self.sweep_rate_max, raw_rate))

        self.logger.info(
            f"Sweep rate set to {self.sweep_rate:.2f} A/s "
            f"(Allowed bounds: {self.sweep_rate_min:.2f} - {self.sweep_rate_max:.2f} A/s)"
        )
        
        # maintain internal states
        self.target_current = 0.0 # Ampere
        self._ramp_thread = None
        self._stop_event = threading.Event() # to stop ramping immediately
        
        # Turn off output on initialization for safety if requested
        if self.conf.get("init_output_off", True):
            self.set_current_direct(0.0)
            self.set_output(False)

    # -------------------------------------------------------------------------
    # Core Ramping Logic
    # -------------------------------------------------------------------------

    def set_current_direct(self, current: float) -> bool:
        """Set current directly to hardware without ramping."""
        current = max(self.curr_min, min(self.curr_max, current))
        self.inst.write(f"SOUR:CURR {current:.3f}")
        return self.check_error()

    def _start_ramp_thread(self, target_current: float):
        """Start or restart the background ramping thread."""
        
        # If some ramping is running, it will be stopped and overwritten by the current ramping.
        if self._ramp_thread and self._ramp_thread.is_alive():
            self._stop_event.set()
            self._ramp_thread.join()

        self._stop_event.clear()

        # change the current value without ramping
        if self.sweep_rate <= 0:
            self.set_current_direct(target_current)
            return

        # run ramping mode
        self._ramp_thread = threading.Thread(
            target=self._ramp_loop, args=(target_current,), daemon=True
        )
        self._ramp_thread.start()

    def _ramp_loop(self, target_current: float):
        """Background loop to step current smoothly."""
        dt = 0.1  # 100 ms period
        step = self.sweep_rate * dt
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

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def set_current(self, target_current: float) -> bool:
        """Set target current. Ramps immediately if Output is ON."""
        self.target_current = max(self.curr_min, min(self.curr_max, target_current))

        # IF the output is ON, the ramping will run immediately.
        if self.get_output_state():
            self._start_ramp_thread(self.target_current)
        # If the output is OFF, the target value is just stored and the ramping will not run.
        else:
            self.logger.info(f"Target current set to {self.target_current} A (Output is OFF).")

        return True

    def set_voltage(self, voltage: float) -> bool:
        """Set voltage limit in Volts (Direct, no ramping)."""
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
            actual_i = self.get_actual_current() # measured current value
            threshold = self.conf.get("off_current_threshold", 0.05)  # The threshold value is set to 50 mA.

            # Check whether the current is small enough and the output can be OFF safely.
            if actual_i > threshold:
                return self.fail_with(
                    f"Cannot turn OFF: Actual current is still {actual_i:.3f} A (>{threshold} A). "
                    "Please set current to 0 A and wait for ramp down first."
                )

            # Then, output OFF
            self.logger.info("Turning OFF power supply output.")
            self._stop_event.set() # Stop ramping first
            self.set_current_direct(0.0) # And set the current value to zero
            self.inst.write("OUTP OFF") # Finally output OFF

        return self.check_error()
    
    def get_current(self) -> float:
        """Get setpoint current in Amperes."""
        ans = self.inst.query("SOUR:CURR?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply: {ans}")
            return 0.0

    def get_actual_current(self) -> float:
        """Measure actual output current in Amperes (MEASure:CURRent)."""
        ans = self.inst.query("MEAS:CURR?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply: {ans}")
            return 0.0

    def get_voltage(self) -> float:
        """Get setpoint voltage in Volts."""
        ans = self.inst.query("SOUR:VOLT?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply: {ans}")
            return 0.0

    def get_actual_voltage(self) -> float:
        """Measure actual output voltage in Volts (MEASure:VOLTage)."""
        ans = self.inst.query("MEAS:VOLT?")
        try:
            return float(ans)
        except ValueError:
            self.logger.exception(f"Unexpected reply: {ans}")
            return 0.0

    def get_output_state(self) -> bool:
        """Get output state (True: ON, False: OFF)."""
        return self.inst.query("OUTP?").strip() == "1"

    # -------------------------------------------------------------------------
    # Standard API for Tweaker
    # -------------------------------------------------------------------------

    def start(self, label: str = "") -> bool:
        return self.set_output(True)

    def stop(self, label: str = "") -> bool:
        return self.set_output(False)

    def get_param_dict_labels(self) -> list[str]:
        return ["electromag"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        if label == "electromag":
            d = P.ParamDict(
                current=P.FloatParam(
                    self.target_current, # Set target_current value via GUI (The actual set_point will be determined by the rampimg logic.)
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
        if label == "electromag":
            ps = P.unwrap(params)
            success = True
            if "current" in ps:
                success &= self.set_current(ps["current"])
            if "voltage" in ps:
                success &= self.set_voltage(ps["voltage"])
            return success
        else:
            return self.fail_with(f"unknown label: {label}")