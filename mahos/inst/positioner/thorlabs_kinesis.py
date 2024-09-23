#!/usr/bin/env python3

"""
Thorlabs Kinesis part of Positioner module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import sys
import time
import threading

from ..instrument import Instrument
from ...msgs import param_msgs as P
from ...util.conv import clip_angle_degrees


# imports for Thorlabs
try:
    import clr
    from System import Decimal
    from System import UInt64
    from System import Action

    sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
    clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
    clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
    clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")

    from Thorlabs.MotionControl import DeviceManagerCLI  # noqa: E402
    from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import CageRotator  # noqa: E402
    from Thorlabs.MotionControl.KCube import DCServoCLI  # noqa: E402
except ImportError:
    print("mahos.inst.positioner: failed to import pythonnet or Thorlabs Kinesis modules")


class _Thorlabs_Kinesis(Instrument):
    """Base class for Thorlabs Kinesis.

    Inherited class must define following methods.

    - _device_prefix
    - _create_device

    You need to install Kinesis software.

    :param serial: (default: "") Serial string to discriminate multiple devices.
        Blank is fine if only one device is connected.
    :type serial: str
    :param range: (default: hardware-defined limit) travel range.
        (lower, upper) bounds of the position.
    :type range: tuple[float, float]

    """

    def _device_prefix(self):
        raise NotImplementedError("_device_prefix must be implemented.")

    def _create_device(self, serial):
        raise NotImplementedError("_create_device must be implemented.")

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)
        self.manager = DeviceManagerCLI.DeviceManagerCLI()
        self.manager.BuildDeviceList()
        devices = list(self.manager.GetDeviceList(self._device_prefix()))
        self.logger.debug("Available Devices: " + ", ".join(devices))

        if not devices:
            self.logger.error("No device detected.")
            raise ValueError("No device detected.")
        if len(devices) == 1:
            if "serial" in self.conf and self.conf["serial"] != devices[0]:
                self.logger.warn(
                    "Given serial {} looks wrong. Opening available one {} anyway.".format(
                        self.conf["serial"], devices[0]
                    )
                )
            self.serial = devices[0]
        else:
            if "serial" not in self.conf:
                msg = "Must specify conf['serial'] as multiple devices are detected."
                msg += "\nAvailable serials: " + ", ".join(devices)
                self.logger.error(msg)
                raise ValueError(msg)
            if self.conf["serial"] not in devices:
                msg = "Specified serial {} is not available. (not in ({}))".format(
                    self.conf["serial"], ", ".join(devices)
                )
                self.logger.error(msg)
                raise ValueError(msg)
            self.serial = self.conf["serial"]

        self.device = self._create_device(self.serial)
        self.device.Connect(self.serial)
        name = self.device.GetDeviceInfo().Name
        self.logger.info(f"Connected to {name} ({self.serial})")

        # Init procedure below is taken from the example code.
        if not self.device.IsSettingsInitialized():
            self.device.WaitForSettingsInitialized(5_000)  # ms
        self.device.StartPolling(250)  # ms
        time.sleep(0.5)
        self.device.EnableDevice()
        time.sleep(0.5)
        # Note that this LoadMotorConfiguration is necessary!
        self.motor_conf = self.device.LoadMotorConfiguration(self.serial)
        self.logger.info("Device name: " + self.motor_conf.DeviceSettingsName)

        self.lock = threading.Lock()
        self.task_id = None
        self.limits = [
            Decimal.ToDouble(self.device.AdvancedMotorLimits.LengthMinimum),
            Decimal.ToDouble(self.device.AdvancedMotorLimits.LengthMaximum),
        ]
        self.logger.debug(f"limits: {self.limits[0]:.3f} {self.limits[1]:.3f}")
        self.range = self.conf.get("range", self.limits)
        if self.range[0] < self.limits[0]:
            msg = f"Given range is out of limit: {self.range[0]} < {self.limits[0]}"
            self.logger.error(msg)
            raise ValueError(msg)
        if self.range[1] > self.limits[1]:
            msg = f"Given range is out of limit: {self.range[1]} > {self.limits[1]}"
            self.logger.error(msg)
            raise ValueError(msg)
        self.logger.info(f"range: {self.range[0]:.3f} {self.range[1]:.3f}")

        if self.is_homed():
            # move to current position to update target pos (self.device.TargetPosition).
            # (target pos becomes always 0 after establishing new connection.)
            self.move(self.get_pos())
            target = self.get_target()
            self.logger.info(f"Device is homed. Current target pos: {target:.3f}")
        else:
            # target being 0 is fine because homing will be performed later.
            self.logger.warn("Device is not homed. Homing is necessary.")

    def done_callback(self, task_id: int):
        with self.lock:
            if task_id == self.task_id:
                self.logger.info(f"Done task {task_id}.")
                self.task_id = None
            else:
                self.logger.error(f"Done task {task_id} but not known task: {self.task_id}")

    def home_wait(self, timeout_ms: int = 60_000) -> bool:
        self.device.Home(timeout_ms)
        return True

    def home(self) -> bool:
        try:
            with self.lock:
                self.task_id = self.device.Home(Action[UInt64](self.done_callback))
                self.logger.info(f"New task {self.task_id} (home)")
            return True
        except DeviceManagerCLI.DeviceMovingException:
            return self.fail_with("Cannot move because device is already moving.")

    def stop_wait(self, timeout_ms: int = 60_000) -> bool:
        self.device.Stop(timeout_ms)
        return True

    def _stop(self) -> bool:
        try:
            with self.lock:
                self.task_id = self.device.Stop(Action[UInt64](self.done_callback))
                self.logger.info(f"New task {self.task_id} (stop)")
            return True
        except Exception:
            # TODO: catch some more specific exceptions
            self.logger.exception("Failed to stop.")
            return False

    def move_wait(self, pos: float, timeout_ms: int = 60_000) -> bool:
        self.device.MoveTo(Decimal(pos), timeout_ms)
        return True

    def move(self, pos: float) -> bool:
        if not self.is_homed():
            return self.fail_with("Cannot move because this device has not been homed yet.")
        if pos < self.range[0] or pos > self.range[1]:
            return self.fail_with(f"Target pos {pos:.3f} is out of range {self.range}.")

        try:
            with self.lock:
                self.task_id = self.device.MoveTo(Decimal(pos), Action[UInt64](self.done_callback))
                self.logger.info(f"New task {self.task_id} (move to {pos:.3f})")
                return True
        except DeviceManagerCLI.DeviceMovingException:
            return self.fail_with("Cannot move because device is already moving.")
        except DeviceManagerCLI.MoveToInvalidPositionException:
            return self.fail_with(f"Cannot move because position {pos:.3f} is out of limit.")

    def get_pos(self) -> float:
        return Decimal.ToDouble(self.device.Position)

    def get_target(self) -> float:
        return Decimal.ToDouble(self.device.TargetPosition)

    def get_status(self) -> dict:
        # Not quite sure about the meaning of these parameters...
        # IsSettled is False regardless of moving / not moving.
        # IsInMotion and IsMoving looks same.
        return {
            "homed": self.device.Status.IsHomed,
            "moving": self.device.Status.IsMoving,
            # "in_motion": self.device.Status.IsInMotion,
            # "settled": self.device.Status.IsSettled,
        }

    def get_all(self) -> dict[str, [float, bool]]:
        """Get all important info about this device packed in a dict.

        :returns pos: current position.
        :returns target: target position.
        :returns range: travel range.
        :returns homed: True if device is homed.
        :returns moving: True if device is moving.

        """

        d = self.get_status()
        d["pos"] = self.get_pos()
        d["target"] = self.get_target()
        d["range"] = self.get_range()

        return d

    def get_range(self) -> tuple[float, float]:
        return self.range

    def is_homed(self) -> bool:
        return self.device.Status.IsHomed

    def close_resources(self):
        if hasattr(self, "device"):
            self.device.StopPolling()
            self.device.Disconnect(True)
            self.device.Dispose()
            self.logger.info(f"Stopped and Disconnected {self.serial}.")

    # Standard API

    def reset(self, label: str = "") -> bool:
        """Perform homing of this device."""

        return self.home()

    def stop(self, label: str = "") -> bool:
        """Stop motion of this device."""

        return self._stop()

    def get_param_dict_labels(self) -> list[str]:
        return ["pos"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        if label == "pos":
            return P.ParamDict(
                target=P.FloatParam(
                    self.get_target(), self.range[0], self.range[1], doc="target position"
                )
            )
        else:
            self.logger.error(f"Unknown label: {label}")
            return None

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "pos":
            return self.move(params["target"])
        else:
            return self.fail_with(f"Unknown label {label}")

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "target":
            return self.move(value)
        else:
            return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None, label: str = ""):
        if key == "all":
            return self.get_all()
        elif key == "pos":
            return self.get_pos()
        elif key == "target":
            return self.get_target()
        elif key == "status":
            return self.get_status()
        elif key == "range":
            return self.get_range()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class Thorlabs_KCube_DCServo(_Thorlabs_Kinesis):
    """Instrument for Thorlabs KCube DC servo motor controller (KDC101).

    You need to install Kinesis software.

    :param serial: (default: "") Serial string to discriminate multiple devices.
        Blank is fine if only one device is connected.
    :type serial: str
    :param range: (default: hardware-defined limit) travel range.
        (lower, upper) bounds of the position.
    :type range: tuple[float, float]

    """

    def _device_prefix(self):
        return DCServoCLI.KCubeDCServo.DevicePrefix

    def _create_device(self, serial):
        return DCServoCLI.KCubeDCServo.CreateKCubeDCServo(serial)


class Thorlabs_CageRotator(_Thorlabs_Kinesis):
    """Instrument for Thorlabs IntegratedStepper-based CageRotator (K10CR1).

    You need to install Kinesis software.

    :param serial: (default: "") Serial string to discriminate multiple devices.
        Blank is fine if only one device is connected.
    :type serial: str
    :param range: (default: hardware-defined limit) travel range.
        (lower, upper) bounds of the position.
    :type range: tuple[float, float]

    """

    def _device_prefix(self):
        return CageRotator.DevicePrefix

    def _create_device(self, serial):
        return CageRotator.CreateCageRotator(serial)

    def move(self, pos: float) -> bool:
        return _Thorlabs_Kinesis.move(self, clip_angle_degrees(pos))

    def get_pos(self) -> float:
        return clip_angle_degrees(Decimal.ToDouble(self.device.Position))
