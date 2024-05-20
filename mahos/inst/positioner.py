#!/usr/bin/env python3

"""
Positioner module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import sys
import time

from .instrument import Instrument


# imports for Thorlabs
try:
    import clr
    from System import Decimal
    from System import UInt64
    from System import Action

    sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
    clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
    clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")

    from Thorlabs.MotionControl import DeviceManagerCLI  # noqa: E402
    from Thorlabs.MotionControl.KCube import DCServoCLI  # noqa: E402
except ImportError:
    print("mahos.inst.positioner: failed to import pythonnet or Thorlabs Kinesis modules")


class KDC101(Instrument):
    """Thorlabs KDC101 DC Servo controller.

    You need to install Kinesis software.

    :param serial: (default: "") Serial string to discriminate multiple devices.
        Blank is fine if only one device is connected.
    :type serial: str

    """

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)
        self.dm = DeviceManagerCLI.DeviceManagerCLI()
        self.dm.BuildDeviceList()
        devices = list(self.dm.GetDeviceList())
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

        self.device = DCServoCLI.KCubeDCServo.CreateKCubeDCServo(self.serial)
        self.device.Connect(self.serial)
        name = self.device.GetDeviceInfo().Name
        self.logger.info(f"Connected to {name} ({self.serial})")

        if not self.device.IsSettingsInitialized():
            self.device.WaitForSettingsInitialized(5000)

        # These wait parameters are taken from the example code.
        self.device.StartPolling(250)  # ms
        time.sleep(0.5)
        self.device.EnableDevice()
        time.sleep(0.5)

        # This is required.
        self.motor_conf = self.device.LoadMotorConfiguration(self.serial)

        self.tasks = []

    def home(self, timeout_ms: int = 60_000) -> bool:
        self.device.Home(timeout_ms)
        return True

    def move(self, pos: float, timeout_ms: int = 60_000) -> bool:
        self.device.MoveTo(Decimal(pos), timeout_ms)
        return True

    def moved_callback(self, task_id: int):
        self.tasks.remove(task_id)
        self.logger.info(f"Done task {task_id}. Remaining tasks: {self.tasks}")

    def move_async(self, pos: float) -> bool:
        self.tasks.append(self.device.MoveTo(Decimal(pos), Action[UInt64](self.moved_callback)))
        return True

    def get_pos(self) -> float:
        return Decimal.ToDouble(self.device.Position)

    def close_resources(self):
        if hasattr(self, "device"):
            self.device.StopPolling()
            self.device.Disconnect()
            self.device.Dispose()
            self.logger.info(f"Stopped and Disconnected {self.serial}.")
