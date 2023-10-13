#!/usr/bin/env python3

"""
Piezo Stage module.

.. This file is a part of MAHOS project.

available class(es):
    E727_3_USB - PI E727.3 Piezo controller using PIPython (GCS USB).
    E727_3_USB_AO - PI E727.3 Piezo controller using PIPython and DAQ AnalogOut.
    AnalogPiezo - Generic Piezo Driver using DAQ AnalogIn and AnalogOut.

"""

from __future__ import annotations
from itertools import product

import numpy as np
from pipython import GCSDevice

from .instrument import Instrument
from .exceptions import InstError
from .daq import AnalogIn, AnalogOut
from ..msgs.confocal_msgs import Axis
from ..util.conf import PresetLoader


class BasePiezo3Axes(Instrument):
    """Base Class for Piezo with 3 axes.

    Listed parameters (confs) are common for all the child classes.

    :param transform: (default: identity matrix) a 3x3 linear transformation matrix
        from (real) target coordinate to command-value coordinate.
        This is usable for tilt correction to compensate the cross-talks in piezo axes.
    :type transform: list[list[float]]
    :param limit_um: (default: [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0)]) limits of positions
        in command coordinate in the following format:
        [[Xmin, Xmax], [Ymin, Ymax], [Ymin, Ymax]].
    :param range_um: travel ranges in target coordinate in the following format:
        [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        This should be identical to limit_um if transform is default (identity matrix), i.e.,
        the target corrdinate and the command-value coordinate are identical.
    :type range_um: list[tuple[float, float]]

    """

    DEFAULT_LIMIT = ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0))
    LIMIT_BOUNDS = ((0.0, 200.0), (0.0, 200.0), (0.0, 200.0))

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        # The cubic-limit in command space
        limit = self.conf.get("limit_um", self.DEFAULT_LIMIT)
        for (low, high), (lb, ub) in zip(limit, self.LIMIT_BOUNDS):
            if low < lb or high > ub:
                raise ValueError(f"limit_um is out of hard-bounds ({lb}, {ub})")
        self.limit: dict[Axis, tuple[float, float]] = {ax: r for ax, r in zip(Axis, limit)}
        self.target: dict[Axis, float] = {ax: 0.0 for ax in Axis}
        self.init_transform_matrix()

    def set_target_pos(self, positions: list[float], axes: list[Axis] | None = None) -> bool:
        """update self.target and move toward it."""

        if axes is None:
            axes = list(Axis)
        positions = list(positions)
        for i, ax in enumerate(axes):
            if ax not in Axis:
                self.logger.error(f"Unknown Axis: {ax}")
                return False
            mn, mx = self.limit[ax]
            if positions[i] < mn:
                positions[i] = mn
            if positions[i] > mx:
                positions[i] = mx
        for ax, pos in zip(axes, positions):
            self.target[ax] = pos
        return self.move()

    def move(self) -> bool:
        raise NotImplementedError("move() is not implemented")

    def set_target_X(self, pos) -> bool:
        return self.set_target_pos([pos], [Axis.X])

    def set_target_Y(self, pos) -> bool:
        return self.set_target_pos([pos], [Axis.Y])

    def set_target_Z(self, pos) -> bool:
        return self.set_target_pos([pos], [Axis.Z])

    def set_target(self, value: dict | list[float]):
        if isinstance(value, dict) and "ax" in value and "pos" in value:
            ax, p = value["ax"], value["pos"]
            if all([isinstance(v, (tuple, list)) for v in (ax, p)]) and len(ax) == len(p):
                axes, pos = ax, p
            elif all([not isinstance(v, (tuple, list)) for v in (ax, p)]):
                axes, pos = [ax], [p]
            else:
                self.logger.error('set("target", value): invalid format for value in dict.')
                return False
        elif isinstance(value, (tuple, list)) and len(value) == 3:
            axes, pos = list(Axis), value
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            axes, pos = [value[0]], [value[1]]
        else:
            self.logger.error('set("target", value): invalid format for value.')
            return False

        if not all([ax in Axis for ax in axes]):
            self.logger.error(f'set("target", value): invalid ax: {axes}.')
            return False

        return self.set_target_pos(pos, axes)

    def init_transform_matrix(self):
        """Init transform matrices and target space range: self.T, self.Tinv and self.range.

        When self.conf["transform"] is undefined,
        This method sets identity transforms and self.range = limit.

        Otherwise, self.T is set as self.conf["transform"], which is the linear transformation
        from target space (x, y, z) to command space (X, Y, Z).
        Also requires user defined cubic-range in target space: self.conf["range_um"].

        """

        limit = self.get_limit()

        if "transform" not in self.conf:
            self.T = self.Tinv = np.eye(3)
            self.range = limit
            return

        self.T = np.array(self.conf["transform"], dtype=np.float64)
        self.Tinv = np.linalg.inv(self.T)
        # self.conf["range_um"] defines cubic-range in target space,
        # limit is cubic-range in command space
        self.check_required_conf(["range_um"])
        low, high = np.array(limit).T
        # validate that the range after transform is inside the limit.
        for corner_point in product(*self.conf["range_um"]):
            p = self.T @ np.array(corner_point)
            if any(p < low) or any(high < p):
                msg = f"range corner point {p} goes outside command space limit {low}, {high}."
                self.logger.error(msg)
                raise ValueError(msg)
            else:
                self.logger.debug("corner point: {:.4f}, {:.4f}, {:.4f}".format(*p))
        # set the given range in target space after validation
        self.range = self.conf["range_um"]

    def get_range(self) -> list[tuple[float, float]]:
        """Get target space range in list (order of x, y, z)."""

        return self.range

    def get_target(self) -> list[float]:
        """Get target positions in list (order of x, y, z)."""

        return [self.target[ax] for ax in Axis]

    def get_limit(self) -> list[tuple[float, float]]:
        """Get command space limit in list (order of x, y, z)."""

        return [self.limit[ax] for ax in Axis]

    def get(self, key: str, args=None):
        if key == "target":
            return self.get_target()
        elif key == "range":
            return self.get_range()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class E727_3_USB(BasePiezo3Axes):
    """E727.3 driver using USB connection of PIPython.

    :param start_servo: (default: True) if True, start servo on init.
    :type start_servo: bool
    :param axes_order: (default: [0, 1, 2]) permutation for piezo axes IDs and (x, y, z).
        When axes_order is [1, 2, 0], the x axis is piezo axis of index 1,
        and the y (z) axis is index 2 (0).
    :type axes_order: tuple[int, int, int]

    """

    def __init__(self, name, conf, prefix=None):
        BasePiezo3Axes.__init__(self, name, conf=conf, prefix=prefix)
        self.init_device()
        self.init_axes()
        if self.conf.get("start_servo", True):
            self.start_servo()

    def init_device(self):
        self.dev = GCSDevice("E-727")
        devs = self.dev.EnumerateUSB(mask=self.conf.get("mask", "E-727"))
        if len(devs) != 1:
            raise InstError(
                self.full_name(),
                f"Cannot initialize as there's no device or multiple devices: {devs}",
            )
        self.dev.ConnectUSB(devs[0])
        self.logger.info(f"Connected to {devs[0]}.")

    def init_axes(self):
        axes_order = self.conf.get("axes_order", [0, 1, 2])
        ids = self.dev.qSAI()
        self.AXIS_IDs = [ids[i] for i in axes_order]
        self.Axis_to_ID = {
            Axis.X: self.AXIS_IDs[0],
            Axis.Y: self.AXIS_IDs[1],
            Axis.Z: self.AXIS_IDs[2],
        }
        self.logger.info(f"got AXIS_IDs: {self.AXIS_IDs}")

    def start_servo(self):
        self.set_servo(True)
        self.update_target_pos()

    def close_usb(self):
        if hasattr(self, "dev"):
            self.dev.CloseConnection()
            self.logger.info("Closed USB connection.")

    def get_error(self):
        return self.dev.qERR()

    def set_servo(self, on: bool) -> bool:
        self.dev.SVO(self.AXIS_IDs, values=[on] * len(self.AXIS_IDs))
        self.logger.info(f"Servo control all {on}.")
        return True

    def move(self) -> bool:
        pos = self.T @ np.array(self.get_target(), dtype=np.float64)
        self.dev.MOV(axes=self.AXIS_IDs, values=pos)
        return True

    def query_servo(self) -> list[bool]:
        r = self.dev.qSVO(self.AXIS_IDs)
        return [r[ax] for ax in self.AXIS_IDs]

    def query_pos(self) -> list[float]:
        r = self.dev.qPOS(axes=self.AXIS_IDs)
        return [r[ax] for ax in self.AXIS_IDs]

    def query_on_target(self) -> list[bool]:
        r = self.dev.qONT(axes=self.AXIS_IDs)
        return [r[ax] for ax in self.AXIS_IDs]

    def query_target_pos(self) -> list[float]:
        r = self.dev.qMOV(axes=self.AXIS_IDs)
        return [r[ax] for ax in self.AXIS_IDs]

    def update_target_pos(self):
        """query target pos and update self.target."""

        pos = self.Tinv @ np.array(self.query_target_pos(), dtype=np.float64)
        for ax, p in zip(Axis, pos):
            self.target[ax] = p

    def get_pos(self):
        return self.Tinv @ np.array(self.query_pos(), dtype=np.float64)

    def get_pos_ont(self):
        return (self.get_pos(), self.query_on_target())

    # Standard API

    def close(self):
        self.close_usb()

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def configure(self, params: dict, label: str = "", group: str = "") -> bool:
        return True

    def set(self, key: str, value=None) -> bool:
        key = key.lower()
        if key == "target":
            return self.set_target(value)
        elif key == "servo":
            if isinstance(value, bool):
                return self.set_servo(value)
            else:
                self.logger.error('set("servo", on: bool): on must be bool.')
                return False
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key == "pos":
            return self.get_pos()
        elif key == "pos_ont":
            return self.get_pos_ont()
        else:
            return BasePiezo3Axes.get(self, key, args)


class E727_3_USB_AO(E727_3_USB):
    """E727.3 driver using USB connection of PIPython and DAQ AnalogOut.

    :param start_servo: (default: True) if True, start servo on init.
    :type start_servo: bool
    :param axes_order: (default: [0, 1, 2]) permutation for piezo axes IDs and (x, y, z).
        When axes_order is [1, 2, 0], the x axis is piezo axis of index 1,
        and the y (z) axis is index 2 (0).
    :type axes_order: tuple[int, int, int]

    :param lines: list of strings to designate DAQ AnalogOut physical channels.
        Order them as [x, y, z].
    :type lines: list[str]
    :param scale_volt_per_um: the AnalogOut voltage scale in V / um.
        the output voltage is determined by: (pos_um - offset_um) * scale_volt_per_um.
    :type scale_volt_per_um: float
    :param offset_um: the AnalogOut voltage offset in um. see scale_volt_per_um too.
    :type offset_um: float

    :param samples_margin: (has preset, default: 0) margin of samples for AnalogOut.
    :type samples_margin: int
    :param write_and_start: (has preset, default: True) if False, DAQ Task is started and then
        scan data is written in start_scan(). True reverses this order.
    :type write_and_start: bool

    """

    def __init__(self, name, conf, prefix=None):
        BasePiezo3Axes.__init__(self, name, conf=conf, prefix=prefix)
        self.init_device()
        self.init_axes()

        self.check_required_conf(("lines", "scale_volt_per_um", "offset_um"))
        lines = self.conf["lines"]
        if len(lines) != len(self.AXIS_IDs):
            raise ValueError("length of AO lines and number of axis don't match.")
        self._scale_volt_per_um = self.conf["scale_volt_per_um"]
        self._offset_um = self.conf["offset_um"]

        ao_name = name + "_ao"
        # bounds in volts converted from command space range
        bounds = [[self.um_to_volt_raw(v) for v in lim] for lim in self.get_limit()]
        ao_conf = {
            "lines": lines,
            "bounds": bounds,
        }
        if "samples_margin" in self.conf:
            ao_conf["samples_margin"] = self.conf["samples_margin"]
        self._ao = AnalogOut(ao_name, ao_conf, prefix=prefix)
        self.load_conf_preset(self._ao.device_type)
        self._write_and_start = self.conf.get("write_and_start", True)

        if self.conf.get("start_servo", True):
            self.start_servo()

    def load_conf_preset(self, dev_type: str):
        loader = PresetLoader(self.logger, PresetLoader.Mode.PARTIAL)
        loader.add_preset("PCIe-6343", [("write_and_start", False)])
        loader.add_preset("USB-6363", [("write_and_start", True)])
        loader.load_preset(self.conf, dev_type)

    def write_scan_array(self, scan_array: np.ndarray) -> bool:
        # scan_array is N x 3 array.
        # Transpose to 3 x N, convert, and transpose again to N x 3.
        volts = self.um_to_volt(scan_array.T).T
        return self._ao.set_output(volts, auto_start=False)

    def start_scan(self, scan_array: np.ndarray) -> bool:
        """start the configured scan with scan_array."""

        if self._write_and_start:
            return self.write_scan_array(scan_array) and self.start()
        else:
            return self.start() and self.write_scan_array(scan_array)

    def join(self, timeout_sec: float):
        return self._ao.join(timeout_sec)

    def close_ao(self):
        if hasattr(self, "_ao"):
            self._ao.close_once()

    def move(self) -> bool:
        return self._ao.set_output_once(self.um_to_volt(self.get_target()))

    def um_to_volt_raw(self, pos_um):
        return (pos_um - self._offset_um) * self._scale_volt_per_um

    def um_to_volt(self, pos_um: np.ndarray):
        pos_tr = self.T @ np.array(pos_um, dtype=np.float64)
        return self.um_to_volt_raw(pos_tr)

    def volt_to_um(self, volt: np.ndarray):
        p = np.array(volt, dtype=np.float64) / self._scale_volt_per_um + self._offset_um
        return self.Tinv @ p

    def get_scan_buffer_size(self):
        return self._ao.get_onboard_buffer_size()

    # Standard API

    def close(self):
        self.close_ao()
        self.close_usb()

    def shutdown(self) -> bool:
        success = (
            self.stop()
            and self.configure({})
            and self.start()
            and self.set_target_pos([0.0] * 3)
            and self.stop()
            and self.set_servo(False)
        )

        self.close_once()

        if success:
            self.logger.info("Ready to power-off.")
        else:
            self.logger.error("Error on shutdown process.")

        return success

    def configure(self, params: dict, label: str = "", group: str = "") -> bool:
        return self._ao.configure(params)

    def start(self) -> bool:
        return self._ao.start()

    def stop(self) -> bool:
        return self._ao.stop()

    def get(self, key: str, args=None):
        if key in ("onboard_buffer_size", "scan_buffer_size"):
            return self._ao.get_onboard_buffer_size()
        elif key == "buffer_size":
            return self._ao.get_buffer_size()
        else:
            return E727_3_USB.get(self, key, args)


class AnalogPiezo3Axes(BasePiezo3Axes):
    """Generic Piezo Driver using DAQ AnalogIn and AnalogOut.

    TODO: NOTE that this class is not tested against real hardware yet.

    AnalogOut is used to output position command to Piezo Driver (amplifier), and
    AnalogIn is used to read the sensor value from Piezo Driver (amplifier).

    :param lines: list of strings to designate DAQ AnalogOut physical channels.
        Order them as [x, y, z].
    :type lines: list[str]
    :param ai_lines: list of strings to designate DAQ AnalogIn physical channels.
        Order them as [x, y, z].
    :type ai_lines: list[str]
    :param scale_volt_per_um: the AnalogOut voltage scale in V / um.
        the output voltage is determined by: (pos_um - offset_um) * scale_volt_per_um.
    :type scale_volt_per_um: float
    :param offset_um: the AnalogOut voltage offset in um. see scale_volt_per_um too.
    :type offset_um: float
    :param ai_scale_volt_per_um: (default: scale_volt_per_um) the AnalogIn voltage scale in V / um.
        the input voltage is considered as: (pos_um - ai_offset_um) * ai_scale_volt_per_um.
    :type ai_scale_volt_per_um: float
    :param ai_offset_um: (default: offset_um) the AnalogIn voltage offset in um.
        see ai_scale_volt_per_um too.
    :type ai_offset_um: float

    :param ont_error: (default: 0.010) Threshold error in um to determine current position is
        on target or not.
    :type ont_error: float
    :param ai_oversample: (default: 10000) Number of sampling points to read current position.
    :type ai_oversample: int

    :param samples_margin: (has preset, default: 0) margin of samples for AnalogOut.
    :type samples_margin: int
    :param write_and_start: (has preset, default: True) if False, DAQ Task is started and then
        scan data is written in start_scan(). True reverses this order.
    :type write_and_start: bool

    """

    def __init__(self, name, conf, prefix=None):
        BasePiezo3Axes.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("lines", "ai_lines", "scale_volt_per_um", "offset_um"))
        lines = self.conf["lines"]
        ai_lines = self.conf["ai_lines"]
        self._scale_volt_per_um = self.conf["scale_volt_per_um"]
        self._offset_um = self.conf["offset_um"]
        self._ai_scale_volt_per_um = self.conf.get("ai_scale_volt_per_um", self._scale_volt_per_um)
        self._ai_offset_um = self.conf.get("ai_offset_um", self._ai_offset_um)
        self._ont_error = self.conf.get("ont_error", 0.010)
        self._ai_oversample = self.conf.get("ai_oversample", 10000)

        ao_name = name + "_ao"
        # bounds in volts converted from command space range
        bounds = [[self.um_to_volt_raw(v) for v in lim] for lim in self.get_limit()]
        ao_conf = {
            "lines": lines,
            "bounds": bounds,
        }
        if "samples_margin" in self.conf:
            ao_conf["samples_margin"] = self.conf["samples_margin"]
        self._ao = AnalogOut(ao_name, ao_conf, prefix=prefix)
        self.load_conf_preset(self._ao.device_type)
        self._write_and_start = self.conf.get("write_and_start", True)

        ai_name = name + "_ai"
        ai_conf = {"lines": ai_lines}
        # bounds in volts converted from command space range
        ai_bounds = [[self.ai_um_to_volt_raw(v) for v in lim] for lim in self.get_limit()]
        self._ai = AnalogIn(ai_name, ai_conf, prefix=prefix)
        self._ai.configure_on_demand({"bounds": ai_bounds})

        self.init_target_pos()

    def load_conf_preset(self, dev_type: str):
        loader = PresetLoader(self.logger, PresetLoader.Mode.PARTIAL)
        loader.add_preset("PCIe-6343", [("write_and_start", False)])
        loader.add_preset("USB-6363", [("write_and_start", True)])
        loader.load_preset(self.conf, dev_type)

    def write_scan_array(self, scan_array: np.ndarray) -> bool:
        # scan_array is N x 3 array.
        # Transpose to 3 x N, convert, and transpose again to N x 3.
        volts = self.um_to_volt(scan_array.T).T
        return self._ao.set_output(volts, auto_start=False)

    def start_scan(self, scan_array: np.ndarray) -> bool:
        """start the configured scan with scan_array."""

        if self._write_and_start:
            return self.write_scan_array(scan_array) and self.start()
        else:
            return self.start() and self.write_scan_array(scan_array)

    def join(self, timeout_sec: float):
        return self._ao.join(timeout_sec)

    def move(self) -> bool:
        return self._ao.set_output_once(self.um_to_volt(self.get_target()))

    def um_to_volt_raw(self, pos_um):
        return (pos_um - self._offset_um) * self._scale_volt_per_um

    def um_to_volt(self, pos_um: np.ndarray):
        pos_tr = self.T @ np.array(pos_um, dtype=np.float64)
        return self.um_to_volt_raw(pos_tr)

    def ai_um_to_volt_raw(self, pos_um):
        return (pos_um - self._ai_offset_um) * self._ai_scale_volt_per_um

    def ai_um_to_volt(self, pos_um: np.ndarray):
        pos_tr = self.T @ np.array(pos_um, dtype=np.float64)
        return self.ai_um_to_volt_raw(pos_tr)

    def ai_volt_to_um(self, volt: np.ndarray):
        p = np.array(volt, dtype=np.float64) / self._ai_scale_volt_per_um + self._ai_offset_um
        return self.Tinv @ p

    def init_target_pos(self):
        """initialize self.target using current position."""

        pos = self.get_pos()
        for ax, p in zip(Axis, pos):
            self.target[ax] = p

    def get_pos(self):
        ai_volt = self._ai.read_on_demand(self._ai_oversample)
        return self.ai_volt_to_um(ai_volt)

    def get_pos_ont(self):
        def ontgt(val):
            return abs(val) < self._ont_error

        pos = self.get_pos()
        ont = [ontgt(p - t) for p, t in zip(pos, self.get_target())]
        return pos, ont

    def get_scan_buffer_size(self):
        return self._ao.get_onboard_buffer_size()

    # Standard API

    def close(self):
        if hasattr(self, "_ao"):
            self._ao.close_once()
        if hasattr(self, "_ai"):
            self._ai.close_once()

    def configure(self, params: dict, label: str = "", group: str = "") -> bool:
        return self._ao.configure(params)

    def start(self) -> bool:
        return self._ao.start()

    def stop(self) -> bool:
        return self._ao.stop()

    def set(self, key: str, value=None) -> bool:
        key = key.lower()
        if key == "target":
            return self.set_target(value)
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key in ("onboard_buffer_size", "scan_buffer_size"):
            return self._ao.get_onboard_buffer_size()
        elif key == "buffer_size":
            return self._ao.get_buffer_size()
        elif key == "pos":
            return self.get_pos()
        elif key == "pos_ont":
            return self.get_pos_ont()
        else:
            return BasePiezo3Axes.get(self, key, args)
