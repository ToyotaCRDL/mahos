#!/usr/bin/env python3

import typing as T
import time
import enum

import numpy as np

from .instrument import Instrument
from ..msgs.confocal_msgs import Axis
from ..msgs.inst_camera_msgs import FrameResult


class Clock_mock(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self.line = self.conf["line"]

    def get_internal_output(self):
        return self.line + "InternalOutput"

    # Standard API

    def configure(self, params: dict) -> bool:
        return True

    def start(self) -> bool:
        self.logger.info("Started dummy clock.")
        return True

    def stop(self) -> bool:
        self.logger.info("Stopped dummy clock.")
        return True

    def get(self, key: str, args=None):
        if key == "internal_output":
            return self.get_internal_output()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class SG_mock(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self._resource = self.conf.get("resource")

        self.power_min, self.power_max = self.conf.get("power_bounds", (None, None))
        self.freq_min, self.freq_max = self.conf.get("freq_bounds", (None, None))

    def set_output(self, on):
        if on:
            self.logger.info("Output ON")
        else:
            self.logger.info("Output OFF")
        return True

    def set_dm_source(self, source: str) -> bool:
        """Set digital (IQ) moduration source.

        By default (on RST*), INTernal is selected.

        """

        if source.upper() not in ("EXT", "EXTERNAL", "INT", "INTERNAL", "SUM"):
            self.logger.error("invalid digital modulation source")
            return False

        return True

    def set_dm(self, on: bool) -> bool:
        if on:
            self.logger.info("DM ON")
        else:
            self.logger.info("DM OFF")
        return True

    def set_modulation(self, on: bool) -> bool:
        """If on is True turn on modulation."""

        if on:
            self.logger.info("Modulation ON.")
        else:
            self.logger.info("Modulation OFF.")
        return True

    def set_init_cont(self, on=True) -> bool:
        if on:
            self.logger.info("INIT:CONT ON")
        else:
            self.logger.info("INIT:CONT OFF")
        return True

    def abort(self) -> bool:
        self.logger.info("ABORT")
        return True

    def get_power_bounds(self):
        return self.power_min, self.power_max

    def get_freq_bounds(self):
        return self.freq_min, self.freq_max

    def get_bounds(self):
        return {
            "power": self.get_power_bounds(),
            "freq": self.get_freq_bounds(),
        }

    def configure_CW(self, freq, power) -> bool:
        """Setup Continuous Wave output with fixed freq and power."""

        self.logger.info("Mock configuration CW mode.")
        return True

    def set_freq_CW(self, freq) -> bool:
        # f = freq * 1E-6
        # self.logger.info(f"Mock set freq CW: {f:.2f} MHz.")
        return True

    # Standard API

    def set(self, key: str, value=None) -> bool:
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
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key == "opc":
            return True
        elif key == "bounds":
            return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def configure(self, params: dict) -> bool:
        return True

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True


class FG_mock(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self._resource = self.conf.get("resource")

        self._ampl_bounds = (0.0, 10.0)
        self._freq_bounds = (0.1, 70e6)

    def set_output(self, on, ch):
        on_off = "ON" if on else "OFF"
        self.logger.info(f"Output{ch} {on_off}")
        return True

    def get_bounds(self):
        return {"ampl": self._ampl_bounds, "freq": self._freq_bounds}

    # Standard API

    def set(self, key: str, value=None) -> bool:
        if key == "output":
            if isinstance(value, dict):
                if "on" in value and "ch" in value:
                    return self.set_output(value["on"], value["ch"])
                else:
                    return self.fail_with("set('output', dict): needs keys on and ch")
            elif isinstance(value, bool):
                return self.set_output(value)
            else:
                return self.fail_with(f"set('output', value): Ill-formed value {value}")
        else:
            return self.fail_with("Unknown set() key.")

    def get(self, key: str, args=None):
        if key == "opc":
            return True
        elif key == "bounds":
            return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def configure(self, params: dict) -> bool:
        return True


class Piezo_mock(Instrument):
    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self.target = [0.0, 0.0, 0.0]

    def set_target(self, value: dict):
        if isinstance(value, dict) and "ax" in value and "pos" in value:
            ax, p = value["ax"], value["pos"]
            if all([isinstance(v, (tuple, list)) for v in (ax, p)]) and len(ax) == len(p):
                axes, pos = ax, p
            elif all([not isinstance(v, (tuple, list)) for v in (ax, p)]):
                axes, pos = [ax], [p]
            else:
                self.logger.error('set("move", value): invalid format for value in dict.')
                return False
        elif isinstance(value, (tuple, list)) and len(value) == 3:
            axes, pos = (Axis.X, Axis.Y, Axis.Z), value
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            axes, pos = [value[0]], [value[1]]
        else:
            self.logger.error('set("move", value): invalid format for value.')
            return False

        for ax, p in zip(axes, pos):
            if ax == Axis.X:
                self.target[0] = p
            elif ax == Axis.Y:
                self.target[1] = p
            elif ax == Axis.Z:
                self.target[2] = p
            else:
                return False
        self.logger.debug(
            "dummy move {} to {}".format(
                ", ".join([ax.name for ax in axes]), ", ".join([f"{p:.4f}" for p in pos])
            )
        )
        return True

    def get_pos(self):
        pos = [v + np.random.normal(0.0, 0.005) for v in self.target]
        return pos

    def get_pos_ont(self):
        def ontgt(val, bound=0.010):
            return abs(val) < bound

        pos = self.get_pos()
        ont = [ontgt(p - t) for p, t in zip(pos, self.target)]
        return pos, ont

    def get_target(self):
        return self.target

    def get_range(self):
        return [(0.0, 200.0)] * 3

    # Standard API

    def shutdown(self) -> bool:
        self.logger.info("Shutdown piezo")
        self.close_once()
        return True

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def configure(self, params: dict) -> bool:
        return True

    def set(self, key: str, value=None):
        if key == "target":
            return self.set_target(value)
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key == "pos":
            return self.get_pos()
        elif key == "pos_ont":
            return self.get_pos_ont()
        elif key == "target":
            return self.get_target()
        elif key == "range":
            return self.get_range()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class Counter_mock(Instrument):
    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self.source = self.conf.get("source")

    def get_data(self):
        if self.stamp:
            return np.random.normal(size=self.samples, loc=100.0), time.time_ns()
        else:
            return np.random.normal(size=self.samples, loc=100.0)

    # Standard API

    def configure(self, params: dict) -> bool:
        self.samples = params["cb_samples"]
        self.stamp = params.get("stamp", False)
        return True

    def start(self) -> bool:
        self.logger.info("Started dummy counting.")
        return True

    def stop(self) -> bool:
        self.logger.info("Stopped dummy counting.")
        return True

    def get(self, key: str, args=None):
        if key == "data":
            return self.get_data()
        elif key == "all_data":
            return [self.get_data()]
        elif key == "unit":
            return "cps"
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class MCS_mock(Instrument):
    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self._range = 10
        self._bin = 0.2e-9
        self._sweeps = 0.0
        from .tdc import MCS

        self.ACQSTATUS = MCS.ACQSTATUS
        self._running = False

    def get_data(self, nDisplay: int):
        data = np.random.normal(0.0, 1.0, size=self._range)
        return data

    def get_status(self, nDisplay: int):
        status = self.ACQSTATUS()
        self._sweeps += 1.0
        status.started = int(self._running)
        status.sweeps = self._sweeps

        return status

    # Standard API

    def configure(self, params: dict) -> bool:
        self.logger.info(f"Dummy conf for MCS: {params}")
        if "range" in params and "bin" in params:
            self._range = int(round(params["range"] / params["bin"]))
            self._bin = params["bin"]
        return True

    def set(self, key: str, value=None):
        if key == "clear":
            return True
        elif key == "sweeps":
            return True
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key == "range_bin":
            return {"range": self._range, "bin": self._bin}
        elif key == "bin":
            return self._bin
        elif key == "data":
            if not isinstance(args, int):
                self.logger.error('get("data", args): args must be int (channel).')
            return self.get_data(args)
        elif key == "status":
            if not isinstance(args, int):
                self.logger.error('get("status", args): args must be int (channel).')
            return self.get_status(args)
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def start(self) -> bool:
        self.logger.info("Started dummy MCS.")
        self._running = True
        self._sweeps = 0.0
        return True

    def resume(self) -> bool:
        self.logger.info("Resumed dummy MCS.")
        self._running = True
        return True

    def stop(self) -> bool:
        self.logger.info("Stopped dummy MCS.")
        self._running = False
        return True


class Spectrometer_mock(Instrument):
    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        if conf.get("base_config"):
            self.set_base_config(conf["base_config"])
        else:
            self._current_base_config = None

        # dummy params
        self._exposure_time = 123.0
        self._exposures = 3
        self._center_wavelength = 680.0

    def capture(self):
        time.sleep(self._exposures * self._exposure_time * 1e-3)

        num = 1001
        x = np.linspace(self._center_wavelength - 100.0, self._center_wavelength + 100.0, num)
        y = np.cos(np.pi * x / 100.0) + np.random.normal(size=len(x))

        return np.vstack((x, y))

    def set_base_config(self, name: str):
        self._current_base_config = name
        return True

    def get_base_config(self):
        return self._current_base_config

    def get_base_configs(self):
        return ["default", "config1", "config2"]

    def set_exposure_time(self, time_ms: float) -> bool:
        if time_ms <= 0.0:
            self.logger.error("time_ms must be a positive float.")
            return False
        self._exposure_time = time_ms
        return True

    def get_exposure_time(self) -> float:
        return self._exposure_time

    def set_exposures_per_frame(self, exposures: int) -> bool:
        if exposures <= 0:
            self.logger.error("Exposures must be a positive integer.")
            return False

        self._exposures = exposures
        return True

    def get_exposures_per_frame(self) -> int:
        return self._exposures

    def set_grating_center_wavelength(self, wavelength_nm: float) -> bool:
        if wavelength_nm <= 0.0:
            self.logger.error("wavelength_nm must be a positive float.")
            return False
        self._center_wavelength = wavelength_nm
        return True

    def get_grating_center_wavelength(self) -> float:
        return self._center_wavelength

    def get(self, key: str, args=None):
        if key == "data":
            return self.capture()
        elif key == "config":
            return {
                "base_config": self.get_base_config(),
                "exposure_time": self.get_exposure_time(),
                "exposures": self.get_exposures_per_frame(),
                "center_wavelength": self.get_grating_center_wavelength(),
            }
        elif key == "base_config":
            return self.get_base_config()
        elif key == "base_configs":
            return self.get_base_configs()
        elif key == "exposure_time":
            return self.get_exposure_time()
        elif key == "exposures":
            return self.get_exposures_per_frame()
        elif key == "center_wavelength":
            return self.get_grating_center_wavelength()
        else:
            self.logger.error(f"Unknown get() key: {key}.")
            return None

    def set(self, key: str, value=None) -> bool:
        if key == "base_config":
            return self.set_base_config(value)
        elif key == "exposure_time":
            return self.set_exposure_time(value)
        elif key == "exposures":
            return self.set_exposures_per_frame(value)
        elif key == "center_wavelength":
            return self.set_grating_center_wavelength(value)
        else:
            return self.fail_with(f"Unknown set() key: {key}.")

    def configure(self, params: dict) -> bool:
        success = True
        if params.get("base_config"):
            success &= self.set_base_config(params["base_config"])
        if params.get("exposure_time"):
            success &= self.set_exposure_time(params["exposure_time"])
        if params.get("exposures"):
            success &= self.set_exposures_per_frame(params["exposures"])
        if params.get("center_wavelength"):
            success &= self.set_grating_center_wavelength(params["center_wavelength"])

        if not success:
            self.fail_with("Failed to configure.")

        return success


class Camera_mock(Instrument):
    class Mode(enum.Enum):
        UNCONFIGURED = 0
        CONTINUOUS = 1
        SOFT_TRIGGER = 2

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self._running = False
        self._mode = self.Mode.UNCONFIGURED
        self._frame_count = 0
        # dummy params
        self._exposure_time = 123.0

    def configure_continuous(self, exposure_time_sec: float) -> bool:
        self._exposure_time = exposure_time_sec
        self._mode = self.Mode.CONTINUOUS
        self._frame_count = 0

        self.logger.info("Configured for continuous capture.")
        return True

    def configure_soft_trigger(
        self,
        exposure_time_sec: float,
        burst_num: int = 1,
        binning: int = 1,
        roi: T.Optional[dict] = None,
    ) -> bool:
        self._exposure_time = exposure_time_sec
        self._mode = self.Mode.SOFT_TRIGGER
        self.logger.info("Configured for software-triggered capture.")

        return True

    def get_frame(self):
        if self._mode == self.Mode.CONTINUOUS:
            self._frame_count += 1
            return FrameResult(
                frame=np.random.normal(loc=10.0, size=(450, 800)),
                time=time.time(),
                count=self._frame_count,
            )
        elif self._mode == self.Mode.SOFT_TRIGGER:
            return FrameResult(frame=np.random.normal(loc=10.0, size=(450, 800)))
        else:
            self.logger.error("get_frame() is called but not running.")
            return FrameResult(frame=None)

    def configure(self, params: dict) -> bool:
        if not self.check_required_params(params, ("mode",)):
            return False

        mode = params["mode"].lower()
        if mode == "continuous":
            if not self.check_required_params(params, ("exposure_time",)):
                return False
            return self.configure_continuous(params["exposure_time"])
        else:
            return self.fail_with(f"Unknown mode {mode}.")

    def start(self) -> bool:
        if self._running:
            self.logger.warn("start() is called while running.")
            return True

        if self._mode is None:
            self.logger.error("Must be configured before start().")
            return False

        if self._mode in (self.Mode.CONTINUOUS, self.Mode.SOFT_TRIGGER):
            self._running = True
            return True
        else:
            return False

    def stop(self) -> bool:
        if not self._running:
            return True

        if self._mode in (self.Mode.CONTINUOUS, self.Mode.SOFT_TRIGGER):
            self._running = False
            return True
        else:  # self.Mode.UNCONFIGURED
            return self.fail_with("stop() is called but mode is unconfigured.")

    def get(self, key: str, args=None):
        if key == "frame":
            return self.get_frame()
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None
