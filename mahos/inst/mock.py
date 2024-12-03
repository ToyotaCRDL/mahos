#!/usr/bin/env python3

"""
Mock Instrument classes for tests.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import time
import enum

import numpy as np

from .instrument import Instrument
from .pg_dtg_core.dtg_core import DTGCoreMixin
from ..msgs.confocal_msgs import Axis
from ..msgs.inst.camera_msgs import FrameResult
from ..msgs.inst.tdc_msgs import ChannelStatus, RawEvents
from ..msgs.inst.pg_msgs import TriggerType, Block, Blocks, BlockSeq, AnalogChannel
from ..msgs.inst.spectrometer_msgs import Temperature
from ..msgs import param_msgs as P


class Clock_mock(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self.line = self.conf["line"]

    def get_internal_output(self):
        return self.line + "InternalOutput"

    # Standard API

    def configure(self, params: dict, label: str = "") -> bool:
        return True

    def start(self, label: str = "") -> bool:
        self.logger.info("Started dummy clock.")
        return True

    def stop(self, label: str = "") -> bool:
        self.logger.info("Stopped dummy clock.")
        return True

    def get(self, key: str, args=None, label: str = ""):
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

    def configure_cw(self, freq, power) -> bool:
        """Setup Continuous Wave output with fixed freq and power."""

        self.logger.info("Mock configuration CW mode.")
        return True

    def configure_iq_ext(self, freq, power) -> bool:
        """Setup external IQ modulation mode."""

        self.logger.info("Mock configuration for external IQ modulation.")
        return True

    def configure_point_trig_freq_sweep(self, start, stop, num, power) -> bool:
        self.logger.info("Mock configuration point trig freq sweep mode.")
        return True

    def set_freq_CW(self, freq) -> bool:
        # f = freq * 1E-6
        # self.logger.info(f"Mock set freq CW: {f:.2f} MHz.")
        return True

    # Standard API

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
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return True
        elif key == "bounds":
            return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "cw":
            return self.configure_cw(params["freq"], params["power"])
        elif label == "iq_ext":
            return self.configure_iq_ext(params["freq"], params["power"])
        elif label == "point_trig_freq_sweep":
            return self.configure_point_trig_freq_sweep(
                params["start"],
                params["stop"],
                params["num"],
                params["power"],
            )
        else:
            self.logger.error(f"Unknown label {label}")
            return False

    def start(self, label: str = "") -> bool:
        return True

    def stop(self, label: str = "") -> bool:
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

    def set(self, key: str, value=None, label: str = "") -> bool:
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

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return True
        elif key == "bounds":
            return self.get_bounds()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def configure(self, params: dict, label: str = "") -> bool:
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
        self.close()
        return True

    def start(self, label: str = "") -> bool:
        return True

    def stop(self, label: str = "") -> bool:
        return True

    def configure(self, params: dict, label: str = "") -> bool:
        return True

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "target":
            return self.set_target(value)
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None, label: str = ""):
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

    def configure(self, params: dict, label: str = "") -> bool:
        self.samples = params["cb_samples"]
        self.stamp = params.get("stamp", False)
        return True

    def start(self, label: str = "") -> bool:
        self.logger.info("Started dummy counting.")
        return True

    def stop(self, label: str = "") -> bool:
        self.logger.info("Stopped dummy counting.")
        return True

    def get(self, key: str, args=None, label: str = ""):
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
        self.resolution_sec = 0.2e-9
        self._bin = 0.2e-9
        self._starts = 0
        self._mean_events = 0.0
        self._tstart = time.time()
        self._running = False

    def get_data(self, nDisplay: int):
        self._mean_events += 1.0
        data = np.random.normal(self._mean_events, 1.0, size=self._range)
        return data

    def get_data_roi(self, nDisplay: int, roi: list[tuple[int, int]]) -> list[np.ndarray] | None:
        data = self.get_data(nDisplay)
        if data is None:
            return None
        data_roi = []
        for start, stop in roi:
            # fill up out-of-bounds in ROI with zeros
            if start < 0:
                d = np.concatenate((np.zeros(abs(start), dtype=data.dtype), data[:stop]))
            else:
                d = data[start:stop]
            if stop > len(data):
                d = np.concatenate((d, np.zeros(stop - len(data), dtype=data.dtype)))
            data_roi.append(d)
        return data_roi

    def get_status(self, nDisplay: int) -> ChannelStatus:
        runtime = time.time() - self._tstart
        # dummy status
        total = 0
        self._starts += 1
        return ChannelStatus(self._running, runtime, total, self._starts)

    def get_raw_events(self) -> RawEvents | None:
        return RawEvents(np.arange(0, 10_000_000, 10, dtype=np.uint64))

    # Standard API

    def configure(self, params: dict, label: str = "") -> bool:
        self.logger.info(f"Dummy conf for MCS: {params}")
        if "range" in params and "bin" in params:
            tbin = params["bin"] or self.resolution_sec
            self._range = int(round(params["range"] / tbin))
            self._bin = tbin
        return True

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "clear":
            return True
        elif key == "sweeps":
            return True
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None, label: str = ""):
        if key == "range_bin":
            return {"range": self._range, "bin": self._bin}
        elif key == "bin":
            return self._bin
        elif key == "data":
            if not isinstance(args, int):
                self.logger.error('get("data", args): args must be int (channel).')
            return self.get_data(args)
        elif key == "data_roi":
            return self.get_data_roi(args["ch"], args["roi"])
        elif key == "status":
            if not isinstance(args, int):
                self.logger.error('get("status", args): args must be int (channel).')
            return self.get_status(args)
        elif key == "raw_events":
            return self.get_raw_events()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def start(self, label: str = "") -> bool:
        self.logger.info("Started dummy MCS.")
        self._running = True
        self._starts = 0
        self._mean_events = 0.0
        self._tstart = time.time()
        return True

    def resume(self, label: str = "") -> bool:
        self.logger.info("Resumed dummy MCS.")
        self._running = True
        return True

    def stop(self, label: str = "") -> bool:
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

    def get_temperature(self) -> Temperature:
        return Temperature(-59.9, -60.0)

    def get(self, key: str, args=None, label: str = ""):
        if key == "data":
            return self.capture()
        elif key == "base_config":
            return self.get_base_config()
        elif key == "base_configs":
            return self.get_base_configs()
        elif key == "temperature":
            return self.get_temperature()
        else:
            self.logger.error(f"Unknown get() key: {key}.")
            return None

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "exposure_time":
            return self.set_exposure_time(value)
        elif key == "exposures":
            return self.set_exposures_per_frame(value)
        elif key == "center_wavelength":
            return self.set_grating_center_wavelength(value)
        else:
            return self.fail_with(f"Unknown set() key: {key}.")

    def configure(self, params: dict, label: str = "") -> bool:
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
        roi: dict | None = None,
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

    def configure(self, params: dict, label: str = "") -> bool:
        if not self.check_required_params(params, ("mode",)):
            return False

        mode = params["mode"].lower()
        if mode == "continuous":
            if not self.check_required_params(params, ("exposure_time",)):
                return False
            return self.configure_continuous(params["exposure_time"])
        else:
            return self.fail_with(f"Unknown mode {mode}.")

    def start(self, label: str = "") -> bool:
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

    def stop(self, label: str = "") -> bool:
        if not self._running:
            return True

        if self._mode in (self.Mode.CONTINUOUS, self.Mode.SOFT_TRIGGER):
            self._running = False
            return True
        else:  # self.Mode.UNCONFIGURED
            return self.fail_with("stop() is called but mode is unconfigured.")

    def get(self, key: str, args=None, label: str = ""):
        if key == "frame":
            return self.get_frame()
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None


class Params_mock(Instrument):
    """Mock for instrument with configurable ParamDict."""

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self.paramA: int = 3
        self.paramB: float = 0.5
        self.paramC: str = "aaa"
        self.paramD: bool = False

    # Standard API

    def start(self, label: str = "") -> bool:
        self.logger.info(f"Start {label}")
        return True

    def stop(self, label: str = "") -> bool:
        self.logger.info(f"Stop {label}")
        return True

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "labelA":
            self.paramA = params["paramA"]
            self.paramB = params["paramB"]
            return True
        elif label == "labelB":
            self.paramC = params["paramC"]
            self.paramD = params["paramD"]
            return True
        else:
            self.logger.error(f"Unknown label {label}")
            return True

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        if label == "labelA":
            return P.ParamDict(
                paramA=P.IntParam(self.paramA, 0, 10), paramB=P.FloatParam(self.paramB, 0.0, 1.0)
            )
        elif label == "labelB":
            return P.ParamDict(paramC=P.StrParam(self.paramC), paramD=P.BoolParam(self.paramD))
        else:
            self.logger.error(f"Unknown label {label}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        return ["labelA", "labelB"]


class DTG5274_mock(Instrument, DTGCoreMixin):
    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix)

        self.check_required_conf(("local_dir",))
        self.LOCAL_DIR = self.conf["local_dir"]
        self.SCAFFOLD = self.conf.get("scaffold_filename", "scaffold.dtg")

        self.CHANNELS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        if "channels" in self.conf:
            self.CHANNELS.update(self.conf["channels"])

        # last total block length and offsets.
        self.length = 0
        self.offsets = None

        self.logger.info(f"opened {name} (mock)")

    def block_granularity(self, freq):
        return 4

    def max_block_len(self):
        return 32000000

    def min_block_len(self, freq):
        return 960

    def max_freq(self):
        return 2.7e9

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_positive: bool = True,
        scaffold_name: str | None = None,
        endless: bool = True,
    ) -> bool:
        """Generate tree using default sequence."""

        tree = self._configure_tree_blocks(
            blocks, freq, trigger_positive, scaffold_name=scaffold_name, endless=endless
        )
        return tree is not None

    def configure_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_positive: bool = True,
        scaffold_name: str | None = None,
        endless: bool = True,
    ) -> bool:
        """Generate tree using explicit sequence."""

        if blockseq.nest_depth() > 2:
            return self.fail_with("maximum BlockSeq nest depth is 2 for DTG.")

        tree = self._configure_tree_blockseq(
            blockseq, freq, trigger_positive, scaffold_name=scaffold_name, endless=endless
        )
        return tree is not None

    # Standard API

    def start(self, label: str = "") -> bool:
        self.logger.info("Start dummy DTG.")
        return True

    def stop(self, label: str = "") -> bool:
        self.logger.info("Stop dummy DTG.")
        return True

    def configure(self, params: dict, label: str = "") -> bool:
        if params.get("n_runs") not in (None, 1):
            return self.fail_with("DTG only supports n_runs None or 1.")
        if "blocks" in params and "freq" in params:
            return self.configure_blocks(
                params["blocks"],
                params["freq"],
                self._trigger_positive(params.get("trigger_type")),
            )
        elif "blockseq" in params and "freq" in params:
            return self.configure_blockseq(
                params["blockseq"],
                params["freq"],
                self._trigger_positive(params.get("trigger_type")),
            )
        else:
            return self.fail_with("These params must be given: 'blocks' | 'blockseq' and 'freq'")

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "trigger":
            return True
        elif key == "clear":
            return True
        else:
            return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None, label: str = ""):
        if key == "length":
            return self.length  # length of last configure_blocks
        elif key == "offsets":
            if args is None:
                return self.offsets  # offsets of last configure_blocks
            elif "blocks" in args and "freq" in args:
                return self.validate_blocks(args["blocks"], args["freq"])
            elif "blockseq" in args and "freq" in args:
                return self.validate_blockseq(args["blockseq"], args["freq"])
            else:
                self.logger.error(f"Invalid args for get(offsets): {args}")
                return None
        elif key == "opc":
            return True
        elif key == "validate":
            if "blocks" in args and "freq" in args:
                return self.validate_blocks(args["blocks"], args["freq"])
            elif "blockseq" in args and "freq" in args:
                return self.validate_blockseq(args["blockseq"], args["freq"])
            else:
                self.logger.error(f"Invalid args for get(validate): {args}")
                return None
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class PulseStreamer_mock(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(("resource",))

        self.CHANNELS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
        if "channels" in conf:
            self.CHANNELS.update(conf["channels"])
        if "analog" in conf:
            self.analog_channels = conf["analog"]["channels"]
            self.analog_values = conf["analog"].get(
                "values", {"00": [0.0, 0.0], "01": [0.0, 0.0], "10": [0.0, 0.0], "11": [0.0, 0.0]}
            )
        else:
            self.analog_channels = []
            self.analog_values = {}
        self._strict = conf.get("strict", True)

        # last total block length and offsets.
        self.length = 0
        self.offsets = None
        self._trigger_type = None

        self.logger.info("Opened PulseStreamer mock at {}.".format(self.conf["resource"]))

    def channels_to_ints(self, channels) -> list[int]:
        def parse(c):
            if isinstance(c, (bytes, str)):
                return self.CHANNELS[c]
            elif isinstance(c, int):
                return c
            else:
                raise TypeError("Invalid type for channel. Use bytes, str or int.")

        if channels is None:
            return []
        elif isinstance(channels, (bytes, str, int)):
            return [parse(channels)]
        else:  # container (list or tuple) of (bytes, str, int)
            return [
                parse(c)
                for c in channels
                if not isinstance(c, AnalogChannel) and c not in self.analog_channels
            ]

    def _included_channels_blocks(self, blocks: Blocks[Block]) -> list[int]:
        """create sorted list of channels in blocks converted to ints."""

        # take set() here because channels_to_ints() may contain non-unique integers
        # (CHANNELS is not always injective), though it's quite rare usecase.
        return sorted(list(set(self.channels_to_ints(blocks.channels()))))

    def _extract_trigger_blocks(self, blocks: Blocks[Block]) -> bool | None:
        trigger = False
        for i, block in enumerate(blocks):
            if block.trigger:
                if i:
                    self.logger.error("Can set trigger for first block only.")
                    return None
                trigger = True
        return trigger

    def _generate_seq_blocks(self, blocks: Blocks[Block]):
        rle_patterns = [(ch, []) for ch in self._included_channels_blocks(blocks)]
        a0_patterns = []
        a1_patterns = []
        _analog_given = bool(blocks.analog_channels())

        for i, block in enumerate(blocks):
            for channels, length in block.total_pattern():
                high_channels = self.channels_to_ints(channels)
                for ch, pat in rle_patterns:
                    pat.append((length, 1 if ch in high_channels else 0))

                if _analog_given:
                    # if any AnalogChannel is involved, use these values instead of
                    # sparse D/A conversion using self.analog_values
                    a0 = block.analog_value(self.analog_channels[0], channels)
                    a0_patterns.append((length, a0))
                    a1 = block.analog_value(self.analog_channels[1], channels)
                    a1_patterns.append((length, a1))
                elif self.analog_channels:
                    # self.analog_channels are included as digital channels in blocks.
                    # perform sparse D/A conversion using self.analog_values
                    label = "".join(
                        ("1" if ch in channels else "0" for ch in self.analog_channels)
                    )
                    a0, a1 = self.analog_values[label]
                    a0_patterns.append((length, a0))
                    a1_patterns.append((length, a1))
        return

    def _scale_blocks(self, blocks: Blocks[Block], freq: float) -> Blocks[Block] | None:
        """Return scaled blocks according to freq, or None if any failure."""

        freq = round(freq)
        base_freq = round(1.0e9)
        if freq > base_freq:
            self.logger.error("PulseStreamer's max freq is 1 GHz.")
            return None
        if base_freq % freq:
            self.logger.error("freq must be a divisor of 1 GHz.")
            return None
        scale = base_freq // freq
        if scale > 1:
            self.logger.info(f"Scaling the blocks by {scale}")
            return blocks.scale(scale)
        else:
            return blocks

    def _adjust_blocks(self, blocks: Blocks[Block]) -> list[int] | None:
        """(mutating) adjust block length so that total_length becomes N * 8 (ns).

        :returns: list of offset values, or None if any failure.

        """

        remainder = blocks.total_length() % 8
        if self._strict and remainder:
            self.logger.error("blocks' total_length is not integer multiple of 8.")
            return None
        if not remainder:
            return [0] * len(blocks)

        lb = blocks[-1]
        if lb.Nrep != 1:
            msg = "blocks' total_length is not integer multiple of 8 and"
            msg += f" last block's Nrep is not 1 ({lb.Nrep})."
            self.logger.error(msg)
            return None
        dur = lb.pattern[-1].duration
        offset = 8 - remainder
        lb.update_duration(-1, dur + offset)
        self.logger.warn(f"Adjusted last block by offset {offset}.")
        return [0] * (len(blocks) - 1) + [offset]

    def validate_blocks(self, blocks: Blocks[Block], freq: float) -> list[int] | None:
        blocks = self._scale_blocks(blocks, freq)
        if blocks is None:
            return None
        return self._adjust_blocks(blocks)

    def validate_blockseq(self, blockseq: BlockSeq[Block], freq: float) -> list[int] | None:
        # Since PulseStreamer doesn't have loop control mechanism,
        # given BlockSeq is equivalent to collapsed one.
        return self.validate_blocks(Blocks([blockseq.collapse()]), freq)

    def configure_blocks(
        self,
        blocks: Blocks[Block],
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Make sequence from blocks."""

        blocks = self._scale_blocks(blocks, freq)
        if blocks is None:
            return False

        trigger = self._extract_trigger_blocks(blocks)
        if trigger is None:
            return False

        self.offsets = self._adjust_blocks(blocks)
        if self.offsets is None:
            return False
        self.length = blocks.total_length()

        try:
            self._generate_seq_blocks(blocks)
        except ValueError:
            self.logger.exception("Failed to generate sequence. Check channel name settings.")
            return False

        msg = f"Configured sequence. length: {self.length} offset: {self.offsets[-1]}"
        msg += f" trigger: {trigger}"
        self.logger.info(msg)
        return True

    def configure_blockseq(
        self,
        blockseq: BlockSeq,
        freq: float,
        trigger_type: TriggerType | None = None,
        n_runs: int | None = None,
    ) -> bool:
        """Make sequence from blockseq."""

        # Since PulseStreamer doesn't have loop control mechanism,
        # given BlockSeq is equivalent to collapsed one.
        return self.configure_blocks(Blocks([blockseq.collapse()]), freq, trigger_type, n_runs)

    def trigger(self) -> bool:
        if self._trigger_type in (TriggerType.IMMEDIATE, TriggerType.SOFTWARE):
            self.logger.info("Issuing dummy software trigger")
            return True
        else:
            msg = "Cannot issue software trigger in hardware trigger mode."
            msg += "\nConsider using PulseStreamerDAQTrigger instread."
            self.logger.error(msg)
            return False

    # Standard API

    def reset(self, label: str = "") -> bool:
        self.logger.info("Dummy reset.")
        return True

    def start(self, label: str = "") -> bool:
        self.logger.info("Start dummy streaming.")
        return True

    def stop(self, label: str = "") -> bool:
        self.logger.info("Stop dummy streaming.")
        return True

    def configure(self, params: dict, label: str = "") -> bool:
        if "blocks" in params and "freq" in params:
            return self.configure_blocks(
                params["blocks"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
                n_runs=params.get("n_runs"),
            )
        elif "blockseq" in params and "freq" in params:
            return self.configure_blockseq(
                params["blockseq"],
                params["freq"],
                trigger_type=params.get("trigger_type"),
                n_runs=params.get("n_runs"),
            )
        else:
            return self.fail_with("These params must be given: 'blocks' | 'blockseq' and 'freq'")

    def set(self, key: str, value=None, label: str = "") -> bool:
        if key == "trigger":
            return self.trigger()
        elif key == "clear":  # for API compatibility
            return True
        else:
            return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None, label: str = ""):
        if key == "length":
            return self.length  # length of last configure
        elif key == "offsets":
            return self.offsets  # offsets of last configure
        elif key == "opc":  # for API compatibility
            return True
        elif key == "validate":
            if "blocks" in args and "freq" in args:
                return self.validate_blocks(args["blocks"], args["freq"])
            elif "blockseq" in args and "freq" in args:
                return self.validate_blockseq(args["blockseq"], args["freq"])
            else:
                self.logger.error(f"Invalid args for get(validate): {args}")
                return None
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class DMM_mock(Instrument):
    class Mode(enum.Enum):
        UNCONFIGURED = 0
        DCV = 1
        DCI = 2

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)

        self._running = False
        self._mode = {1: self.Mode.UNCONFIGURED, 2: self.Mode.UNCONFIGURED}

        # just for debug
        self._bias = self.conf.get("bias", 0.0)

    def configure_DCV(self, ch: int = 1) -> bool:
        self._mode[ch] = self.Mode.DCV
        self.logger.info(f"Configured ch {ch} for DC Voltage measurement.")
        return True

    def configure_DCI(self, ch: int = 1) -> bool:
        self._mode[ch] = self.Mode.DCI
        self.logger.info(f"Configured ch {ch} for DC Current measurement.")
        return True

    def get_data(self, ch: int = 1):
        if self._mode[ch] == self.Mode.DCV:
            return np.random.normal(self._bias, 1.0, size=1)[0]
        elif self._mode[ch] == self.Mode.DCI:
            return np.random.normal(self._bias, 0.1, size=1)[0]
        else:
            self.logger.error("get_data() is called but not configured.")
            return None

    def get_unit(self, ch: int = 1) -> str:
        if self._mode[ch] == self.Mode.DCV:
            return "V"
        elif self._mode[ch] == self.Mode.DCI:
            return "A"
        else:
            self.logger.error("get_unit() is called but not configured.")
            return ""

    def get(self, key: str, args=None, label: str = ""):
        if key == "opc":
            return True
        elif key == "data":
            return self.get_data(int(label[2]))
        elif key == "unit":
            return self.get_unit(int(label[2]))
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None

    def get_param_dict_labels(self) -> list[str]:
        return ["ch1_dcv", "ch1_dci", "ch2_dcv", "ch2_dci"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        if label in self.get_param_dict_labels():
            return P.ParamDict(
                nplc=P.IntChoiceParam(10, [1, 2, 10, 100]),
                trigger=P.StrChoiceParam("IMM", ["IMM", "BUS", "EXT"]),
            )
        else:
            return self.fail_with(f"unknown label {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        label = label.lower()
        if label not in self.get_param_dict_labels():
            self.fail_with(f"unknown label: {label}")
        ch = int(label[2])
        meas = label[4:]
        if meas == "dcv":
            return self.configure_DCV(ch)
        else:  # "dci"
            return self.configure_DCI(ch)

    def start(self, label: str = "") -> bool:
        ch = int(label[2])
        if self._mode[ch] == self.Mode.UNCONFIGURED:
            return self.fail_with("start() is called but not configured.")
        return True

    def stop(self, label: str = "") -> bool:
        ch = int(label[2])
        if self._mode[ch] == self.Mode.UNCONFIGURED:
            return self.fail_with("start() is called but not configured.")
        return True


class Positioner_mock(Instrument):
    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self.range = self.conf.get("range", [0.0, 10.0])
        self.pos = 0.0
        self.target = 0.0
        self.homed = False
        self.moving = False

    def move(self, pos: float) -> bool:
        if pos < self.range[0] or pos > self.range[1]:
            return self.fail_with(f"Target pos {pos:.3f} is out of range {self.range}.")
        self.target = pos
        self.pos = pos
        self.logger.info(f"Dummy move to {pos:.3f}")
        return True

    def get_status(self) -> dict:
        return {
            "homed": self.homed,
            "moving": self.moving,
        }

    def get_all(self) -> dict[str, [float, bool]]:
        d = self.get_status()
        d["pos"] = self.pos
        d["target"] = self.target
        d["range"] = self.range

        return d

    def reset(self, label: str = "") -> bool:
        #  Homing (zero-point adjustment) of the positioner
        self.homed = True
        self.logger.info("Dummy homing performed")
        return True

    def stop(self, label: str = "") -> bool:
        self.logger.info("Dummy stop")
        return True

    def get_param_dict_labels(self) -> list[str]:
        return ["pos"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue] | None:
        """Get ParamDict for `label`."""

        if label == "pos":
            return P.ParamDict(
                target=P.FloatParam(
                    self.get_target(), self.limits[0], self.limits[1], doc="target position"
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
            return self.pos
        elif key == "target":
            return self.target
        elif key == "status":
            return self.get_status()
        elif key == "range":
            return self.get_range()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
