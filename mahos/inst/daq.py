#!/usr/bin/env python3

"""
NI-DAQ module.

.. This file is a part of MAHOS project.

This module requires PyDAQmx library.

"""

from __future__ import annotations
import time
from typing import Callable, Optional

import numpy as np
import PyDAQmx as D

from ..util.locked_queue import LockedQueue
from .instrument import Instrument
from .exceptions import InstError


def estimate_buffer_size(rate=None):
    if rate is None:
        return 10000

    if rate > 1e6:
        return 1000000
    elif rate > 10000:
        return 100000
    elif rate > 100:
        return 10000
    else:
        return 1000


def _edge_polarity(rising: bool):
    if rising:
        return D.DAQmx_Val_Rising
    else:
        return D.DAQmx_Val_Falling


def _low_or_high(high: bool):
    if high:
        return D.DAQmx_Val_High
    else:
        return D.DAQmx_Val_Low


def _samples_finite_or_cont(finite: bool):
    if finite:
        return D.DAQmx_Val_FiniteSamps
    else:
        return D.DAQmx_Val_ContSamps


def _samp_timing_clock_or_ondemand(clock: bool):
    if clock:
        return D.DAQmx_Val_SampClk
    else:
        return D.DAQmx_Val_OnDemand


def _device_name(ident: str) -> str:
    """Extract device name from resource (line, counter, etc.) identifier."""

    return ident.strip("/").split("/")[0]


class ConfigurableTask(Instrument):
    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self.running = False
        self.finite = False
        self.task = None

    def join(self, timeout_sec: float):
        self.task.WaitUntilTaskDone(timeout_sec)

    # Standard API

    def close(self):
        self.stop()

    def start(self) -> bool:
        if self.running:
            self.logger.warn("start() is called while running.")
            return True
        if self.task is None:
            self.logger.error("Must call configure() before start().")
            return False

        self.task.StartTask()
        self.running = True
        self.logger.debug("Started Task.")
        return True

    def stop(self) -> bool:
        if not self.running:
            return True
        self.running = False

        if not self.finite:
            self.task.StopTask()
        self.task.ClearTask()
        self.task = None
        self.logger.debug("Stopped and Cleared Task.")
        return True


class ClockSource(ConfigurableTask):
    """A configurable DAQ Task class to provide a clock source.

    :param counter: DAQ's counter name.
    :type counter: str

    """

    def __init__(self, name, conf, prefix=None):
        ConfigurableTask.__init__(self, name, conf=conf, prefix=prefix)
        self.check_required_conf(("counter",))
        self.counter = conf["counter"]

    def get_internal_output(self):
        return self.counter + "InternalOutput"

    # Standard API

    def configure(self, params: dict) -> bool:
        self.task = D.Task()

        if not self.check_required_params(params, ("freq", "samples")):
            return False
        freq = params["freq"]
        samples = params["samples"]

        # optional params
        self.finite = params.get("finite", True)
        idle_state = _low_or_high(params.get("idle_state", False))  # default to low idle state
        initial_delay = params.get("initial_delay", 0.0)
        duty = params.get("duty", 0.5)

        self.task.CreateCOPulseChanFreq(
            self.counter, "", D.DAQmx_Val_Hz, idle_state, initial_delay, freq, duty
        )
        self.task.CfgImplicitTiming(_samples_finite_or_cont(self.finite), samples)

        msg = f"Clock configured. freq: {freq:.2f} Hz."
        if self.finite:
            msg += f" {samples} pulses."
        self.logger.debug(msg)

        return True

    def get(self, key: str, args=None):
        if key == "internal_output":
            return self.get_internal_output()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class ClockDivider(ConfigurableTask):
    """A configurable DAQ Task class to provide a clock divider.

    :param counter: DAQ's counter name.
    :type counter: str

    """

    def __init__(self, name, conf, prefix=None):
        ConfigurableTask.__init__(self, name, conf=conf, prefix=prefix)
        self.check_required_conf(("counter",))
        self.counter = conf["counter"]

    def get_internal_output(self):
        return self.counter + "InternalOutput"

    # Standard API

    def configure(self, params: dict) -> bool:
        self.task = D.Task()

        if not self.check_required_params(params, ("source", "ratio", "samples")):
            return False
        source = params["source"]
        ratio = params["ratio"]
        samples = params["samples"]

        # ~ 50% duty (exactly 50% if ratio is even)
        low, high = ratio // 2 + ratio % 2, ratio // 2

        # optional params
        self.finite = params.get("finite", True)
        idle_state = _low_or_high(params.get("idle_state", False))  # default to low idle state
        initial_delay = params.get("initial_delay", 0)

        self.task.CreateCOPulseChanTicks(
            self.counter,
            "",
            source,
            idle_state,
            initial_delay,
            low,
            high,
        )
        self.task.CfgImplicitTiming(_samples_finite_or_cont(self.finite), samples)

        msg = f"Divider configured. ratio: {ratio}."
        if self.finite:
            msg += f" {samples} pulses."
        self.logger.debug(msg)

        return True

    def get(self, key: str, args=None):
        if key == "internal_output":
            return self.get_internal_output()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class AnalogOut(ConfigurableTask):
    """A configurable DAQ Task class for Analog Output voltage channel(s).

    :param list lines: Sequence of strings to designate DAQ's physical channels.
    :param list bounds: bounds (min, max) values of the output voltage per channels.

    """

    def __init__(self, name, conf, prefix=None):
        ConfigurableTask.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("lines", "bounds"))
        self.lines = self.conf["lines"]
        self.bounds = self.conf["bounds"]
        if len(self.lines) != len(self.bounds):
            raise ValueError("Length of lines and bounds must match.")

    def clip(self, volts: np.ndarray) -> np.ndarray:
        """Clip the voltage values if it exceeds max or min bounds."""

        return np.vstack(
            [vs.clip(bound[0], bound[1]) for vs, bound in zip(volts.T, self.bounds)]
        ).T

    def set_output(self, volts: np.ndarray, auto_start: bool) -> bool:
        """Set output analog voltages.

        Voltage values are clipped automatically.

        :param volts: 2D array-like for voltages. 0th-axis is samples. 1st-axis is channels.

        """

        if not self.running:
            self.logger.error("output() is called while not running.")
            return False
        if not isinstance(volts, (np.ndarray, tuple, list)):
            self.logger.error("volts must be ndarray, tuple, or list.")
            return False

        volts = np.array(volts, dtype=np.float64)
        volts = self.clip(volts)
        samples_per_ch, channels = volts.shape

        if len(self.lines) != channels:
            self.logger.error("Number of output values is different from number of channels.")
            return False

        self.logger.debug(
            f"Writing {samples_per_ch} (samp.) x {channels} (ch.) ="
            + f" {samples_per_ch * channels} data to buffer"
        )
        written = D.int32()
        self.task.WriteAnalogF64(
            samples_per_ch,
            auto_start,
            self.timeout_sec,
            D.DAQmx_Val_GroupByScanNumber,
            volts.flatten(order="C"),
            D.byref(written),
            None,
        )
        if written.value != samples_per_ch:
            self.logger.error(
                "Failed to write all AO samples: {} out of {} per channels".format(
                    written.value, samples_per_ch
                )
            )
            return False

        return True

    def set_output_once(self, volts: np.ndarray) -> bool:
        volts = np.array(volts, dtype=np.float64)
        if volts.ndim == 0:  # scalar value: single values for all channels.
            volts = volts[None, None].repeat(len(self.lines), axis=1)
        elif volts.ndim == 1:  # 1D: simgle sample values per channels.
            volts = volts[None, :]
        elif volts.ndim > 2:
            self.logger.error("Dim of volts array is larger than 2.")
            return False

        return self.set_output(volts, auto_start=True)

    def get_onboard_buffer_size(self) -> int | None:
        if self.task is None:
            return None
        size = D.uInt32()
        self.task.GetBufOutputOnbrdBufSize(D.byref(size))
        return size.value

    def get_buffer_size(self) -> int | None:
        if self.task is None:
            return None
        size = D.uInt32()
        self.task.GetBufOutputBufSize(D.byref(size))
        return size.value

    # Standard API

    def configure(self, params: dict) -> bool:
        self.clock_mode = params.get("clock_mode", False)
        if self.clock_mode:
            if not self.check_required_params(params, ("clock", "samples")):
                return False
            clock = params["clock"]
            samples = params["samples"]
            rate = params.get("rate", 10000.0)
            clock_dir = _edge_polarity(params.get("clock_dir", True))
            self.finite = params.get("finite", True)
        else:
            self.finite = False

        self.timeout_sec = params.get("timeout_sec", 5.0)

        samp_timing = _samp_timing_clock_or_ondemand(self.clock_mode)

        self.task = D.Task()

        for line, bound in zip(self.lines, self.bounds):
            self.task.CreateAOVoltageChan(line, "", bound[0], bound[1], D.DAQmx_Val_Volts, None)
        self.task.SetSampTimingType(samp_timing)
        if self.clock_mode:
            self.task.CfgSampClkTiming(
                clock,
                rate,
                clock_dir,
                _samples_finite_or_cont(self.finite),
                # a bit larger samples to assure buffer size
                samples + 1,
            )

        self.logger.debug(f"Configured. clock_mode: {self.clock_mode}")

        return True

    def set(self, key: str, value=None) -> bool:
        if key == "voltage":
            return self.set_output_once(value)
        else:
            return self.fail_with(f"unknown set() key: {key}")

    def get(self, key: str, args=None):
        if key == "onboard_buffer_size":
            return self.get_onboard_buffer_size()
        elif key == "buffer_size":
            return self.get_buffer_size()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class AnalogInTask(D.Task):
    def __init__(
        self,
        name: str,
        line_num: int,
        every: bool,
        oversample: int,
        drop_first: int,
        cb_samples: int,
        everyN_handler: Callable[[np.ndarray | list[np.ndarray]], None],
        every1_handler: Callable[[], None],
        done_handler: Callable[[int], None],
    ):
        D.Task.__init__(self)

        self._name = name
        self._line_num = line_num
        self._every = every
        self._oversample = oversample
        self._cb_samples = cb_samples

        self._everyN_handler = everyN_handler
        self._every1_handler = every1_handler
        self._done_handler = done_handler

        self._i = 0
        self._drop_left = abs(drop_first)
        if self._every:
            self._drop_samples = self._drop_left
        else:
            self._drop_samples = self._drop_left * self._cb_samples
        self._read = D.int32()

    def _read_samples(self, samples):
        data = np.zeros(samples, dtype=np.float64)
        self.ReadAnalogF64(
            samples,
            10.0,
            D.DAQmx_Val_GroupByChannel,
            data,
            len(data),
            D.byref(self._read),
            None,
        )
        if samples != self._read.value:
            raise InstError(
                self._name,
                "Fail to read requested number ({}) of samples! read: {}".format(
                    samples, self._read.value
                ),
            )
        return data

    def EveryNCallback(self) -> int:
        if self._drop_left > 0:
            self._drop_left -= 1
            if self._drop_left == 0:
                # just finished drop period, perform dummy read.
                data = self._read_samples(self._drop_samples)
            return 0  # should return 0 anyway

        if self._every:
            self._every1_handler()

            self._i += 1
            if self._i % self._cb_samples != 0:
                return 0  # should return 0 anyway

        data = self._read_samples(self._cb_samples)
        ret = []
        for i in range(self._line_num):
            # Slice data for i-th channel.
            d = data[i * self._cb_samples : (i + 1) * self._cb_samples]
            # Take mean for each oversample samples.
            # Note that self._cb_samples % self._oversample is always 0.
            ret.append(
                d.reshape(self._cb_samples // self._oversample, self._oversample).mean(axis=1)
            )

        if self._line_num == 1:
            self._everyN_handler(ret[0])
        else:
            self._everyN_handler(ret)

        return 0  # should return 0 anyway

    def DoneCallback(self, status) -> int:
        self._done_handler(status)

        return 0  # should return 0 anyway


class AnalogIn(ConfigurableTask):
    """A configurable DAQ Task class for Analog Input voltage channel.

    This class supports buffered clock-mode and on-demand-mode.
    In clock-mode, readings are synchronized to a clock, buffered, and read at each `cb_samples`.
    In on-demand-mode, data are read on get("data") request.

    :param lines: Sequence of strings to designate DAQ's physical channels.
    :type lines: list[str]

    :param clock_mode: If True (False), configures as clock-mode (on-demand-mode).
    :type clock_mode: bool
    :param bounds: bounds (min, max) values of the expected input voltages per channels.
    :type bounds: list[tuple[float, float]]
    :param finite: (default True) Switch if finite mode or infinite mode.
    :type finite: bool
    :param every: (default False) Switch if every1 mode or not.
    :type every: bool
    :param oversample: (default 1) cb_samples and samples are multiplied by `oversample`,
                       and everyN_handler is passed mean of `oversample` readings.
    :type oversample: int
    :param stamp: (default False) Attach timestamp for each samples.
    :type stamp: bool
    :param drop_first: (default 0) drop the data on first N callbacks.
    :type drop_first: int

    Finite or Infinite mode
    -----------------------

    In finite mode (finite=True), task is finished with `samples`.
    In inifinite mode (finite=False), task continues infinitely but it can be stopped by stop().
    (`samples` is used to determine the buffer size.)

    Callback handlers
    -----------------

    You'd pass three callback handler functions: every1_handler, everyN_handler, and done_handler.
    If every1 mode (every=True), EveryNCallBack is called with N=1 regardress of `cb_samples`.
    And every1_handler is called at each callbacks.
    EveryNCallBack aquires data and call everyN_handler at each `cb_samples` callbacks.
    If every1 mode is not active (every=False), EveryNCallBack is called
    after acquiring `cb_samples`, and everyN_handler is called for each callback.
    In this mode every1_handler is never called.

    """

    def __init__(self, name, conf=None, prefix=None):
        ConfigurableTask.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("lines",))
        self.lines = conf["lines"]

        self.buffer_size = self.conf.get("buffer_size", 10000)
        self.queue = LockedQueue(self.buffer_size)
        self._stamp = False

    def _null_every1_handler(self):
        pass

    def _null_done_handler(self, status: int):
        pass

    def get_buffer_size(self) -> int | None:
        if self.task is None:
            return None
        size = D.uInt32()
        self.task.GetBufInputBufSize(D.byref(size))
        return size.value

    def get_max_rate(self) -> float | None:
        rate = D.float64()
        if len(self.lines) == 1:
            D.GetDevAIMaxSingleChanRate(_device_name(self.lines[0]), D.byref(rate))
        else:
            D.GetDevAIMaxMultiChanRate(_device_name(self.lines[0]), D.byref(rate))
        return rate.value

    def _append_data(self, data: np.ndarray | list[np.ndarray]):
        if not self.queue.append((data, time.time_ns()) if self._stamp else data):
            self.logger.warn("queue is overflowing. The oldest data is discarded.")

    def pop_opt(
        self,
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float] | None:
        """Get data from buffer.

        :returns: np.ndarray if len(lines) == 1 and stamp == False.
                  list[np.ndarray] if len(lines) > 1 and stamp == False.
                  tuple[np.ndarray, float] if len(lines) == 1 and stamp == True.
                  tuple[list[np.ndarray], float] if len(lines) > 1 and stamp == True.
                  None if buffer is empty.

        """

        return self.queue.pop_opt()

    def pop_all_opt(
        self,
    ) -> list[np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]] | None:
        """Get all data from buffer as list.

        :returns: list[np.ndarray] if len(lines) == 1 and stamp == False.
                  list[list[np.ndarray]] if len(lines) > 1 and stamp == False.
                  list[tuple[np.ndarray, float]] if len(lines) == 1 and stamp == True.
                  list[tuple[list[np.ndarray], float]] if len(lines) > 1 and stamp == True.
                  None if buffer is empty.

        """

        return self.queue.pop_all_opt()

    def pop_block(
        self,
    ) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]:
        """Get data from buffer.

        If buffer is empty, this function blocks until data is ready.

        see pop_block() for return value types.

        """

        return self.queue.pop_block()

    def pop_all_block(
        self,
    ) -> list[np.ndarray | list[np.ndarray] | tuple[np.ndarray | list[np.ndarray], float]]:
        """Get all data from buffer as list.

        If buffer is empty, this function blocks until data is ready.

        see pop_all_opt() for return value types.

        """

        return self.queue.pop_all_block()

    def _get_bounds(self, params: dict) -> tuple[bool, list[tuple[float, float]]]:
        bounds = params.get("bounds", [(-10.0, 10.0)] * len(self.lines))
        if len(bounds) == 2 and isinstance(bounds[0], (float, np.floating, int, np.integer)):
            bounds = [bounds] * len(self.lines)
        if len(bounds) != len(self.lines):
            self.logger.error(f"len(bounds) is invalid: {len(bounds)} != {len(self.lines)}.")
            return False, []
        return True, bounds

    def configure_on_demand(self, params: dict) -> bool:
        success, bounds = self._get_bounds(params)
        if not success:
            return False

        self.finite = False
        self.task = D.Task()
        for line, bound in zip(self.lines, bounds):
            self.task.CreateAIVoltageChan(
                line, "", D.DAQmx_Val_RSE, bound[0], bound[1], D.DAQmx_Val_Volts, None
            )
        self.task.SetSampTimingType(_samp_timing_clock_or_ondemand(False))
        self.logger.debug("Configured OnDemand mode.")
        return True

    def configure_clock(self, params: dict) -> bool:
        if not self.check_required_params(params, ("clock", "cb_samples", "samples")):
            return False
        clock = params["clock"]
        cb_samples = params["cb_samples"]
        samples = params["samples"]

        rate = params.get("rate", 10000.0)
        clock_dir = _edge_polarity(params.get("clock_dir", True))
        every = params.get("every", False)
        oversample = params.get("oversample", 1)
        self._stamp = params.get("stamp", False)
        drop_first = params.get("drop_first", 0)
        self.finite = params.get("finite", True)

        cb_samples *= oversample
        samples *= oversample
        if every and samples % cb_samples != 0:
            self.logger.error("samples must be integer multiple of cb_samples.")
            return False

        success, bounds = self._get_bounds(params)
        if not success:
            return False

        self.queue = LockedQueue(self.buffer_size)

        every1_handler = params.get("every1_handler", self._null_every1_handler)
        done_handler = params.get("done_handler", self._null_done_handler)
        self.task = AnalogInTask(
            self.full_name(),
            len(self.lines),
            every,
            oversample,
            drop_first,
            cb_samples,
            self._append_data,
            every1_handler,
            done_handler,
        )

        for line, bound in zip(self.lines, bounds):
            self.task.CreateAIVoltageChan(
                line, "", D.DAQmx_Val_RSE, bound[0], bound[1], D.DAQmx_Val_Volts, None
            )
        self.task.SetSampTimingType(_samp_timing_clock_or_ondemand(True))

        self.task.CfgSampClkTiming(
            clock,
            rate,
            clock_dir,
            _samples_finite_or_cont(self.finite),
            # a bit larger samples to assure buffer size
            samples + 1,
        )

        self.logger.debug(f"Buffer size: {self.get_buffer_size()}")

        if every:
            self.task.AutoRegisterEveryNSamplesEvent(D.DAQmx_Val_Acquired_Into_Buffer, 1, 0)
        else:
            self.task.AutoRegisterEveryNSamplesEvent(
                D.DAQmx_Val_Acquired_Into_Buffer, cb_samples, 0
            )

        if "done_handler" in params:
            self.task.AutoRegisterDoneEvent(0)

        return True

    def read_on_demand(self, oversample: int = 1) -> float | np.ndarray:
        """Read out analog voltages on demand.

        if len(self.lines) is 1, the value is returned in float.
        Otherwise values are returned in ndarray (size: len(self.lines), type: float64).

        :param oversample: number of samples per channels, which is used for averaging.

        """

        line_num = len(self.lines)
        volt = np.zeros(oversample * line_num, dtype=np.float64)
        read_samps = D.int32()

        self.task.ReadAnalogF64(
            oversample,
            10.0,
            D.DAQmx_Val_GroupByChannel,
            volt,
            len(volt),
            D.byref(read_samps),
            None,
        )
        ret = [np.mean(volt[i * oversample : (i + 1) * oversample]) for i in range(line_num)]

        if line_num == 1:
            return ret[0]
        else:
            return ret

    # Standard API

    def configure(self, params: dict) -> bool:
        self.clock_mode = params.get("clock_mode", False)

        if self.clock_mode:
            return self.configure_clock(params)
        else:
            return self.configure_on_demand(params)

    def get(self, key: str, args=None):
        if key == "data":
            if not self.clock_mode:
                return self.read_on_demand()
            if args:
                return self.pop_block()
            else:
                return self.pop_opt()
        elif key == "all_data":
            if args:
                return self.pop_all_block()
            else:
                return self.pop_all_opt()
        elif key == "unit":
            return "V"
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class BufferedEdgeCounterTask(D.Task):
    def __init__(
        self,
        name: str,
        every: bool,
        diff: bool,
        gate: bool,
        drop_first: int,
        cb_samples: int,
        everyN_handler: Callable[[np.ndarray], None],
        every1_handler: Callable[[], None],
        done_handler: Callable[[int], None],
    ):
        D.Task.__init__(self)

        self._name = name
        self._every = every
        self._diff = diff
        self._gate = gate
        self._cb_samples = cb_samples

        self._everyN_handler = everyN_handler
        self._every1_handler = every1_handler
        self._done_handler = done_handler

        self._last_data = 0
        self._i = 0
        self._drop_left = abs(drop_first)
        if self._every:
            self._drop_samples = self._drop_left
        else:
            self._drop_samples = self._drop_left * self._cb_samples
        self._read = D.int32()

    def _read_samples(self, samples):
        data = np.zeros(samples, dtype=np.uint32)
        self.ReadCounterU32(samples, 10.0, data, len(data), D.byref(self._read), None)
        if samples != self._read.value:
            raise InstError(
                self._name,
                "Fail to read requested number ({}) of samples! read: {}".format(
                    samples, self._read.value
                ),
            )
        return data

    def EveryNCallback(self) -> int:
        if self._drop_left > 0:
            self._drop_left -= 1
            if self._drop_left == 0:
                # just finished drop period, perform dummy read.
                data = self._read_samples(self._drop_samples)
            return 0  # should return 0 anyway

        if self._every:
            self._every1_handler()

            self._i += 1
            if self._i % self._cb_samples != 0:
                return 0  # should return 0 anyway

        data = self._read_samples(self._cb_samples)

        if self._gate:
            ar = np.diff(data)[::2]
        elif self._diff:
            ar = np.concatenate(
                (np.array((data[0] - self._last_data,), dtype=np.uint32), np.diff(data))
            )
            self._last_data = data[-1]
        else:
            ar = data

        self._everyN_handler(ar)

        return 0  # should return 0 anyway

    def DoneCallback(self, status) -> int:
        self._done_handler(status)

        return 0  # should return 0 anyway


class BufferedEdgeCounter(ConfigurableTask):
    """A configurable DAQ Task class for edge counting.

    Counts are buffered and read at each `cb_samples`.

    :param counter: The device name for counter (like /Dev1/Ctr0).
    :type counter: str
    :param source: The pin name for counter source (like /Dev1/PFI0).
    :type source: str
    :param source_dir: (default: True) Source direction. True (False) for rising (falling) edge.
    :type source_dir: bool
    :param buffer_size: (default: 10000) Software buffer (queue) size.
    :type buffer_size: int

    :param finite: (default True) Switch if finite mode or infinite mode.
    :type finite: bool
    :param every: (default False) Switch if every1 mode or not.
    :type every: bool
    :param diff: (default True) If True, everyN_handler is passed differential data
                 (<count of n-th sample> - <count of (n-1)-th sample>).
    :type diff: bool
    :param gate: (default False) If True, cb_samples and samples are automatically
                 doubled and everyN_handler is passed gated data
                 (<count of (2n+1)-th sample> - <count of (2n)-th sample>).
    :type gate: bool
    :param stamp: (default False) Attach timestamp for each samples.
    :type stamp: bool
    :param drop_first: (default 0) drop the data on first N callbacks.
    :type drop_first: int

    Finite or Infinite mode
    -----------------------

    In finite mode (finite=True), task is finished with `samples`.
    In inifinite mode (finite=False), task continues infinitely but it can be stopped by stop().
    (`samples` is used to determine the buffer size.)

    In infinite mode, buffer size is automatically assigned by NI-DAQmx.
    On some device, we cannot call EveryNCallBack at arbitrary `cb_samples`.
    (For example, buffer size must be even integer multiple of cb_samples.)
    Setting every=True may resolve such a situation.

    Callback handlers
    -----------------

    You'd pass three callback handler functions: every1_handler, everyN_handler, and done_handler.
    If every1 mode (every=True), EveryNCallBack is called with N=1 regardress of `cb_samples`.
    And every1_handler is called at each callbacks.
    EveryNCallBack aquires data and call everyN_handler at each `cb_samples` callbacks.
    If every1 mode is not active (every=False), EveryNCallBack is called
    after acquiring `cb_samples`, and everyN_handler is called for each callback.
    In this mode every1_handler is never called.

    """

    def __init__(self, name, conf=None, prefix=None):
        ConfigurableTask.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("counter", "source"))
        self.counter = conf["counter"]
        self.source = conf["source"]
        self.source_dir = _edge_polarity(conf.get("source_dir", True))

        self.buffer_size = self.conf.get("buffer_size", 10000)
        self.queue = LockedQueue(self.buffer_size)
        self._stamp = False

    def _null_every1_handler(self):
        pass

    def _null_done_handler(self, status: int):
        pass

    def get_buffer_size(self) -> int | None:
        if self.task is None:
            return None
        size = D.uInt32()
        self.task.GetBufInputBufSize(D.byref(size))
        return size.value

    def get_max_rate(self) -> float | None:
        rate = D.float64()
        D.DAQmxGetDevCIMaxTimebase(_device_name(self.counter), D.byref(rate))
        return rate.value

    def _append_data(self, data: np.ndarray):
        if not self.queue.append((data, time.time_ns()) if self._stamp else data):
            self.logger.warn("queue is overflowing. The oldest data is discarded.")

    def pop_opt(self) -> np.ndarray | tuple[np.ndarray, float] | None:
        """Get data from buffer. If buffer is empty, returns None."""

        return self.queue.pop_opt()

    def pop_all_opt(self) -> list[np.ndarray | tuple[np.ndarray, float]] | None:
        """Get all data from buffer as list. If buffer is empty, returns None."""

        return self.queue.pop_all_opt()

    def pop_block(self) -> np.ndarray | tuple[np.ndarray, float]:
        """Get data from buffer. If buffer is empty, this function blocks until data is ready."""

        return self.queue.pop_block()

    def pop_all_block(self) -> list[np.ndarray | tuple[np.ndarray, float]]:
        """Get all data from buffer as list.

        If buffer is empty, this function blocks until data is ready.

        """

        return self.queue.pop_all_block()

    # Standard API

    def configure(self, params: dict) -> bool:
        if not self.check_required_params(params, ("clock", "cb_samples", "samples")):
            return False
        clock = params["clock"]
        cb_samples = params["cb_samples"]
        samples = params["samples"]
        clock_dir = _edge_polarity(params.get("clock_dir", True))

        self.finite = params.get("finite", True)
        every = params.get("every", False)
        diff = params.get("diff", True)
        gate = params.get("gate", False)
        self._stamp = params.get("stamp", False)
        rate = params.get("rate", 10000.0)
        drop_first = params.get("drop_first", 0)

        if gate:
            cb_samples *= 2
            samples *= 2
        if every and samples % cb_samples != 0:
            self.logger.error("samples must be integer multiple of cb_samples.")
            return False
        if diff and cb_samples <= 1:
            self.logger.error("samples must be greater than 1 in diff mode.")
            return False

        self.queue = LockedQueue(self.buffer_size)

        every1_handler = params.get("every1_handler", self._null_every1_handler)
        done_handler = params.get("done_handler", self._null_done_handler)
        self.task = BufferedEdgeCounterTask(
            self.full_name(),
            every,
            diff,
            gate,
            drop_first,
            cb_samples,
            self._append_data,
            every1_handler,
            done_handler,
        )

        self.task.CreateCICountEdgesChan(self.counter, "", self.source_dir, 0, D.DAQmx_Val_CountUp)
        self.task.SetCICountEdgesTerm(self.counter, self.source)

        self.task.CfgSampClkTiming(
            clock,
            rate,
            clock_dir,
            _samples_finite_or_cont(self.finite),
            # a bit larger samples to assure buffer size
            samples + 1,
        )

        self.logger.debug(f"Buffer size: {self.get_buffer_size()}")

        if every:
            self.task.AutoRegisterEveryNSamplesEvent(D.DAQmx_Val_Acquired_Into_Buffer, 1, 0)
        else:
            self.task.AutoRegisterEveryNSamplesEvent(
                D.DAQmx_Val_Acquired_Into_Buffer, cb_samples, 0
            )

        if "done_handler" in params:
            self.task.AutoRegisterDoneEvent(0)

        return True

    def get(self, key: str, args=None):
        if key == "data":
            if args:
                return self.pop_block()
            else:
                return self.pop_opt()
        elif key == "all_data":
            if args:
                return self.pop_all_block()
            else:
                return self.pop_all_opt()
        elif key == "unit":
            return ""
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class SingleShotTask(Instrument):
    """Base class for simple DAQ Task which won't need task stop/restart."""

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf=conf, prefix=prefix)
        self.running = False
        self.task = D.Task()

    # Standard API

    def close(self):
        self.stop()

    def start(self) -> bool:
        if self.running:
            return True
        self.running = True

        self.task.StartTask()
        return True

    def stop(self) -> bool:
        if not self.running:
            return True
        self.running = False

        self.task.StopTask()
        self.task.ClearTask()
        self.logger.debug("Stopped and Cleared Task.")
        return True


class DigitalOut(SingleShotTask):
    """A DAQ Task class for a Digital Output channels.

    :param lines: Sequence of strings to designate DAQ's physical channels.
    :type lines: list[str]
    :param auto_start: If True, StartTask() is called in the constructor.
    :type auto_start: bool
    :param command: dict of command name to predefined output data.
    :type command: dict[str, list[bool | int]]

    """

    def __init__(self, name, conf, prefix=None):
        SingleShotTask.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("lines",))
        self.lines = self.conf["lines"]
        self.timeout_sec = self.conf.get("timeout_sec", 10.0)

        self.data_low = np.zeros(len(self.lines), dtype=np.uint8)
        self.data_high = np.ones(len(self.lines), dtype=np.uint8)

        self._commands = {}
        if "command" in self.conf:
            for cmd_name, data in self.conf["command"].items():
                d = self._convert(data)
                if d is not None:
                    self._commands[cmd_name] = d

        for line in self.lines:
            self.task.CreateDOChan(line, "", D.DAQmx_Val_ChanForAllLines)

        if self.conf.get("auto_start", True):
            self.start()
            self.set_output_low()

    def _convert(self, data) -> Optional[np.ndarray]:
        if isinstance(data, (list, tuple, np.ndarray)):
            if len(data) != len(self.lines):
                msg = f"Data length mismatch. Given {len(data)} but needs {len(self.lines)}"
                self.logger.error(msg)
                return None
            return np.array([int(bool(v)) for v in data], dtype=np.uint8)
        elif data:
            return self.data_high
        else:
            return self.data_low

    def _write_once(self, data) -> bool:
        samples_per_ch = 1
        written = D.int32()
        self.task.WriteDigitalLines(
            samples_per_ch,
            1,
            self.timeout_sec,
            D.DAQmx_Val_GroupByChannel,
            data,
            D.byref(written),
            None,
        )
        if written.value != samples_per_ch:
            self.logger.error(
                "Failed to write all DO samples: {} out of {} per channels".format(
                    written.value, samples_per_ch
                )
            )
            return False
        return True

    def set_output(self, data) -> bool:
        d = self._convert(data)
        if d is None:
            return False
        return self._write_once(d)

    def set_command(self, name: str) -> bool:
        if name not in self._commands:
            self.logger.error(f"No such command: {name}.")
            return False
        return self._write_once(self._commands[name])

    def set_output_low(self) -> bool:
        return self._write_once(self.data_low)

    def set_output_high(self) -> bool:
        return self._write_once(self.data_high)

    def set_output_pulse(self) -> bool:
        return self.set_output_high() and self.set_output_low()

    def set_output_pulse_neg(self) -> bool:
        return self.set_output_low() and self.set_output_high()

    # Standard API

    def set(self, key: str, value=None) -> bool:
        if key.startswith("out"):
            return self.set_output(value)
        elif key == "command":
            return self.set_command(value)
        elif key == "low":
            return self.set_output_low()
        elif key == "high":
            return self.set_output_high()
        elif key == "pulse":
            return self.set_output_pulse()
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False


### Deprecated Classes below ###


class DOPulser(SingleShotTask):
    """A DAQ Task class for a Digital Output channel, especially for pulsing.

    :param line: str to designate DAQ's physical channels.
    :type line: str
    :param auto_start: If True, StartTask() is called in the constructor.
    :type auto_start: bool

    """

    def __init__(self, name, conf, prefix=None):
        SingleShotTask.__init__(self, name, conf=conf, prefix=prefix)

        self.data0 = np.zeros(1, dtype=np.uint8)
        self.data1 = np.ones(1, dtype=np.uint8)

        line = self.conf.get("line")
        if line is None:
            raise ValueError("line is not given.")

        self.task.CreateDOChan(line, "", D.DAQmx_Val_ChanForAllLines)

        if self.conf.get("auto_start", True):
            self.start()
            self.set_output_low()

    def set_output_low(self) -> bool:
        self.task.WriteDigitalLines(1, 1, 10.0, D.DAQmx_Val_GroupByChannel, self.data0, None, None)
        return True

    def set_output_high(self) -> bool:
        self.task.WriteDigitalLines(1, 1, 10.0, D.DAQmx_Val_GroupByChannel, self.data1, None, None)
        return True

    def set_output_pulse(self) -> bool:
        self.task.WriteDigitalLines(1, 1, 10.0, D.DAQmx_Val_GroupByChannel, self.data1, None, None)
        self.task.WriteDigitalLines(1, 1, 10.0, D.DAQmx_Val_GroupByChannel, self.data0, None, None)
        return True

    # Standard API

    def set(self, key: str, value=None) -> bool:
        if key == "low":
            return self.set_output_low()
        elif key == "high":
            return self.set_output_high()
        elif key == "pulse":
            return self.set_output_pulse()
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False


class DIPoller(SingleShotTask):
    """A DAQ Task class for Digital Input channel(s), especially for polling input(s).

    :param line: str to designate DAQ's physical channels.
    :type line: str
    :param line_num: Number of channel lines.
    :type line_num: int
    :param sample_per_ch: Sample number for each channel.
    :type sample_per_ch: int
    :param polarity: Set False if lines are active-low (you want to wait for low level).
        DIPoller.wait_all will be alias of wait_all_pos (wait_all_neg) if polarity is True (False).
    :type polarity: bool
    :param auto_start: If True, StartTask() is called in the constructor.
    :type auto_start: bool

    """

    def __init__(self, name, conf, prefix=None):
        SingleShotTask.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("line", "line_num"))
        line = self.conf["line"]
        self.line_num = self.conf["line_num"]

        self.polarity = self.conf.get("polarity", True)
        self.sample_per_ch = self.conf.get("sample_per_ch", 10000)

        self.dataIn = np.zeros(self.line_num * self.sample_per_ch, dtype=np.uint8)
        self.readIn, self.bytesPerSamp = D.int32(), D.int32()

        self.task.CreateDIChan(line, "", D.DAQmx_Val_ChanForAllLines)

        if self.polarity:
            self.wait_all = self.wait_all_pos
        else:
            self.wait_all = self.wait_all_neg

        if self.conf.get("auto_start", True):
            self.start()

    def set_sample_per_ch(self, sample_per_ch) -> bool:
        self.sample_per_ch = sample_per_ch
        self.dataIn = np.zeros(self.line_num * self.sample_per_ch, dtype=np.uint8)
        return True

    def wait_all_pos(self):
        for i in range(100000):
            self.task.ReadDigitalLines(
                self.sample_per_ch,
                10.0,
                D.DAQmx_Val_GroupByChannel,
                self.dataIn,
                len(self.dataIn),
                D.byref(self.readIn),
                D.byref(self.bytesPerSamp),
                None,
            )
            if all(self.dataIn[: self.readIn.value * self.line_num]):
                return i
        return -1

    def wait_all_neg(self):
        for i in range(100000):
            self.task.ReadDigitalLines(
                self.sample_per_ch,
                10.0,
                D.DAQmx_Val_GroupByChannel,
                self.dataIn,
                len(self.dataIn),
                D.byref(self.readIn),
                D.byref(self.bytesPerSamp),
                None,
            )
            if not any(self.dataIn[: self.readIn.value * self.line_num]):
                return i
        return -1

    # Standard API

    def set(self, key: str, value=None) -> bool:
        if key == "sample":
            return self.set_sample_per_ch(value)
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key == "pos":
            return self.wait_all_pos()
        elif key == "neg":
            return self.wait_all_neg()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class DICounter(SingleShotTask):
    """A DAQ Task class for Counter Input channel(s), especially for couting slower signal."""

    def __init__(self, name, conf, prefix=None):
        SingleShotTask.__init__(self, name, conf=conf, prefix=prefix)

        line = self.conf.get("line")
        if line is None:
            raise ValueError("line is not given.")
        count_dir = _edge_polarity(self.conf.get("rising", True))

        self._count = 0
        self.task.CreateCICountEdgesChan(line, "", count_dir, 0, D.DAQmx_Val_CountUp)

        if self.conf.get("auto_start", True):
            self.start()

    def get_data(self, timeout=1):
        data = D.c_uint32(0)
        self.task.ReadCounterScalarU32(timeout, D.byref(data), None)
        data = data.value
        self._count += data
        return data

    # Standard API

    def get(self, key: str, args=None):
        if key == "data":
            return self.get_data()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None


class AOPutter(SingleShotTask):
    """[DEPRECATED] Use AnalogOut instead."""

    def __init__(self, name, conf, prefix=None):
        SingleShotTask.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("line", "line_num", "min_val", "max_val"))
        line = self.conf["line"]
        self.line_num = self.conf["line_num"]
        self.min_val = self.conf["min_val"]
        self.max_val = self.conf["max_val"]
        self.bounds = (self.min_val, self.max_val)

        self.task.CreateAOVoltageChan(
            line, "", self.min_val, self.max_val, D.DAQmx_Val_Volts, None
        )

        if self.conf.get("auto_start", True):
            self.start()

    def clip(self, v):
        """Clip the voltage value if it exceeds max or min bounds."""

        if v > self.max_val:
            v = self.max_val
        if v < self.min_val:
            v = self.min_val

        return v

    def set_output(self, volt) -> bool:
        """Output analog voltages. Volt can be int (for 1 channel), tuple, list or ndarray.
        Voltages are clipped automatically.

        """

        if isinstance(volt, (int, float)):
            if self.line_num != 1:
                self.logger.error("Scalar output value is passed for multiple channels.")
                return False
            volt = self.clip(volt)
        elif isinstance(volt, (tuple, list, np.ndarray)):
            if self.line_num != len(volt):
                self.logger.error("Number of output values is different from number of channels.")
                return False
            volt = [self.clip(v) for v in volt]
        else:
            self.logger.error("Voltage must be int, tuple, list or ndarray.")
            return False

        self.task.WriteAnalogF64(
            1, 1, 10.0, D.DAQmx_Val_GroupByChannel, np.array(volt, dtype=np.float64), None, None
        )

        return True

    # Standard API

    def set(self, key: str, value=None) -> bool:
        if key == "voltage":
            return self.set_output(value)
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False


class AIReader(SingleShotTask):
    """[DEPRECATED] Use AnalogIn instead."""

    def __init__(self, name, conf, prefix=None):
        SingleShotTask.__init__(self, name, conf=conf, prefix=prefix)

        self.check_required_conf(("line", "line_num", "min_val", "max_val"))
        line = self.conf["line"]
        self.line_num = self.conf["line_num"]
        self.min_val = self.conf["min_val"]
        self.max_val = self.conf["max_val"]
        self.bounds = (self.min_val, self.max_val)

        self.task.CreateAIVoltageChan(
            line, "", D.DAQmx_Val_RSE, self.min_val, self.max_val, D.DAQmx_Val_Volts, None
        )

        if self.conf.get("auto_start", True):
            self.start()

    def get_data(self, average: int = 10):
        """Read out analog voltages.

        if line_num is 1, the value is returned in float.
        Otherwise values are returned in ndarray (size: line_num, type: float64).

        Average specifies number of samples per channels, which is used for averaging.

        """

        volt = np.zeros(average * self.line_num, dtype=np.float64)
        read_samps = D.int32()

        self.task.ReadAnalogF64(
            average, 1, D.DAQmx_Val_GroupByChannel, volt, len(volt), D.byref(read_samps), None
        )
        ret = [np.average(volt[i * average : (i + 1) * average]) for i in range(self.line_num)]

        if self.line_num == 1:
            return ret[0]
        else:
            return ret

    # Standard API

    def get(self, key: str, args=None):
        if key == "data":
            return self.get_data()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
