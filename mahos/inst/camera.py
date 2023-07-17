#!/usr/bin/env python3

"""
Camera module.

.. This file is a part of MAHOS project.

"""

import os
import typing as T
import threading
import time
import enum

import numpy as np
from pypylon import pylon

from .instrument import Instrument
from ..util.locked_queue import LockedQueue
from ..msgs.inst_camera_msgs import FrameResult


class ThorlabsCamera(Instrument):
    """Wrapper for Thorlabs Scientific Camera SDK.

    You need following:
    1. Install SDK's Python wrapper library (thorlabs_tsi_sdk).
    2. Place SDK native DLLs somewhere.

    :param dll_path: (Required) The path to the directory containing SDK DLLs.

    """

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(("dll_path",))
        p = os.path.expanduser(self.conf["dll_path"])
        os.environ["PATH"] += os.pathsep + p
        os.add_dll_directory(p)

        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

        self.sdk = TLCameraSDK()
        cameras = self.sdk.discover_available_cameras()
        if not cameras:
            self.logger.error("No camera detected.")
            raise ValueError("No camera detected.")

        # TODO: multiple cameras
        self.camera = self.sdk.open_camera(cameras[0])
        self._mode = None
        self._queue_size = self.conf.get("queue_size", 8)
        self._queue = LockedQueue(self._queue_size)
        self.running = False

    def close(self):
        if hasattr(self, "camera"):
            self.camera.dispose()
        if hasattr(self, "sdk"):
            self.sdk.dispose()

    def configure_continuous(self, exposure_time_sec: float) -> bool:
        self.camera.exposure_time_us = int(round(exposure_time_sec * 1e6))
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.image_poll_timeout_ms = 0
        self._mode = "continuous"
        self._queue = LockedQueue(self._queue_size)

        self.logger.info("Configured for continuous capture.")
        return True

    def poll_continuous(self, ev: threading.Event):
        while not ev.is_set():
            frame = self.camera.get_pending_frame_or_null()
            if frame is not None:
                self._queue.append(
                    FrameResult(
                        frame=frame.image_buffer, count=frame.frame_count, time=time.time()
                    )
                )

    def get_frame(self) -> FrameResult:
        if self._queue is None:
            return FrameResult(frame=None)
        return self._queue.pop_block()

    # Standard API

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
        if self.running:
            self.logger.warn("start() is called while running.")
            return True

        if self._mode is None:
            self.logger.error("Must be configured before start().")
            return False

        if self._mode == "continuous":
            self.camera.arm(2)
            self.camera.issue_software_trigger()
            self.poll_stop_ev = threading.Event()
            self.poll_thread = threading.Thread(
                target=self.poll_continuous, args=(self.poll_stop_ev,)
            )
            self.poll_thread.start()

            self.running = True

            return True
        else:
            return False

    def stop(self) -> bool:
        if not self.running:
            return True
        self.running = False

        if self._mode == "continuous":
            self.poll_stop_ev.set()
            self.poll_thread.join()
            self.camera.disarm()
            self._mode = None
            return True
        else:
            return False

    def get(self, key: str, args=None):
        if key == "frame":
            return self.get_frame()
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None


class BaslerPylonCamera(Instrument):
    """Wrapper for Basler Pylon."""

    class Mode(enum.Enum):
        UNCONFIGURED = 0
        CONTINUOUS = 1
        SOFT_TRIGGER = 2
        HARD_TRIGGER = 3

    def __init__(self, name, conf=None, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self._emulation = self.conf.get("emulation", False)
        if self._emulation:
            os.environ["PYLON_CAMEMU"] = "1"

        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if not devices:
            self.logger.error("No camera detected.")
            raise ValueError("No camera detected.")

        self._trigger_source = self.conf.get("trigger_source", "Line1")
        self._trigger_wait_line = self.conf.get("trigger_wait_line", "Line2")
        self._trigger_wait_invert = bool(self.conf.get("trigger_wait_invert", False))
        self._pixel_format = self.conf.get("pixel_format", "")

        # TODO: multiple cameras
        self.camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
        self._mode = self.Mode.UNCONFIGURED
        self._running = False
        self._queue_size = self.conf.get("queue_size", 8)
        self._queue = LockedQueue(self._queue_size)
        self._frame_count = 0
        self._burst_num = 1

        self.logger.info("Opened Camera: " + self.camera.GetDeviceInfo().GetModelName())

    def close(self):
        if hasattr(self, "camera"):
            self.camera.Close()

    def load_default(self):
        was_closed = not self.camera.IsOpen()
        if was_closed:
            self.camera.Open()
        self.camera.UserSetSelector = "Default"
        self.camera.UserSetLoad.Execute()
        if was_closed:
            self.camera.Close()
        self.logger.info("Loaded 'Default' Preset.")

    def reset(self) -> bool:
        self.load_default()
        return True

    def hasattr(self, name) -> bool:
        try:
            return hasattr(self.camera, name)
        except pylon.LogicalErrorException:
            return False

    def set_pixel_format(self, pixel_format: str = "") -> bool:
        fmt = pixel_format or self._pixel_format
        if not fmt:
            self.logger.warn("Skipping to set pixel format.")
            return True
        try:
            self.camera.PixelFormat = fmt
            return True
        except (pylon.LogicalErrorException, pylon.AccessException):
            self.logger.exception("Could not set pixel format.")
            return False

    def set_exposure_time(self, exposure_time_sec: float) -> bool:
        # unit is us
        if self.hasattr("ExposureTime"):
            # Cameras based on SFNC 2.0 or later, e.g., USB cameras
            self.camera.ExposureTime = exposure_time_sec * 1e6
            return True
        elif self.hasattr("ExposureTimeAbs"):
            self.camera.ExposureTimeAbs = exposure_time_sec * 1e6
            return True
        else:
            self.logger.error("Cannot set ExposureTime.")
            return False

    def set_frame_rate(self, frame_rate_Hz: T.Optional[float]) -> bool:
        if frame_rate_Hz is None:
            self.camera.AcquisitionFrameRateEnable = False
            self.logger.info("FrameRate limit disabled.")
            return True
        self.camera.AcquisitionFrameRateEnable = True
        if self.hasattr("AcquisitionFrameRate"):
            self.camera.AcquisitionFrameRate = frame_rate_Hz
            self.logger.info(f"FrameRate limit {frame_rate_Hz:.2f} Hz")
            return True
        elif self.hasattr("AcquisitionFrameRateAbs"):
            self.camera.AcquisitionFrameRateAbs = frame_rate_Hz
            self.logger.info(f"FrameRate limit {frame_rate_Hz:.2f} Hz")
            return True
        else:
            self.logger.error("Cannot set FrameRate limit.")
            return False

    def set_binning(self, binning: int) -> bool:
        if self._emulation:
            self.logger.warn("No binning option for emulation.")
            return True

        try:
            self.camera.BinningHorizontal = binning
            self.camera.BinningVertical = binning
            # binning mode to average
            self.camera.BinningHorizontalMode.FromString("Average")
            self.camera.BinningVerticalMode.FromString("Average")
            return True
        except (pylon.LogicalErrorException, pylon.AccessException):
            self.logger.exception("Could not set binning.")
            return False

    def set_roi(self, roi: dict) -> bool:
        # ignore roi with any disabled value
        if any([roi.get(k) is None for k in ("width", "height", "woffset", "hoffset")]):
            return True

        try:
            self.camera.Width = roi["width"]
            self.camera.Height = roi["height"]
            self.camera.OffsetX = roi["woffset"]
            self.camera.OffsetY = roi["hoffset"]
            return True
        except Exception:
            self.logger.exception("Could not set roi.")
            return False

    def set_frame_start_trigger(self, on: bool) -> bool:
        if self._emulation:
            self.logger.warn("no FrameStart trigger in emulation")
            return True

        triggers = self.camera.TriggerSelector.GetSymbolics()
        if "FrameStart" in triggers:
            self.camera.TriggerSelector.SetValue("FrameStart")
            self.camera.TriggerMode.SetValue("On" if on else "Off")
            return True
        else:
            return self.fail_with("FrameStart trigger is not available.")

    def set_frame_burst_trigger(self, on: bool) -> bool:
        triggers = self.camera.TriggerSelector.GetSymbolics()
        for key in ("FrameBurstStart", "AcquisitionStart"):
            if key in triggers:
                self.camera.TriggerSelector.SetValue(key)
                self.camera.TriggerMode.SetValue("On" if on else "Off")
                return True
        else:
            return self.fail_with("FrameBurstStart / AcquisitionStart trigger is not available.")

    def set_burst_frames(self, num: int) -> bool:
        if self.hasattr("AcquisitionBurstFrameCount"):
            self.camera.AcquisitionBurstFrameCount = num
            return True
        elif self.hasattr("AcquisitionFrameCount"):
            self.camera.AcquisitionFrameCount = num
            return True
        else:
            return False

    def _fail_with_close(self, msg: str = "") -> bool:
        if msg:
            self.logger.error(msg)
        self.camera.Close()
        return False

    def configure_continuous(
        self,
        exposure_time_sec: float,
        frame_rate_Hz: T.Optional[float] = None,
        binning: int = 0,
        roi: T.Optional[dict] = None,
    ) -> bool:
        self.load_default()
        self.camera.RegisterConfiguration(
            pylon.AcquireContinuousConfiguration(),
            pylon.RegistrationMode_ReplaceAll,
            pylon.Cleanup_Delete,
        )
        # RegisterConfiguration must be called before Open()
        self.camera.Open()

        if not self.set_pixel_format():
            return self._fail_with_close()
        if not self.set_exposure_time(exposure_time_sec):
            return self._fail_with_close()
        if not self.set_frame_rate(frame_rate_Hz):
            return self._fail_with_close()
        if binning > 0 and not self.set_binning(binning):
            return self._fail_with_close()
        if roi is not None and not self.set_roi(roi):
            return self._fail_with_close()

        self._mode = self.Mode.CONTINUOUS
        self._queue = LockedQueue(self._queue_size)
        self._frame_count = 0
        self._burst_num = 1

        self.logger.info("Configured for continuous capture.")
        self.logger.debug(
            f"TriggerSource: {self.camera.TriggerSource.Value}"
            + f" TriggerMode: {self.camera.TriggerMode.Value}"
        )

        if self.hasattr("ResultingFrameRate"):
            fps = self.camera.ResultingFrameRate.Value
        elif self.hasattr("ResultingFrameRateAbs"):
            fps = self.camera.ResultingFrameRateAbs.Value
        else:
            fps = 0.0
        self.logger.info(f"Expected max fps: {fps:.2f}")
        return True

    def configure_soft_trigger(
        self,
        exposure_time_sec: float,
        burst_num: int = 1,
        binning: int = 0,
        roi: T.Optional[dict] = None,
    ) -> bool:
        self.load_default()
        # self.camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(),
        #                                   pylon.RegistrationMode_ReplaceAll,
        #                                   pylon.Cleanup_Delete)
        # RegisterConfiguration must be called before Open()
        self.camera.Open()

        try:
            if self._emulation:
                # burst is not available in emulation,
                # but we must use set_frame_burst_trigger here.
                if burst_num != 1:
                    return self._fail_with_close("Cannot set burst (burst_num > 1) in emulation.")
                if not self.set_frame_burst_trigger(True):
                    return self._fail_with_close()
            else:
                if burst_num == 1:
                    if not (
                        self.set_frame_burst_trigger(False) and self.set_frame_start_trigger(True)
                    ):
                        return self._fail_with_close()
                else:
                    if not (
                        self.set_frame_start_trigger(False)
                        and self.set_frame_burst_trigger(True)
                        and self.set_burst_frames(burst_num)
                    ):
                        return self._fail_with_close()
            self.camera.TriggerSource.SetValue("Software")
        except Exception:
            self.logger.exception("Failed to configure software-trigger.")
            return self._fail_with_close()

        if not self.set_pixel_format():
            return self._fail_with_close()
        if not self.set_exposure_time(exposure_time_sec):
            return self._fail_with_close()
        if binning > 0 and not self.set_binning(binning):
            return self._fail_with_close()
        if roi is not None and not self.set_roi(roi):
            return self._fail_with_close()

        self._mode = self.Mode.SOFT_TRIGGER
        self._burst_num = burst_num
        self.logger.info("Configured for software-triggered capture.")
        self.logger.debug(
            f"TriggerSource: {self.camera.TriggerSource.Value}"
            + f" TriggerMode: {self.camera.TriggerMode.Value}"
        )

        return True

    def configure_hard_trigger(
        self,
        exposure_time_sec: float,
        burst_num: int = 1,
        binning: int = 0,
        roi: T.Optional[dict] = None,
        trigger_source: T.Optional[str] = None,
        trigger_positive: bool = True,
        trigger_wait_line: T.Optional[str] = None,
        trigger_wait_invert: T.Optional[bool] = None,
    ) -> bool:
        self.load_default()
        self.camera.Open()

        try:
            if burst_num == 1:
                if not (
                    self.set_frame_burst_trigger(False) and self.set_frame_start_trigger(True)
                ):
                    return self._fail_with_close()
            else:
                if not (
                    self.set_frame_start_trigger(False)
                    and self.set_frame_burst_trigger(True)
                    and self.set_burst_frames(burst_num)
                ):
                    return self._fail_with_close()
            self.camera.TriggerSource.SetValue(trigger_source or self._trigger_source)
            self.camera.TriggerActivation.SetValue(
                "RisingEdge" if trigger_positive else "FallingEdge"
            )

            self.camera.LineSelector.SetValue(trigger_wait_line or self._trigger_wait_line)
            self.camera.LineInverter.SetValue(
                self._trigger_wait_invert if trigger_wait_invert is None else trigger_wait_invert
            )
            if burst_num == 1:
                self.camera.LineSource.SetValue("FrameTriggerWait")
            else:
                self.camera.LineSource.SetValue("FrameBurstTriggerWait")
        except Exception:
            self.logger.exception("Failed to configure hardware-trigger.")
            return self._fail_with_close()

        if not self.set_pixel_format():
            return self._fail_with_close()
        if not self.set_exposure_time(exposure_time_sec):
            return self._fail_with_close()
        if binning > 0 and not self.set_binning(binning):
            return self._fail_with_close()
        if roi is not None and not self.set_roi(roi):
            return self._fail_with_close()

        self._mode = self.Mode.HARD_TRIGGER
        self._queue = LockedQueue(self._queue_size)
        self._frame_count = 0
        self._burst_num = burst_num
        self.logger.info("Configured for hardware-triggered capture.")
        self.logger.debug(
            f"TriggerSource: {self.camera.TriggerSource.Value}"
            + f" TriggerMode: {self.camera.TriggerMode.Value}"
        )

        return True

    def poll_loop_continuous(self, ev: threading.Event):
        while not ev.is_set() and self.camera.IsGrabbing():
            try:
                res: pylon.GrabResult = self.camera.RetrieveResult(
                    2000, pylon.TimeoutHandling_ThrowException
                )
                with res:
                    if res.GrabSucceeded():
                        self._queue.append(
                            FrameResult(frame=res.Array, count=self._frame_count, time=time.time())
                        )
                        self._frame_count += 1
                    else:
                        self.logger.error("GrabSucceeded() returned False.")
            except pylon.TimeoutException:
                self.logger.error("RetrieveResult() timed out.")

    def poll_loop_hard_trig(self, ev: threading.Event):
        while not ev.is_set() and self.camera.IsGrabbing():
            frames = []
            while len(frames) < self._burst_num:
                if ev.is_set():  # avoid infinite loop on failiure case.
                    return
                try:
                    res: pylon.GrabResult = self.camera.RetrieveResult(
                        1000, pylon.TimeoutHandling_ThrowException
                    )
                    with res:
                        if res.GrabSucceeded():
                            frames.append(res.Array)
                        else:
                            self.logger.error("GrabSucceeded() returned False.")
                except pylon.TimeoutException:
                    self.logger.warn("RetrieveResult() timed out.")
            self._queue.append(
                FrameResult(frame=self._process_burst_frames(frames), time=time.time())
            )

    def _process_burst_frames(self, frames: T.List[np.array]) -> np.array:
        if len(frames) == 1:
            return frames[0]
        return np.array(frames).mean(
            axis=0
        )  # TODO: types? this will be float64. option for float32?

    def get_frame_soft_trig_imm(self) -> T.Optional[np.ndarray]:
        """Issue soft ware trigger and get grab result immediately."""

        if self._burst_num == 1:
            # Because there is no equivalent function for FrameBurstTrigger
            # (something like WaitForFrameBurstTriggerReady),
            # we have to skip the wait for burst mode.
            try:
                self.camera.WaitForFrameTriggerReady(2000, pylon.TimeoutHandling_ThrowException)
            except pylon.TimeoutException:
                self.logger.error("WaitForFrameTriggerReady() timed out.")
                return None
        self.camera.ExecuteSoftwareTrigger()

        try:
            frames = []
            for i in range(self._burst_num):
                res: pylon.GrabResult = self.camera.RetrieveResult(
                    2000, pylon.TimeoutHandling_ThrowException
                )
                with res:
                    if res.GrabSucceeded():
                        frames.append(res.Array)
                    else:
                        self.logger.error("GrabSucceeded() returned False.")
                        return None
            return self._process_burst_frames(frames)
        except pylon.TimeoutException:
            self.logger.error("RetrieveResult() timed out.")
            return None

    def get_frame(self, timeout_sec: T.Optional[float] = None) -> FrameResult:
        if not self._running:
            return self.fail_with("get_frame() is called but not running.")

        if self._mode == self.Mode.CONTINUOUS:
            if self._queue is None:
                return FrameResult(frame=None)
            return self._queue.pop_block(timeout_sec=timeout_sec)
        elif self._mode == self.Mode.HARD_TRIGGER:
            if self._queue is None:
                return FrameResult(frame=None)
            # TODO: more error to check (camera's skipped image count?).
            full = self._queue.is_full()
            r: FrameResult = self._queue.pop_block(timeout_sec=timeout_sec)
            r.invalid = full
            return r
        elif self._mode == self.Mode.SOFT_TRIGGER:
            return FrameResult(frame=self.get_frame_soft_trig_imm())
        else:
            return self.fail_with("get_frame() is called but not configured correctly.")

    # Standard API

    def configure(self, params: dict) -> bool:
        if not self.check_required_params(params, ("mode",)):
            return False

        mode = params["mode"].lower()
        if mode == "continuous":
            if not self.check_required_params(params, ("exposure_time",)):
                return False
            return self.configure_continuous(
                params["exposure_time"],
                frame_rate_Hz=params.get("frame_rate", 0),
                binning=params.get("binning", 0),
                roi=params.get("roi"),
            )
        elif mode == "soft_trigger":
            if not self.check_required_params(params, ("exposure_time",)):
                return False
            return self.configure_soft_trigger(
                params["exposure_time"],
                binning=params.get("binning", 0),
                roi=params.get("roi"),
            )
        elif mode == "hard_trigger":
            if not self.check_required_params(params, ("exposure_time",)):
                return False
            return self.configure_hard_trigger(
                params["exposure_time"],
                binning=params.get("binning", 0),
                roi=params.get("roi"),
            )
        else:
            return self.fail_with(f"Unknown mode {mode}.")

    def start(self) -> bool:
        if self._running:
            self.logger.warn("start() is called while running.")
            return True

        if self._mode == self.Mode.CONTINUOUS:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.poll_stop_ev = threading.Event()
            self.poll_thread = threading.Thread(
                target=self.poll_loop_continuous, args=(self.poll_stop_ev,)
            )
            self.poll_thread.start()
            self._running = True
            return True
        elif self._mode == self.Mode.SOFT_TRIGGER:
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self._running = True
            return True
        elif self._mode == self.Mode.HARD_TRIGGER:
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.poll_stop_ev = threading.Event()
            self.poll_thread = threading.Thread(
                target=self.poll_loop_hard_trig, args=(self.poll_stop_ev,)
            )
            self.poll_thread.start()
            self._running = True
            return True
        else:  # self.Mode.UNCONFIGURED
            return self.fail_with("Must be configured before start().")

    def stop(self) -> bool:
        if not self._running:
            return True

        if self._mode in (self.Mode.CONTINUOUS, self.Mode.HARD_TRIGGER):
            self.poll_stop_ev.set()
            self.poll_thread.join()
            self.camera.StopGrabbing()
            self.camera.Close()
            self._running = False
            return True
        elif self._mode == self.Mode.SOFT_TRIGGER:
            self.camera.StopGrabbing()
            self.camera.Close()
            self._running = False
            return True
        else:  # self.Mode.UNCONFIGURED
            return self.fail_with("stop() is called but mode is unconfigured.")

    def set(self, key: str, value=None) -> bool:
        if key == "exposure_time":
            self.set_exposure_time(value)
        else:
            return self.fail_with(f"Unknown set() key: {key}")

    def get(self, key: str, args=None):
        if key == "frame":
            return self.get_frame(timeout_sec=args)
        else:
            self.logger.error(f"Unknown get() key: {key}")
            return None
