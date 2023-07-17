#!/usr/bin/env python3

"""
Worker for Imaging ODMR.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations
import enum

import numpy as np

from ..msgs.iodmr_msgs import IODMRData
from ..msgs import param_msgs as P
from ..msgs.inst_pg_msgs import TriggerType, Block, Blocks
from ..inst.overlay.iodmr_sweeper_interface import IODMRSweeperInterface
from ..inst.sg_interface import SGInterface
from ..inst.pg_interface import PGInterface
from ..inst.camera_interface import CameraInterface
from .common_worker import Worker


class WorkStatus(enum.Enum):
    Normal = 0  # normally worked (or skipped working): nothing to do.
    SweepDone = 1  # completed capturing one sweep: data should be published.
    Error = 2  # error occured capturing: stop.


class ISweeperBase(Worker):
    def _make_param_dict(self, bounds):
        f_min, f_max = bounds["freq"]
        p_min, p_max = bounds["power"]
        d = P.ParamDict(
            start=P.FloatParam(2.74e9, f_min, f_max),
            stop=P.FloatParam(3.00e9, f_min, f_max),
            num=P.IntParam(51, 2, 10000),
            power=P.FloatParam(p_min, p_min, p_max),
            sweeps=P.IntParam(0, 0, 100000),
            roi=P.ParamDict(
                width=P.IntParam(100, minimum=0, optional=True, enable=False),
                height=P.IntParam(100, minimum=0, optional=True, enable=False),
                woffset=P.IntParam(0, minimum=0, optional=True, enable=False),
                hoffset=P.IntParam(0, minimum=0, optional=True, enable=False),
            ),
            binning=P.IntParam(1, 0, 4),
            exposure_delay=P.FloatParam(25e-3, 0.0, 1.0),
            exposure_time=P.FloatParam(1e-3, 1e-5, 1.0),
            burst_num=P.IntParam(1, 1, 10000),
            resume=P.BoolParam(False),
            ident=P.UUIDParam(optional=True, enable=False),
        )
        return d


class ISweeperOverlay(ISweeperBase):
    """ISweeper using Overlay."""

    def __init__(self, cli, logger):
        Worker.__init__(self, cli, logger)
        self.sweeper = IODMRSweeperInterface(cli, "isweeper")
        self.add_instrument(self.sweeper)

        self.data = IODMRData()

    def get_param_dict(self, name: str) -> P.ParamDict[str, P.PDValue] | None:
        bounds = self.sweeper.get_bounds()
        if bounds is None:
            self.logger.error("Failed to get bounds.")
            return None
        return self._make_param_dict(bounds)

    def start(self, params: None | P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is not None:
            params = P.unwrap(params)
        success = self.sweeper.lock()
        if params is not None:
            success &= self.sweeper.configure(params)
        success &= self.sweeper.start()

        if not success:
            return self.fail_with_release("Error starting isweeper.")

        if params is not None and not params["resume"]:
            # new measurement.
            self.data = IODMRData(params)
            self.data.start()
            self.logger.info("Started isweeper.")
        else:
            # resume.
            self.data.update_params(params)
            self.data.resume()
            self.logger.info("Resuming isweeper.")

        return True

    def update_data(self, frames):
        def frames_to_line(fs):
            # shape: (num_freq, 1)
            return np.mean(fs, axis=(1, 2))[:, np.newaxis]

        d = self.data
        d.data_latest = frames
        if d.sweeps:
            d.data_sum = d.data_sum + frames.astype(np.float64)
            d.data_history = np.append(d.data_history, frames_to_line(frames), axis=1)
        else:  # first data
            d.data_sum = frames.astype(np.float64)
            d.data_history = frames_to_line(frames)
        d.sweeps += 1

    def work(self) -> WorkStatus:
        if not self.data.running:
            return WorkStatus.Normal

        frames = self.sweeper.get_frames()
        if frames is not None:
            self.update_data(frames)
            return WorkStatus.SweepDone
        else:
            return WorkStatus.Normal

    def is_finished(self) -> bool:
        if not self.data.has_params() or not self.data.sweeps:
            return False
        if self.data.params.get("sweeps", 0) <= 0:
            return False  # no acquisitions limit defined.
        return self.data.sweeps >= self.data.params["sweeps"]

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = self.sweeper.stop() and self.sweeper.release()

        self.data.finalize()
        if success:
            self.logger.info("Stopped isweeper.")
        else:
            self.logger.error("Error stopping isweeper.")
        return success

    def data_msg(self) -> IODMRData | None:
        if not self.data.sweeps:
            return None
        return self.data


class ISweeperDirect(ISweeperBase):
    """ISweeper using direct hardware calls."""

    def __init__(self, cli, logger):
        Worker.__init__(self, cli, logger)
        self.sg = SGInterface(cli, "sg")
        self.pg = PGInterface(cli, "pg")
        self.camera = CameraInterface(cli, "camera")
        self.add_instruments(self.sg, self.pg, self.camera)

        self.data = IODMRData()
        self._frames = []

    def get_param_dict(self, name: str) -> dict | None:
        bounds = self.sg.get_bounds()
        if bounds is None:
            self.logger.error("Failed to get bounds.")
            return None
        return self._make_param_dict(bounds)

    def configure_inst(self, params: dict) -> bool:
        # SG
        if not self.sg.configure_point_trig_freq_sweep(
            params["start"], params["stop"], params["num"], params["power"]
        ):
            self.logger.error("Failed to configure SG.")
            return False

        # PG
        # TODO: assuming 1 GHz freq (1 ns pattern) here
        freq = 1.0e9
        pattern = [
            ("sg_trig", 1000),
            (None, int(round(params["exposure_delay"] * 1e9))),
            ("camera_trig", 5000),
            (None, 100),
        ]
        blocks = Blocks([Block("CW1", pattern, trigger=True)])
        if not self.pg.configure_blocks(
            blocks, freq, trigger_type=TriggerType.HARDWARE_FALLING, n_runs=1
        ):
            self.logger.error("Failed to configure PG.")
            return False

        # Camera
        if not self.camera.configure_hard_trigger(
            params["exposure_time"], binning=params.get("binning", 0), roi=params.get("roi")
        ):
            self.logger.error("Failed to configure Camera.")
            return False

        return True

    def start(self, params: dict | None) -> bool:
        resume = params is None or ("resume" in params and params["resume"])

        if not self.lock_instruments():
            return self.fail_with_release("Error acquiring instrument locks.")

        if not self.configure_inst(params):
            return self.fail_with_release("Error configuring instruments.")

        if not (
            self.camera.start()
            and self.sg.set_output(True)
            and self.sg.set_init_cont(True)
            and self.pg.start()
        ):
            return self.fail_with_release("Error starting isweeper.")

        if not resume:
            # new measurement.
            self.data = IODMRData(params)
            self.data.start()
            self.logger.info("Starting isweeper.")
        else:
            # TODO: check ident if resume?
            self.data.update_params(params)
            self.data.resume()
            self.logger.info("Resuming isweeper.")

        # start first sweep here
        self.pg.trigger()
        self._frames = []

        return True

    def update_frames(self, frames):
        def frames_to_line(fs):
            # shape: (num_freq, 1)
            return np.mean(fs, axis=(1, 2))[:, np.newaxis]

        d = self.data
        d.data_latest = frames
        if d.sweeps:
            d.data_sum = d.data_sum + frames.astype(np.float64)
            d.data_history = np.append(d.data_history, frames_to_line(frames), axis=1)
        else:  # first data
            d.data_sum = frames.astype(np.float64)
            d.data_history = frames_to_line(frames)
        d.sweeps += 1

    def update_frame(self, frame) -> bool:
        """returns True if sweep is done."""

        self._frames.append(frame)
        if len(self._frames) == self.data.params["num"]:
            # sweep is done
            self.update_frames(np.array(self._frames))
            self._frames = []
            return True
        else:
            return False

    def work(self) -> WorkStatus:
        if not self.data.running:
            return WorkStatus.Normal

        res = self.camera.get_frame()
        if res.is_invalid():
            self.logger.error("Captured invalid FrameResult. Quitting.")
            return WorkStatus.Error

        if not res.is_empty() and self.update_frame(res.frame):
            return WorkStatus.SweepDone
        return WorkStatus.Normal

    def is_finished(self) -> bool:
        if self.data.params is None or not self.data.sweeps:
            return False
        if self.data.params.get("sweeps", 0) <= 0:
            return False  # no acquisitions limit defined.
        return self.data.sweeps >= self.data.params["sweeps"]

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = (
            self.camera.stop()
            and self.camera.release()
            and self.pg.stop()
            and self.pg.release()
            and self.sg.set_output(False)
            and self.sg.set_abort()
            and self.sg.set_init_cont(False)
            and self.sg.release()
        )

        self.data.finalize()
        if success:
            self.logger.info("Stopped isweeper.")
        else:
            self.logger.error("Error stopping isweeper.")
        return success

    def data_msg(self) -> IODMRData | None:
        if not self.data.sweeps:
            return None
        return self.data
