#!/usr/bin/env python3

"""
Worker for Recorder.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import time

from ..util.timer import IntervalTimer
from ..msgs.recorder_msgs import RecorderData
from ..inst.interface import InstrumentInterface
from ..msgs import param_msgs as P
from .common_worker import Worker


class Collector(Worker):
    def __init__(self, cli, logger, conf: dict, mode: dict | None = None):
        Worker.__init__(self, cli, logger)

        self.interval_sec = conf.get("interval_sec", 1.0)
        self.insts = self.cli.insts()
        self.add_instruments([InstrumentInterface(self.cli, inst) for inst in self.insts])

        self._label = ""
        self.mode_dicts = mode or {"all": {inst: [inst, ""] for inst in self.insts}}

        self.data = RecorderData()
        self.timer = None

    def get_param_dict_labels(self) -> list[str]:
        return list(self.mode_dicts.keys())

    def get_param_dict(self, label: str) -> P.ParamDict[str, P.PDValue] | None:
        if label not in self.mode_dicts:
            self.logger.error(f"Invalid label {label}")
            return None

        pd = P.ParamDict()
        pd["max_len"] = P.IntParam(1000, 1, 100_000_000, doc="maximum data length")
        pd["interval"] = P.FloatParam(
            self.interval_sec, 0.01, 100.0, unit="s", doc="polling interval"
        )
        pd["lock"] = P.BoolParam(True, doc="acquire the locks of instruments")
        for channel, (inst, inst_label) in self.mode_dicts[label].items():
            d = self.cli.get_param_dict(inst, inst_label)
            if d is None:
                self.logger.error(
                    f"Failed to generate param dict for {channel} ({inst}, {inst_label})."
                )
                return None
            pd[channel] = d
        return pd

    def start(
        self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue], label: str = ""
    ) -> bool:
        if params is not None:
            params = P.unwrap(params)

        if label not in self.mode_dicts:
            self.logger.error(f"Invalid mode label {label}")
            return False

        self._label = label
        used_insts = set([inst for inst, inst_label in self.mode_dicts[label].values()])

        if params.get("lock", True):
            for inst in used_insts:
                if not self.cli.lock(inst):
                    return self.fail_with_release(f"Failed to lock instrument {inst}")
        else:
            self.logger.info("Skipping to acquire locks of instruments")

        units = []
        for channel, (inst, inst_label) in self.mode_dicts[label].items():
            if channel not in params:
                return self.fail_with_release(
                    f"Channel {channel} ({inst}, {inst_label}) is not contained in params."
                )
            if not self.cli.configure(inst, params[channel], inst_label):
                return self.fail_with_release(
                    f"Failed to configure channel {channel} ({inst}, {inst_label})"
                )
            units.append((channel, self.cli.get(inst, "unit", label=inst_label) or ""))

        for channel, (inst, inst_label) in self.mode_dicts[label].items():
            if not self.cli.start(inst, label=inst_label):
                return self.fail_with_release(
                    f"Failed to start channel {channel} ({inst}, {inst_label})"
                )

        self.timer = IntervalTimer(params.get("interval", self.interval_sec))

        self.data = RecorderData(params, label)
        self.data.set_units(units)
        self.data.start()
        self.logger.info("Started collector.")

        return True

    def reset(self, label) -> bool:
        if self.data.running:
            self.logger.error("reset() is called while running.")
            return False

        if label not in self.mode_dicts:
            self.logger.error(f"Invalid mode label {label}")
            return False

        success = True
        for channel, (inst, inst_label) in self.mode_dicts[label].items():
            if not self.cli.reset(inst, label=inst_label):
                self.logger.error(f"Failed to reset channel {channel} ({inst}, {inst_label})")
                success = False
        return success

    def work(self) -> bool:
        # TODO: treatment of time stamp is quite rough now

        if not self.data.running:
            return False

        if not self.timer.check():
            return False

        t_start = time.time()
        data = {}
        for channel, (inst, inst_label) in self.mode_dicts[self._label].items():
            d = self.cli.get(inst, "data", label=inst_label)
            if d is None:
                return False
            data[channel] = d
        t_finish = time.time()
        t = (t_start + t_finish) / 2.0
        self.data.append(t, data)
        return True

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = True
        for channel, (inst, inst_label) in self.mode_dicts[self._label].items():
            success &= self.cli.stop(inst, label=inst_label) and self.cli.release(inst)
        self.timer = None
        self.data.finalize()

        if success:
            self.logger.info("Stopped collector.")
        else:
            self.logger.error("Error stopping collector.")
        return success

    def data_msg(self) -> RecorderData:
        return self.data
