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

        self.interval_sec = conf.get("interval_sec", 0.1)
        self.insts = self.cli.insts()
        self.add_instruments(*[InstrumentInterface(self.cli, inst) for inst in self.insts])
        self.used_insts = []

        self.mode_inst_label = mode or {"all": {inst: "" for inst in self.insts}}

        self.data = RecorderData()
        self.timer = None

    def get_param_dict_labels(self) -> list[str]:
        return list(self.mode_inst_label.keys())

    def get_param_dict(self, mode: str) -> P.ParamDict[str, P.PDValue] | None:
        if mode not in self.mode_inst_label:
            self.logger.error(f"Invalid mode {mode}")
            return None

        pd = P.ParamDict()
        pd["mode"] = P.StrChoiceParam(mode, self.get_param_dict_labels())
        pd["max_len"] = P.IntParam(1000, 1, 100_000_000)
        for inst, label in self.mode_inst_label[mode].items():
            d = self.cli.get_param_dict(inst, label)
            if d is None:
                self.logger.error(f"Failed to generate param dict for {inst}.")
                return None
            pd[inst] = d
        return pd

    def start(self, params: P.ParamDict[str, P.PDValue] | dict[str, P.RawPDValue]) -> bool:
        if params is not None:
            params = P.unwrap(params)

        if "mode" not in params or params["mode"] not in self.mode_inst_label:
            self.logger.error("mode must be in params")
            return False

        self.used_insts = list(self.mode_inst_label[params["mode"]].keys())

        for inst in self.used_insts:
            if not self.cli.lock(inst):
                return self.fail_with_release(f"Failed to lock instrument {inst}")

        units = []
        for inst in self.used_insts:
            if inst not in params:
                return self.fail_with_release(f"Instrument {inst} is not contained in params.")
            label = self.mode_inst_label[params["mode"]][inst]
            if not self.cli.configure(inst, params[inst], label):
                return self.fail_with_release(f"Failed to configure instrument {inst}")
            units.append((inst, self.cli.get(inst, "unit") or ""))

        for inst in self.used_insts:
            if not self.cli.start(inst):
                return self.fail_with_release(f"Failed to start instrument {inst}")

        self.timer = IntervalTimer(self.interval_sec)

        self.data = RecorderData(params)
        self.data.set_units(units)
        self.data.start()
        self.logger.info("Started collector.")

        return True

    def work(self) -> bool:
        # TODO: treatment of time stamp is quite rough now

        if not self.data.running:
            return False

        if not self.timer.check():
            return False

        t_start = time.time()
        data = {}
        for inst in self.used_insts:
            d = self.cli.get(inst, "data")
            if d is None:
                return False
            data[inst] = d
        t_finish = time.time()
        t = (t_start + t_finish) / 2.0
        self.data.append(t, data)
        return True

    def stop(self) -> bool:
        # avoid double-stop (abort status can be broken)
        if not self.data.running:
            return False

        success = all([self.cli.stop(inst) and self.cli.release(inst) for inst in self.used_insts])

        self.timer = None
        self.data.finalize()

        if success:
            self.logger.info("Stopped collector.")
        else:
            self.logger.error("Error stopping collector.")
        return success

    def data_msg(self) -> RecorderData:
        return self.data
