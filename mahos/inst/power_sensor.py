#!/usr/bin/env python3

"""
RF / MW Power Sensor module

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import enum
import time
import ctypes as C

import numpy as np

from .instrument import Instrument
from ..msgs import param_msgs as P


class RS_NRPZ(Instrument):
    """Instrument for Rohde & Schwarz NRP-Z Power Sensor.

    Requires the DLL (rsnrpz_64.dll) which is installed along with
    `R&S NRP Toolkit <https://www.rohde-schwarz.com/jp/software/nrp-z2x1/>`_.

    """

    class Mode(enum.Enum):
        UNCONFIGURED = 0
        CONT_AVG = 1

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        self.check_required_conf(["resources"])
        resources = [s.upper() for s in self.conf["resources"]]

        fn = "rsnrpz_64.dll"
        self.dll = C.windll.LoadLibrary(fn)
        self.logger.info(f"Loaded {fn}")

        self._sessions = []
        for i, res in enumerate(resources):
            session = C.c_uint32()
            n = C.create_string_buffer(res.encode())
            err = self.dll.rsnrpz_init(n, True, True, C.byref(session))
            if err < 0:
                # cannot use self.check_error here because self._sessions hasn't been initialized.
                msg = C.create_string_buffer(512)
                self.dll.rsnrpz_error_message(session, err, C.byref(msg))
                m = msg.value.decode()
                self.logger.error(m)
                raise RuntimeError(m)

            self.logger.info(f"rsnrpz_init resource: {res} session: {session.value} -> {err}")
            self._sessions.append(session.value)

        # Don't use rsnrpz_GetSensorInfo here because it looks a bit strange:
        # The result is not tied to session id.

        self._running = False
        self._mode = self.Mode.UNCONFIGURED

    def check_error(self, session: int, err: int) -> bool:
        if not err:
            return True
        msg = C.create_string_buffer(512)
        self.dll.rsnrpz_error_message(session, err, C.byref(msg))
        self.logger.error(f"Error ({err}) " + msg.value.decode())
        return False

    def configure_cont_average(self, params) -> bool:
        self.set_unit(params.get("unit", "dBm"))

        RSNRPZ_SENSOR_MODE_CONTAV = 0
        RSNRPZ_TRIGGER_SOURCE_IMMEDIATE = 3
        for session in self._sessions:
            ch = 1
            success = (
                self.check_error(
                    session,
                    self.dll.rsnrpz_chan_mode(session, ch, RSNRPZ_SENSOR_MODE_CONTAV),
                )
                and self.check_error(
                    session,
                    self.dll.rsnrpz_chan_setCorrectionFrequency(
                        session, ch, C.c_double(params.get("freq", 1.0e9))
                    ),
                )
                and self.check_error(session, self.dll.rsnrpz_avg_setAutoEnabled(session, ch, 0))
                and self.check_error(
                    session,
                    self.dll.rsnrpz_avg_configureAvgManual(session, ch, params.get("samples", 1)),
                )
                and self.check_error(
                    session,
                    self.dll.rsnrpz_chan_setContAvAperture(
                        session, ch, C.c_double(params.get("aperture", 100e-6))
                    ),
                )
                and self.check_error(
                    session,
                    self.dll.rsnrpz_trigger_setSource(
                        session, ch, RSNRPZ_TRIGGER_SOURCE_IMMEDIATE
                    ),
                )
                and self.check_error(
                    session,
                    self.dll.rsnrpz_chan_setContAvSmoothingEnabled(
                        session, ch, params.get("smoothing", True)
                    ),
                )
            )
            if not success:
                return False
        self._mode = self.Mode.CONT_AVG
        return True

    def set_unit(self, unit: str) -> bool:
        unit = unit.lower()
        if unit == "dbm":
            self._unit = "dBm"
            return True
        elif unit == "dbw":
            self._unit = "dBW"
            return True
        elif unit == "w":
            self._unit = "W"
            return True
        elif unit == "mw":
            self._unit = "mW"
            return True
        return self.fail_with(f"Invalid unit: {unit}")

    def convert_power(self, power_W: float):
        if self._unit == "dBm":
            return 10 * np.log10(np.abs(power_W)) + 30.0
        elif self._unit == "dBW":
            return 10 * np.log10(np.abs(power_W))
        elif self._unit == "mW":
            return power_W * 1e3
        else:
            return power_W

    def _get_data_cont_avg(self, ch: int) -> float | None:
        session = self._sessions[ch]
        if not self.check_error(session, self.dll.rsnrpz_chans_initiate(session)):
            return None
        compl = C.c_uint16(0)
        res = C.c_double()
        while not compl.value:
            if not self.check_error(
                session,
                self.dll.rsnrpz_chan_isMeasurementComplete(session, 1, C.byref(compl)),
            ):
                return None
            time.sleep(0.01)
        if not self.check_error(
            session,
            self.dll.rsnrpz_meass_fetchMeasurement(session, 1, C.byref(res)),
        ):
            return None
        return self.convert_power(res.value)

    def get_data_cont_avg(self, ch: int | None = None) -> float | None | list[float | None]:
        if len(self._sessions) == 1:
            return self._get_data_cont_avg(0)
        if ch is None:
            return [self._get_data_cont_avg(ch) for ch in range(len(self._sessions))]
        return self._get_data_cont_avg(ch)

    def get_data(self, ch: int | None = None) -> float | None | list[float | None]:
        if self._mode == self.Mode.CONT_AVG:
            return self.get_data_cont_avg(ch)
        else:  # self.Mode.UNCONFIGURED
            self.logger.error("get_data() is called but not started.")
            return None

    def get_unit(self) -> str:
        return self._unit

    # Standard API

    def close(self):
        for session in self._sessions:
            self.dll.rsnrpz_close(session)
            self.logger.info(f"Closed session {session}")

    def start(self, label: str = "") -> bool:
        """Start measurement."""

        if self._running:
            self.logger.warn("start() is called while running.")
            return True

        if self._mode == self.Mode.CONT_AVG:
            self._running = True
            return True
        else:  # self.Mode.UNCONFIGURED
            return self.fail_with("Must be configured before start().")

    def stop(self, label: str = "") -> bool:
        """Stop measurement."""

        if not self._running:
            return True

        if self._mode == self.Mode.CONT_AVG:
            self._running = False
            return True
        else:  # self.Mode.UNCONFIGURED
            return self.fail_with("stop() is called but mode is unconfigured.")

    def get_param_dict_labels(self) -> list[str]:
        return ["cont_avg"]

    def get_param_dict(self, label: str = "") -> P.ParamDict[str, P.PDValue]:
        if label == "cont_avg":
            return P.ParamDict(
                unit=P.StrChoiceParam("dBm", ["dBm", "dBW", "mW", "W"]),
                freq=P.FloatParam(
                    1e9, 1e3, 20e9, SI_prefix=True, unit="Hz", doc="correction frequency"
                ),
                samples=P.IntParam(4, 1, 1000, doc="number of averaging samples"),
                aperture=P.FloatParam(
                    100e-6, 100e-6, 1e-3, SI_prefix=True, unit="s", doc="measurement aperture"
                ),
                smoothing=P.BoolParam(True, doc="enable smoothing"),
            )
        else:
            return self.fail_with(f"unknown label {label}")

    def configure(self, params: dict, label: str = "") -> bool:
        if label == "cont_avg":
            return self.configure_cont_average(params)
        else:
            return self.fail_with(f"unknown label: {label}")

    def get(self, key: str, args=None):
        if key == "data":
            return self.get_data(args)
        elif key == "unit":
            return self.get_unit()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
