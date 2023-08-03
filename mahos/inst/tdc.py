#!/usr/bin/env python3

"""
Time to Digital Converter (Time Digitizer) module

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import ctypes as C
import re
import os

import numpy as np

from .instrument import Instrument


def c_str(s: str) -> C.c_char_p:
    """Convert Python string to C Constant String (pointer to NULL terminated of chars)"""

    return C.c_char_p(bytes(s, encoding="utf-8"))


class MCS(Instrument):
    """Wrapper Class of DMCSX.dll for Fast ComTec MCS6 / MCS8 Series.

    :param base_configs: Mapping from base config name to actual file name.
        Used in configure_base_range_bin() etc.
    :type base_configs: dict[str, str]
    :param home: (default: "C:\\mcs8x64") Home directory for mcs8 software.
    :type home: str
    :param ext_ref_clock: use external reference clock source.
    :type ext_ref_clock: bool
    :param remove_lst: (default: True) Remove *.lst file after loading it.
    :type remove_lst: bool
    :param lst_channels: (default: [8, 9]) Collected channels for lst file.
        Under a setting, default value [8, 9] corresponds to STOP1 and STOP2.
        As correspondence is unclear after reading the manual (maybe dependent on the setting),
        it is recommended to inspect the output lst file first.
    :type lst_channels: list[int]

    DLL Modifications
    =================

    We've made some modification on DMCSX.dll itself.

    RunCmd
    ------

    In original code, return type was void.
    Fixed this function to return int.
    Return 0 on success and some negative values on error.

    """

    MAXCNT = 448
    nDev = 0

    EOS_DEADTIME = 100e-9  # end-of-sweep deadtime

    class ACQSTATUS(C.Structure):
        _fields_ = [
            ("started", C.c_uint),
            ("maxval", C.c_uint),
            ("runtime", C.c_double),  # in seconds
            ("ofls", C.c_double),
            ("totalsum", C.c_double),
            ("roisum", C.c_double),
            ("roirate", C.c_double),
            ("sweeps", C.c_double),
            ("starts", C.c_double),
            ("zeros", C.c_double),
        ]

    class ACQSETTING(C.Structure):
        _fields_ = [
            ("range", C.c_int),
            ("cftfak", C.c_int),
            ("roimin", C.c_int),
            ("roimax", C.c_int),
            ("nregions", C.c_int),
            ("caluse", C.c_int),
            ("calpoints", C.c_int),
            ("param", C.c_int),
            ("offset", C.c_int),
            ("xdim", C.c_int),
            ("bitshift", C.c_uint),  # // LOWORD: Binwidth = 2 ^ (bitshift)
            # // HIWORD: Threshold for Coinc
            ("active", C.c_int),
            ("eventpreset", C.c_double),
            ("dummy1", C.c_double),
            ("dummy2", C.c_double),
            ("dummy3", C.c_double),
        ]

    # this was called ACQMCSSETTING in MCS6A.
    # the structure is identical except rename of "maxchan" to "periods".
    class BOARDSETTING(C.Structure):
        _fields_ = [
            ("sweepmode", C.c_int),
            ("prena", C.c_int),
            ("cycles", C.c_int),
            ("sequences", C.c_int),
            ("syncout", C.c_int),
            ("digio", C.c_int),
            ("digval", C.c_int),
            ("dac0", C.c_int),
            ("dac1", C.c_int),
            ("dac2", C.c_int),
            ("dac3", C.c_int),
            ("dac4", C.c_int),
            ("dac5", C.c_int),
            ("fdac", C.c_int),
            ("tagbits", C.c_int),
            ("extclk", C.c_int),
            # MCS8A: periods // number of periods in folded mode, sweeplength = range * periods
            # MCS6A: maxchan
            ("periods_maxchan", C.c_int),
            ("serno", C.c_int),
            ("ddruse", C.c_int),
            ("active", C.c_int),
            ("holdafter", C.c_double),
            ("swpreset", C.c_double),
            ("fstchan", C.c_double),
            ("timepreset", C.c_double),
        ]

    class DATSETTING(C.Structure):
        _fields_ = [
            ("savedata", C.c_int),
            ("autoinc", C.c_int),
            ("fmt", C.c_int),
            ("mpafmt", C.c_int),
            ("sephead", C.c_int),
            ("smpts", C.c_int),
            ("caluse", C.c_int),
            ("filename", C.c_char * 256),
            ("specfile", C.c_char * 256),
            ("command", C.c_char * 256),
        ]

    def __init__(self, name, conf, prefix=None):
        Instrument.__init__(self, name, conf, prefix=prefix)

        fn = self.conf.get("dll_name", "DMCS8.dll")
        self.resolution_sec = self.conf.get("resolution_sec", 0.2e-9)
        self.dll = C.windll.LoadLibrary(fn)
        self.logger.info(f"Loaded {fn}")

        self._base_configs = self.conf.get("base_configs", {})
        self._home = self.conf.get("home", "C:\\mcs8x64")
        self._save_file_name = None
        self._remove_lst = self.conf.get("remove_lst", True)
        self._lst_channels = self.conf.get("lst_channels", [8, 9])
        self.logger.debug(f"available base config files: {self._base_configs}")

        # Ref. clock setting is not affected by load_config() and
        # persistent during MCS software is alive.
        self.ext_ref_clock = self.conf.get("ext_ref_clock", False)
        self.set_reference_clock(bool(self.ext_ref_clock))

    def run_command(self, cmd: str) -> bool:
        """Run a command in MCS Server. Return True on success."""

        ret = self.dll.RunCmd(0, c_str(cmd))
        if ret:
            self.logger.error(f"RunCmd() returned {ret}")
            return False
        else:
            return True

    def load_config(self, fn: str) -> bool:
        """Load config (*.set) file for MCS."""

        return self.run_command(f"loadcnf {fn}")

    def clear(self) -> bool:
        """Clear all spectra data."""

        return self.run_command("erase")

    def get_data_point(self, i: int, nDisplay: int):
        """Get intensity of one point in spectrum.

        nDisplay is the channel number.

        """

        return self.dll.GetSpec(i, nDisplay)

    def get_data(self, nDisplay: int) -> np.ndarray | None:
        """Get whole spectrum at <nDisplay>.

        Spectrum data is returned as numpy.ndarray.
        If Display is 1D spectrum, return 1D array.
        If Display is 2D mapped spectrum, return 2D array.

        """

        xmax, ymax = C.c_long(), C.c_long()
        _range = self.dll.GetDatInfo(nDisplay, C.byref(xmax), C.byref(ymax))
        # print(_range, xmax.value, ymax.value)

        if _range <= 0:
            self.logger.error("No Data in specified Display.")
            return None

        buf = (C.c_ulong * _range)()
        if self.dll.LVGetDat(buf, nDisplay) != 0:
            self.logger.error("Error reading spectrum.")
            return None
        # print(np.array(buf).shape)

        if ymax.value > 1:
            return np.reshape(np.array(buf), (xmax.value, ymax.value)).T
        else:
            return np.array(buf)

    def get_count(self, nDisplay: int) -> np.ndarray | None:
        cnt = (C.c_double * self.MAXCNT)()
        if self.dll.LVGetCnt(cnt, nDisplay) == 0:
            return np.array(cnt)
        else:
            return None

    def get_status(self, nDisplay: int) -> ACQSTATUS:
        status = self.ACQSTATUS()
        self.dll.GetStatusData(C.byref(status), nDisplay)

        return status

    def get_setting(self, nDisplay: int) -> ACQSETTING:
        setting = self.ACQSETTING()
        self.dll.GetSettingData(C.byref(setting), nDisplay)
        return setting

    def set_setting(self, nDisplay: int, setting: ACQSETTING):
        self.dll.StoreSettingData(C.byref(setting), nDisplay)
        self.dll.NewSetting(self.nDev)

    def get_data_setting(self) -> DATSETTING:
        setting = self.DATSETTING()
        self.dll.GetDatSetting(C.byref(setting))
        return setting

    def set_data_setting(self, setting: DATSETTING):
        self.dll.StoreDatSetting(C.byref(setting))
        self.dll.NewSetting(self.nDev)

    def set_save_file_name(self, name: str) -> bool:
        if not name.endswith(".mpa"):
            name += ".mpa"

        setting = self.get_data_setting()
        setting.filename = name.encode()
        self.set_data_setting(setting)
        self._save_file_name = name
        return True

    def remove_saved_file(self, name: str) -> bool:
        f = os.path.join(self._home, name)
        if not os.path.exists(f):
            return self.fail_with(f"File doesn't exist: {f}")
        os.remove(f)
        return True

    def set_range(self, value: int) -> bool:
        return self.run_command(f"range={value}")

    def get_range(self) -> int:
        setting = self.get_setting(0)
        return setting.range

    def set_bitshift(self, value: int) -> bool:
        return self.run_command(f"bitshift={value}")

    def get_bitshift(self) -> int:
        setting = self.get_setting(0)
        return setting.bitshift & 0x0000FFFF  # mask high word.

    def get_binwidth(self) -> int:
        return 2 ** self.get_bitshift()

    def get_range_bin(self) -> dict:
        """Get range and time bin (in sec)."""

        _range = self.get_range()
        tbin = self.resolution_sec * self.get_binwidth()

        return {"range": _range, "bin": tbin}

    def get_timebin(self) -> float:
        """Get time bin (in sec)."""

        return self.resolution_sec * self.get_binwidth()

    def set_sweep_preset(self, preset: float, enable: bool) -> bool:
        """set sweep preset. Preset value is actually used only if enable is True."""

        s = self.get_mcssetting()

        if enable:
            s.prena = s.prena | (1 << 2)
        else:
            s.prena = s.prena & ~(1 << 2)
        s.swpreset = preset

        self.set_mcssetting(s)
        return True

    def set_reference_clock(self, external: bool) -> bool:
        s = self.get_mcssetting()
        if external:
            s.extclk = 1
            self.logger.info("Reference clock: External")
        else:
            s.extclk = 0
            self.logger.info("Reference clock: Internal")
        self.set_mcssetting(s)
        return True

    def get_mcssetting(self) -> BOARDSETTING:
        s = self.BOARDSETTING()
        self.dll.GetMCSSetting(C.byref(s), self.nDev)
        return s

    def set_mcssetting(self, setting: BOARDSETTING):
        self.dll.StoreMCSSetting(C.byref(setting), self.nDev)
        self.dll.NewSetting(self.nDev)

    def load_raw_events(self) -> list[np.ndarray] | None:
        if not self._save_file_name:
            self.logger.error("save file name has not been set.")
            return None

        file_name = os.path.splitext(self._save_file_name)[0] + ".lst"
        file_name = os.path.join(self._home, file_name)

        ret = self.load_lst_file(file_name)

        if self._remove_lst:
            self.remove_saved_file(file_name)

        if ret is None:
            return None

        _, format_info, data = ret
        return self.convert_raw_events(format_info, data)

    def convert_raw_events(self, format_info, data) -> list[np.ndarray]:
        cl, ch = format_info["channel"]
        tl, th = format_info["timedata"]
        ch_data = (data >> cl) & 2 ** (ch - cl + 1) - 1

        results = []
        for i in self._lst_channels:
            timedata = (data[ch_data == i] >> tl) & (2 ** (th - tl + 1) - 1)
            results.append(timedata)
        return results

    def load_lst_file(self, file_name: str) -> tuple[str, dict, np.ndarray] | None:
        """Load the lst file to get data."""

        if not os.path.exists(file_name):
            self.logger.error(f"File doesn't exist: {file_name}")
            return None

        setting = self.get_data_setting()
        if setting.mpafmt == 0:
            binary = False
        elif setting.mpafmt == 1:
            binary = True
        else:
            self.logger.error(f"Unsupported DATSETTING.mpafmt: {setting.mapfmt}")
            return None

        format_info = {
            "datalength": None,
            "channel": None,
            "edge": None,
            "timedata": None,
            "datalost": None,
        }

        header = []
        f = open(file_name, "rb" if binary else "r")

        pat_b = re.compile(r"^;datalength=(\d+)bytes")
        pat_c = re.compile(r"^;bit(\d+)\.\.(\d+):channel")
        pat_e = re.compile(r"^;bit(\d+):edge")
        pat_t = re.compile(r"^;bit(\d+)\.\.(\d+):timedata")
        pat_l = re.compile(r"^;bit(\d+):data_lost")

        for _ in range(200):
            l = f.readline().strip()
            if binary:
                l = l.decode("utf-8")

            ll = l.replace(" ", "")
            if (m := pat_b.match(ll)) is not None:
                format_info["datalength"] = int(m.group(1))
            elif (m := pat_c.match(ll)) is not None:
                format_info["channel"] = (int(m.group(1)), int(m.group(2)))
            elif (m := pat_e.match(ll)) is not None:
                format_info["edge"] = int(m.group(1))
            elif (m := pat_t.match(ll)) is not None:
                format_info["timedata"] = (int(m.group(1)), int(m.group(2)))
            elif (m := pat_l.match(ll)) is not None:
                format_info["datalost"] = int(m.group(1))

            if l == "[DATA]":
                break
            header.append(l)
        else:
            self.logger.error("[DATA] line not found in lst file.")
            return None

        self.logger.debug(f"Loaded header. {format_info}")

        if not binary:
            data = np.array([int(l, base=16) for l in f.readlines()], dtype=np.uint64)
        else:
            dsize = os.path.getsize(file_name) - f.tell()
            dlen = format_info["datalength"]
            if dsize % dlen:
                self.logger.error(
                    f"data size {dsize} is not integer multiple of datalength {dlen}"
                )
                return None
            data = np.fromfile(f, dtype=np.uint64)

        return header, format_info, data

    def configure_base_range_bin_save(
        self,
        base_config: str,
        trange: float,
        tbin: float,
        save_file: str,
    ) -> bool:
        """Load base config, set range and timebin in sec, and save file name.

        Note that actual timebin maybe rounded.

        """

        if not self.configure_base_range_bin(base_config, trange, tbin):
            return False

        return self.set_save_file_name(save_file)

    def configure_base_range_bin(self, base_config: str, trange: float, tbin: float) -> bool:
        """Load base config and set range and timebin in sec.

        Note that actual timebin maybe rounded.

        """

        if base_config not in self._base_configs:
            return self.fail_with("Unknown base config name")

        if not self.load_config(self._base_configs[base_config]):
            return False

        return self.configure_range_bin(trange, tbin)

    def configure_range_bin(self, trange: float, tbin: float) -> bool:
        """Set range and timebin in sec.

        Note that actual timebin maybe rounded.

        """

        if tbin == 0.0:
            tbin = self.resolution_sec
        success = self.set_bitshift(int(np.round(np.log2(tbin / self.resolution_sec))))
        tbin = self.resolution_sec * self.get_binwidth()
        success &= self.set_range(int(round(trange / tbin)))

        return success

    # Standard API

    def start(self) -> bool:
        """Clear data and start a new acquisition."""

        return self.run_command("start")

    def stop(self) -> bool:
        """Stop acqusition if running."""

        return self.run_command("halt")

    def resume(self) -> bool:
        """Resume acqusition."""

        return self.run_command("cont")

    def configure(self, params: dict) -> bool:
        if all([k in params for k in ("base_config", "range", "bin", "save_file")]):
            return self.configure_base_range_bin_save(
                params["base_config"], params["range"], params["bin"], params["save_file"]
            )
        elif all([k in params for k in ("base_config", "range", "bin")]):
            return self.configure_base_range_bin(
                params["base_config"], params["range"], params["bin"]
            )
        elif all([k in params for k in ("range", "bin")]):
            return self.configure_range_bin(params["range"], params["bin"])
        else:
            self.logger.error("invalid configure() params keys.")
            return False

    def set(self, key: str, value=None) -> bool:
        if key == "clear":
            return self.clear()
        elif key == "sweeps":
            return self.set_sweep_preset(value, bool(value))
        elif key == "save_file":
            return self.set_save_file_name(value)
        else:
            self.logger.error(f"unknown set() key: {key}")
            return False

    def get(self, key: str, args=None):
        if key == "range_bin":
            return self.get_range_bin()
        elif key == "bin":
            return self.get_timebin()
        elif key == "data":
            if not isinstance(args, int):
                self.logger.error('get("data", args): args must be int (channel).')
            return self.get_data(args)
        elif key == "status":
            if not isinstance(args, int):
                self.logger.error('get("status", args): args must be int (channel).')
            return self.get_status(args)
        elif key == "raw_events":
            return self.load_raw_events()
        else:
            self.logger.error(f"unknown get() key: {key}")
            return None
