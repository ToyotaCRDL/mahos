#!/usr/bin/env python3

"""
HDF5 file transport for AWG waveform data.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import typing as T

import h5py
import numpy as np


FORMAT = "mahos.awg.waveforms"
VERSION = 1


def save_waveforms(file_name: str, analog: dict[int, np.ndarray], digital: dict[str, T.Any]):
    """Save analog waveforms and RLE digital tracks to an uncompressed HDF5 file."""

    with h5py.File(file_name, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["version"] = VERSION

        analog_group = f.create_group("analog")
        for channel, samples in analog.items():
            analog_group.create_dataset(str(channel), data=np.asarray(samples), compression=None)

        digital_group = f.create_group("digital")
        for index, (name, runs) in enumerate(digital.items()):
            track = digital_group.create_group(str(index))
            track.attrs["name"] = name
            levels, counts = zip(*runs) if runs else ((), ())
            track.create_dataset("levels", data=np.asarray(levels, dtype=bool), compression=None)
            track.create_dataset(
                "counts", data=np.asarray(counts, dtype=np.uint64), compression=None
            )


def load_waveforms(
    file_name: str,
) -> tuple[dict[int, np.ndarray], dict[str, list[tuple[bool, int]]]]:
    """Load and validate analog waveforms and RLE digital tracks from an HDF5 file."""

    with h5py.File(file_name, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"invalid AWG waveform file format: {f.attrs.get('format')!r}")
        if f.attrs.get("version") != VERSION:
            raise ValueError(f"unsupported AWG waveform file version: {f.attrs.get('version')!r}")
        if "analog" not in f or "digital" not in f:
            raise ValueError("AWG waveform file must contain analog and digital groups")

        analog = {}
        for raw_channel, dataset in f["analog"].items():
            try:
                channel = int(raw_channel)
            except ValueError:
                raise ValueError(f"invalid analog channel dataset name: {raw_channel!r}") from None
            if raw_channel != str(channel) or channel < 0 or channel in analog:
                raise ValueError(f"invalid analog channel dataset name: {raw_channel!r}")
            analog[channel] = dataset[()]

        digital = {}
        for track in f["digital"].values():
            if "name" not in track.attrs or "levels" not in track or "counts" not in track:
                raise ValueError("invalid digital track in AWG waveform file")
            name = track.attrs["name"]
            if not isinstance(name, str) or name in digital:
                raise ValueError(f"invalid or duplicate digital track name: {name!r}")
            levels = np.asarray(track["levels"][()])
            counts = np.asarray(track["counts"][()])
            if levels.ndim != 1 or counts.ndim != 1 or levels.size != counts.size:
                raise ValueError(f"invalid RLE arrays for digital track {name!r}")
            if levels.dtype.kind == "b":
                normalized_levels = levels
            elif (
                levels.dtype.kind in "iuf"
                and np.all(np.isfinite(levels))
                and np.all((levels == 0) | (levels == 1))
            ):
                normalized_levels = levels.astype(bool)
            else:
                raise ValueError(f"invalid RLE levels for digital track {name!r}")
            if counts.dtype.kind not in "iu" or np.any(counts < 0):
                raise ValueError(f"invalid RLE counts for digital track {name!r}")
            digital[name] = [
                (bool(level), int(count)) for level, count in zip(normalized_levels, counts)
            ]

    return analog, digital
