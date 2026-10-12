#!/usr/bin/env python3

"""Shared-filesystem transport primitives.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
import typing as T
import uuid


class SharedFileTransport:
    """Manage safely named, atomically published files in one shared directory."""

    PREFIXES = {
        "save": "mahos_save_",
        "load": "mahos_load_",
        "export": "mahos_export_",
        "awg": "mahos_awg_",
    }

    def __init__(self, directory: str | os.PathLike[str]):
        if not isinstance(directory, (str, os.PathLike)) or not os.fspath(directory):
            raise ValueError("file transport directory must be a non-empty path")
        self.directory = os.path.abspath(os.path.expanduser(os.fspath(directory)))
        if not os.path.isdir(self.directory):
            raise FileNotFoundError(
                f"File transport directory {self.directory!r} does not exist or is not a directory"
            )

    @staticmethod
    def validate_basename(name: str) -> str:
        """Validate and return a safe non-empty file basename."""

        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            or os.path.basename(name) != name
        ):
            raise ValueError(f"file name must be a basename: {name!r}")
        return name

    def resolve(self, name: str) -> str:
        """Resolve a validated basename inside the transport directory."""

        return os.path.join(self.directory, self.validate_basename(name))

    @staticmethod
    def suffix(path: str | os.PathLike[str]) -> str:
        """Return all suffixes of a path (for example, ``.data.h5``)."""

        basename = os.path.basename(os.fspath(path))
        if basename.startswith(".") and basename not in (".", ".."):
            return basename
        return "".join(Path(basename).suffixes)

    def new_name(self, purpose: str, source_name: str | os.PathLike[str] = "") -> str:
        """Return a UUID-based basename partitioned by transport purpose."""

        try:
            prefix = self.PREFIXES[purpose]
        except KeyError:
            raise ValueError(f"unknown file transport purpose: {purpose!r}") from None
        suffix = ".h5" if purpose == "awg" else self.suffix(source_name)
        return f"{prefix}{uuid.uuid4().hex}{suffix}"

    def _staging_path(self, final_path: str) -> str:
        suffix = self.suffix(final_path)
        basename = os.path.basename(final_path)
        stem = basename[: -len(suffix)] if suffix else basename
        name = f"{stem}.staging-{uuid.uuid4().hex}{suffix}"
        return os.path.join(os.path.dirname(final_path), name)

    def publish(self, name: str, writer: T.Callable[[str], T.Any]) -> str:
        """Call ``writer`` on a staging path and atomically publish it as ``name``."""

        path, sidecars = self.publish_with_sidecars(name, writer)
        for _, sidecar_path in sidecars:
            self.remove_path(sidecar_path)
        return path

    @staticmethod
    def _staging_sidecars(staging_path: str) -> list[tuple[str, str]]:
        """Return ``(name suffix, path)`` pairs derived from a primary staging path."""

        head, _ = os.path.splitext(staging_path)
        head_name = os.path.basename(head)
        sidecars = []
        for entry in os.scandir(os.path.dirname(staging_path)):
            if (
                entry.path == staging_path
                or not entry.name.startswith(head_name)
                or not entry.is_file(follow_symlinks=False)
            ):
                continue
            suffix = entry.path[len(head) :]
            if suffix:
                sidecars.append((suffix, entry.path))
        return sorted(sidecars)

    def publish_with_sidecars(
        self, name: str, writer: T.Callable[[str], T.Any]
    ) -> tuple[str, list[tuple[str, str]]]:
        """Publish a primary file and sibling files derived from its stem.

        The returned sidecar pairs contain the suffix relative to the primary stem and the
        published path. For example, a writer output named ``plot_trace.png`` for primary
        ``plot.png`` has the suffix ``_trace.png``.

        """

        final_path = self.resolve(name)
        staging_path = self._staging_path(final_path)
        final_head, _ = os.path.splitext(final_path)
        published_sidecars = []
        try:
            writer(staging_path)
            for suffix, sidecar_path in self._staging_sidecars(staging_path):
                published_path = final_head + suffix
                if published_path == final_path:
                    raise ValueError(f"sidecar conflicts with primary artifact: {suffix!r}")
                os.replace(sidecar_path, published_path)
                published_sidecars.append((suffix, published_path))
            os.replace(staging_path, final_path)
        except BaseException:
            self.remove_path(staging_path)
            for _, sidecar_path in self._staging_sidecars(staging_path):
                self.remove_path(sidecar_path)
            for _, sidecar_path in published_sidecars:
                self.remove_path(sidecar_path)
            self.remove_path(final_path)
            raise
        return final_path, published_sidecars

    @staticmethod
    def atomic_copy(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> str:
        """Copy a file and atomically publish it at ``destination``."""

        source_path = os.fspath(source)
        final_path = os.path.abspath(os.path.expanduser(os.fspath(destination)))
        directory = os.path.dirname(final_path)
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Destination directory {directory!r} does not exist")
        suffix = SharedFileTransport.suffix(final_path)
        base = os.path.basename(final_path)
        stem = base[: -len(suffix)] if suffix else base
        staging_path = os.path.join(directory, f"{stem}.staging-{uuid.uuid4().hex}{suffix}")
        try:
            shutil.copyfile(source_path, staging_path)
            os.replace(staging_path, final_path)
        except BaseException:
            SharedFileTransport.remove_path(staging_path)
            raise
        return final_path

    def copy_from(self, name: str, destination: str | os.PathLike[str]) -> str:
        """Atomically copy transport artifact ``name`` to ``destination``."""

        return self.atomic_copy(self.resolve(name), destination)

    def stage_copy(self, source: str | os.PathLike[str], purpose: str) -> str:
        """Atomically copy ``source`` into this directory and return its new basename."""

        name = self.new_name(purpose, source)
        final_path = self.resolve(name)
        self.atomic_copy(source, final_path)
        return name

    @staticmethod
    def remove_path(path: str | os.PathLike[str]) -> bool:
        """Best-effort removal returning whether an artifact is absent afterwards."""

        try:
            os.remove(path)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    def remove(self, name: str) -> bool:
        """Best-effort removal of a validated transport basename."""

        return self.remove_path(self.resolve(name))

    def cleanup(self, prefix: str, ttl: float, now: float | None = None) -> list[str]:
        """Remove artifacts older than ``ttl`` whose basenames start with ``prefix``."""

        if not isinstance(prefix, str) or not prefix or prefix not in self.PREFIXES.values():
            raise ValueError(f"invalid artifact prefix: {prefix!r}")
        if ttl < 0:
            raise ValueError("ttl must be non-negative")
        threshold = (time.time() if now is None else now) - ttl
        removed = []
        for entry in os.scandir(self.directory):
            if not entry.name.startswith(prefix) or not entry.is_file(follow_symlinks=False):
                continue
            try:
                old = entry.stat(follow_symlinks=False).st_mtime <= threshold
            except OSError:
                continue
            if old and self.remove(entry.name):
                removed.append(entry.name)
        return removed
