#!/usr/bin/env python3

"""
Config utility module.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

Defines utilities for configs.

"""

from __future__ import annotations
import typing as T
import enum

import mahos.util.validation as V


_MISSING = object()


class ConfAccessorMixin(object):
    """Mixin providing validated access to ``self.conf`` entries."""

    def _conf_get(self, key: str, default: object = _MISSING) -> T.Any:
        if key in self.conf:
            return self.conf[key]
        if default is not _MISSING:
            return default
        raise V.ValidationError(f"Required configuration {key} is missing.")

    @T.overload
    def _conf_bool(self, key: str) -> V.Boolean: ...

    @T.overload
    def _conf_bool(self, key: str, default: V.Boolean) -> V.Boolean: ...

    def _conf_bool(self, key: str, default: object = _MISSING) -> V.Boolean:
        v = self._conf_get(key, default)
        return V.check_bool(v, key)

    @T.overload
    def _conf_str(self, key: str) -> str: ...

    @T.overload
    def _conf_str(self, key: str, default: str) -> str: ...

    def _conf_str(self, key: str, default: object = _MISSING) -> str:
        v = self._conf_get(key, default)
        return V.check_str(v, key)

    @T.overload
    def _conf_nonempty_str(self, key: str) -> str: ...

    @T.overload
    def _conf_nonempty_str(self, key: str, default: str) -> str: ...

    def _conf_nonempty_str(self, key: str, default: object = _MISSING) -> str:
        v = self._conf_get(key, default)
        return V.check_nonempty_str(v, key)

    @T.overload
    def _conf_int(self, key: str) -> V.Integer: ...

    @T.overload
    def _conf_int(self, key: str, default: V.Integer) -> V.Integer: ...

    def _conf_int(self, key: str, default: object = _MISSING) -> V.Integer:
        v = self._conf_get(key, default)
        return V.check_int(v, key)

    @T.overload
    def _conf_pos_int(self, key: str) -> V.Integer: ...

    @T.overload
    def _conf_pos_int(self, key: str, default: V.Integer) -> V.Integer: ...

    def _conf_pos_int(self, key: str, default: object = _MISSING) -> V.Integer:
        v = self._conf_get(key, default)
        return V.check_pos_int(v, key)

    @T.overload
    def _conf_nonneg_int(self, key: str) -> V.Integer: ...

    @T.overload
    def _conf_nonneg_int(self, key: str, default: V.Integer) -> V.Integer: ...

    def _conf_nonneg_int(self, key: str, default: object = _MISSING) -> V.Integer:
        v = self._conf_get(key, default)
        return V.check_nonneg_int(v, key)

    @T.overload
    def _conf_float(self, key: str) -> V.Float: ...

    @T.overload
    def _conf_float(self, key: str, default: V.Float) -> V.Float: ...

    def _conf_float(self, key: str, default: object = _MISSING) -> V.Float:
        v = self._conf_get(key, default)
        return V.check_float(v, key)

    @T.overload
    def _conf_pos_float(self, key: str) -> V.Float: ...

    @T.overload
    def _conf_pos_float(self, key: str, default: V.Float) -> V.Float: ...

    def _conf_pos_float(self, key: str, default: object = _MISSING) -> V.Float:
        v = self._conf_get(key, default)
        return V.check_pos_float(v, key)

    @T.overload
    def _conf_nonneg_float(self, key: str) -> V.Float: ...

    @T.overload
    def _conf_nonneg_float(self, key: str, default: V.Float) -> V.Float: ...

    def _conf_nonneg_float(self, key: str, default: object = _MISSING) -> V.Float:
        v = self._conf_get(key, default)
        return V.check_nonneg_float(v, key)

    @T.overload
    def _conf_num(self, key: str) -> V.Number: ...

    @T.overload
    def _conf_num(self, key: str, default: V.Number) -> V.Number: ...

    def _conf_num(self, key: str, default: object = _MISSING) -> V.Number:
        v = self._conf_get(key, default)
        return V.check_num(v, key)

    @T.overload
    def _conf_pos_num(self, key: str) -> V.Number: ...

    @T.overload
    def _conf_pos_num(self, key: str, default: V.Number) -> V.Number: ...

    def _conf_pos_num(self, key: str, default: object = _MISSING) -> V.Number:
        v = self._conf_get(key, default)
        return V.check_pos_num(v, key)

    @T.overload
    def _conf_nonneg_num(self, key: str) -> V.Number: ...

    @T.overload
    def _conf_nonneg_num(self, key: str, default: V.Number) -> V.Number: ...

    def _conf_nonneg_num(self, key: str, default: object = _MISSING) -> V.Number:
        v = self._conf_get(key, default)
        return V.check_nonneg_num(v, key)

    @T.overload
    def _conf_integers(self, key: str, length: int | None = None) -> V.IntegerSequence: ...

    @T.overload
    def _conf_integers(
        self,
        key: str,
        length: int | None,
        default: V.IntegerSequence,
    ) -> V.IntegerSequence: ...

    def _conf_integers(
        self,
        key: str,
        length: int | None = None,
        default: object = _MISSING,
    ) -> V.IntegerSequence:
        v = self._conf_get(key, default)
        return V.check_integers(v, length, key)

    @T.overload
    def _conf_pos_integers(self, key: str, length: int | None = None) -> V.IntegerSequence: ...

    @T.overload
    def _conf_pos_integers(
        self,
        key: str,
        length: int | None,
        default: V.IntegerSequence,
    ) -> V.IntegerSequence: ...

    def _conf_pos_integers(
        self,
        key: str,
        length: int | None = None,
        default: object = _MISSING,
    ) -> V.IntegerSequence:
        v = self._conf_get(key, default)
        return V.check_pos_integers(v, length, key)

    @T.overload
    def _conf_nonneg_integers(self, key: str, length: int | None = None) -> V.IntegerSequence: ...

    @T.overload
    def _conf_nonneg_integers(
        self,
        key: str,
        length: int | None,
        default: V.IntegerSequence,
    ) -> V.IntegerSequence: ...

    def _conf_nonneg_integers(
        self,
        key: str,
        length: int | None = None,
        default: object = _MISSING,
    ) -> V.IntegerSequence:
        v = self._conf_get(key, default)
        return V.check_nonneg_integers(v, length, key)

    @T.overload
    def _conf_numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @T.overload
    def _conf_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def _conf_numbers(
        self,
        key: str,
        length: int | None = None,
        default: object = _MISSING,
    ) -> V.NumberSequence:
        v = self._conf_get(key, default)
        return V.check_numbers(v, length, key)

    @T.overload
    def _conf_pos_numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @T.overload
    def _conf_pos_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def _conf_pos_numbers(
        self,
        key: str,
        length: int | None = None,
        default: object = _MISSING,
    ) -> V.NumberSequence:
        v = self._conf_get(key, default)
        return V.check_pos_numbers(v, length, key)

    @T.overload
    def _conf_nonneg_numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @T.overload
    def _conf_nonneg_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def _conf_nonneg_numbers(
        self,
        key: str,
        length: int | None = None,
        default: object = _MISSING,
    ) -> V.NumberSequence:
        v = self._conf_get(key, default)
        return V.check_nonneg_numbers(v, length, key)

    @T.overload
    def _conf_ascending_numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @T.overload
    def _conf_ascending_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def _conf_ascending_numbers(
        self,
        key: str,
        length: int | None = None,
        default: object = _MISSING,
    ) -> V.NumberSequence:
        v = self._conf_get(key, default)
        return V.check_ascending_numbers(v, length, key)

    @T.overload
    def _conf_descending_numbers(
        self, key: str, length: int | None = None
    ) -> V.NumberSequence: ...

    @T.overload
    def _conf_descending_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def _conf_descending_numbers(
        self,
        key: str,
        length: int | None = None,
        default: object = _MISSING,
    ) -> V.NumberSequence:
        v = self._conf_get(key, default)
        return V.check_descending_numbers(v, length, key)


class PresetLoader(object):
    class Mode(enum.Enum):
        EXACT = 0
        PARTIAL = 1
        FORWARD = 2
        BACKWARD = 3

    def __init__(self, logger, mode: Mode = Mode.EXACT):
        self.logger = logger
        self.mode = mode
        self.presets = {}

    def add_preset(self, name: str, preset: list[tuple[str, T.Any]]):
        self.presets[name] = preset

    def load_or_warn(self, conf, key, value):
        if key in conf:
            msg = f"conf[{key}] = {conf[key]} exists. Not loading preset value {value}."
            self.logger.warn(msg)
        else:
            conf[key] = value
            self.logger.debug(f"Load conf[{key}] = {conf[key]}")

    def search_preset(self, name: str):
        for key in self.presets:
            if (
                (self.mode == self.Mode.EXACT and name == key)
                or (self.mode == self.Mode.PARTIAL and key in name)
                or (self.mode == self.Mode.FORWARD and name.startswith(key))
                or (self.mode == self.Mode.BACKWARD and name.endswith(key))
            ):
                return key
        return None

    def load_preset(self, conf: dict, name: str):
        n = self.search_preset(name)
        if n is None:
            self.logger.warn(f"Cannot load preset due to unknown name {name}.")
            return
        self.logger.info(f"Loading preset {n}.")
        for key, value in self.presets[n]:
            self.load_or_warn(conf, key, value)
