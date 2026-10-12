#!/usr/bin/env python3

"""
Parameter validation and access utilities.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, TypeVar, overload

import mahos.util.validation as V


_MISSING = object()
_SequenceT = TypeVar("_SequenceT")


class ParamError(V.ValidationError):
    """Error raised for a missing or invalid parameter."""


class ParamAccessor:
    """Provide validated access to an unwrapped parameter mapping."""

    def __init__(self, params: Mapping[str, Any], prefix: str = ""):
        self._params = params
        self._prefix = prefix

    def _path(self, key: str) -> str:
        return f"{self._prefix}.{key}" if self._prefix else key

    def _get(self, key: str, default: object = _MISSING) -> tuple[str, Any]:
        path = self._path(key)
        if key in self._params:
            return path, self._params[key]
        if default is not _MISSING:
            return path, default
        raise ParamError(f"Required parameter {path} is missing.")

    def _raise_invalid(self, path: str, expected: str, value: Any) -> None:
        raise ParamError(f"{path} must be {expected}. Got {type(value).__name__}: {value!r}")

    def _validate(self, validator: Callable[[Any, str], Any], value: Any, path: str) -> Any:
        try:
            return validator(value, path)
        except V.ValidationError as e:
            raise ParamError(str(e)) from None

    def _sequence(
        self,
        validator: Callable[[Any, int | None, str], _SequenceT],
        key: str,
        length: int | None,
        default: object = _MISSING,
    ) -> _SequenceT:
        path, value = self._get(key, default)
        try:
            return validator(value, length, path)
        except V.ValidationError as e:
            raise ParamError(str(e)) from None

    def __contains__(self, key: object) -> bool:
        return key in self._params

    def get(self, key: str, default: object = None) -> Any:
        """Return an unvalidated parameter value."""

        return self._params.get(key, default)

    @overload
    def bool(self, key: str) -> V.Boolean: ...

    @overload
    def bool(self, key: str, default: V.Boolean) -> V.Boolean: ...

    def bool(self, key: str, default: object = _MISSING) -> V.Boolean:
        """Return a boolean parameter."""

        path, value = self._get(key, default)
        return self._validate(V.check_bool, value, path)

    @overload
    def str(self, key: str) -> str: ...

    @overload
    def str(self, key: str, default: str) -> str: ...

    def str(self, key: str, default: object = _MISSING) -> str:
        """Return a string parameter."""

        path, value = self._get(key, default)
        return self._validate(V.check_str, value, path)

    @overload
    def nonempty_str(self, key: str) -> str: ...

    @overload
    def nonempty_str(self, key: str, default: str) -> str: ...

    def nonempty_str(self, key: str, default: object = _MISSING) -> str:
        """Return a non-empty string parameter."""

        path, value = self._get(key, default)
        return self._validate(V.check_nonempty_str, value, path)

    @overload
    def int(self, key: str) -> V.Integer: ...

    @overload
    def int(self, key: str, default: V.Integer) -> V.Integer: ...

    def int(self, key: str, default: object = _MISSING) -> V.Integer:
        """Return an integer parameter, rejecting booleans."""

        path, value = self._get(key, default)
        return self._validate(V.check_int, value, path)

    @overload
    def pos_int(self, key: str) -> V.Integer: ...

    @overload
    def pos_int(self, key: str, default: V.Integer) -> V.Integer: ...

    def pos_int(self, key: str, default: object = _MISSING) -> V.Integer:
        """Return a positive integer parameter, rejecting booleans."""

        path, value = self._get(key, default)
        return self._validate(V.check_pos_int, value, path)

    @overload
    def nonneg_int(self, key: str) -> V.Integer: ...

    @overload
    def nonneg_int(self, key: str, default: V.Integer) -> V.Integer: ...

    def nonneg_int(self, key: str, default: object = _MISSING) -> V.Integer:
        """Return a non-negative integer parameter, rejecting booleans."""

        path, value = self._get(key, default)
        return self._validate(V.check_nonneg_int, value, path)

    @overload
    def float(self, key: str) -> V.Float: ...

    @overload
    def float(self, key: str, default: V.Float) -> V.Float: ...

    def float(self, key: str, default: object = _MISSING) -> V.Float:
        """Return a finite floating-point parameter."""

        path, value = self._get(key, default)
        return self._validate(V.check_float, value, path)

    @overload
    def pos_float(self, key: str) -> V.Float: ...

    @overload
    def pos_float(self, key: str, default: V.Float) -> V.Float: ...

    def pos_float(self, key: str, default: object = _MISSING) -> V.Float:
        """Return a positive finite floating-point parameter."""

        path, value = self._get(key, default)
        return self._validate(V.check_pos_float, value, path)

    @overload
    def nonneg_float(self, key: str) -> V.Float: ...

    @overload
    def nonneg_float(self, key: str, default: V.Float) -> V.Float: ...

    def nonneg_float(self, key: str, default: object = _MISSING) -> V.Float:
        """Return a non-negative finite floating-point parameter."""

        path, value = self._get(key, default)
        return self._validate(V.check_nonneg_float, value, path)

    @overload
    def num(self, key: str) -> V.Number: ...

    @overload
    def num(self, key: str, default: V.Number) -> V.Number: ...

    def num(self, key: str, default: object = _MISSING) -> V.Number:
        """Return a finite numeric parameter, rejecting booleans."""

        path, value = self._get(key, default)
        return self._validate(V.check_num, value, path)

    @overload
    def pos_num(self, key: str) -> V.Number: ...

    @overload
    def pos_num(self, key: str, default: V.Number) -> V.Number: ...

    def pos_num(self, key: str, default: object = _MISSING) -> V.Number:
        """Return a positive finite numeric parameter, rejecting booleans."""

        path, value = self._get(key, default)
        return self._validate(V.check_pos_num, value, path)

    @overload
    def nonneg_num(self, key: str) -> V.Number: ...

    @overload
    def nonneg_num(self, key: str, default: V.Number) -> V.Number: ...

    def nonneg_num(self, key: str, default: object = _MISSING) -> V.Number:
        """Return a non-negative finite numeric parameter, rejecting booleans."""

        path, value = self._get(key, default)
        return self._validate(V.check_nonneg_num, value, path)

    @overload
    def integers(self, key: str, length: int | None = None) -> V.IntegerSequence: ...

    @overload
    def integers(
        self,
        key: str,
        length: int | None,
        default: V.IntegerSequence,
    ) -> V.IntegerSequence: ...

    def integers(
        self, key: str, length: int | None = None, default: object = _MISSING
    ) -> V.IntegerSequence:
        """Return a list or tuple of integers without conversion, rejecting booleans."""

        return self._sequence(V.check_integers, key, length, default)

    @overload
    def pos_integers(self, key: str, length: int | None = None) -> V.IntegerSequence: ...

    @overload
    def pos_integers(
        self,
        key: str,
        length: int | None,
        default: V.IntegerSequence,
    ) -> V.IntegerSequence: ...

    def pos_integers(
        self, key: str, length: int | None = None, default: object = _MISSING
    ) -> V.IntegerSequence:
        """Return positive integers without conversion, rejecting booleans."""

        return self._sequence(V.check_pos_integers, key, length, default)

    @overload
    def nonneg_integers(self, key: str, length: int | None = None) -> V.IntegerSequence: ...

    @overload
    def nonneg_integers(
        self,
        key: str,
        length: int | None,
        default: V.IntegerSequence,
    ) -> V.IntegerSequence: ...

    def nonneg_integers(
        self, key: str, length: int | None = None, default: object = _MISSING
    ) -> V.IntegerSequence:
        """Return non-negative integers without conversion, rejecting booleans."""

        return self._sequence(V.check_nonneg_integers, key, length, default)

    @overload
    def numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @overload
    def numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def numbers(
        self, key: str, length: int | None = None, default: object = _MISSING
    ) -> V.NumberSequence:
        """Return a list or tuple of finite numbers without conversion."""

        return self._sequence(V.check_numbers, key, length, default)

    @overload
    def pos_numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @overload
    def pos_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def pos_numbers(
        self, key: str, length: int | None = None, default: object = _MISSING
    ) -> V.NumberSequence:
        """Return positive finite numbers without conversion, rejecting booleans."""

        return self._sequence(V.check_pos_numbers, key, length, default)

    @overload
    def nonneg_numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @overload
    def nonneg_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def nonneg_numbers(
        self, key: str, length: int | None = None, default: object = _MISSING
    ) -> V.NumberSequence:
        """Return non-negative finite numbers without conversion, rejecting booleans."""

        return self._sequence(V.check_nonneg_numbers, key, length, default)

    @overload
    def ascending_numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @overload
    def ascending_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def ascending_numbers(
        self, key: str, length: int | None = None, default: object = _MISSING
    ) -> V.NumberSequence:
        """Return strictly ascending finite numbers without conversion."""

        return self._sequence(V.check_ascending_numbers, key, length, default)

    @overload
    def descending_numbers(self, key: str, length: int | None = None) -> V.NumberSequence: ...

    @overload
    def descending_numbers(
        self,
        key: str,
        length: int | None,
        default: V.NumberSequence,
    ) -> V.NumberSequence: ...

    def descending_numbers(
        self, key: str, length: int | None = None, default: object = _MISSING
    ) -> V.NumberSequence:
        """Return strictly descending finite numbers without conversion."""

        return self._sequence(V.check_descending_numbers, key, length, default)

    @overload
    def child(self, key: str) -> "ParamAccessor": ...

    @overload
    def child(self, key: str, default: Mapping[str, Any]) -> "ParamAccessor": ...

    def child(self, key: str, default: object = _MISSING) -> "ParamAccessor":
        """Return an accessor for a nested mapping, preserving its full path."""

        path, value = self._get(key, default)
        if not isinstance(value, Mapping):
            self._raise_invalid(path, "mapping", value)
        return ParamAccessor(value, path)
