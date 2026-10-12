#!/usr/bin/env python3

"""
Generic value validation utilities.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeVar

import numpy as np


Boolean = bool | np.bool_
Integer = int | np.integer
Float = float | np.floating
Number = Integer | Float
IntegerSequence = list[Integer] | tuple[Integer, ...]
NumberSequence = list[Number] | tuple[Number, ...]
_SequenceT = TypeVar("_SequenceT", IntegerSequence, NumberSequence)


class ValidationError(ValueError):
    """Error raised when a value fails validation."""


def _is_num(value) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and (not isinstance(value, (float, np.floating)) or math.isfinite(value))
    )


def _raise_invalid(value, name: str, expected: str, subject: str) -> None:
    if name:
        raise ValidationError(f"{name} must be {expected}. Got {type(value).__name__}: {value!r}")
    raise ValidationError(
        f"{subject} is invalid. Expected {expected}. Got {type(value).__name__}: {value!r}"
    )


def check_bool(value: Boolean, name: str = "") -> Boolean:
    """Return a boolean value without conversion, or raise :class:`ValidationError`."""

    if not isinstance(value, (bool, np.bool_)):
        _raise_invalid(value, name, "bool", "A boolean value")
    return value


def check_str(value: str, name: str = "") -> str:
    """Return a string value without conversion, or raise :class:`ValidationError`."""

    if not isinstance(value, str):
        _raise_invalid(value, name, "str", "A string value")
    return value


def check_nonempty_str(value: str, name: str = "") -> str:
    """Return a non-empty string without conversion, or raise :class:`ValidationError`."""

    if not isinstance(value, str) or not value:
        _raise_invalid(value, name, "non-empty str", "A string value")
    return value


def check_int(value: Integer, name: str = "") -> Integer:
    """Return an integer value without conversion, rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _raise_invalid(value, name, "int", "An integer value")
    return value


def check_pos_int(value: Integer, name: str = "") -> Integer:
    """Return a positive integer value without conversion, rejecting booleans."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        _raise_invalid(value, name, "positive int", "An integer value")
    return value


def check_nonneg_int(value: Integer, name: str = "") -> Integer:
    """Return a non-negative integer value without conversion, rejecting booleans."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        _raise_invalid(value, name, "non-negative int", "An integer value")
    return value


def check_float(value: Float, name: str = "") -> Float:
    """Return a finite floating-point value without conversion."""

    if not isinstance(value, (float, np.floating)) or not math.isfinite(value):
        _raise_invalid(value, name, "finite float", "A floating-point value")
    return value


def check_pos_float(value: Float, name: str = "") -> Float:
    """Return a positive finite floating-point value without conversion."""

    if not isinstance(value, (float, np.floating)) or not math.isfinite(value) or value <= 0.0:
        _raise_invalid(value, name, "positive finite float", "A floating-point value")
    return value


def check_nonneg_float(value: Float, name: str = "") -> Float:
    """Return a non-negative finite floating-point value without conversion."""

    if not isinstance(value, (float, np.floating)) or not math.isfinite(value) or value < 0.0:
        _raise_invalid(value, name, "non-negative finite float", "A floating-point value")
    return value


def check_num(value: Number, name: str = "") -> Number:
    """Return a finite numeric value without conversion, rejecting booleans."""

    if not _is_num(value):
        _raise_invalid(value, name, "finite number", "A numeric value")
    return value


def check_pos_num(value: Number, name: str = "") -> Number:
    """Return a positive finite numeric value without conversion, rejecting booleans."""

    if not _is_num(value) or value <= 0:
        _raise_invalid(value, name, "positive finite number", "A numeric value")
    return value


def check_nonneg_num(value: Number, name: str = "") -> Number:
    """Return a non-negative finite numeric value without conversion, rejecting booleans."""

    if not _is_num(value) or value < 0:
        _raise_invalid(value, name, "non-negative finite number", "A numeric value")
    return value


def _check_sequence(
    value: _SequenceT,
    length: int | None,
    name: str,
    element: str,
    checker: Callable,
) -> _SequenceT:
    expected = f"list or tuple of {element}"
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError(f"length must be positive. Got {length}")
        expected = f"list or tuple of {length} {element}"
    if not isinstance(value, (list, tuple)) or length is not None and len(value) != length:
        _raise_invalid(value, name, expected, "A numeric sequence")
    for i, elem in enumerate(value):
        checker(elem, f"{name}[{i}]" if name else f"Element {i}")
    return value


def check_integers(
    value: IntegerSequence, length: int | None = None, name: str = ""
) -> IntegerSequence:
    """Return a list or tuple of integers without conversion, rejecting booleans."""

    return _check_sequence(value, length, name, "integers", check_int)


def check_pos_integers(
    value: IntegerSequence, length: int | None = None, name: str = ""
) -> IntegerSequence:
    """Return a list or tuple of positive integers without conversion, rejecting booleans."""

    return _check_sequence(value, length, name, "positive integers", check_pos_int)


def check_nonneg_integers(
    value: IntegerSequence, length: int | None = None, name: str = ""
) -> IntegerSequence:
    """Return a list or tuple of non-negative integers without conversion, rejecting booleans."""

    return _check_sequence(value, length, name, "non-negative integers", check_nonneg_int)


def check_numbers(
    value: NumberSequence, length: int | None = None, name: str = ""
) -> NumberSequence:
    """Return a list or tuple of finite numbers without conversion."""

    return _check_sequence(value, length, name, "numbers", check_num)


def check_pos_numbers(
    value: NumberSequence, length: int | None = None, name: str = ""
) -> NumberSequence:
    """Return a list or tuple of positive finite numbers without conversion."""

    return _check_sequence(value, length, name, "positive numbers", check_pos_num)


def check_nonneg_numbers(
    value: NumberSequence, length: int | None = None, name: str = ""
) -> NumberSequence:
    """Return a list or tuple of non-negative finite numbers without conversion."""

    return _check_sequence(value, length, name, "non-negative numbers", check_nonneg_num)


def check_ascending_numbers(
    value: NumberSequence, length: int | None = None, name: str = ""
) -> NumberSequence:
    """Return strictly ascending finite numbers without conversion."""

    numbers = check_numbers(value, length, name)
    if any(a >= b for a, b in zip(numbers, numbers[1:])):
        _raise_invalid(value, name, "strictly ascending numbers", "A numeric sequence")
    return value


def check_descending_numbers(
    value: NumberSequence, length: int | None = None, name: str = ""
) -> NumberSequence:
    """Return strictly descending finite numbers without conversion."""

    numbers = check_numbers(value, length, name)
    if any(a <= b for a, b in zip(numbers, numbers[1:])):
        _raise_invalid(value, name, "strictly descending numbers", "A numeric sequence")
    return value
