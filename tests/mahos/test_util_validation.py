#!/usr/bin/env python3

"""
Tests for mahos.util.validation.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import numpy as np
import pytest

import mahos.util.validation as V


@pytest.mark.parametrize(
    ("checker", "value"),
    [
        (V.check_bool, True),
        (V.check_bool, np.bool_(True)),
        (V.check_str, "value"),
        (V.check_nonempty_str, "value"),
        (V.check_int, 1),
        (V.check_int, np.int32(1)),
        (V.check_pos_int, np.int64(1)),
        (V.check_nonneg_int, np.uint32(0)),
        (V.check_float, 1.0),
        (V.check_float, np.float32(1.0)),
        (V.check_pos_float, np.float64(1.0)),
        (V.check_nonneg_float, 0.0),
        (V.check_num, np.int16(-1)),
        (V.check_num, np.float32(-1.0)),
        (V.check_pos_num, np.float64(1.0)),
        (V.check_nonneg_num, np.int32(0)),
    ],
)
def test_check_preserves_value(checker, value):
    assert checker(value) is value


@pytest.mark.parametrize(
    ("checker", "value"),
    [
        (V.check_bool, 1),
        (V.check_str, None),
        (V.check_nonempty_str, ""),
        (V.check_int, True),
        (V.check_int, np.bool_(True)),
        (V.check_pos_int, 0),
        (V.check_nonneg_int, -1),
        (V.check_float, 1),
        (V.check_float, float("nan")),
        (V.check_pos_float, 0.0),
        (V.check_nonneg_float, -1.0),
        (V.check_num, True),
        (V.check_num, np.bool_(False)),
        (V.check_num, float("inf")),
        (V.check_num, 1.0j),
        (V.check_pos_num, 0),
        (V.check_nonneg_num, -1),
    ],
)
def test_check_rejects_invalid_value(checker, value):
    with pytest.raises(V.ValidationError):
        checker(value)


def test_validation_error_message():
    with pytest.raises(
        V.ValidationError,
        match=r"sample_rate must be positive finite number\. Got str: 'fast'",
    ):
        V.check_pos_num("fast", "sample_rate")

    with pytest.raises(V.ValidationError, match="A numeric value is invalid"):
        V.check_num(float("nan"))


@pytest.mark.parametrize(
    ("checker", "value"),
    [
        (V.check_integers, [1, np.int32(2)]),
        (V.check_pos_integers, (1, np.uint32(2))),
        (V.check_nonneg_integers, [np.int64(0), 1]),
        (V.check_numbers, [1, np.float32(2.0)]),
        (V.check_pos_numbers, (1, np.float32(2.0))),
        (V.check_nonneg_numbers, [np.int32(0), 1.0]),
        (V.check_ascending_numbers, (1, np.float32(2.0))),
        (V.check_descending_numbers, [np.int32(2), 1]),
    ],
)
def test_check_number_sequence_preserves_value(checker, value):
    assert checker(value, length=2) is value


@pytest.mark.parametrize(
    ("checker", "value", "match"),
    [
        (V.check_integers, [1, 2.0], r"bounds\[1\] must be int"),
        (V.check_pos_integers, [1, 0], r"bounds\[1\] must be positive int"),
        (V.check_nonneg_integers, [1, -1], r"bounds\[1\] must be non-negative int"),
        (V.check_numbers, [1], "list or tuple of 2 numbers"),
        (V.check_numbers, [1, float("inf")], r"bounds\[1\] must be finite number"),
        (V.check_pos_numbers, [1, 0.0], r"bounds\[1\] must be positive finite number"),
        (
            V.check_nonneg_numbers,
            [1, -1.0],
            r"bounds\[1\] must be non-negative finite number",
        ),
        (V.check_ascending_numbers, [1, 1], "strictly ascending numbers"),
        (V.check_descending_numbers, [1, 2], "strictly descending numbers"),
    ],
)
def test_check_number_sequence_rejects_invalid_value(checker, value, match):
    with pytest.raises(V.ValidationError, match=match):
        checker(value, length=2, name="bounds")


def test_check_numbers_rejects_invalid_length():
    with pytest.raises(ValueError, match="length must be positive"):
        V.check_numbers([], length=0)
