#!/usr/bin/env python3

"""
Tests for mahos.util.param.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import numpy as np
import pytest

from mahos.util.param import ParamAccessor, ParamError
from mahos.util.validation import ValidationError


def test_param_accessor():
    params = {
        "name": "test",
        "empty_name": "",
        "enabled": True,
        "count": 2,
        "offset": 0,
        "rate": 1.5,
        "zero_float": 0.0,
        "nested": {},
        "integers": [0, np.int32(1)],
        "positive_integers": (1, np.int64(2)),
        "positive_numbers": [1, np.float32(2.0)],
        "numbers": [1, 2.0],
        "bounds": [-1.0, 1.0],
        "levels": (3, 2, 1),
    }
    p = ParamAccessor(params)

    assert p.str("name") == "test"
    assert p.nonempty_str("name") == "test"
    assert p.bool("enabled") is True
    assert p.int("count") == 2
    assert p.pos_int("count") == 2
    assert p.nonneg_int("offset") == 0
    assert p.float("rate") == 1.5
    assert p.pos_float("rate") == 1.5
    assert p.nonneg_float("zero_float") == 0.0
    assert p.num("rate") == 1.5
    assert p.pos_num("rate") == 1.5
    assert p.nonneg_num("offset") == 0
    assert "nested" in p
    assert p.get("nested") == {}
    assert p.integers("integers") is params["integers"]
    assert p.pos_integers("positive_integers") is params["positive_integers"]
    assert p.nonneg_integers("integers") is params["integers"]
    assert p.numbers("numbers") is params["numbers"]
    assert p.pos_numbers("positive_numbers") is params["positive_numbers"]
    assert p.nonneg_numbers("numbers") is params["numbers"]
    assert p.ascending_numbers("bounds") is params["bounds"]
    assert p.descending_numbers("levels") is params["levels"]
    assert p.int("missing", 3) == 3
    default_numbers = [1, 2]
    assert p.numbers("missing_numbers", None, default_numbers) is default_numbers

    with pytest.raises(ParamError, match="Required parameter missing is missing"):
        p.int("missing")
    with pytest.raises(ParamError, match="empty_name must be non-empty str"):
        p.nonempty_str("empty_name")
    with pytest.raises(ParamError, match="strictly ascending"):
        p.ascending_numbers("levels", 3)
    with pytest.raises(ParamError, match=r"bounds\[1\] must be finite number"):
        ParamAccessor({"bounds": [0.0, float("inf")]}).ascending_numbers("bounds", 2)
    with pytest.raises(ParamError, match="list or tuple of 3 numbers"):
        p.numbers("numbers", 3)


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("num", True),
        ("num", float("nan")),
        ("num", float("inf")),
        ("num", -float("inf")),
        ("pos_num", 0.0),
        ("nonneg_num", -1.0),
        ("pos_int", 0),
        ("nonneg_int", -1),
        ("float", 1),
        ("float", float("nan")),
        ("float", float("inf")),
        ("pos_float", 0.0),
        ("nonneg_float", -1.0),
        ("integers", [1, 2.0]),
        ("pos_integers", [1, 0]),
        ("nonneg_integers", [0, -1]),
        ("pos_numbers", [1.0, 0]),
        ("nonneg_numbers", [0, -1.0]),
    ],
)
def test_param_accessor_rejects_invalid_numbers(method, value):
    with pytest.raises(ParamError):
        getattr(ParamAccessor({"value": value}), method)("value")


def test_param_accessor_nested_path():
    p = ParamAccessor({"timing": {"trigger": {"width": -1.0}}})

    trigger = p.child("timing").child("trigger")
    assert trigger.num("width") == -1.0
    with pytest.raises(ParamError, match=r"timing\.trigger\.width must be positive"):
        trigger.pos_num("width")


def test_param_accessor_preserves_numpy_scalar():
    value = np.float32(1.0)

    assert ParamAccessor({"value": value}).float("value") is value
    assert ParamAccessor({"value": value}).pos_num("value") is value
    assert issubclass(ParamError, ValidationError)


def test_param_accessor_float_default():
    value = np.float64(1.0)

    assert ParamAccessor({}).pos_float("value", value) is value
