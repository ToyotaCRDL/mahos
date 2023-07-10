#!/usr/bin/env python3

import pytest

from mahos.msgs.param_msgs import FloatParam, IntParam, StrParam
from mahos.msgs.param_msgs import ParamDict, isclose

valid0 = ParamDict({"a": IntParam(123), "b": FloatParam(2.0), "c": StrParam("aaa")})

valid1 = ParamDict(
    {
        "a": IntParam(123),
        "b": FloatParam(2.5),
        "c": StrParam("aaa"),
        "d": {"d-a": [IntParam(1234), FloatParam(3.0)]},
        "e": [{"e_a": StrParam("bbb"), "e_b": IntParam(2)}, FloatParam(1.0)],
    }
)

valid1_nest = ParamDict(
    {
        "a": IntParam(123),
        "b": FloatParam(2.5),
        "c": StrParam("aaa"),
        "d": ParamDict({"d-a": [IntParam(1234), FloatParam(3.0)]}),
        "e": [ParamDict({"e_a": StrParam("bbb"), "e_b": IntParam(2)}), FloatParam(1.0)],
    }
)

valid1_close = ParamDict(
    {
        "a": IntParam(123),
        "b": FloatParam(2.5000000001),
        "c": StrParam("aaa"),
        "d": {"d-a": [IntParam(1234), FloatParam(2.9999999999)]},
        "e": [{"e_a": StrParam("bbb"), "e_b": IntParam(2)}, FloatParam(1.0)],
    }
)
valid1_close_more = ParamDict(
    {
        "a": IntParam(123),
        "b": FloatParam(2.5000000001),
        "c": StrParam("aaa"),
        "d": {"d-a": [IntParam(1234), FloatParam(2.9999999999)]},
        "e": [{"e_a": StrParam("bbb"), "e_b": IntParam(2)}, FloatParam(1.0)],
        "extra": StrParam("extra"),
    }
)

invalid_key_should_be_str = ParamDict({1: FloatParam(2.0)})
invalid_value_should_be_param = ParamDict({"a": FloatParam(2.0), "b": 123.4})
invalid1 = ParamDict(
    {
        "a": 123,
        "b": FloatParam(2.5),
        "c": StrParam("aaa"),
        "d": {"d-a": [1234, FloatParam(3.0)]},
        "e": [{"e_a": StrParam("bbb"), "e_b": IntParam(2)}, FloatParam(1.0)],
    }
)


def test_param():
    assert IntParam(1).value() == 1
    assert IntParam(1, optional=True, enable=False).value() is None
    assert IntParam(1, doc="this is int").doc() == "this is int"


def test_param_dict_validation():
    assert valid0.validate()
    assert valid1.validate()
    assert valid1_nest.validate()
    assert not invalid_key_should_be_str.validate()
    assert not invalid_value_should_be_param.validate()


def test_param_dict_unwrap():
    unwrapped_valid0 = valid0.unwrap()
    assert isinstance(unwrapped_valid0, dict)
    assert unwrapped_valid0["a"] == 123
    assert unwrapped_valid0["c"] == "aaa"

    unwrapped_invalid1 = invalid1.unwrap()
    assert unwrapped_invalid1["a"] == 123
    assert unwrapped_invalid1["d"]["d-a"][0] == 1234

    with pytest.raises(TypeError):
        invalid1.unwrap_exn()


def test_param_dict_isclose():
    assert valid1.isclose(valid1_close)
    assert isclose(valid1.unwrap(), valid1_nest.unwrap())
    # contains different type (ParamDict vs dict)
    assert not valid1.isclose(valid1_nest)
    assert isclose(valid1.unwrap(), valid1_close.unwrap())
    assert not valid1.isclose(valid1_close_more)
    assert not valid1_close_more.isclose(valid1_close)
    assert not isclose(valid1, valid1_close.unwrap())
    assert not isclose(valid1, 123.4)
    assert not isclose(valid1, None)
    assert not isclose(12, None)


def test_param_dict_flatten():
    assert valid1["d"]["d-a"][1] == valid1.getf("d.d-a[1]")
    assert valid1.flatten()["d.d-a[1]"] == valid1.getf("d.d-a[1]")
    assert valid1.flatten()["e[0].e_a"] == valid1.getf("e[0].e_a")
