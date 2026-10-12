#!/usr/bin/env python3

"""
Tests for mahos.util.conf.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import numpy as np
import pytest

from mahos.util.conf import ConfAccessorMixin, PresetLoader
from mahos.util.validation import ValidationError


class DummyLogger:
    def __init__(self):
        self.messages = []

    def warn(self, msg):
        self.messages.append(("warn", msg))

    def info(self, msg):
        self.messages.append(("info", msg))

    def debug(self, msg):
        self.messages.append(("debug", msg))


class ConfOwner(ConfAccessorMixin):
    def __init__(self, conf):
        self.conf = conf


def test_conf_accessor_mixin():
    owner = ConfOwner({"count": 2, "rate": 1.5, "label": "test", "empty_name": ""})

    assert owner._conf_pos_int("count") == 2
    assert owner._conf_pos_num("rate") == 1.5
    assert owner._conf_nonempty_str("label") == "test"
    assert owner._conf_bool("enabled", True) is True
    with pytest.raises(ValidationError, match="Required configuration name is missing"):
        owner._conf_str("name")

    for method in ("_conf_float", "_conf_pos_float", "_conf_nonneg_float"):
        with pytest.raises(ValueError):
            getattr(ConfOwner({"value": float("nan")}), method)("value", 0.0)
    for method in ("_conf_num", "_conf_pos_num", "_conf_nonneg_num"):
        with pytest.raises(ValueError):
            getattr(ConfOwner({"value": float("inf")}), method)("value", 0.0)

    with pytest.raises(ValidationError, match="count must be int"):
        ConfOwner({"count": "2"})._conf_int("count")
    with pytest.raises(ValidationError, match="empty_name must be non-empty str"):
        owner._conf_nonempty_str("empty_name")


def test_conf_accessor_preserves_numpy_scalar():
    value = np.int32(1)
    assert ConfOwner({"value": value})._conf_pos_int("value") is value


def test_conf_accessor_number_sequences():
    integers = [np.int32(0), 1]
    positive_integers = (np.int64(1), 2)
    numbers = [np.int32(1), 2.0]
    positive_numbers = (np.float32(1.0), 2)
    ascending = [1, np.float32(2.0)]
    descending = (2.0, np.int64(1))
    owner = ConfOwner(
        {
            "integers": integers,
            "positive_integers": positive_integers,
            "numbers": numbers,
            "positive_numbers": positive_numbers,
            "ascending": ascending,
            "descending": descending,
        }
    )

    assert owner._conf_integers("integers", 2) is integers
    assert owner._conf_pos_integers("positive_integers", 2) is positive_integers
    assert owner._conf_nonneg_integers("integers", 2) is integers
    assert owner._conf_numbers("numbers", 2) is numbers
    assert owner._conf_pos_numbers("positive_numbers", 2) is positive_numbers
    assert owner._conf_nonneg_numbers("numbers", 2) is numbers
    assert owner._conf_ascending_numbers("ascending", 2) is ascending
    assert owner._conf_descending_numbers("descending", 2) is descending
    default = [0, 1]
    assert owner._conf_numbers("default", 2, default) is default

    with pytest.raises(ValidationError, match="strictly ascending"):
        ConfOwner({"values": [2, 1]})._conf_ascending_numbers("values", 2)
    with pytest.raises(ValidationError, match=r"values\[1\] must be finite number"):
        ConfOwner({"values": [1, float("inf")]})._conf_numbers("values", 2)
    with pytest.raises(ValidationError, match=r"values\[1\] must be positive int"):
        ConfOwner({"values": [1, 0]})._conf_pos_integers("values", 2)


def test_preset_loader_exact_match():
    logger = DummyLogger()
    loader = PresetLoader(logger)
    loader.add_preset("basic", [("alpha", 1), ("beta", 2)])

    conf = {}
    loader.load_preset(conf, "basic")

    assert conf == {"alpha": 1, "beta": 2}
    assert any(level == "info" for level, _ in logger.messages)


def test_preset_loader_preserves_existing_keys():
    logger = DummyLogger()
    loader = PresetLoader(logger)
    loader.add_preset("basic", [("alpha", 1), ("beta", 2)])

    conf = {"alpha": 10}
    loader.load_preset(conf, "basic")

    assert conf["alpha"] == 10
    assert conf["beta"] == 2
    assert any(level == "warn" for level, _ in logger.messages)


def test_preset_loader_matching_modes():
    logger = DummyLogger()
    loader = PresetLoader(logger, mode=PresetLoader.Mode.PARTIAL)
    loader.add_preset("basic", [("alpha", 1)])
    assert loader.search_preset("basic-v2") == "basic"

    loader = PresetLoader(logger, mode=PresetLoader.Mode.FORWARD)
    loader.add_preset("basic", [("alpha", 1)])
    assert loader.search_preset("basic-v2") == "basic"

    loader = PresetLoader(logger, mode=PresetLoader.Mode.BACKWARD)
    loader.add_preset("v2", [("alpha", 1)])
    assert loader.search_preset("basic-v2") == "v2"


def test_preset_loader_unknown_name():
    logger = DummyLogger()
    loader = PresetLoader(logger)
    loader.add_preset("basic", [("alpha", 1)])

    conf = {}
    loader.load_preset(conf, "missing")

    assert conf == {}
    assert any(level == "warn" for level, _ in logger.messages)
