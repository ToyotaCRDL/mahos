#!/usr/bin/env python3

"""
Config utility module.

.. This file is a part of MAHOS project.

Defines utilities for configs.

"""

from __future__ import annotations
import typing as T
import enum


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
