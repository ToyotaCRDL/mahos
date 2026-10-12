#!/usr/bin/env python3

"""
Deprecated compatibility module for :mod:`mahos.util.queue`.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import warnings

from mahos.util.queue import RollingQueue


warnings.warn(
    "mahos.util.locked_queue is deprecated; use mahos.util.queue.RollingQueue instead.",
    DeprecationWarning,
    stacklevel=2,
)


class LockedQueue(RollingQueue):
    def pop_block(self, timeout_sec: float | None = None, interval_sec=0.001):
        return RollingQueue.pop_block(self, timeout_sec)

    def pop_all_block(self, timeout_sec: float | None = None, interval_sec=0.001):
        return RollingQueue.pop_all_block(self, timeout_sec)
