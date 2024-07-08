#!/usr/bin/env python3

"""
Qt signal-based client of Recorder.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

from .client import QBasicMeasClient
from ..msgs.recorder_msgs import ResetReq


class QRecorderClient(QBasicMeasClient):
    """Qt-based client for Recorder."""

    def reset(self, label: str) -> bool:
        res = self.req.request(ResetReq(label))
        return res.success
