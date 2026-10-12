#!/usr/bin/env python3

"""Transport-aware request mixins for Confocal and Trace clients.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import os

from mahos.meas.file_transport import FileTransportClientMixin
from mahos_dq.msgs.confocal_msgs import (
    ExportImageArtifactReq,
    ExportImageReq,
    ExportTraceArtifactReq,
    ExportTraceReq,
    ExportViewArtifactReq,
    ExportViewReq,
    Image,
    LoadImageArtifactReq,
    LoadImageReq,
    LoadTraceArtifactReq,
    LoadTraceReq,
    SaveImageArtifactReq,
    SaveImageReq,
    SaveTraceArtifactReq,
    SaveTraceReq,
    ScanDirection,
    Trace,
)
from mahos_dq.msgs.confocal_tracker_msgs import (
    LoadParamsArtifactReq,
    LoadParamsReq,
    SaveParamsArtifactReq,
    SaveParamsReq,
)


class ConfocalImageReqMixin(FileTransportClientMixin):
    """Implement transport-aware Confocal image requests."""

    def save_image(
        self, file_name: str, direction: ScanDirection | None = None, note: str = ""
    ) -> bool:
        return self.request_output(
            SaveImageReq(file_name, direction=direction, note=note),
            SaveImageArtifactReq(os.path.basename(file_name), direction=direction, note=note),
            file_name,
        )

    def export_image(
        self, file_name: str, direction: ScanDirection | None = None, params=None
    ) -> bool:
        return self.request_output(
            ExportImageReq(file_name, direction, params),
            ExportImageArtifactReq(os.path.basename(file_name), direction, params),
            file_name,
        )

    def export_view(self, file_name: str, params=None) -> bool:
        return self.request_output(
            ExportViewReq(file_name, params),
            ExportViewArtifactReq(os.path.basename(file_name), params),
            file_name,
        )

    def load_image(self, file_name: str) -> Image | None:
        rep = self.request_input(LoadImageReq(file_name), LoadImageArtifactReq, file_name)
        return rep.ret if rep.success else None


class ConfocalTraceReqMixin(FileTransportClientMixin):
    """Implement transport-aware Confocal trace requests."""

    def save_trace(self, file_name: str, note: str = "") -> bool:
        return self.request_output(
            SaveTraceReq(file_name, note=note),
            SaveTraceArtifactReq(os.path.basename(file_name), note=note),
            file_name,
        )

    def export_trace(self, file_name: str, params=None) -> bool:
        return self.request_output(
            ExportTraceReq(file_name, params=params),
            ExportTraceArtifactReq(os.path.basename(file_name), params=params),
            file_name,
        )

    def load_trace(self, file_name: str) -> Trace | None:
        rep = self.request_input(LoadTraceReq(file_name), LoadTraceArtifactReq, file_name)
        return rep.ret if rep.success else None


class ConfocalTrackerReqMixin(FileTransportClientMixin):
    """Implement transport-aware ConfocalTracker parameter requests."""

    def save_params(self, params: dict, file_name: str | None = None) -> bool:
        if file_name is None or self.file_transport is None:
            return self.req.request(SaveParamsReq(params, file_name=file_name)).success
        return self.request_output(
            SaveParamsReq(params, file_name=file_name),
            SaveParamsArtifactReq(os.path.basename(file_name), params),
            file_name,
        )

    def load_params(self, file_name: str | None = None) -> dict | None:
        if file_name is None or self.file_transport is None:
            rep = self.req.request(LoadParamsReq(file_name))
        else:
            rep = self.request_input(LoadParamsReq(file_name), LoadParamsArtifactReq, file_name)
        return rep.ret if rep.success else None
