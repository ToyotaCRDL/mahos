#!/usr/bin/env python3

"""Tests for Confocal shared-file transport."""

from mahos.msgs.common_msgs import Reply
from mahos.node.log import DummyLogger
from mahos.util.file_transport import SharedFileTransport
from mahos_dq.meas.confocal import Confocal, TraceNode
from mahos_dq.meas.confocal_req import (
    ConfocalImageReqMixin,
    ConfocalTraceReqMixin,
    ConfocalTrackerReqMixin,
)
from mahos_dq.meas.confocal_tracker import ConfocalTracker
from mahos_dq.msgs.confocal_msgs import (
    ExportImageReq,
    ExportTraceReq,
    ExportViewReq,
    LoadImageReq,
    LoadTraceReq,
    SaveImageReq,
    SaveTraceReq,
    ScanDirection,
)
from mahos_dq.msgs.confocal_tracker_msgs import LoadParamsReq, SaveParamsReq


class _Requester:
    def __init__(self, handler):
        self.handler = handler
        self.messages = []

    def request(self, msg):
        self.messages.append(msg)
        return self.handler(msg)


class _ConfocalTransportClient(ConfocalImageReqMixin, ConfocalTraceReqMixin):
    pass


class _TrackerTransportClient(ConfocalTrackerReqMixin):
    pass


def test_confocal_client_legacy_requests_without_transport():
    client = _ConfocalTransportClient()
    client.file_transport = None
    client.req = _Requester(lambda msg: Reply(True, ret="loaded"))

    assert client.save_image("/node/image.h5", ScanDirection.XY, note="image")
    assert client.export_image("/node/image.png")
    assert client.export_view("/node/view.png")
    assert client.load_image("/node/image.h5") == "loaded"
    assert client.save_trace("/node/trace.h5", note="trace")
    assert client.export_trace("/node/trace.png")
    assert client.load_trace("/node/trace.h5") == "loaded"
    assert [type(msg) for msg in client.req.messages] == [
        SaveImageReq,
        ExportImageReq,
        ExportViewReq,
        LoadImageReq,
        SaveTraceReq,
        ExportTraceReq,
        LoadTraceReq,
    ]


def test_confocal_transport_roundtrip_with_shared_directory(tmp_path):
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    shared.mkdir()
    local.mkdir()

    calls = []
    node = object.__new__(Confocal)
    node._closed = True
    node.file_transport = SharedFileTransport(shared)
    node._file_transport_output_purposes = ("save", "export")
    node.logger = DummyLogger("test_confocal_transport")

    def output(msg):
        calls.append(msg)
        with open(msg.file_name, "wb") as f:
            f.write(type(msg).__name__.encode())
        return Reply(True)

    def load(msg):
        calls.append(msg)
        with open(msg.file_name, "rb") as f:
            return Reply(True, ret=f.read())

    node.save_image = output
    node.export_image = output
    node.export_view = output
    node.load_image = load
    node.save_trace = output
    node.export_trace = output
    node.load_trace = load

    client = _ConfocalTransportClient()
    client.file_transport = SharedFileTransport(shared)
    client.req = _Requester(node.handle_req)

    assert not client.save_image(str(local / "missing" / "image.h5"))
    assert list(shared.iterdir()) == []

    image = local / "chosen.image.h5"
    assert client.save_image(str(image), ScanDirection.XZ, note="image note")
    assert image.read_bytes() == b"SaveImageReq"
    assert calls[-1].direction == ScanDirection.XZ
    assert calls[-1].note == "image note"

    image_plot = local / "image.png"
    image_params = {"dpi": 120}
    assert client.export_image(str(image_plot), ScanDirection.YZ, image_params)
    assert image_plot.read_bytes() == b"ExportImageReq"
    assert calls[-1].params == image_params

    view = local / "view.svg"
    assert client.export_view(str(view), {"color": "red"})
    assert view.read_bytes() == b"ExportViewReq"

    image_source = local / "source.image.h5"
    image_source.write_bytes(b"image input")
    assert client.load_image(str(image_source)) == b"image input"

    trace = local / "chosen.trace.h5"
    assert client.save_trace(str(trace), note="trace note")
    assert trace.read_bytes() == b"SaveTraceReq"
    assert calls[-1].note == "trace note"

    trace_plot = local / "trace.png"
    trace_params = {"color": "blue"}
    assert client.export_trace(str(trace_plot), trace_params)
    assert trace_plot.read_bytes() == b"ExportTraceReq"
    assert calls[-1].params == trace_params

    trace_source = local / "source.trace.h5"
    trace_source.write_bytes(b"trace input")
    assert client.load_trace(str(trace_source)) == b"trace input"
    assert list(shared.iterdir()) == []


def test_confocal_artifact_failure_cleanup(tmp_path):
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    shared.mkdir()
    local.mkdir()

    node = object.__new__(Confocal)
    node._closed = True
    node.file_transport = SharedFileTransport(shared)
    node._file_transport_output_purposes = ("save", "export")
    node.logger = DummyLogger("test_confocal_transport_failure")
    node.save_image = lambda msg: Reply(False, "no image")

    client = _ConfocalTransportClient()
    client.file_transport = SharedFileTransport(shared)
    client.req = _Requester(node.handle_req)

    assert not client.save_image(str(local / "image.h5"))
    assert list(shared.iterdir()) == []


def test_trace_node_transport_handlers(tmp_path):
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    shared.mkdir()
    local.mkdir()

    node = object.__new__(TraceNode)
    node._closed = True
    node.file_transport = SharedFileTransport(shared)
    node._file_transport_output_purposes = ("save", "export")
    node.logger = DummyLogger("test_trace_node_transport")

    def output(msg):
        with open(msg.file_name, "wb") as f:
            f.write(type(msg).__name__.encode())
        return Reply(True)

    def load(msg):
        with open(msg.file_name, "rb") as f:
            return Reply(True, ret=f.read())

    node.save_trace = output
    node.export_trace = output
    node.load_trace = load

    client = _ConfocalTransportClient()
    client.file_transport = SharedFileTransport(shared)
    client.req = _Requester(node.handle_req)

    saved = local / "trace.h5"
    exported = local / "trace.png"
    source = local / "source.h5"
    source.write_bytes(b"trace input")
    assert client.save_trace(str(saved))
    assert saved.read_bytes() == b"SaveTraceReq"
    assert client.export_trace(str(exported))
    assert exported.read_bytes() == b"ExportTraceReq"
    assert client.load_trace(str(source)) == b"trace input"
    assert list(shared.iterdir()) == []


def test_tracker_transport_explicit_and_default_paths(tmp_path):
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    shared.mkdir()
    local.mkdir()

    node = object.__new__(ConfocalTracker)
    node._closed = True
    node.file_transport = SharedFileTransport(shared)
    node._file_transport_output_purposes = ("save",)
    node.logger = DummyLogger("test_confocal_tracker_transport")

    client = _TrackerTransportClient()
    client.file_transport = SharedFileTransport(shared)
    client.req = _Requester(node.handle_req)

    params = {"interval_sec": 1.0, "mode": "test"}
    destination = local / "tracker.params.pkl"
    assert client.save_params(params, str(destination))
    assert client.load_params(str(destination)) == params
    assert list(shared.iterdir()) == []

    recorder = _Requester(lambda msg: Reply(True, ret=params))
    client.req = recorder
    assert client.save_params(params)
    assert client.load_params() == params
    assert [type(msg) for msg in recorder.messages] == [SaveParamsReq, LoadParamsReq]
