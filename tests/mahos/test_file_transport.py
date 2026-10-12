#!/usr/bin/env python3

"""Tests for shared-file transport primitives."""

import os
import time

import pytest

from mahos.meas.common_meas import BasicMeasNode, BasicMeasReqMixin
from mahos.meas.file_transport import FileTransportNodeMixin
from mahos.msgs.common_msgs import (
    CleanupArtifactReq,
    ExportDataReq,
    ExportArtifactReq,
    FileArtifact,
    FileArtifactBundle,
    LoadDataReq,
    Reply,
    SaveArtifactReq,
    SaveDataReq,
)
from mahos.node.log import DummyLogger
from mahos.util.file_transport import SharedFileTransport


def test_names_resolution_and_compound_suffix(tmp_path):
    transport = SharedFileTransport(tmp_path)
    for purpose, prefix in transport.PREFIXES.items():
        name = transport.new_name(purpose, "data.pkl.bz2")
        assert name.startswith(prefix)
        assert name.endswith(".h5" if purpose == "awg" else ".pkl.bz2")
        assert transport.resolve(name) == str(tmp_path / name)

    for name in ("", ".", "..", "../x", "a/b", r"a\b"):
        with pytest.raises(ValueError):
            transport.resolve(name)


def test_atomic_publication_copy_and_failure_cleanup(tmp_path):
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    shared.mkdir()
    local.mkdir()
    transport = SharedFileTransport(shared)
    name = transport.new_name("save", "result.data.h5")
    seen = []

    def write(path):
        seen.append(path)
        assert os.path.basename(path).startswith(name.removesuffix(".data.h5") + ".staging-")
        assert path.endswith(".data.h5")
        with open(path, "wb") as f:
            f.write(b"transport")

    transport.publish(name, write)
    destination = local / "chosen.data.h5"
    transport.copy_from(name, destination)
    assert destination.read_bytes() == b"transport"
    assert transport.remove(name)
    assert transport.remove(name)

    failed_name = transport.new_name("export", ".png")

    def fail(path):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("failed writer")

    with pytest.raises(RuntimeError):
        transport.publish(failed_name, fail)
    assert list(shared.iterdir()) == []


def test_stage_copy_and_prefix_scoped_cleanup(tmp_path):
    transport = SharedFileTransport(tmp_path)
    source = tmp_path / "source.h5"
    source.write_bytes(b"data")
    load_name = transport.stage_copy(source, "load")
    save_name = transport.new_name("save", ".h5")
    export_name = transport.new_name("export", ".png")
    awg_name = transport.new_name("awg")
    for name in (save_name, export_name, awg_name):
        transport.publish(name, lambda path: open(path, "wb").close())

    old = time.time() - 100.0
    for name in (load_name, save_name, export_name, awg_name):
        os.utime(transport.resolve(name), (old, old))

    removed = transport.cleanup(transport.PREFIXES["save"], ttl=10.0)
    assert removed == [save_name]
    assert not os.path.exists(transport.resolve(save_name))
    for name in (load_name, export_name, awg_name):
        assert os.path.exists(transport.resolve(name))


def test_cleanup_includes_interrupted_staging_file(tmp_path):
    transport = SharedFileTransport(tmp_path)
    name = transport.new_name("save", "result.data.h5")
    staging_path = transport._staging_path(transport.resolve(name))
    with open(staging_path, "wb") as f:
        f.write(b"partial")
    os.utime(staging_path, (0.0, 0.0))

    staging_name = os.path.basename(staging_path)
    assert staging_name.startswith(transport.PREFIXES["save"])
    assert staging_name.endswith(".data.h5")
    assert transport.cleanup(transport.PREFIXES["save"], ttl=1.0) == [staging_name]
    assert not os.path.exists(staging_path)


def test_node_mixin_purpose_scoped_cleanup(tmp_path):
    transport = SharedFileTransport(tmp_path)
    save_name = transport.new_name("save", ".h5")
    export_name = transport.new_name("export", ".png")
    for name in (save_name, export_name):
        transport.publish(name, lambda path: open(path, "wb").close())
        os.utime(transport.resolve(name), (0.0, 0.0))

    node = FileTransportNodeMixin()
    node.conf = {"file_transport_dir": str(tmp_path)}
    node.logger = DummyLogger("test_file_transport_node_mixin")
    node.init_file_transport(("save",), ttl=1.0)

    assert not os.path.exists(transport.resolve(save_name))
    assert os.path.exists(transport.resolve(export_name))
    export = FileArtifact(export_name, os.path.getsize(transport.resolve(export_name)))
    assert not node.cleanup_artifact(CleanupArtifactReq(export)).success
    assert not node.publish_artifact("plot.png", "export", lambda path: Reply(True)).success


class _Requester:
    def __init__(self, node):
        self.node = node

    def request(self, msg):
        return self.node._handle_req(msg)


class _RecordingRequester:
    def __init__(self):
        self.messages = []

    def request(self, msg):
        self.messages.append(msg)
        return Reply(True, ret="loaded-data")


class _TransportClient(BasicMeasReqMixin):
    pass


def test_basic_meas_client_legacy_requests_without_transport():
    client = _TransportClient()
    client.file_transport = None
    client.req = _RecordingRequester()

    assert BasicMeasReqMixin.save_data(client, "/local/data.h5")
    assert BasicMeasReqMixin.export_data(client, "/local/data.png")
    assert BasicMeasReqMixin.load_data(client, "/local/data.h5") == "loaded-data"
    assert [type(msg) for msg in client.req.messages] == [SaveDataReq, ExportDataReq, LoadDataReq]


def test_basic_meas_client_roundtrip_and_buffer_name(tmp_path):
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    shared.mkdir()
    local.mkdir()

    node = object.__new__(BasicMeasNode)
    node._closed = True
    node.file_transport = SharedFileTransport(shared)
    node._file_transport_output_purposes = ("save", "export")
    loaded = []

    def save(msg):
        with open(msg.file_name, "wb") as f:
            f.write(b"saved")
        return Reply(True)

    def export(msg):
        with open(msg.file_name, "wb") as f:
            f.write(b"exported")
        head, ext = os.path.splitext(msg.file_name)
        with open(head + "_trace" + ext, "wb") as f:
            f.write(b"trace")
        return Reply(True)

    def load(msg):
        loaded.append((open(msg.file_name, "rb").read(), msg.buffer_name, msg.to_buffer))
        return Reply(True, ret="loaded-data")

    node.save_data = save
    node.export_data = export
    node.load_data = load

    client = _TransportClient()
    client.file_transport = SharedFileTransport(shared)
    client.req = _Requester(node)

    save_path = local / "chosen.data.h5"
    assert BasicMeasReqMixin.save_data(client, str(save_path))
    assert save_path.read_bytes() == b"saved"
    export_path = local / "plot.png"
    assert BasicMeasReqMixin.export_data(client, str(export_path))
    assert export_path.read_bytes() == b"exported"
    assert (local / "plot_trace.png").read_bytes() == b"trace"

    source = local / "original.data.h5"
    source.write_bytes(b"input")
    assert BasicMeasReqMixin.load_data(client, str(source), to_buffer=True) == "loaded-data"
    assert loaded == [(b"input", "original.data.h5", True)]
    assert list(shared.iterdir()) == []


def test_artifact_failure_cleanup(tmp_path):
    node = object.__new__(BasicMeasNode)
    node._closed = True
    node.file_transport = SharedFileTransport(tmp_path)
    node._file_transport_output_purposes = ("save", "export")
    node.save_data = lambda msg: Reply(False, "no data")
    node.logger = type("Logger", (), {"exception": lambda *args: None})()
    rep = node._handle_req(SaveArtifactReq("result.h5"))
    assert not rep.success
    assert list(tmp_path.iterdir()) == []

    def fail_export(msg):
        with open(msg.file_name, "wb") as f:
            f.write(b"partial")
        head, ext = os.path.splitext(msg.file_name)
        with open(head + "_sidecar" + ext, "wb") as f:
            f.write(b"partial sidecar")
        return Reply(False, "failed export")

    node.export_data = fail_export
    rep = node._handle_req(ExportArtifactReq("plot.png"))
    assert not rep.success
    assert list(tmp_path.iterdir()) == []


def test_artifact_bundle_cleanup_scope(tmp_path):
    transport = SharedFileTransport(tmp_path)
    primary_name = transport.new_name("export", "plot.png")
    sidecar_name = os.path.splitext(primary_name)[0] + "_trace.png"
    for name in (primary_name, sidecar_name):
        transport.publish(name, lambda path: open(path, "wb").close())

    node = FileTransportNodeMixin()
    node.conf = {"file_transport_dir": str(tmp_path)}
    node.logger = DummyLogger("test_artifact_bundle_cleanup_scope")
    node.init_file_transport()
    primary = FileArtifact(primary_name, 0)
    sidecar = FileArtifact(sidecar_name, 0)

    assert node.cleanup_artifact(
        CleanupArtifactReq(FileArtifactBundle(primary, [("plot_trace.png", sidecar)]))
    ).success
    assert list(tmp_path.iterdir()) == []
