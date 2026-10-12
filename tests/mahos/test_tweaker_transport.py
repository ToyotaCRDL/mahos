#!/usr/bin/env python3

"""Tests for Tweaker state embedding and shared-file transport."""

from mahos.meas.pos_tweaker import PosTweaker
from mahos.meas.pos_tweaker_io import PosTweakerIO
from mahos.meas.tweaker import TweakSaver, Tweaker, TweakerFileReqMixin
from mahos.meas.tweaker_io import TweakerIO
from mahos.msgs import param_msgs as P
from mahos.msgs.common_msgs import Reply
from mahos.msgs.pos_tweaker_msgs import PosTweakerState
from mahos.msgs.tweaker_msgs import GetStateReq, LoadReq, SaveReq, TweakerState
from mahos.node.log import DummyLogger
from mahos.util.file_transport import SharedFileTransport


class _Requester:
    def __init__(self, handler):
        self.handler = handler
        self.messages = []

    def request(self, msg):
        self.messages.append(msg)
        return self.handler(msg)


class _TweakerTransportClient(TweakerFileReqMixin):
    pass


def _tweaker_state(value=1):
    return TweakerState(
        {"inst::label": P.ParamDict({"value": P.IntParam(value)})},
        {"inst::label": True},
    )


def test_state_messages_serialize():
    tweaker = _tweaker_state()
    restored = TweakerState.deserialize(tweaker.serialize())
    assert restored.param_dicts["inst::label"]["value"].value() == 1
    assert restored.start_stop_states == {"inst::label": True}

    pos = PosTweakerState({"x": {"pos": 1.0, "target": 2.0, "homed": True}})
    restored_pos = PosTweakerState.deserialize(pos.serialize())
    assert restored_pos.axis_states == pos.axis_states


def test_tweak_saver_embeds_returned_state_locally(tmp_path):
    measurement = tmp_path / "measurement.h5"
    saver = object.__new__(TweakSaver)
    saver._closed = True
    saver.logger = DummyLogger("test_tweak_saver")
    saver.tweaker_io = TweakerIO(saver.logger)
    saver.pos_tweaker_io = PosTweakerIO(saver.logger)
    saver.req = _Requester(lambda msg: Reply(True, ret=_tweaker_state()))

    assert saver.save(str(measurement), "__remote::tweaker__")
    assert [type(msg) for msg in saver.req.messages] == [GetStateReq]
    loaded = saver.tweaker_io.load_data(str(measurement), "__remote::tweaker__")
    assert loaded["inst::label"]["value"] == 1

    saver.req = _Requester(
        lambda msg: Reply(
            True,
            ret=PosTweakerState({"x": {"pos": 1.0, "target": 2.0, "homed": True}}),
        )
    )
    assert saver.save(str(measurement), "__remote::pos_tweaker__")
    loaded_pos = saver.pos_tweaker_io.load_data(str(measurement), "__remote::pos_tweaker__")
    assert loaded_pos["x"]["target"] == 2.0


def test_tweaker_client_legacy_requests_without_transport():
    client = _TweakerTransportClient()
    client.file_transport = None
    client.req = _Requester(lambda msg: Reply(True, ret="loaded"))

    assert client.save_file("/node/path/state.tweak.h5")
    assert client.load_file("/node/path/state.tweak.h5").ret == "loaded"
    assert [type(msg) for msg in client.req.messages] == [SaveReq, LoadReq]


def test_tweaker_transport_roundtrip_with_shared_directory(tmp_path):
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    shared.mkdir()
    local.mkdir()

    state = _tweaker_state()

    node = object.__new__(Tweaker)
    node._closed = True
    node.logger = DummyLogger("test_tweaker_node")
    node.file_transport = SharedFileTransport(shared)
    node._file_transport_output_purposes = ("save",)
    node.io = TweakerIO(node.logger)
    node._param_dicts = state.param_dicts
    node._start_stop_states = state.start_stop_states
    state_rep = node.handle_req(GetStateReq())
    assert isinstance(state_rep.ret, TweakerState)

    client = _TweakerTransportClient()
    client.file_transport = SharedFileTransport(shared)
    client.req = _Requester(node.handle_req)

    assert not client.save_file(str(local / "missing" / "state.tweak.h5"))
    assert list(shared.iterdir()) == []

    destination = local / "chosen.state.tweak.h5"
    assert client.save_file(str(destination))
    assert destination.exists()
    assert list(shared.iterdir()) == []

    node._param_dicts["inst::label"]["value"].set(0)
    rep = client.load_file(str(destination))
    assert rep.success
    assert rep.ret["inst::label"]["value"].value() == 1
    assert list(shared.iterdir()) == []

    invalid = local / "invalid.h5"
    invalid.write_bytes(b"not hdf5")
    assert not client.load_file(str(invalid)).success
    assert list(shared.iterdir()) == []


def test_pos_tweaker_transport_roundtrip(tmp_path):
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    shared.mkdir()
    local.mkdir()

    targets = []
    positioner = type(
        "Positioner", (), {"set_target": lambda self, value: targets.append(value)}
    )()
    node = object.__new__(PosTweaker)
    node._closed = True
    node.logger = DummyLogger("test_pos_tweaker_node")
    node.file_transport = SharedFileTransport(shared)
    node._file_transport_output_purposes = ("save",)
    node.io = PosTweakerIO(node.logger)
    node._axis_states = {"x": {"pos": 1.0, "target": 2.0, "homed": True}}
    node._axis_positioners = {"x": positioner}
    state_rep = node.handle_req(GetStateReq())
    assert isinstance(state_rep.ret, PosTweakerState)

    client = _TweakerTransportClient()
    client.file_transport = SharedFileTransport(shared)
    client.req = _Requester(node.handle_req)

    destination = local / "state.ptweak.h5"
    assert client.save_file(str(destination))
    assert client.load_file(str(destination)).success
    assert targets == [2.0]
    assert list(shared.iterdir()) == []
