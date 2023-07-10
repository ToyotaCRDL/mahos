#!/usr/bin/env python3

from mahos.msgs.confocal_msgs import ConfocalState
from mahos.msgs.common_msgs import BinaryState
from util import expect_value

from fixtures import (
    ctx,
    gconf,
    manager,
    server,
    confocal,
    tracker,
    odmr,
    manager_conf,
    confocal_name,
    tracker_name,
    odmr_name,
)


def expect_locked(cli, name, poll_timeout_ms):
    get = lambda: cli.is_locked(name)
    return expect_value(get, True, poll_timeout_ms, trials=500)


def expect_release(cli, name, poll_timeout_ms):
    get = lambda: cli.is_locked(name)
    return expect_value(get, False, poll_timeout_ms, trials=500)


def expect_states(cli, states, poll_timeout_ms):
    def get():
        s = cli.get_status()
        if s is None:
            return None
        ss = s.states
        return tuple(((name, ss[name]) for name, state in states))

    return expect_value(get, states, poll_timeout_ms, trials=500)


def test_state_manager(manager, server, confocal, tracker, odmr, manager_conf):
    m_poll_timeout_ms = manager_conf["poll_timeout_ms"]

    for cli in (server, confocal, tracker, odmr, manager):
        cli.wait()

    assert not manager.command("non-existent-command")
    assert not manager.restore("non-existent-command")

    states_idle = (
        (odmr_name, BinaryState.IDLE),
        (confocal_name, ConfocalState.IDLE),
        (tracker_name, BinaryState.IDLE),
    )
    states_active = (
        (odmr_name, BinaryState.ACTIVE),
        (confocal_name, ConfocalState.IDLE),
        (tracker_name, BinaryState.IDLE),
    )

    assert expect_states(manager, states_idle, m_poll_timeout_ms)

    params = odmr.get_param_dict("cw")
    params["num"].set(10)
    assert odmr.change_state(BinaryState.ACTIVE, params)

    assert expect_states(manager, states_active, m_poll_timeout_ms)

    assert manager.command("prepare_scan")

    assert expect_states(manager, states_idle, m_poll_timeout_ms)

    assert manager.restore("prepare_scan")

    assert expect_states(manager, states_active, m_poll_timeout_ms)
