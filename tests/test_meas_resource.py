#!/usr/bin/env python3

from mahos.msgs.confocal_msgs import ConfocalState
from mahos.msgs.common_msgs import BinaryState
from util import expect_value

from fixtures import ctx, gconf, server, confocal_ev, odmr_ev, server_conf, confocal_conf


def expect_locked(cli, name, poll_timeout_ms):
    get = lambda: cli.is_locked(name)
    return expect_value(get, True, poll_timeout_ms, trials=300)


def expect_release(cli, name, poll_timeout_ms):
    get = lambda: cli.is_locked(name)
    return expect_value(get, False, poll_timeout_ms, trials=300)


def expect_confocal_state(cli, state, poll_timeout_ms):
    get = lambda: cli.get_status().state
    return expect_value(get, state, poll_timeout_ms, trials=300)


def test_meas_resource_conflicts(server, confocal_ev, odmr_ev, server_conf, confocal_conf):
    s_poll_timeout_ms = server_conf["poll_timeout_ms"]
    c_poll_timeout_ms = confocal_conf["poll_timeout_ms"]

    confocal, c_shutdown_ev = confocal_ev
    odmr, o_shutdown_ev = odmr_ev

    for cli in (server, confocal, odmr):
        cli.wait()

    # conflict Confocal and user client
    ## sg is not related to Confocal
    assert server.lock("sg")
    assert confocal.change_state(ConfocalState.INTERACT)
    assert confocal.change_state(ConfocalState.IDLE)
    ## when piezo is locked, cannot go to INTERACT state.
    assert server.lock("piezo")
    assert expect_locked(server, "piezo", s_poll_timeout_ms)
    assert not confocal.change_state(ConfocalState.INTERACT)
    assert expect_confocal_state(confocal, ConfocalState.IDLE, c_poll_timeout_ms)
    assert server.release("piezo")
    ## going to INTERACT falls back to PIEZO because piezo is released but pd is locked.
    assert server.lock("pd0")
    assert expect_locked(server, "pd0", s_poll_timeout_ms)
    assert not confocal.change_state(ConfocalState.INTERACT)
    assert expect_confocal_state(confocal, ConfocalState.PIEZO, c_poll_timeout_ms)
    assert server.release("pd0")
    assert confocal.change_state(ConfocalState.INTERACT)
    assert confocal.change_state(ConfocalState.IDLE)
    assert server.release("sg")

    # conflict ODMR and user client
    params = odmr.get_param_dict("cw")
    params["num"].set(10)
    assert server.lock("piezo")
    assert expect_locked(server, "piezo", s_poll_timeout_ms)
    assert odmr.change_state(BinaryState.ACTIVE, params)
    assert odmr.change_state(BinaryState.IDLE)
    assert server.lock("sg")
    assert expect_locked(server, "sg", s_poll_timeout_ms)
    assert not odmr.change_state(BinaryState.ACTIVE, params)
    assert server.release("sg")
    # assert expect_release(server, "sg", s_poll_timeout_ms)
    assert odmr.change_state(BinaryState.ACTIVE, params)
    assert odmr.change_state(BinaryState.IDLE)
    assert server.release("piezo")

    # conflict Confocal and ODMR
    ## ODMR can't start because pd is locked by Confocal.
    assert confocal.change_state(ConfocalState.INTERACT)
    assert not odmr.change_state(BinaryState.ACTIVE, params)
    assert confocal.change_state(ConfocalState.IDLE)
    ## Confocal can't start because pd is locked by ODMR.
    assert odmr.change_state(BinaryState.ACTIVE, params)
    assert not confocal.change_state(ConfocalState.INTERACT)
    ### (falls back to PIEZO)
    assert expect_confocal_state(confocal, ConfocalState.PIEZO, c_poll_timeout_ms)
    assert odmr.change_state(BinaryState.IDLE)
    ## Sanity check. Individual start and stop.
    assert odmr.change_state(BinaryState.ACTIVE, params)
    assert odmr.change_state(BinaryState.IDLE)
    assert confocal.change_state(ConfocalState.INTERACT)
    assert confocal.change_state(ConfocalState.IDLE)

    # shutdown (deletion of Confocal and ODMR) should release the resources.
    assert confocal.change_state(ConfocalState.PIEZO)
    assert odmr.change_state(BinaryState.ACTIVE, params)
    confocal.close()
    odmr.close()
    c_shutdown_ev.set()
    o_shutdown_ev.set()
    assert expect_release(server, "piezo", s_poll_timeout_ms)
    assert expect_release(server, "sg", s_poll_timeout_ms)
