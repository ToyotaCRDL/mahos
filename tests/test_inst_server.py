#!/usr/bin/env python3

from mahos.msgs.confocal_msgs import Axis
from mahos.inst.overlay.confocal_scanner_mock import DUMMY_CAPABILITY

from fixtures import ctx, gconf, server_2clients


def test_inst(server_2clients):
    client, client2 = server_2clients

    client.wait()
    client2.wait()

    # non-existent failures
    assert not client.lock("non-existent-inst")
    assert not client.release("non-existent-inst")
    assert not client("non-existent-inst", "set_output", on=True).success
    # func name and argument errors
    assert not client("sg", "non-existent-func").success
    assert not client("sg", "set_output").success
    assert not client("sg", "set_output", non_existent_arg="abc").success

    # standardized APIs
    assert client.set("sg", "output", True)
    assert not client.set("piezo", "target", {"ax": None})  # failing call (invalid arg)
    assert client2.set("piezo", "target", {"ax": Axis.X, "pos": 5.6})

    # lock and release
    ## lock from a client
    assert client.lock("sg")
    assert client.lock("sg")  # double lock by same cli is OK
    ## cannot release or lock by different client
    assert not client2.lock("sg")
    assert not client2.release("sg")
    ## release from a client
    assert client.release("sg")
    assert client.release("sg")  # double release is fine
    ## lock again and use APIs from locking client
    assert client.lock("sg")
    assert client("sg", "set_output", on=True).success
    assert client.set("sg", "output", False)
    ## client2 fails because client locks
    assert not client2.set("sg", "output", True)
    ## client2 can access after release
    assert client.release("sg")
    assert client2.set("sg", "output", True)


def test_overlay(server_2clients):
    client, client2 = server_2clients

    client.wait()
    client2.wait()

    # func name and argument errors
    assert not client("scanner", "non-existent-func").success
    assert not client("scanner", "get_line", non_existent_arg="abc").success

    # standardized APIs
    assert client.get("scanner", "capability") == DUMMY_CAPABILITY
    assert client.start("scanner")
    assert client.stop("scanner")

    # lock
    ## client cannot lock if part is locked by client2
    assert client2.lock("piezo")
    assert not client.lock("scanner")
    assert client2.release("piezo")
    ## fine to lock the part and then overlay by the same client
    assert client.lock("piezo")
    assert client.lock("scanner")
    assert client.lock("scanner")
    ## cannot release or lock by different client
    assert not client2.lock("scanner")
    assert not client2.lock("piezo")  # for parts
    assert not client2.release("scanner")
    assert not client2.release("piezo")
    ## fine to operate non-part inst
    assert client2.lock("sg")
    assert client2.release("sg")
    assert client2.set("sg", "output", True)
    ## release from the client
    assert client.release("scanner")
    assert client.release("scanner")  # double release is fine
    ## lock again and use APIs from locking client
    assert client.lock("scanner")
    assert client.get("scanner", "capability") == DUMMY_CAPABILITY
    ## client2 fails because client locks
    assert not client2("scanner", "get_line").success
    assert not client2("piezo", "set_target").success
    ## client2 can access after free
    assert client.release("scanner")
    assert client2.get("scanner", "capability") == DUMMY_CAPABILITY
    assert client2.set("piezo", "target", {"ax": Axis.X, "pos": 5.6})
