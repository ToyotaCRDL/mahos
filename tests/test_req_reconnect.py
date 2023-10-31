#!/usr/bin/env python3

"""
Tests for reconnection function of node request.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import time

from fixtures import ctx, gconf, server_restart, server_conf


def wait_new_status(client, old_status, poll_timeout_ms, trials=500):
    for i in range(trials):
        if client.get_status() is not old_status:
            return True
        time.sleep(1e-3 * poll_timeout_ms)
    else:
        return False


def test_req_reconnect(gconf, server_restart, server_conf):
    server, client = server_restart

    poll_timeout_ms = server_conf["poll_timeout_ms"]

    client.wait()
    assert client.lock("sg")

    # server is down here and thus failure
    server.stop()
    assert not client.lock("sg")

    # restart the server and retry from client
    server.start()
    status = client.get_status()
    assert wait_new_status(client, status, poll_timeout_ms)
    assert client.lock("sg")
