#!/usr/bin/env python3

from util import expect_value

from fixtures import ctx, gconf, param_server, param_server_2clients, param_server_conf


def expect_param(cli, key, value, poll_timeout_ms):
    return expect_value(lambda: cli.get_param(key), value, poll_timeout_ms, trials=500)


def test_param_server(param_server_2clients, param_server_conf):
    poll_timeout_ms = param_server_conf["poll_timeout_ms"]

    client, client2 = param_server_2clients

    client.wait()
    client2.wait()

    assert client.set_param("hoge", 123)
    assert expect_param(client2, "hoge", 123, poll_timeout_ms)
    assert expect_param(client, "hoge", 123, poll_timeout_ms)
    assert client2.set_param("hoge", "abc")
    assert expect_param(client, "hoge", "abc", poll_timeout_ms)
    assert expect_param(client2, "hoge", "abc", poll_timeout_ms)

    assert client.set_param("fuga", 1234)
    assert expect_param(client, "fuga", 1234, poll_timeout_ms)
