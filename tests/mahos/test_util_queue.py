#!/usr/bin/env python3

"""
Tests for mahos.util.queue.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

import threading
import time

from mahos.util.queue import RollingQueue


def test_append_and_overflow():
    q = RollingQueue(2)
    assert q.append("a")
    assert q.append("b")
    assert not q.append("c")

    assert len(q) == 2
    assert q.pop_opt() == "b"
    assert q.pop_opt() == "c"


def test_pop_opt_empty():
    q = RollingQueue(1)
    assert q.pop_opt() is None


def test_inspection_while_lock_held():
    q = RollingQueue(1)

    with q.lock:
        assert len(q) == 0
        assert not q.is_full()

    q.append("x")

    with q.lock:
        assert len(q) == 1
        assert q.is_full()


def test_pop_all_opt():
    q = RollingQueue(3)
    q.append(1)
    q.append(2)
    assert q.pop_all_opt() == [1, 2]
    assert len(q) == 0
    assert q.pop_all_opt() is None


def test_pop_block_timeout():
    q = RollingQueue(1)
    assert q.pop_block(timeout_sec=0.01) is None


def test_pop_block_waits_for_append():
    q = RollingQueue(1)
    ret = []

    t = threading.Thread(target=lambda: ret.append(q.pop_block(timeout_sec=1.0)))
    t.start()
    time.sleep(0.01)

    assert q.append("x")
    t.join(timeout=1.0)

    assert not t.is_alive()
    assert ret == ["x"]


def test_pop_all_block_with_data():
    q = RollingQueue(2)
    q.append("x")
    q.append("y")
    assert q.pop_all_block(timeout_sec=0.01) == ["x", "y"]
    assert len(q) == 0


def test_pop_opt_with_status_does_not_report_full_queue():
    q = RollingQueue(2)
    q.append("a")
    q.append("b")

    data, overflowed = q.pop_opt_with_status()
    assert data == "a"
    assert not overflowed

    data, overflowed = q.pop_opt_with_status()
    assert data == "b"
    assert not overflowed


def test_pop_opt_with_status_reports_overflow():
    q = RollingQueue(2)
    assert q.append("a")
    assert q.append("b")
    assert not q.append("c")

    data, overflowed = q.pop_opt_with_status()
    assert data == "b"
    assert overflowed

    data, overflowed = q.pop_opt_with_status()
    assert data == "c"
    assert not overflowed


def test_pop_block_with_status_timeout():
    q = RollingQueue(1)
    assert q.pop_block_with_status(timeout_sec=0.01) == (None, False)
