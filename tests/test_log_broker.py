#!/usr/bin/env python3

import time

from fixtures import ctx, gconf, log_broker, dummy_logging


dummy_name = ("localhost", "dummy")
dummy_interval_sec = 0.01


def pick_bodies(logs):
    return [l[-1] for l in logs]


def test_log(log_broker, dummy_logging):
    N = 10

    # wait for log
    for i in range(1000):
        logs = log_broker.get_logs()
        if len(logs) == N:
            break
        time.sleep(dummy_interval_sec)

    log_bodies = pick_bodies(logs)
    n = int(log_bodies[0][:-1])
    assert log_bodies == [str(i) + "\n" for i in range(n, n + N)]
    assert log_broker.pop_log()[-1] == log_bodies[0]
