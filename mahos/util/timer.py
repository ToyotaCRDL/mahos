#!/usr/bin/env python3

import typing as T
import time
import math


def _round_down(x: float, n_digits: int) -> float:
    k = 10**n_digits
    return math.floor(x * k) / k


def seconds_to_hms(t: float, n_digits: int = 1) -> T.Tuple[int, int, T.Union[float, int]]:
    """Convert time in seconds `t` to triple of (hours, minutes, secs).

    This is useful for displaying purpose.
    On typical usecase, `t` is a time difference (difference of time.time()).

    :param n_digits: round down the seconds to this digit.
                     to avoid displaying "60" when sec is like 59.9999.
    :returns: (hours in int, minutes in int, secs) secs is int if n_digits is 1, float otherwise.

    """

    hours, t = int(round(t // 3600)), t % 3600
    minutes, secs = int(round(t // 60)), t % 60
    secs = _round_down(secs, n_digits)
    if n_digits == 1:
        secs = int(secs)
    return hours, minutes, secs


class IntervalTimer(object):
    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self.last = time.time()

    def check(self):
        now = time.time()
        duration = now - self.last
        if duration > self.interval_sec:
            self.last = now
            return True
        else:
            return False

    def clone(self):
        return IntervalTimer(self.interval_sec)


class OneshotTimer(object):
    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self.start = time.time()
        self._forced = False

    def force_activation(self):
        """force activation of this timer. subsequent check() returns always True."""

        self._forced = True

    def check(self) -> bool:
        """return True if given interval has been elapsed, or force-activated."""

        if self._forced:
            return True

        duration = time.time() - self.start
        if duration > self.interval_sec:
            return True
        else:
            return False

    def clone(self):
        """return OneshotTimer with same interval."""

        return OneshotTimer(self.interval_sec)


class StopWatch(object):
    def __init__(self, start: float = time.time()):
        self.start = start
        self.last = self.start

    def elapsed(self) -> T.Tuple[float, float]:
        now = time.time()
        total = now - self.start
        lap = now - self.last
        self.last = now
        return total, lap

    def to_str(self, t: float) -> str:
        hours, minutes, secs = seconds_to_hms(t, n_digits=3)
        return f"{t:.3f} sec. ({hours:02d}:{minutes:02d}:{secs:04.3f})"

    def elapsed_str(self) -> str:
        total, lap = self.elapsed()
        return "{} / lap {}".format(self.to_str(total), self.to_str(lap))


class IntervalSleeper(object):
    def __init__(self, interval_sec: float):
        self.interval_ns = int(round(interval_sec * 1e9))
        self.last = None

    def time_to_sleep(self, now: float):
        # in case time jumped backwards
        if self.last > now:
            self.last = now

        elapsed = now - self.last
        return self.interval_ns - elapsed

    def sleep(self):
        if self.last is None:
            self.last = time.time_ns()
            return

        now = time.time_ns()
        to_sleep = self.time_to_sleep(now)
        if to_sleep >= 0:
            time.sleep(to_sleep * 1e-9)

        # in case time jumped forwards or too slow
        if now - self.last > self.interval_ns * 2:
            self.last = now
        else:
            self.last = self.last + self.interval_ns


class FPSCounter(object):
    def __init__(self):
        self.frames = 0
        self.start_time = None
        self._fps = 0.0

    def tick(self):
        if not self.start_time:
            self.start_time = time.perf_counter()
        else:
            self.frames += 1
            elapsed = time.perf_counter() - self.start_time
            self._fps = self.frames / elapsed
        return self._fps

    def fps(self) -> float:
        return self._fps
