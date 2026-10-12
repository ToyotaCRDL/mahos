#!/usr/bin/env python3

"""
Thread-safe fixed-size queue utilities.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations
import time
from threading import Condition, RLock
from collections import deque


class RollingQueue(object):
    """Thread-safe fixed-size queue that drops the oldest item on overflow.

    The queue keeps at most ``size`` items. When :meth:`append` is called while
    full, the newest item is still appended, the oldest item is discarded, and
    ``False`` is returned. Status pop methods report whether at least one item
    has been discarded since queue creation or the previous successful status
    pop. Blocking pop methods wait on a condition variable.

    """

    def __init__(self, size: int):
        self._size = size
        self.buffer = deque(maxlen=size)
        self.lock = RLock()
        self.cond = Condition(self.lock)
        self._overflowed = False

    def __len__(self):
        with self.lock:
            return len(self.buffer)

    def size(self) -> int:
        return self._size

    def is_full(self) -> bool:
        with self.lock:
            return self._is_full_locked()

    def _is_full_locked(self) -> bool:
        return len(self.buffer) >= self._size

    def append(self, data) -> bool:
        """Append data and return False if the oldest item was discarded."""

        with self.cond:
            ok = not self._is_full_locked()
            if not ok:
                self._overflowed = True
            self.buffer.append(data)
            self.cond.notify()
        return ok

    def pop_opt(self):
        """pop single data from queue. return None immediately if queue is empty."""

        with self.cond:
            if self.buffer:
                return self.buffer.popleft()
            else:
                return None

    def pop_opt_with_status(self):
        """Pop single data and return ``(data, overflowed)`` immediately.

        ``overflowed`` is True when an append has discarded an item since queue
        creation or the previous successful status pop.
        """

        with self.cond:
            if self.buffer:
                return self._pop_with_status_locked()
            else:
                return None, False

    def pop_all_opt(self):
        """pop all data from queue. return None immediately if queue is empty."""

        with self.cond:
            if self.buffer:
                ret = list(self.buffer)
                self.buffer.clear()
                return ret
            else:
                return None

    def pop_block(self, timeout_sec: float | None = None):
        """pop single data from queue, blocking if the queue is empty.

        returns None if queue is still empty after timeout_sec.

        :param timeout_sec: timeout of blocking. if None or zero, block is unlimited.

        """

        with self.cond:
            if not self._wait_for_data_locked(timeout_sec):
                return None
            return self.buffer.popleft()

    def pop_block_with_status(self, timeout_sec: float | None = None):
        """Pop single data and return ``(data, overflowed)``, blocking if empty.

        ``overflowed`` is True when an append has discarded an item since queue
        creation or the previous successful status pop.

        :param timeout_sec: timeout of blocking. if None or zero, block is unlimited.

        """

        with self.cond:
            if not self._wait_for_data_locked(timeout_sec):
                return None, False
            return self._pop_with_status_locked()

    def pop_all_block(self, timeout_sec: float | None = None):
        """pop all data from queue, blocking if the queue is empty.

        returns None if queue is still empty after timeout_sec.

        :param timeout_sec: timeout of blocking. if None or zero, block is unlimited.

        """

        with self.cond:
            if not self._wait_for_data_locked(timeout_sec):
                return None
            ret = list(self.buffer)
            self.buffer.clear()
            return ret

    def _pop_with_status_locked(self):
        overflowed = self._overflowed
        self._overflowed = False
        return self.buffer.popleft(), overflowed

    def _wait_for_data_locked(self, timeout_sec: float | None) -> bool:
        if self.buffer:
            return True

        if not timeout_sec:
            while not self.buffer:
                self.cond.wait()
            return True

        deadline = time.monotonic() + timeout_sec
        while not self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.cond.wait(timeout=remaining)
        return True
