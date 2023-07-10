#!/usr/bin/env python3

import time
from threading import Lock
from collections import deque
import copy
import typing as T


class LockedQueue(object):
    """Queue with Lock."""

    def __init__(self, size: int):
        self._size = size
        self.buffer = deque(maxlen=size)
        self.lock = Lock()

    def __len__(self):
        return len(self.buffer)

    def size(self) -> int:
        return self._size

    def is_full(self) -> bool:
        return len(self) >= self.size()

    def append(self, data) -> bool:
        """append data in queue. return False if queue is overflowing."""

        with self.lock:
            ok = not self.is_full()
            self.buffer.append(data)
        return ok

    def pop_opt(self):
        """pop single data from queue. return None immediately if queue is empty."""

        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            else:
                return None

    def pop_all_opt(self):
        """pop all data from queue. return None immediately if queue is empty."""

        with self.lock:
            if self.buffer:
                ret = list(copy.copy(self.buffer))
                self.buffer.clear()
                return ret
            else:
                return None

    def pop_block(self, timeout_sec: T.Optional[float] = None, interval_sec: float = 0.001):
        """pop single data from queue, blocking if the queue is empty.

        returns None if queue is still empty after timeout_sec.

        :param timeout_sec: timeout of blocking. if None or zero, block is unlimited.
        :param interval_sec: interval for empty check.

        """

        tstart = time.time()
        while True:
            with self.lock:
                if self.buffer:
                    return self.buffer.popleft()
            if timeout_sec and time.time() - tstart > timeout_sec:
                return None
            time.sleep(interval_sec)

    def pop_all_block(self, timeout_sec: T.Optional[float] = None, interval_sec: float = 0.001):
        """pop all data from queue, blocking if the queue is empty.

        returns None if queue is still empty after timeout_sec.

        :param timeout_sec: timeout of blocking. if None or zero, block is unlimited.
        :param interval_sec: interval for empty check.

        """

        tstart = time.time()
        while True:
            with self.lock:
                if self.buffer:
                    ret = list(copy.copy(self.buffer))
                    self.buffer.clear()
                    return ret
            if timeout_sec and time.time() - tstart > timeout_sec:
                return None
            time.sleep(interval_sec)
