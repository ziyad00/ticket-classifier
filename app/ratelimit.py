"""In-memory sliding-window rate limiter, keyed by caller.

Per process only — with several replicas each one enforces its own window.
Enough to stop a single client from looping curl; not a substitute for
a proxy-level limiter in front of the service.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max_per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.max_per_minute <= 0:  # disabled
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= now - 60:
                q.popleft()
            if len(q) >= self.max_per_minute:
                return False
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory under many distinct callers
                for k in [k for k, v in self._hits.items() if not v or v[-1] <= now - 60]:
                    del self._hits[k]
            return True
