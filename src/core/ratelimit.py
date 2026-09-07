"""
Rate limiting.

In-memory, deliberately not slowapi or Redis: this app is single-worker by design, so a
dict and a lock give exactly the same guarantee as a Redis token bucket would, with no
extra service to run. The moment this app runs multiple workers, THIS STOPS BEING CORRECT -
each worker would enforce its own limit and the effective rate would multiply by the worker
count.

What it protects against:

* **Password guessing.** `/api/login` is otherwise as fast as the network.
* **CPU exhaustion.** Argon2 is deliberately expensive (~50-100ms of dedicated CPU per
  verification), which slows an attacker down and hands them a cheap denial-of-service in
  the same breath. A few hundred concurrent login attempts will saturate the machine.
* **Runaway cost.** Every /chat call spends LLM tokens, and every upload spends disk and
  minutes of CPU.

Limits are per (bucket, key), where key is usually the client IP or a user id.
"""
import threading
import time
from typing import Dict, Optional, Tuple

from src.core.logging import get_logger

log = get_logger(__name__)


class RateLimit:
    """One named limit: `allowance` events per `per_seconds`, per key."""

    __slots__ = ("name", "allowance", "per_seconds")

    def __init__(self, name: str, allowance: int, per_seconds: float):
        self.name = name
        self.allowance = allowance
        self.per_seconds = per_seconds


# Tuned for "a person using the app", not for a load test. Login is the strictest because
# it is the one an attacker actually wants.
LOGIN = RateLimit("login", allowance=8, per_seconds=60)
SIGNUP = RateLimit("signup", allowance=5, per_seconds=3600)
UPLOAD = RateLimit("upload", allowance=30, per_seconds=3600)
CHAT = RateLimit("chat", allowance=60, per_seconds=3600)

_lock = threading.Lock()
# (limit name, key) -> [event timestamps], oldest first.
_events: Dict[Tuple[str, str], list] = {}
_last_sweep = 0.0


def _sweep(now: float) -> None:
    """Drops empty buckets so the dict cannot grow forever on a long-running server."""
    global _last_sweep
    if now - _last_sweep < 300:
        return
    _last_sweep = now
    stale = [k for k, stamps in _events.items() if not stamps or now - stamps[-1] > 3600]
    for key in stale:
        _events.pop(key, None)


def _check_memory(limit: RateLimit, key: str) -> Optional[int]:
    """The original in-process sliding window. Unchanged - see the module docstring."""
    now = time.monotonic()
    with _lock:
        _sweep(now)
        bucket = _events.setdefault((limit.name, key), [])

        cutoff = now - limit.per_seconds
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)

        if len(bucket) >= limit.allowance:
            retry_after = int(bucket[0] + limit.per_seconds - now) + 1
            log.warning("Rate limit '%s' hit by %s (retry in %ds)", limit.name, key, retry_after)
            return retry_after

        bucket.append(now)
        return None


def check(limit: RateLimit, key: str) -> Optional[int]:
    """
    Records an event and returns None if it is allowed, or the seconds to wait if not.

    Sliding window rather than fixed buckets: a fixed window lets an attacker fire the full
    allowance at 59.9s and again at 60.1s.
    """
    return _check_memory(limit, key)


def reset() -> None:
    """Clears all buckets. For tests only."""
    with _lock:
        _events.clear()


def client_key(request) -> str:
    """
    A key for an unauthenticated caller.

    `request.client.host` is the peer address - which behind a reverse proxy is the proxy,
    making every user share one bucket. `--proxy-headers` (uvicorn) rewrites it from
    X-Forwarded-For, so run it that way if you put a proxy in front; trusting the header
    here directly would let anyone forge a fresh identity per request.
    """
    return request.client.host if request.client else "unknown"
