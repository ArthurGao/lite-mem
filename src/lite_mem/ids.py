"""UUIDv7 (RFC 9562): time-ordered ids, so a memory's id sorts with its creation.

Monotonic within a process (RFC 9562 §6.2, method 3): an id is a 48-bit millisecond and a 74-bit
counter (the bits around the version and variant fields). A new millisecond starts the counter at
random; the same millisecond, or a clock that stepped back, adds one to the last id. Random bits
alone made the order of two ids in one millisecond random, and lists order ties by id (found by the
SQLite store, which writes several memories within a millisecond). Across processes, ids of one
millisecond are unordered."""

import os
import threading
import time
import uuid

_MAX_MS = (1 << 48) - 1
_COUNTER = 1 << 74
_lock = threading.Lock()
_last = (0, 0)  # (ms, counter) of the last id


def _pack(ms: int, counter: int) -> int:
    a, b = counter >> 62, counter & ((1 << 62) - 1)  # 12 + 62 bits
    return (ms << 80) | (0x7 << 76) | (a << 64) | (0b10 << 62) | b


def _after(last: tuple[int, int], now_ms: int, rand: int) -> tuple[int, int]:
    ms, counter = last
    if now_ms > ms:
        return now_ms, rand % _COUNTER
    counter += 1
    if counter == _COUNTER:  # this millisecond is used up: borrow the next one
        ms, counter = ms + 1, 0
    if ms > _MAX_MS:
        raise OverflowError("UUIDv7 timestamps end in the year 10889")
    return ms, counter


def uuid7() -> uuid.UUID:
    global _last
    with _lock:
        rand = int.from_bytes(os.urandom(10), "big")
        _last = _after(_last, time.time_ns() // 1_000_000, rand)
        return uuid.UUID(int=_pack(*_last))
