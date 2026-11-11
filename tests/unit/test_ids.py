"""UUIDv7 ids sort in the order they were made (design §3): list order breaks ties by id."""

import threading
import uuid

import pytest

from lite_mem.ids import uuid7


def test_ids_made_in_one_millisecond_still_sort_in_order() -> None:
    """Found by the SQLite store (S7b): it writes several memories within one millisecond, and
    random bits made their order random."""
    made = [uuid7() for _ in range(5000)]
    assert made == sorted(made) and len(set(made)) == len(made)
    assert all(i.version == 7 and i.variant == uuid.RFC_4122 for i in made)


def test_ids_from_many_threads_are_unique() -> None:
    made: list[uuid.UUID] = []

    def make() -> None:
        made.extend(uuid7() for _ in range(2000))

    threads = [threading.Thread(target=make) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(made)) == len(made) == 8000


def test_the_counter_carries_into_the_next_millisecond_and_never_wraps() -> None:
    """Codex S7b: at the last timestamp the old increment wrapped to zero."""
    from lite_mem.ids import _COUNTER, _MAX_MS, _after

    assert _after((5, _COUNTER - 1), 5, 0) == (6, 0)  # used up: borrow the next millisecond
    assert _after((5, 7), 3, 0) == (5, 8)  # the clock stepped back: still after the last id
    with pytest.raises(OverflowError):
        _after((_MAX_MS, _COUNTER - 1), _MAX_MS, 0)
