"""Shared fixtures. PostgreSQL tests use `LITEMEM_TEST_DATABASE_URL` — set explicitly, never
inherited from the app's URL — and only ever the `lite_mem_test` schema, which they drop and
recreate."""

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg import sql

from lite_mem.db import migrate
from lite_mem.engine import MemoryEngine
from lite_mem.stores.pg import PgStore
from lite_mem.testing import FakeEmbedder

TEST_SCHEMA = "lite_mem_test"


@pytest.fixture(autouse=True)
def _isolated_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Settings must not pick up the developer's environment or `.env` in tests."""
    for name in list(os.environ):
        if name.startswith("LITEMEM_") and name != "LITEMEM_TEST_DATABASE_URL":
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture
def pg_url() -> str:
    url = os.environ.get("LITEMEM_TEST_DATABASE_URL")
    if not url:
        pytest.fail("LITEMEM_TEST_DATABASE_URL is not set: PostgreSQL tests need it (.env.example)")
    return url


@pytest.fixture
def pg(pg_url: str) -> Iterator[psycopg.Connection]:
    """A connection with an empty `lite_mem_test` schema (dropped again afterwards)."""
    with psycopg.connect(pg_url, autocommit=True) as conn:
        drop = sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(TEST_SCHEMA))
        conn.execute(drop)
        yield conn
        conn.execute(drop)


class Clock:
    """A controllable clock: starts at a fixed instant, moves only when told."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def tick(self, seconds: float = 60) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def migrated(pg: psycopg.Connection) -> psycopg.Connection:
    migrate(pg, TEST_SCHEMA)
    return pg


@pytest.fixture
def store(migrated: psycopg.Connection, pg_url: str) -> Iterator[PgStore]:
    s = PgStore(pg_url, TEST_SCHEMA)
    yield s
    s.close()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def engine(store: PgStore, clock: Clock) -> MemoryEngine:
    return MemoryEngine(store, "app", FakeEmbedder(), clock=clock)
