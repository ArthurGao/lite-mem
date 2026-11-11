"""Shared fixtures. PostgreSQL tests use `LITEMEM_TEST_DATABASE_URL` — set explicitly, never
inherited from the app's URL — and only ever the `lite_mem_test` schema, which they drop and
recreate."""

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql

from lite_mem.db import migrate
from lite_mem.engine import MemoryEngine
from lite_mem.factory import Components
from lite_mem.resolve import AddOnlyResolver
from lite_mem.stores.base import Store
from lite_mem.stores.pg import PgStore, PgTx
from lite_mem.stores.sqlite import SqliteStore, SqliteTx
from lite_mem.testing import FakeEmbedder, RuleExtractor
from lite_mem.types import Scope

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
def pg_store(migrated: psycopg.Connection, pg_url: str) -> Iterator[PgStore]:
    """PostgreSQL only: for tests about PgStore itself (its SQL, its schema, its connections)."""
    s = PgStore(pg_url, TEST_SCHEMA)
    yield s
    s.close()


@pytest.fixture(params=["pg", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Store]:
    """Every store-level test runs on both stores: the Store contract (S7b)."""
    if request.param == "pg":
        yield request.getfixturevalue("pg_store")
        return
    s = SqliteStore(tmp_path / "lite-mem.db")
    yield s
    s.close()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def engine(store: Store, clock: Clock) -> MemoryEngine:
    return MemoryEngine(store, "app", FakeEmbedder(), clock=clock)


def fake_components(store: Store) -> Components:
    """What the CLI and the HTTP service share, with fakes: no model anywhere (each user message
    is one fact, every fact is added)."""
    return Components(store, FakeEmbedder(), RuleExtractor(), AddOnlyResolver())


def fake_engine(store: Store, namespace: str = "app") -> MemoryEngine:
    return fake_components(store).engine(namespace)


def tx_class(store: Store) -> type:
    """The store's transaction class, where a test injects a fault (any store)."""
    return PgTx if isinstance(store, PgStore) else SqliteTx


def count_rows(store: Store, table: str, user_id: str | None = None) -> int:
    """Rows in one of the store's tables (one user's, when given); `memory_vectors` is per
    dimension in PostgreSQL (the tests' fake embedder: 64)."""
    pg = isinstance(store, PgStore)
    name = "memory_vectors_64" if pg and table == "memory_vectors" else table
    where = f" WHERE user_id = {'%s' if pg else '?'}" if user_id else ""
    with store.transaction() as tx:
        conn: Any = tx._conn  # type: ignore[attr-defined]
        row = conn.execute(
            f"SELECT count(*) AS n FROM {name}{where}", (user_id,) if user_id else ()
        )
        return int(row.fetchone()["n"])


def current(engine: MemoryEngine, scope: Scope | None = None) -> list[str]:
    """The scope's current memories' contents, sorted (default: user "ada")."""
    return sorted(m.content for m in engine.list_memories(scope or Scope(user_id="ada")))
