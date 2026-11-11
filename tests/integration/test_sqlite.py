"""S7b: the embedded store end to end — chosen by `LITEMEM_DATABASE_URL=sqlite:///<file>`, through
the real factory and CLI (only the embedding model is faked). The Store contract itself runs on
both stores through the parametrised `store` fixture (tests/conftest.py)."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lite_mem import factory
from lite_mem.cli import app
from lite_mem.config import Settings
from lite_mem.errors import ConfigError
from lite_mem.stores.sqlite import SqliteStore
from lite_mem.testing import FakeEmbedder
from lite_mem.types import Fact, Scope


def test_a_sqlite_url_opens_the_embedded_store(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'memory.db'}"
    store = factory.open_store(Settings(database_url=url))
    assert isinstance(store, SqliteStore)
    store.close()
    assert (tmp_path / "memory.db").exists()
    with pytest.raises(ConfigError, match="sqlite:///"):
        factory.open_store(Settings(database_url="sqlite://"))


def test_the_cli_runs_on_a_sqlite_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setenv("LITEMEM_DATABASE_URL", f"sqlite:///{tmp_path / 'memory.db'}")
    monkeypatch.setenv("LITEMEM_NAMESPACE", "app")
    cli = CliRunner()

    def run(*args: str) -> Any:
        result = cli.invoke(app, list(args))
        assert result.exit_code == 0, result.output
        return result.output

    assert "ready" in run("migrate")
    run("add", "--user", "ada", "--raw", "Lives in Berlin")
    [listed] = [json.loads(line) for line in run("list", "--user", "ada").splitlines()]
    assert listed["content"] == "Lives in Berlin"
    assert "Lives in Berlin" in run("search", "--user", "ada", "Berlin")
    assert [json.loads(line)["op"] for line in run("history", listed["id"]).splitlines()] == ["ADD"]
    assert json.loads(run("maintain"))["expired"] == 0
    assert "erased 1" in run("erase", "--user", "ada", "--yes")


def test_the_file_keeps_its_memories_across_opens(tmp_path: Path) -> None:
    from lite_mem.engine import MemoryEngine

    path = tmp_path / "memory.db"
    first = SqliteStore(path)
    MemoryEngine(first, "app", FakeEmbedder()).remember(Fact("Lives in Berlin"), Scope("ada"))
    first.close()
    again = SqliteStore(path)
    engine = MemoryEngine(again, "app", FakeEmbedder())
    assert [m.content for m in engine.list_memories(Scope("ada"))] == ["Lives in Berlin"]
    again.close()


def test_close_waits_for_a_running_transaction(tmp_path: Path) -> None:
    """Codex S7b: close() closed connections under a transaction another thread was using."""
    import threading

    from lite_mem.errors import StoreError

    store = SqliteStore(tmp_path / "memory.db")
    inside, release, done = threading.Event(), threading.Event(), {}

    def work() -> None:
        with store.transaction() as tx:
            inside.set()
            release.wait(5)
            done["rows"] = tx.scopes("app", None)  # still usable: close() has waited

    worker = threading.Thread(target=work)
    worker.start()
    inside.wait(5)
    closer = threading.Thread(target=store.close)
    closer.start()
    closer.join(0.3)
    assert closer.is_alive()  # waiting for the worker's transaction
    release.set()
    worker.join(5)
    closer.join(5)
    assert done == {"rows": []} and not closer.is_alive()
    with pytest.raises(StoreError, match="closed"):
        with store.transaction():
            pass


def test_shared_rows_are_found_through_the_index(tmp_path: Path) -> None:
    """Efficiency review: `ns AND (own OR shared)` made SQLite scan the whole namespace."""
    from lite_mem.stores.base import Filters
    from lite_mem.stores.sqlite import _where

    store = SqliteStore(tmp_path / "memory.db")
    where, params = _where(Filters(Scope("ada"), include_shared=True))
    with store.transaction() as tx:
        plan = [r[3] for r in tx._conn.execute(  # type: ignore[attr-defined]
            f"EXPLAIN QUERY PLAN SELECT m.id FROM memories m WHERE {where}", {**params, "ns": "app"}
        )]  # fmt: skip
    store.close()
    assert not any("memories_user" in step for step in plan), plan
    assert sum("memories_current" in step for step in plan) == 2, plan
