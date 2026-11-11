"""The mem0 comparison adapter, with mem0 itself replaced at its import (the boundary)."""

import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import psycopg
import pytest

from lite_mem.config import Settings
from lite_mem.eval_mem0 import Mem0System
from tests.conftest import TEST_SCHEMA


class _FakeMemory:
    configs: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "_FakeMemory":
        cls.configs.append(config)
        return cls()

    def delete(self, memory_id: str) -> None:
        raise AssertionError("a run must never delete memories it did not create")


@pytest.fixture
def fake_mem0(monkeypatch: pytest.MonkeyPatch) -> type[_FakeMemory]:
    _FakeMemory.configs = []
    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(Memory=_FakeMemory))
    return _FakeMemory


def _table_exists(pg_url: str, name: str) -> bool:
    with psycopg.connect(pg_url) as conn:
        row = conn.execute("SELECT to_regclass(%s)", (f"{TEST_SCHEMA}.{name}",)).fetchone()
    return row is not None and row[0] is not None


def test_each_run_is_isolated_and_cleans_up_only_its_own(
    pg: Any, pg_url: str, fake_mem0: type[_FakeMemory]
) -> None:
    settings = Settings.model_validate(
        {
            "database_url": pg_url,
            "llm": {"model": "some-llm", "base_url": "http://llm.test/v1"},
            "embedding": {"model": "some-embedder", "dim": 8},
        }
    )
    first, second = (
        Mem0System(settings, schema=TEST_SCHEMA),
        Mem0System(settings, schema=TEST_SCHEMA),
    )
    a, b = fake_mem0.configs
    assert a["llm"]["config"]["model"] == "some-llm"  # the same models as the lite-mem variants
    assert a["llm"]["config"]["openai_base_url"] == "http://llm.test/v1"
    assert (a["embedder"]["config"]["model"], a["embedder"]["config"]["embedding_dims"]) == (
        "some-embedder",
        8,
    )
    collections = [c["vector_store"]["config"]["collection_name"] for c in (a, b)]
    assert collections[0] != collections[1]  # never the data of another run
    assert a["history_db_path"] != b["history_db_path"]
    first.reset(["emma"])  # nothing to delete in a fresh collection
    with psycopg.connect(pg_url, autocommit=True) as conn:
        for name in collections:  # what mem0 would create
            conn.execute(f'CREATE TABLE "{TEST_SCHEMA}"."{name}" (id int)')
    first.close()
    assert not _table_exists(pg_url, collections[0])
    assert _table_exists(pg_url, collections[1])
    assert not Path(a["history_db_path"]).parent.exists()
    second.close()
    assert not _table_exists(pg_url, collections[1])
