"""The CLI over an engine with fakes (the factory is replaced at its boundary)."""

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from lite_mem import factory
from lite_mem.cli import app
from lite_mem.engine import MemoryEngine
from lite_mem.extract import RuleExtractor
from lite_mem.resolve import AddOnlyResolver
from lite_mem.stores.pg import PgStore
from lite_mem.testing import FakeEmbedder
from tests.conftest import TEST_SCHEMA


@pytest.fixture
def cli(store: PgStore, pg_url: str, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """`store` is asked for its migrated schema; the CLI opens its own pool on it."""

    def fake_engine(*_: Any, **__: Any) -> MemoryEngine:
        return MemoryEngine(
            PgStore(pg_url, TEST_SCHEMA), "app", FakeEmbedder(),
            extractor=RuleExtractor(), resolver=AddOnlyResolver(),
        )  # fmt: skip

    monkeypatch.setattr(factory, "open_engine", fake_engine)
    return CliRunner()


def _rows(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def test_add_search_list_history_erase(cli: CliRunner) -> None:
    added = cli.invoke(app, ["add", "Lives in Berlin", "Likes green tea", "--user", "ada"])
    assert added.exit_code == 0, added.output
    assert [r["op"] for r in _rows(added.output)] == ["ADD", "ADD"]
    found = _rows(cli.invoke(app, ["search", "green tea", "--user", "ada", "--top-k", "1"]).output)
    assert [r["content"] for r in found] == ["Likes green tea"]
    listed = _rows(cli.invoke(app, ["list", "--user", "ada"]).output)
    assert sorted(r["content"] for r in listed) == ["Likes green tea", "Lives in Berlin"]
    history = _rows(cli.invoke(app, ["history", found[0]["id"]]).output)
    assert [h["op"] for h in history] == ["ADD"]
    refused = cli.invoke(app, ["erase", "--user", "ada"])
    assert refused.exit_code == 1 and "--yes" in refused.output
    erased = cli.invoke(app, ["erase", "--user", "ada", "--yes"])
    assert "erased 2 memories" in erased.output
    assert _rows(cli.invoke(app, ["list", "--user", "ada"]).output) == []


def test_an_engine_error_is_reported_not_raised(cli: CliRunner) -> None:
    result = cli.invoke(app, ["history", "00000000-0000-0000-0000-000000000000"])
    assert result.exit_code == 1 and "no memory" in result.output
