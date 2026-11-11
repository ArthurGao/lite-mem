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


def test_eval_runs_the_variants_and_prints_the_table(
    pg: Any, pg_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from pathlib import Path

    def fake_engine(settings: Any, *_: Any, **__: Any) -> MemoryEngine:
        return MemoryEngine(
            PgStore(pg_url, settings.db_schema), settings.namespace, FakeEmbedder(),
            extractor=RuleExtractor(), resolver=AddOnlyResolver(),
        )  # fmt: skip

    monkeypatch.setattr(factory, "open_engine", fake_engine)
    monkeypatch.setenv("LITEMEM_DATABASE_URL", pg_url)
    dataset = Path(__file__).parents[2] / "eval" / "dataset.json"
    out = tmp_path / "run.json"
    result = CliRunner().invoke(
        app,
        ["eval", "-v", "resolve", "-v", "add_only", "--dataset", str(dataset),
         "--schema", TEST_SCHEMA, "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [r["system"] for r in saved["reports"]] == ["resolve", "add_only"]
    assert saved["settings"]["add_only"]["pipeline"]["resolver"] == "add_only"
    assert "database_url" not in saved["settings"]["resolve"]
    assert pg_url not in out.read_text(encoding="utf-8")
    rows = [line for line in result.output.splitlines() if line.startswith("| ")]
    assert [r.split("|")[1].strip() for r in rows[2:]] == ["resolve", "add_only"]
    assert "19/19" in rows[2]  # the rule extractor keeps every user message
    unknown = CliRunner().invoke(app, ["eval", "-v", "nope", "--dataset", str(dataset)])
    assert unknown.exit_code == 1 and "unknown variant" in unknown.output


def test_eval_sets_up_mem0_before_the_long_lite_mem_runs(
    pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from lite_mem import eval_mem0

    opened: list[str] = []

    def no_mem0(*_: Any, **__: Any) -> None:
        raise ImportError("The 'ollama' library is required.")

    def track(settings: Any, *_: Any, **__: Any) -> MemoryEngine:
        opened.append(settings.namespace)
        raise AssertionError("no lite-mem run may start before mem0 is set up")

    monkeypatch.setattr(eval_mem0, "Mem0System", no_mem0)
    monkeypatch.setattr(factory, "open_engine", track)
    monkeypatch.setenv("LITEMEM_DATABASE_URL", pg_url)
    dataset = Path(__file__).parents[2] / "eval" / "dataset.json"
    result = CliRunner().invoke(
        app, ["eval", "-v", "resolve", "--mem0", "--dataset", str(dataset), "--schema", TEST_SCHEMA]
    )
    assert isinstance(result.exception, ImportError) and opened == []
