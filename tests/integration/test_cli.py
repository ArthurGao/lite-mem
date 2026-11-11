"""The CLI over an engine with fakes (the factory is replaced at its boundary)."""

import json
from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

from lite_mem import factory
from lite_mem.cli import app
from lite_mem.engine import MemoryEngine
from lite_mem.stores.pg import PgStore
from tests.conftest import TEST_SCHEMA, fake_engine


@pytest.fixture
def cli(pg_store: PgStore, pg_url: str, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """`store` is asked for its migrated schema; the CLI opens its own pool on it."""

    monkeypatch.setattr(
        factory, "open_engine", lambda *_, **__: fake_engine(PgStore(pg_url, TEST_SCHEMA))
    )
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

    def variant_engine(settings: Any, *_: Any, **__: Any) -> MemoryEngine:
        return fake_engine(PgStore(pg_url, settings.db_schema), settings.namespace)

    monkeypatch.setattr(factory, "open_engine", variant_engine)
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
        raise RuntimeError(f"cannot connect to {pg_url}")  # third-party errors may echo the URL

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
    assert result.exit_code == 1 and opened == []
    assert "error:" in result.output
    password = psycopg.conninfo.conninfo_to_dict(pg_url).get("password")
    assert password and password not in result.output
    assert app.pretty_exceptions_show_locals is False  # a crash never prints locals (the URL)


def test_a_failing_mem0_run_keeps_the_lite_mem_results_and_hides_the_url(
    pg: Any, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from lite_mem import eval_mem0

    class BrokenMem0:
        name = "mem0-2.2.1"

        def __init__(self, *_: Any) -> None: ...

        def reset(self, user_ids: list[str]) -> None: ...

        def add(self, user_id: str, messages: Any) -> None:
            raise RuntimeError(f"lost connection to {pg_url}")

        def close(self) -> None: ...

    def variant_engine(settings: Any, *_: Any, **__: Any) -> MemoryEngine:
        return fake_engine(PgStore(pg_url, settings.db_schema), settings.namespace)

    monkeypatch.setattr(eval_mem0, "Mem0System", BrokenMem0)
    monkeypatch.setattr(factory, "open_engine", variant_engine)
    monkeypatch.setenv("LITEMEM_DATABASE_URL", pg_url)
    dataset = Path(__file__).parents[2] / "eval" / "dataset.json"
    result = CliRunner().invoke(
        app,
        ["eval", "-v", "add_only", "--mem0", "--dataset", str(dataset), "--schema", TEST_SCHEMA],
    )
    assert result.exit_code == 0, result.output
    assert "| add_only |" in result.output and "mem0 run failed: RuntimeError" in result.output
    password = psycopg.conninfo.conninfo_to_dict(pg_url).get("password")
    assert password and password not in result.output


def test_key_new_prints_the_token_once_and_the_entry_to_configure() -> None:
    from lite_mem.config import ApiKey
    from lite_mem.server.auth import authenticate

    result = CliRunner().invoke(app, ["key", "new", "steward", "--namespace", "steward/*"])
    assert result.exit_code == 0, result.output
    token = next(w for w in result.output.split() if w.startswith("lm_steward_"))
    entry = ApiKey.model_validate_json(
        next(line for line in result.output.splitlines() if line.startswith("{"))
    )
    assert entry.namespaces == ["steward/*"] and token.split("_", 2)[2] not in entry.sha256
    assert authenticate(f"Bearer {token}", [entry]) == entry


def test_serve_refuses_a_public_address_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    started: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: started.append(kw))
    monkeypatch.setenv("LITEMEM_NAMESPACE", "app")
    refused = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0"])
    assert refused.exit_code == 1 and "API key" in refused.output and started == []
    ok = CliRunner().invoke(app, ["serve"])
    assert ok.exit_code == 0, ok.output
    assert (started[0]["host"], started[0]["port"]) == ("127.0.0.1", 8710)


def test_maintain_prints_what_it_changed(cli: CliRunner, pg_store: PgStore) -> None:
    from datetime import UTC, datetime, timedelta

    from lite_mem.engine import MemoryEngine
    from lite_mem.testing import FakeEmbedder
    from lite_mem.types import Fact, Scope

    now = datetime.now(UTC)  # the CLI's engine runs on the real clock: write it two hours ago
    earlier = MemoryEngine(pg_store, "app", FakeEmbedder(), clock=lambda: now - timedelta(hours=2))
    gone = now - timedelta(hours=1)
    earlier.remember(Fact("parked on level 3", expires_at=gone), Scope(user_id="ada"))
    result = cli.invoke(app, ["maintain", "--user", "ada"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"expired": 1, "merged": 0, "summarised": 0, "forgotten": 0}
