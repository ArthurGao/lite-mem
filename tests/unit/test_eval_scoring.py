import hashlib
import json
from pathlib import Path

from lite_mem import eval as ev
from lite_mem.config import Settings
from lite_mem.errors import ModelError

ROOT = Path(__file__).parents[2]


def test_present_and_stale_follow_the_keywords() -> None:
    now = ["住在上海", "Moved from Auckland to Wellington", "Used to be vegetarian, eats fish now"]
    assert ev.remembered(now, ["Shanghai", "上海"])
    assert not ev.remembered(now, ["北京", "Beijing"])
    assert not ev.stale(now, ev.Stale(old=["Auckland"], unless=["Wellington"]))  # a change story
    assert ev.stale(["Lives in Auckland"], ev.Stale(old=["Auckland"], unless=["Wellington"]))
    assert ev.stale(["password 482915"], ev.Stale(old=["482915"]))


def test_naming_the_new_value_does_not_excuse_the_old_one() -> None:
    """Only wording that puts the old value in the past does (Codex S5 review)."""
    data = ev.load(ROOT / "eval" / "dataset.json")
    checks = [s for u in data.users for x in u.sessions for s in x.stale]
    auckland = next(s for s in checks if s.old == ["Auckland"])
    assert ev.stale(["Emma lives in Auckland and works in Wellington"], auckland)
    assert not ev.stale(["Emma moved from Auckland to Wellington"], auckland)
    beijing = next(s for s in checks if "北京" in s.old)
    assert ev.stale(["住在北京和上海"], beijing)
    assert not ev.stale(["以前住在北京，现在住上海"], beijing)


class _Canned:
    """A system that remembers exactly what each user said (add-only, no model)."""

    name = "canned"

    def __init__(self) -> None:
        self.memory: dict[str, list[str]] = {}

    def add(self, user_id: str, messages: list[dict[str, str]]) -> None:
        said = [m["content"] for m in messages if m["role"] == "user"]
        self.memory.setdefault(user_id, []).extend(said)

    def current(self, user_id: str) -> list[str]:
        return list(self.memory.get(user_id, []))

    def search(self, user_id: str, query: str, k: int) -> list[str]:
        return self.current(user_id)[-k:]

    def reset(self, user_ids: list[str]) -> None:
        self.memory.clear()

    def maintain(self, user_id: str) -> int:
        return 0


def test_the_run_scores_a_system_and_the_table_shows_it() -> None:
    data = ev.load(ROOT / "eval" / "dataset.json")
    report = ev.run(_Canned(), data, k=3)
    assert (report.present, report.questions) == (19, 11)
    assert report.remembered == report.present  # it keeps everything said...
    assert report.stale >= 4  # ...including what changed: Beijing, the old job, the password
    assert "still remembers '482915'" in " ".join(report.misses)
    assert report.memories == 11 and len(report.add_seconds) == 11
    assert "| canned | 19/19 (100%) |" in ev.table([report]) and "hit@3" in ev.table([report])


def test_described_settings_keep_no_credentials() -> None:
    settings = Settings.model_validate(
        {
            "database_url": "postgresql://u:dbpw@db.test/x",
            "llm": {
                "base_url": "http://u:urlpw@llm.test:8080/v1?api_key=qkey",
                "api_key": "llmkey",
                "roles": {"resolve": {"base_url": "https://r.test/v1?token=rtok"}},
            },
            "embedding": {"base_url": "http://e.test/api", "api_key": "embkey"},
        }
    )
    described = ev.describe(settings, {"llm", "embedding", "pipeline"})
    text = json.dumps(described)
    for secret in ("dbpw", "urlpw", "qkey", "llmkey", "rtok", "embkey", "database_url"):
        assert secret not in text
    assert described["llm"]["base_url"] == "http://llm.test:8080/v1"
    assert described["llm"]["roles"]["resolve"]["base_url"] == "https://r.test/v1"
    assert described["embedding"]["base_url"] == "http://e.test/api"
    assert described["llm"]["model"] == settings.llm.model


class _Merging(_Canned):
    """Maintenance after each session keeps only the latest thing the user said."""

    name = "merging"

    def maintain(self, user_id: str) -> int:
        said = self.memory.get(user_id, [])
        closed, self.memory[user_id] = len(said[:-1]), said[-1:]
        return closed


def test_maintenance_runs_after_each_session_and_is_counted() -> None:
    report = ev.run(_Merging(), ev.load(ROOT / "eval" / "dataset.json"), k=3)
    assert report.memories == 3  # one per user at the end
    assert report.closed == 11 - 3  # every memory but each user's last was closed
    assert "| merging |" in ev.table([report]) and "| closed |" in ev.table([report])


class _Flaky(_Canned):
    """Every third write fails the way a model endpoint does (Ollama answers 500 now and then)."""

    name = "flaky"

    def __init__(self) -> None:
        super().__init__()
        self.writes = 0

    def add(self, user_id: str, messages: list[dict[str, str]]) -> None:
        self.writes += 1
        if self.writes % 3 == 0:
            raise ModelError("resolve (openai:qwen3:8b) failed: InternalServerError")
        super().add(user_id, messages)


def test_a_failed_call_is_counted_and_the_run_goes_on() -> None:
    report = ev.run(_Flaky(), ev.load(ROOT / "eval" / "dataset.json"), k=3)
    assert report.errors == 3 and len(report.add_seconds) == 11  # every session was still tried
    assert sum("failed: InternalServerError" in m for m in report.misses) == 3
    assert "| errors |" in ev.table([report])


def test_a_saved_run_records_what_produced_it(tmp_path: Path) -> None:
    data_path = ROOT / "eval" / "dataset.json"
    report = ev.run(_Canned(), ev.load(data_path), k=3)
    out = tmp_path / "run.json"
    ev.save(out, [report], dataset=data_path, settings={"canned": {"llm": {"model": "m"}}})
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["dataset_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert saved["k"] == 3 and saved["settings"] == {"canned": {"llm": {"model": "m"}}}
    (row,) = saved["reports"]
    assert (row["system"], row["remembered"], row["present"], row["recall"]) == (
        "canned",
        19,
        19,
        1.0,
    )
    assert row["misses"] == report.misses and "git" in saved and "at" in saved


def test_variants_override_the_settings_and_get_their_own_namespace() -> None:
    base = Settings(_env_file=None, namespace="app")  # type: ignore[call-arg]
    s = ev.variant_settings(base, "add_only", "lite_mem_eval")
    assert (s.namespace, s.db_schema) == ("eval/add-only", "lite_mem_eval")
    assert (s.pipeline.resolver, s.llm.reasoning) == ("add_only", "none")
    assert ev.variant_settings(base, "thinking", "lite_mem_eval").llm.reasoning is None
