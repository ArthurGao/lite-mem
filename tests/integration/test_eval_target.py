"""lite-mem's evals on lite-eval (design §8, §12 S11).

Layer 2 of lite-eval design §18.16: the frozen `eval/multihop.py` (tests/legacy_multihop.py) and
the lite-eval suite, in one process, return the same hits for every question and so the same
hop1 / hop2 / both counts, for every config of the suite. CI runs it with FakeEmbedder; `-m ollama`
runs it with the real bge-m3 and also checks the counts recorded on 2026-11-12.
"""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from lite_eval.datasets.loaders import import_dataset
from lite_eval.datasets.local import load_dataset
from lite_eval.runner import RunResult, run_suite
from lite_eval.store import load_run

from lite_mem import eval_target, factory
from lite_mem.engine import MAX_FACT_CHARS, MemoryEngine
from lite_mem.eval_target import CachedEmbedder
from lite_mem.testing import FakeEmbedder
from tests import legacy_multihop as legacy

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval" / "lite-eval"
SUITE = EVAL / "suites" / "multihop.yaml"
DATASET = EVAL / "datasets" / "multihop"
RECORDED = REPO / "eval" / "results" / "2026-11-12-s10-multihop.json"
CASES = [json.loads(line) for line in (DATASET / "cases.jsonl").read_text().splitlines()]
CASES_BY_ID = {c["id"]: c for c in CASES}
SAMPLE = Path(__file__).parents[1] / "data" / "longmemeval_sample.json"


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> list[FakeEmbedder]:
    """Every engine either side opens embeds with a FakeEmbedder (the model boundary)."""
    made: list[FakeEmbedder] = []

    def build(config: Any) -> FakeEmbedder:
        made.append(FakeEmbedder())
        return made[-1]

    monkeypatch.setattr(factory, "build_embedder", build)
    return made


@pytest.fixture(autouse=True)
def _temp_here(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The frozen script's throwaway database goes under tmp_path, not the system temp dir."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def legacy_data() -> dict[str, Any]:
    """The old `eval/multihop.json`, rebuilt from the dataset (users in file order)."""
    users = list(dict.fromkeys(c["input"]["user"] for c in CASES))
    return {
        "users": [
            {
                "user_id": user,
                "memories": json.loads((DATASET / "corpus" / f"{user}.json").read_text()),
                "questions": [
                    {
                        "q": c["input"]["question"],
                        "answer": c["expected"]["answer"],
                        "hops": c["expected"]["hops"],
                    }
                    for c in CASES
                    if c["input"]["user"] == user
                ],
            }
            for user in users
        ]
    }


# sha256 of the `users` of `eval/multihop.json` at 7491e36 (before S11), as canonical JSON
ORIGINAL_USERS = "8426f924b6e04313847bd756ad230d4b97e21aa78fb8d8ccfe7ff2478b937cbd"


def test_the_dataset_holds_exactly_the_original_two_hop_data() -> None:
    """Codex S11: the equivalence below rebuilds the old input from the new dataset, so a label
    changed in the move would change both sides; this pins the dataset to the original file."""
    users = legacy_data()["users"]
    text = json.dumps(users, sort_keys=True, ensure_ascii=False)
    assert hashlib.sha256(text.encode()).hexdigest() == ORIGINAL_USERS


def test_a_setup_that_fails_leaves_no_temporary_directory(
    tmp_path: Path, fake_embedder: list[FakeEmbedder], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex S11: lite-eval tears down only targets whose setup succeeded."""

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(eval_target, "open_components", broken)
    for target in (eval_target.multihop(5), eval_target.longmemeval(5)):
        dataset = load_dataset(DATASET)
        with pytest.raises(RuntimeError, match="store unavailable"):
            asyncio.run(target.setup(dataset))
    assert list(tmp_path.glob("lite-mem-eval-*")) == []


def by_config(results: list[RunResult]) -> dict[str, RunResult]:
    return {str(r.record.target.config["name"]): r for r in results}


def assert_same_as_legacy(results: list[RunResult]) -> dict[tuple[int, int], dict[str, int]]:
    """Every config's hits and hop scores equal the frozen script's; returns its counts."""
    data = legacy_data()
    counts = {}
    assert len(results) == 8
    for result in results:
        config = result.record.target.config
        top_k, follow_up = int(config["top_k"]), int(config["follow_up"])
        old = legacy.run(data, top_k, follow_up)
        old_rows = {(r["user"], r["q"]): r for r in old["rows"]}
        outputs = {o.case_id: o for o in load_run(result.directory).outputs}
        scores = {(s.case_id, s.scorer): s.value for s in result.scores}
        assert len(outputs) == len(old_rows) == 20
        for case_id, output in outputs.items():
            case = CASES_BY_ID[case_id]
            row = old_rows[(case["input"]["user"], case["input"]["question"])]
            assert output.error is None
            assert output.value == row["found"], (config["name"], case_id)  # the same hits
            assert scores[(case_id, "hop1")] == float(row["hop1"])
            assert scores[(case_id, "hop2")] == float(row["hop2"])
            assert scores[(case_id, "all_hops")] == float(row["hop1"] and row["hop2"])
        n = len(outputs)
        metrics = result.record.summary.metrics
        new_counts = {m: round(metrics[k].value * n) for m, k in
                      (("hop1", "hop1"), ("hop2", "hop2"), ("both", "all_hops"))}  # fmt: skip
        old_counts = {m: old["summary"][m] for m in ("hop1", "hop2", "both")}
        assert new_counts == old_counts, config["name"]
        counts[(top_k, follow_up)] = new_counts
    return counts


def test_multihop_on_lite_eval_returns_what_the_old_script_returned(
    tmp_path: Path, fake_embedder: list[FakeEmbedder]
) -> None:
    results = run_suite(SUITE, root=tmp_path / "root", no_cache=True)
    # The eight configs share one setup: one engine, the memories stored once
    assert len(fake_embedder) == 1
    assert all(w.code != "setup_not_shared" for r in results for w in r.record.warnings)
    counts = assert_same_as_legacy(results)
    assert len(set(map(json.dumps, counts.values()))) > 1  # the configs differ: not a constant


@pytest.mark.ollama
def test_multihop_with_bge_m3_matches_the_old_script_and_the_recorded_counts(
    tmp_path: Path,
) -> None:
    results = run_suite(SUITE, root=tmp_path / "root", no_cache=True)
    counts = assert_same_as_legacy(results)
    recorded = json.loads(RECORDED.read_text())
    expected = {(r["top_k"], r["follow_up"]): r for r in recorded["engine_follow_up"]["runs"]}
    expected |= {(r["top_k"], 0): r | {"follow_up": 0} for r in recorded["baseline"]}
    for key, got in counts.items():
        assert got == {m: expected[key][m] for m in ("hop1", "hop2", "both")}, key


# ------------------------------------------------------------------ LongMemEval


def longmemeval_suite(tmp_path: Path) -> Path:
    raw = SAMPLE.read_text().replace('"LONG"', json.dumps("x" * (MAX_FACT_CHARS + 1)))
    src = tmp_path / "longmemeval_sample.json"
    src.write_text(raw)
    dataset = import_dataset("longmemeval", src, tmp_path / "datasets" / "lme")
    suite = tmp_path / "suites" / "lme.yaml"
    suite.parent.mkdir()
    suite.write_text(
        json.dumps(
            {
                "name": "longmemeval",
                "dataset": str(dataset),
                "scorers": ["recall@1", "hit@2"],
                "targets": [
                    {
                        "python": "lite_mem.eval_target:longmemeval",
                        "configs": [{"name": "k1", "top_k": 1}, {"name": "k3", "top_k": 3}],
                    }
                ],
            }
        )
    )
    return suite


def test_longmemeval_cases_are_answered_with_session_ids_from_user_turns_only(
    tmp_path: Path, fake_embedder: list[FakeEmbedder], monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[str] = []
    add = MemoryEngine.add

    def counted(self: MemoryEngine, messages: Any, *args: Any, **kwargs: Any) -> Any:
        writes.append(kwargs["metadata"]["session_id"])
        return add(self, messages, *args, **kwargs)

    monkeypatch.setattr(MemoryEngine, "add", counted)
    results = by_config(run_suite(longmemeval_suite(tmp_path), root=tmp_path, no_cache=True))
    # Each question's database is built once (one write per session with a kept turn) and
    # searched by both configs, not built again for the second
    assert sorted(writes) == ["s-cat", "s-garden", "s-garden", "s-trip", "s-trip"]
    outputs = {o.case_id: o for o in load_run(results["k3"].directory).outputs}
    pet = outputs["q-pet"]
    assert pet.error is None
    # Ids only (no benchmark text in the record); the cat is in s-cat's user turn. The garden
    # session's assistant mentions the cat too, but assistant turns are not memories.
    assert pet.value[0] == {"id": "s-cat"}
    assert sorted(h["id"] for h in pet.value) == ["s-cat", "s-garden", "s-trip"]
    [store] = pet.trace.spans
    # 4 non-blank user turns; the one over MAX_FACT_CHARS, which the engine refuses (the whole
    # write, found on the oracle run: 2 of 470 questions failed), is left out and counted
    assert store.attrs == {"turns": 4, "stored": 3, "sessions": 3}
    assert outputs["q-trip"].value == [{"id": "s-trip"}, {"id": "s-garden"}]
    scores = {(s.case_id, s.scorer): s.value for s in results["k1"].scores}
    assert scores == {
        ("q-pet", "recall@1"): 1.0,
        ("q-pet", "hit@2"): 1.0,
        ("q-trip", "recall@1"): 1.0,
        ("q-trip", "hit@2"): 1.0,
    }
    # Both configs share the leader's cache: each distinct text reached the model once (the
    # probe of the dimension aside, FakeEmbedder learns it without a call)
    [model] = fake_embedder  # built once, by the leader; the follower shares its cache
    sent = [t for batch in model.calls for t in batch]
    questions = ["What is the name of my cat?", "Which town is the June trip to?"]
    turns = [
        "I adopted a cat and named the cat Miso.",
        "Any tips for growing tomatoes in a garden?",
        "We are planning a trip to Nelson in June.",
    ]
    assert sorted(sent) == sorted(turns + questions)


def test_the_embedding_cache_returns_the_vectors_the_model_gave(
    fake_embedder: list[FakeEmbedder],
) -> None:
    model, reference = FakeEmbedder(), FakeEmbedder()
    cached = CachedEmbedder(model)
    # exact vectors, in order, duplicates included
    assert cached.embed(["a b", "c", "a b"]) == reference.embed(["a b", "c", "a b"])
    assert cached.embed(["c", "d"]) == reference.embed(["c", "d"])
    assert model.calls == [["a b", "c"], ["d"]]
