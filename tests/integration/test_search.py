"""S3 acceptance (design §12): hybrid search — BM25 equal to the formula, RRF, weights, scope,
types, metadata and as_of filters — with search entries committed with the row."""

from datetime import timedelta
from typing import Any

import psycopg
import pytest
from psycopg import sql

from lite_mem.engine import MemoryEngine
from lite_mem.errors import ModelError
from lite_mem.keyword import idf, term_counts, term_score, terms
from lite_mem.stores.base import Filters, Store
from lite_mem.stores.pg import PgStore
from lite_mem.testing import FakeEmbedder
from lite_mem.types import Fact, Scope
from tests.conftest import TEST_SCHEMA, Clock, count_rows, tx_class

ADA = Scope(user_id="ada")
BOB = Scope(user_id="bob")


def _found(engine: MemoryEngine, query: str, scope: Scope = ADA, **kw: Any) -> list[str]:
    return [h.memory.content for h in engine.search(query, scope, **kw)]


def _formula_order(docs: list[str], query: str) -> list[str]:
    """`docs` ranked by keyword.py's BM25 with statistics over exactly these docs."""
    q = sorted(set(terms(query)))
    counts = [term_counts(d) for d in docs]
    n, avgdl = len(docs), sum(sum(c.values()) for c in counts) / len(docs)
    df = {t: sum(1 for c in counts if t in c) for t in q}
    scored = [
        (-sum(idf(n, df[t]) * term_score(c[t], sum(c.values()), avgdl) for t in q if t in c), d)
        for d, c in zip(docs, counts, strict=True)
        if any(t in c for t in q)
    ]
    return [d for _, d in sorted(scored)]


def test_bm25_statistics_are_the_scopes_not_the_namespaces(
    engine: MemoryEngine, store: Store
) -> None:
    """S8b: one user's ranking must not depend on what other users said (and its cost must grow
    with the user, not the namespace). Bob's many teas would make "tea" common namespace-wide."""
    mine = ["tea tea tea", "morning walk"]
    for d in mine:
        engine.remember(Fact(d), ADA)
    for n in range(30):
        engine.remember(Fact(f"tea number {n}"), BOB)
    query = "morning tea"
    namespace_order = _formula_order(mine + [f"tea number {n}" for n in range(30)], query)
    assert [d for d in namespace_order if d in mine] != _formula_order(mine, query)  # it matters
    with store.transaction() as tx:
        ids = tx.text_search("app", query, Filters(ADA), 10)
        got = {m.id: m.content for m in tx.get_memories("app", ids)}
    assert [got[i] for i in ids] == _formula_order(mine, query)


def test_sql_bm25_ranks_exactly_as_the_formula(engine: MemoryEngine, store: Store) -> None:
    docs = [
        # written first, so it has the smaller id: without length normalisation it would tie
        # with the short one below and win the tie; the formula ranks the short one higher
        "kombucha lover who also drinks plenty of water juice soda milk and lemonade daily",
        "kombucha fan",
        "green tea every morning",
        "tea tea tea with milk",
        "coffee in the morning, never tea",
        "lives in Berlin near the river",
    ]
    for d in docs:
        engine.remember(Fact(d), ADA)
    query = "morning tea kombucha"
    q = sorted(set(terms(query)))
    counts = [term_counts(d) for d in docs]
    n, avgdl = len(docs), sum(sum(c.values()) for c in counts) / len(docs)
    df = {t: sum(1 for c in counts if t in c) for t in q}
    expected = sorted(
        (
            (
                -sum(idf(n, df[t]) * term_score(c[t], sum(c.values()), avgdl) for t in q if t in c),
                d,
            )
            for d, c in zip(docs, counts, strict=True)
            if any(t in c for t in q)
        ),
    )
    with store.transaction() as tx:
        ids = tx.text_search("app", query, Filters(ADA), 10)
        got = {m.id: m.content for m in tx.get_memories("app", ids)}
    assert [got[i] for i in ids] == [d for _, d in expected]


def test_hybrid_search_finds_the_memory_that_answers(engine: MemoryEngine) -> None:
    for d in ["Lives in Berlin", "Prefers green tea", "Works as a nurse", "Has a cat named Mo"]:
        engine.remember(Fact(d), ADA)
    [best, *_] = engine.search("what tea does she like", ADA)
    assert best.memory.content == "Prefers green tea" and best.relevance > 0


def test_chinese_memories_are_found_by_their_words(engine: MemoryEngine) -> None:
    engine.remember(Fact("我住在上海浦东"), ADA)
    engine.remember(Fact("喜欢喝绿茶"), ADA)
    assert _found(engine, "住在哪里 上海", top_k=1) == ["我住在上海浦东"]


def test_search_stays_in_its_scope_and_namespace(store: Store, clock: Clock) -> None:
    app = MemoryEngine(store, "app", FakeEmbedder(), clock=clock)
    other = MemoryEngine(store, "other", FakeEmbedder(), clock=clock)
    app.remember(Fact("Ada likes green tea"), ADA)
    app.remember(Fact("Bob likes green tea"), BOB)
    app.remember(Fact("The shop sells green tea"), Scope())
    other.remember(Fact("Ada likes green tea in another app"), ADA)
    assert sorted(_found(app, "green tea")) == ["Ada likes green tea", "The shop sells green tea"]
    assert _found(app, "green tea", include_shared=False) == ["Ada likes green tea"]
    assert _found(other, "green tea") == ["Ada likes green tea in another app"]


def test_a_superseded_value_is_not_found_now_but_is_as_of_then(
    engine: MemoryEngine, clock: Clock
) -> None:
    engine.remember(Fact("Lives in Berlin", type="profile", key="city"), ADA)
    then = clock.now
    clock.tick()
    engine.remember(Fact("Lives in Shanghai", type="profile", key="city"), ADA)
    assert _found(engine, "where does she live Berlin Shanghai") == ["Lives in Shanghai"]
    assert _found(engine, "where does she live Berlin Shanghai", as_of=then) == ["Lives in Berlin"]


def test_a_correction_reindexes_the_text(engine: MemoryEngine, store: Store) -> None:
    [e] = engine.remember(Fact("Likes grean tee"), ADA)
    engine.update(e.memory_id, "Likes green tea")
    assert _found(engine, "green tea") == ["Likes green tea"]
    with store.transaction() as tx:  # the old words are gone from the text index
        assert tx.text_search("app", "grean tee", Filters(ADA), 10) == []
        assert tx.text_search("app", "green tea", Filters(ADA), 10) == [e.memory_id]


def test_weights_age_importance_and_confidence(engine: MemoryEngine, clock: Clock) -> None:
    engine.remember(Fact("Asked about the refund policy", type="episodic"), ADA)
    clock.tick(timedelta(days=90).total_seconds())
    engine.remember(Fact("Asked about the refund policy again", type="episodic"), ADA)
    hits = engine.search("refund policy", ADA)
    assert hits[0].memory.content == "Asked about the refund policy again"  # 90 days: 1/8
    old = next(h for h in hits if h.memory.content == "Asked about the refund policy")
    assert old.score == pytest.approx(old.relevance * (1 + 0.3 * 0.5) * 0.5**3)

    engine.remember(Fact("Prefers email", importance=1.0), BOB)
    engine.remember(Fact("Prefers email maybe", confidence=0.3), BOB)
    [sure, unsure] = engine.search("prefers email", BOB)
    assert sure.memory.content == "Prefers email"
    assert unsure.score == pytest.approx(unsure.relevance * (1 + 0.3 * 0.5) * 0.8)


def test_type_and_metadata_filters(engine: MemoryEngine) -> None:
    engine.remember(Fact("Prefers short answers", type="procedural", metadata={"src": "chat"}), ADA)
    engine.remember(Fact("Short answers were asked for once", type="episodic"), ADA)
    assert _found(engine, "short answers", types=["procedural"]) == ["Prefers short answers"]
    assert _found(engine, "short answers", metadata={"src": "chat"}) == ["Prefers short answers"]
    assert _found(engine, "short answers", metadata={"src": "email"}) == []


def test_a_hit_counts_as_an_access(engine: MemoryEngine, clock: Clock) -> None:
    [e] = engine.remember(Fact("Lives in Berlin"), ADA)
    engine.search("Berlin", ADA)
    engine.search("Berlin", ADA)
    assert engine.get(e.memory_id).access_count == 2


def test_an_index_failure_rolls_the_write_back(
    engine: MemoryEngine, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("index down")

    monkeypatch.setattr(tx_class(store), "index_memory", broken)
    with pytest.raises(RuntimeError):
        engine.remember(Fact("Lives in Berlin"), ADA)
    monkeypatch.undo()
    assert engine.list_memories(ADA) == []


def test_a_failed_embedding_writes_nothing(store: Store, clock: Clock) -> None:
    class Down(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise ModelError("embedding down")

    engine = MemoryEngine(store, "app", Down(), clock=clock)
    with pytest.raises(ModelError):
        engine.remember(Fact("Lives in Berlin"), ADA)
    assert engine.list_memories(ADA) == []


def test_erase_takes_the_search_entries_too(engine: MemoryEngine, store: Store) -> None:
    engine.remember(Fact("Lives in Berlin"), ADA)
    engine.remember(Fact("Lives in Paris"), BOB)
    engine.erase("ada")
    assert count_rows(store, "memory_vectors") == 1  # Bob's
    assert count_rows(store, "memory_postings") == len(term_counts("Lives in Paris"))
    assert _found(engine, "Berlin") == []


@pytest.mark.ollama
def test_a_real_round_trip_with_ollama(store: Store, clock: Clock) -> None:
    from lite_mem.config import EmbeddingConfig
    from lite_mem.embedding import OllamaEmbedder

    engine = MemoryEngine(store, "real", OllamaEmbedder(EmbeddingConfig()), clock=clock)
    for d in ["Lives in Berlin", "Prefers green tea", "Works night shifts as a nurse"]:
        engine.remember(Fact(d), ADA)
    assert _found(engine, "what does she drink", top_k=1) == ["Prefers green tea"]
    assert _found(engine, "她做什么工作", top_k=1) == ["Works night shifts as a nurse"]


def test_a_crowd_of_nearer_vectors_in_other_scopes_does_not_hide_ones_own(
    pg_store: Store, clock: Clock
) -> None:
    """The filter must take part in the nearest-neighbour search, not cut its result afterwards
    (PostgreSQL: with an HNSW index the planner may use; SqliteStore always ranks exactly)."""
    store = pg_store
    engine = MemoryEngine(store, "app", FakeEmbedder(), clock=clock)
    for i in range(300):  # nearer to the query than Ada's own memory, and all someone else's
        engine.remember(Fact(f"green tea green tea {i}"), Scope(user_id=f"u{i}"))
    engine.remember(Fact("drinks green tea sometimes with lemon and honey"), ADA)
    with store.transaction() as tx:
        # an index scan gives up after this many tuples (default 20000; small here, as in a big
        # namespace with many nearer strangers)
        tx._conn.execute("SET LOCAL hnsw.max_scan_tuples = 50")
        tx._conn.execute("SET LOCAL enable_seqscan = off")  # as the planner does at scale
        [vector] = FakeEmbedder().embed(["green tea"])
        nearest = tx.vector_search("app", vector, Filters(ADA), 5)
    assert len(nearest) == 1  # her one memory, found despite 300 nearer strangers


def test_exact_mode_sees_exactly_one_scope(engine: MemoryEngine, store: Store) -> None:
    """What S4's related-memory lookup uses: the scope itself — no shared rows, and a scope
    without a session does not reach into that user's sessions."""
    engine.remember(Fact("green tea, said in session 1"), Scope("ada", session_id="s1"))
    engine.remember(Fact("green tea, always"), ADA)
    engine.remember(Fact("green tea, the shop's"), Scope())
    [vector] = FakeEmbedder().embed(["green tea"])

    def contents(f: Filters) -> list[str]:
        with store.transaction() as tx:
            ids = [i for i, _ in tx.vector_search("app", vector, f, 10)]
            return sorted(m.content for m in tx.get_memories("app", ids))

    assert contents(Filters(ADA, mode="exact", include_shared=True)) == ["green tea, always"]
    assert contents(Filters(Scope("ada", session_id="s1"), mode="exact")) == [
        "green tea, said in session 1"
    ]
    assert len(contents(Filters(ADA, include_shared=True))) == 3  # visible: all three


def test_vector_search_reports_cosine_similarity(engine: MemoryEngine, store: Store) -> None:
    engine.remember(Fact("green tea"), ADA)
    engine.remember(Fact("black coffee"), ADA)
    [vector] = FakeEmbedder().embed(["green tea"])
    with store.transaction() as tx:
        [(_, same), (_, other)] = tx.vector_search("app", vector, Filters(ADA), 10)
    assert same == pytest.approx(1.0) and 0 <= other < same


def test_vector_tables_carry_no_unused_ann_index(
    pg_store: PgStore, migrated: psycopg.Connection, clock: Clock
) -> None:
    """S8: since S3 every vector search ranks exactly among the scope's rows, so an HNSW index
    was only a cost on each write. Binding creates none; migration 0004 drops old ones."""
    from lite_mem.db import migrate

    MemoryEngine(pg_store, "app", FakeEmbedder(), clock=clock)  # binds: the 64-d table
    schema = sql.Identifier(TEST_SCHEMA)

    def ann_indexes() -> list[str]:
        rows = migrated.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s AND indexdef ILIKE %s",
            (TEST_SCHEMA, "%using hnsw%"),
        ).fetchall()
        return [r[0] for r in rows]

    assert ann_indexes() == []
    migrated.execute(  # as S3 to S7 built it
        sql.SQL("CREATE INDEX memory_vectors_64_hnsw ON {}.memory_vectors_64 "
                "USING hnsw (embedding vector_cosine_ops)").format(schema)
    )  # fmt: skip
    migrated.execute(
        sql.SQL("DELETE FROM {}.schema_migrations WHERE version LIKE '0004%%'").format(schema)
    )
    migrate(migrated, TEST_SCHEMA)
    assert ann_indexes() == []


def test_wide_embeddings_fit_without_an_ann_index(pg_store: PgStore, clock: Clock) -> None:
    """HNSW capped vectors at 2000 dimensions; without it the `vector` type's 16 000 apply."""
    engine = MemoryEngine(pg_store, "wide", FakeEmbedder(dim=3072), clock=clock)
    engine.remember(Fact("Lives in Berlin"), ADA)
    assert [h.memory.content for h in engine.search("Berlin", ADA)] == ["Lives in Berlin"]
