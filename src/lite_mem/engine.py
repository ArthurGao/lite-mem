"""`MemoryEngine`: the one core behind the library and the HTTP service (design §1, §2).

An engine is bound to one namespace and one embedding model (design §4). Every write goes through
`_apply`, one operation at a time inside one transaction: the memory rows, their history and the
idempotency record commit together."""

import dataclasses
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from lite_mem.config import MaintainConfig, Namespace, SearchConfig
from lite_mem.consolidate import Consolidator
from lite_mem.embedding import Embedder
from lite_mem.errors import ConfigError, Conflict, InvalidRequest, NotFound, redact
from lite_mem.extract import Extractor
from lite_mem.fusion import rrf
from lite_mem.ids import uuid7
from lite_mem.keyword import TOKENIZER, terms
from lite_mem.resolve import Resolver
from lite_mem.sensitive import Finding, Policy, find, mask
from lite_mem.stores.base import Filters, Store, Tx
from lite_mem.types import (
    MEMORY_TYPES,
    EventOp,
    Fact,
    HistoryEntry,
    Memory,
    MemoryEvent,
    MemoryHit,
    MemoryType,
    Message,
    Scope,
)

MAX_FACT_CHARS = 2000
MAX_TOP_K = 100
# A follow-up search (S10) searches each of the first round's top `follow_up` hits for what it
# adds to the question, and takes the best new hits from each (measured 2026-11-12: two-hop
# recall 13/20 -> 20/20 with one seed and two hits each, top 5).
MAX_FOLLOW_UP = 3
FOLLOW_UP_HITS = 2
# Bounds every interface shares (library, CLI, HTTP): a write's size, a list's length.
MAX_MESSAGES = 200
MAX_MESSAGE_CHARS = 20_000
MAX_LIST = 10_000

Clock = Callable[[], datetime]
_EVENTS = TypeAdapter(list[MemoryEvent])  # idempotency results, stored as JSON
log = logging.getLogger("lite_mem")


def _utcnow() -> datetime:
    return datetime.now(UTC)


# The write operations `_apply` knows (design §3). A new memory is an Add; the memories it
# `replaces` are closed with superseded_by the new one: one for a model's UPDATE (SUPERSEDE),
# several for consolidation (MERGE).


@dataclasses.dataclass(frozen=True)
class Add:
    fact: Fact
    replaces: tuple[uuid.UUID, ...] = ()
    closes_as: Literal["SUPERSEDE", "MERGE"] = "SUPERSEDE"
    vector: list[float] | None = None  # set by _write, before the transaction opens


@dataclasses.dataclass(frozen=True)
class Correct:
    memory_id: uuid.UUID
    content: str
    vector: list[float] | None = None


@dataclasses.dataclass(frozen=True)
class Retire:
    memory_id: uuid.UUID


@dataclasses.dataclass(frozen=True)
class Expire:
    """Close a memory (`maintain`) at `at`, by default its own expiry time: `as_of` keeps seeing
    it before then."""

    memory_id: uuid.UUID
    at: datetime | None = None


Op = Add | Correct | Retire | Expire


@dataclasses.dataclass(frozen=True)
class MaintenanceReport:
    """What one `maintain` run changed (design §6), and its events: what a caller mirroring this
    engine (llm-steward, S9) maps, as it maps an `add`'s."""

    expired: int = 0
    merged: int = 0  # memories closed into a merged one
    summarised: int = 0  # episodic memories closed into a summary
    forgotten: int = 0  # memories expired for low weight
    events: tuple[MemoryEvent, ...] = ()

    @property
    def closed(self) -> int:
        return self.expired + self.merged + self.summarised + self.forgotten


@dataclasses.dataclass(eq=False)
class _Step:
    """One planned write (identity matters: steps are removed by identity)."""

    op: Add | Retire
    reason: str | None
    produced_by: str | None


def _new_memory(fact: Fact, scope: Scope, now: datetime, memory_id: uuid.UUID) -> Memory:
    return Memory(
        id=memory_id, scope=scope, type=fact.type, key=fact.key, content=fact.content,
        importance=fact.importance, confidence=fact.confidence, valid_from=now, valid_to=None,
        superseded_by=None, expires_at=fact.expires_at, source=fact.source,
        metadata=fact.metadata, access_count=0, created_at=now, updated_at=now,
    )  # fmt: skip


def _inherit(sources: Sequence[Memory], content: str) -> Fact:
    """A new fact made of existing memories (a merge, a summary, a compound fact's remainder): one
    rule for what it keeps. The oldest source's type and metadata; the highest importance and
    confidence; it expires only when every source would (the latest of their times)."""
    ends = [m.expires_at for m in sources]
    return Fact(
        content, type=sources[0].type, importance=max(m.importance for m in sources),
        confidence=max(m.confidence for m in sources), metadata=dict(sources[0].metadata),
        expires_at=None if None in ends else max(e for e in ends if e is not None),
    )  # fmt: skip


class MemoryEngine:
    def __init__(
        self,
        store: Store,
        namespace: str,
        embedder: Embedder,
        *,
        extractor: Extractor | None = None,
        resolver: Resolver | None = None,
        related: int = 5,
        search: SearchConfig | None = None,
        sensitive: Policy = "drop",
        consolidator: Consolidator | None = None,
        maintenance: MaintainConfig | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self._ns = valid_namespace(namespace)
        self._store = store
        self._embedder = embedder
        self._extractor = extractor
        self._resolver = resolver
        self._related = related  # existing memories each new fact is compared with
        self._search = search or SearchConfig()
        self._sensitive = sensitive  # what happens to a secret (lite_mem.sensitive)
        self._consolidator = consolidator  # None: maintain neither merges nor summarises
        self._maintenance = maintenance or MaintainConfig()
        self._now = clock
        with store.transaction() as tx:
            tx.bind_namespace(
                self._ns,
                provider=embedder.provider,
                model=embedder.model,
                dim=embedder.dim,
                tokenizer=TOKENIZER,
            )

    @property
    def namespace(self) -> str:
        return self._ns

    def close(self) -> None:
        """Release the store's connections."""
        self._store.close()

    # writes

    def add(
        self,
        messages: Sequence[Message],
        scope: Scope,
        *,
        infer: bool = True,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> list[MemoryEvent]:
        """Remember what a conversation says. `infer=False`: each user message is stored as said
        (one semantic memory each). `infer=True` (design §2): the extractor proposes facts, each
        is compared with its related memories in exactly this scope and resolved (ADD, UPDATE as
        SUPERSEDE, DELETE, NOOP), and the whole batch is written in one transaction. Model calls
        run outside any transaction; a retry with the same key replays without calling them."""
        if len(messages) > MAX_MESSAGES:
            raise InvalidRequest(f"at most {MAX_MESSAGES} messages per write")
        if any(len(m.content) > MAX_MESSAGE_CHARS for m in messages):
            raise InvalidRequest(f"a message is at most {MAX_MESSAGE_CHARS} characters")
        metadata = self._stated_metadata(metadata or {})
        if infer:
            return self._infer(messages, scope, metadata, idempotency_key)
        facts = [
            Fact(m.content, metadata=dict(metadata))
            for m in messages
            if m.role == "user" and m.content.strip()
        ]
        return self._write(scope, [Add(f) for f in self._derived(facts)], idempotency_key)

    def remember(
        self, fact: Fact, scope: Scope, *, idempotency_key: str | None = None
    ) -> list[MemoryEvent]:
        """Store one fact as given (a profile fact supersedes the key's current value)."""
        return self._write(scope, [Add(self._stated_fact(fact))], idempotency_key)

    def update(
        self,
        memory_id: uuid.UUID,
        content: str,
        *,
        reason: str | None = None,
        produced_by: str | None = None,
    ) -> Memory:
        """Correct a current memory's wording in place (CORRECT: same id; the history keeps the
        old text). A changed fact is a new memory that supersedes the old one instead."""
        return self._change(Correct(memory_id, self._stated(content)), reason, produced_by)

    def delete(
        self,
        memory_id: uuid.UUID,
        *,
        reason: str | None = None,
        produced_by: str | None = None,
    ) -> Memory:
        """Retire a current memory: it stops being current, `as_of` still sees it."""
        return self._change(Retire(memory_id), reason, produced_by)

    def erase(self, user_id: str) -> int:
        """Delete everything about one user — memories, history, stored write results — for
        GDPR-style requests. The only hard delete (design §4). Returns how many memories."""
        if not user_id.strip():
            raise InvalidRequest("erase needs a user_id")
        with self._store.transaction() as tx:
            return tx.erase_user(self._ns, user_id)

    # maintenance (design §6): offline, never on the write path

    def maintain(self, user_id: str | None = None) -> MaintenanceReport:
        """Consolidate and forget, for one user or the whole namespace (design §6). Safe to
        repeat. Model calls run outside transactions; a group whose sources changed meanwhile is
        skipped (the next run sees it again)."""
        cfg, now = self._maintenance, self._now()
        events: list[MemoryEvent] = []
        with self._store.transaction() as tx:
            for memory in tx.expired(self._ns, now, user_id):
                events += self._closing(tx, Expire(memory.id), "its expires_at passed")
            expired = len(events)
            scopes = tx.scopes(self._ns, user_id)
        merged, summarised, forgotten = [], [], []
        for scope in scopes:
            if self._consolidator is not None and cfg.merge_threshold is not None:
                for kind in cfg.merge_types:  # one type per group: a merge never changes type
                    merged += self._merge_similar(scope, kind, self._consolidator, now)
            if self._consolidator is not None and cfg.summarise_after_days is not None:
                summarised += self._summarise_episodes(scope, self._consolidator, now)
            if cfg.forget_below is not None:
                forgotten += self._forget_idle(scope, now)
        events += merged + summarised + forgotten
        return MaintenanceReport(
            expired, len(merged), len(summarised), len(forgotten), tuple(events)
        )

    def _merge_similar(
        self, scope: Scope, kind: MemoryType, consolidator: Consolidator, now: datetime
    ) -> list[MemoryEvent]:
        threshold = self._maintenance.merge_threshold
        assert threshold is not None
        exact = Filters(scope, mode="exact", types=(kind,), now=now)
        with self._store.transaction() as tx:
            memories = tx.list_memories(self._ns, exact)
            vectors = tx.vectors(self._ns, [m.id for m in memories])
        by_id = {m.id: m for m in memories}
        grouped: set[uuid.UUID] = set()
        groups: list[list[Memory]] = []
        for memory in memories:  # oldest first: each memory joins one group at most
            if memory.id in grouped or memory.id not in vectors:
                continue
            # One short transaction per lookup: on SQLite every transaction holds the write lock,
            # and a long grouping pass would make every other read and write wait.
            with self._store.transaction() as tx:
                near = tx.vector_search(self._ns, vectors[memory.id], exact, self._related + 1)
            group = [memory] + [
                by_id[i] for i, sim in near
                if i in by_id and i != memory.id and i not in grouped and sim >= threshold
            ]  # fmt: skip
            if len(group) > 1:
                grouped.update(m.id for m in group)
                groups.append(group)
        steps: list[_Step] = []
        for group in groups:
            decision = consolidator.merge(group)
            if decision.content is not None:
                fact = _inherit(group, decision.content)
                steps += self._merging(scope, fact, group, decision.reason, decision.produced_by)
        return self._commit_merges(scope, steps)

    def _summarise_episodes(
        self, scope: Scope, consolidator: Consolidator, now: datetime
    ) -> list[MemoryEvent]:
        cfg = self._maintenance
        assert cfg.summarise_after_days is not None
        cutoff = now - timedelta(days=cfg.summarise_after_days)
        exact = Filters(scope, mode="exact", types=("episodic",), now=now)
        with self._store.transaction() as tx:
            old = [m for m in tx.list_memories(self._ns, exact) if m.valid_from < cutoff]
        if len(old) < cfg.summarise_min:
            return []
        steps: list[_Step] = []
        for summary in consolidator.summarise(old):
            sources = [m for m in old if m.id in summary.sources]
            fact = dataclasses.replace(_inherit(sources, summary.content), type="semantic")
            why = f"summarises {len(sources)} episodes"
            steps += self._merging(scope, fact, sources, why, summary.produced_by)
        return self._commit_merges(scope, steps)

    def _merging(
        self, scope: Scope, fact: Fact, sources: Sequence[Memory], why: str, by: str | None
    ) -> list[_Step]:
        """The step that merges `sources` into `fact`, if the fact passes the screening."""
        ids = tuple(m.id for m in sources)
        return [
            _Step(Add(f, replaces=ids, closes_as="MERGE"), why, by)
            for f in self._screened([fact], scope)
        ]

    def _commit_merges(self, scope: Scope, steps: list[_Step]) -> list[MemoryEvent]:
        """Write each merge whose sources all still hold in exactly this scope, checked under
        their locks (taken in id order, so two runs cannot deadlock or both merge them)."""
        ops = self._embedded([s.op for s in steps])
        events: list[MemoryEvent] = []
        with self._store.transaction() as tx:
            now = self._now()
            for step, op in zip(steps, ops, strict=True):
                assert isinstance(op, Add)
                locked = [tx.get_memory(self._ns, i, lock=True) for i in sorted(op.replaces)]
                if all(m is not None and m.holds_at(now) and m.scope == scope for m in locked):
                    events += self._apply(tx, scope, op, step.produced_by, step.reason)
        return [e for e in events if e.op == "MERGE"]

    def _forget_idle(self, scope: Scope, now: datetime) -> list[MemoryEvent]:
        cfg = self._maintenance
        assert cfg.forget_below is not None
        decaying = tuple(
            t for t in MEMORY_TYPES if t != "profile" and self._search.half_lives.get(t)
        )
        if not decaying:
            return []
        before = now - timedelta(days=cfg.forget_idle_days)
        exact = Filters(scope, mode="exact", types=decaying, now=now)
        events: list[MemoryEvent] = []
        with self._store.transaction() as tx:
            why = f"weight below {cfg.forget_below}, idle {cfg.forget_idle_days} days"
            for memory in tx.idle(self._ns, exact, before):
                if self._weight(memory, now) < cfg.forget_below:
                    events += self._closing(tx, Expire(memory.id, at=now), why)
        return events

    def _closing(self, tx: Tx, op: Expire, why: str) -> list[MemoryEvent]:
        """Expire a memory found by a read, if it is still open under its lock (a concurrent write
        may have closed it meanwhile: then nothing is done)."""
        memory = tx.get_memory(self._ns, op.memory_id, lock=True)
        if memory is None or not memory.current:
            return []
        return self._apply(tx, memory.scope, op, None, why)

    # reads

    def health(self, *, deep: bool = True) -> dict[str, str]:
        """See `health()`."""
        return health(self._store, self._embedder, deep=deep)

    def get(self, memory_id: uuid.UUID) -> Memory:
        with self._store.transaction() as tx:
            return self._existing(tx, memory_id)

    def list_memories(
        self,
        scope: Scope,
        *,
        include_shared: bool = False,
        as_of: datetime | None = None,
        types: Sequence[MemoryType] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """The scope's current memories, or those in effect at `as_of`, oldest first; a page of
        them with `limit` (at most MAX_LIST) and `offset`."""
        if limit is not None and not 1 <= limit <= MAX_LIST:
            raise InvalidRequest(f"limit must be between 1 and {MAX_LIST}")
        if offset < 0:
            raise InvalidRequest("offset must not be negative")
        filters = _filters(scope, include_shared, as_of, types, None, self._now())
        with self._store.transaction() as tx:
            return tx.list_memories(self._ns, filters, limit=limit, offset=offset)

    def search(
        self,
        query: str,
        scope: Scope,
        *,
        top_k: int = 10,
        include_shared: bool = True,
        types: Sequence[MemoryType] | None = None,
        metadata: dict[str, Any] | None = None,
        as_of: datetime | None = None,
        follow_up: int = 0,
    ) -> list[MemoryHit]:
        """The memories that answer `query`, best first (design §5): vector and BM25 candidates
        fused by RRF, then weighted by importance, age (per type) and confidence.

        `follow_up` (0 to MAX_FOLLOW_UP, off by default) adds a second round for questions that
        need two memories ("where does my sister work": "Lily is my sister", then what Lily
        does): the top `follow_up` hits are searched again for the words the question lacks, and
        their best new hits (`via` = the hit they came from) follow the first round's top half
        (rounded up).
        Scores from the two rounds are not comparable; the order is the ranking."""
        if not query.strip():
            raise InvalidRequest("search needs a query")
        if not 1 <= top_k <= MAX_TOP_K:
            raise InvalidRequest(f"top_k must be between 1 and {MAX_TOP_K}")
        if not 0 <= follow_up <= min(MAX_FOLLOW_UP, top_k):
            raise InvalidRequest(f"follow_up must be between 0 and {MAX_FOLLOW_UP} and <= top_k")
        filters = _filters(scope, include_shared, as_of, types, metadata, self._now())
        [vector] = self._embedder.embed([query])  # before the transaction: a model call
        with self._store.transaction() as tx:
            hits = self._rank(tx, query, vector, filters, top_k)
            if not follow_up:
                return self._touched(tx, hits)
        probes = self._probes(query, hits[:follow_up])  # between the rounds: a model call
        with self._store.transaction() as tx:
            # ranked again, so everything returned holds in this one snapshot (what changed
            # during the model call is not returned)
            first = self._rank(tx, query, vector, filters, top_k)
            return self._touched(tx, self._followed(tx, first, follow_up, probes, filters, top_k))

    def _touched(self, tx: Tx, hits: list[MemoryHit]) -> list[MemoryHit]:
        """A caller's retrieval counts as an access of every hit it returns."""
        tx.touch(self._ns, [h.memory.id for h in hits], self._now())
        return hits

    def _probes(
        self, query: str, seeds: list[MemoryHit]
    ) -> list[tuple[uuid.UUID, str, list[float]]]:
        """Each seed's terms the question lacks (seeds that add none are dropped), embedded in
        one call: what the second round of `search` looks for."""
        asked = set(terms(query))
        lacking = ((h.memory.id, [t for t in terms(h.memory.content) if t not in asked])
                   for h in seeds)  # fmt: skip
        texts = [(seed, " ".join(words)) for seed, words in lacking if words]
        vectors = self._embedder.embed([text for _, text in texts]) if texts else []
        return [(seed, text, v) for (seed, text), v in zip(texts, vectors, strict=True)]

    def _followed(
        self,
        tx: Tx,
        first: list[MemoryHit],
        seeds: int,
        probes: list[tuple[uuid.UUID, str, list[float]]],
        filters: Filters,
        top_k: int,
    ) -> list[MemoryHit]:
        """The second round's merge (design §5): the first round's top half, rounded up, stays as
        it is — a single-hop answer never moves down — then each probe's best new hits (at most
        FOLLOW_UP_HITS), then the rest of the first round, cut to `top_k`."""
        hits = first[: (top_k + 1) // 2]
        seen = {h.memory.id for h in first[: max(seeds, len(hits))]}  # the seeds, the kept
        for seed, text, vector in probes:
            new = [h for h in self._rank(tx, text, vector, filters, top_k)
                   if h.memory.id not in seen][:FOLLOW_UP_HITS]  # fmt: skip
            hits += [dataclasses.replace(h, via=seed) for h in new]
            seen |= {h.memory.id for h in new}
        kept = {h.memory.id for h in hits}
        return (hits + [h for h in first if h.memory.id not in kept])[:top_k]

    def _rank(
        self,
        tx: Tx,
        query: str,
        vector: list[float],
        filters: Filters,
        top_k: int,
        candidates: int | None = None,
    ) -> list[MemoryHit]:
        """Hybrid ranking inside `tx` with a vector already computed: vector and BM25 candidates,
        RRF, then the weights. Internal lookups (related memories, S4) use it without counting
        an access."""
        cfg, now = self._search, self._now()
        n = candidates or cfg.candidates
        by_vector = [i for i, _ in tx.vector_search(self._ns, vector, filters, n)]
        by_text = tx.text_search(self._ns, query, filters, n)
        relevance = rrf([by_vector, by_text], k=cfg.rrf_k)
        found = tx.get_memories(self._ns, list(relevance))
        hits = (
            MemoryHit(m, relevance[m.id] * self._weight(m, now), relevance[m.id]) for m in found
        )
        return sorted(hits, key=lambda h: (-round(h.score, 12), h.memory.id))[:top_k]

    def _weight(self, memory: Memory, now: datetime) -> float:
        """Importance x decay (by the type's half-life) x confidence (design §5)."""
        cfg = self._search
        weight = 1 + cfg.importance_boost * memory.importance
        half_life = cfg.half_lives.get(memory.type)
        if half_life:
            age_days = max((now - memory.valid_from).total_seconds(), 0) / 86400
            weight *= 0.5 ** (age_days / half_life)
        if memory.confidence < cfg.low_confidence:
            weight *= cfg.low_confidence_factor
        return weight

    def history(self, memory_id: uuid.UUID) -> list[HistoryEntry]:
        """Every change to the memory, oldest first (each memory has at least its ADD)."""
        with self._store.transaction() as tx:
            entries = tx.history(self._ns, memory_id)
        if not entries:
            raise NotFound(f"no memory {memory_id} in namespace {self._ns!r}")
        return entries

    # internals

    def _write(
        self, scope: Scope, adds: Sequence[Add], idempotency_key: str | None
    ) -> list[MemoryEvent]:
        """Write a caller's facts in one transaction; a key already used replays its result."""
        _check_scope(scope)
        now = self._now()
        checked = [dataclasses.replace(op, fact=_checked(op.fact, scope)) for op in adds]
        for op in checked:
            if op.fact.expires_at is not None and op.fact.expires_at <= now:
                raise InvalidRequest("expires_at is already past: such a memory would never hold")
        if idempotency_key is not None:
            replay = self.replay(idempotency_key, scope)
            if replay is not None:
                return replay
        ready = self._embedded(checked)  # model calls stay outside the transaction
        return self._commit(
            scope,
            idempotency_key,
            lambda tx: [e for op in ready for e in self._apply(tx, scope, op, None, None)],
        )

    def _commit(
        self, scope: Scope, key: str | None, body: Callable[[Tx], list[MemoryEvent]]
    ) -> list[MemoryEvent]:
        """One write transaction: claim the idempotency key (a concurrent holder's result is
        replayed), run `body`, store its events under the key."""
        with self._store.transaction() as tx:
            if key is not None:
                held = tx.claim_idempotency(self._ns, key, scope.user_id)
                if held is not None:  # a concurrent write with the same key got there first
                    return self._replayed(held, scope, key)
            events = body(tx)
            if key is not None:
                tx.store_idempotency(self._ns, key, _EVENTS.dump_python(events, mode="json"))
            return events

    def replay(self, idempotency_key: str, scope: Scope) -> list[MemoryEvent] | None:
        """The events of the committed write that used this key, or None. Cheap and read-only:
        callers with expensive work (extraction) check it before doing that work."""
        with self._store.transaction() as tx:
            held = tx.lookup_idempotency(self._ns, idempotency_key)
        return None if held is None else self._replayed(held, scope, idempotency_key)

    def _replayed(self, held: tuple[str | None, Any], scope: Scope, key: str) -> list[MemoryEvent]:
        user_id, result = held
        if user_id != scope.user_id:  # never hand one user's write to another
            raise Conflict(f"idempotency key {key!r} was already used for another scope")
        return _EVENTS.validate_python(result)

    def _infer(
        self,
        messages: Sequence[Message],
        scope: Scope,
        metadata: dict[str, Any] | None,
        idempotency_key: str | None,
    ) -> list[MemoryEvent]:
        if self._extractor is None or self._resolver is None:
            raise ConfigError("infer=true needs an extractor and a resolver (see open_engine)")
        _check_scope(scope)
        if idempotency_key is not None:
            replay = self.replay(idempotency_key, scope)
            if replay is not None:
                return replay  # before any model call
        extraction = self._extractor.extract(list(messages))
        given = [dataclasses.replace(f, metadata=dict(metadata or {})) for f in extraction.facts]
        adds = self._embedded([Add(f) for f in self._screened(given, scope)])
        exact, k = Filters(scope, mode="exact", now=self._now()), self._related
        with self._store.transaction() as tx:  # read-only: what each fact may conflict with
            stored = [
                [h.memory for h in self._rank(tx, a.fact.content, a.vector or [], exact, k, 4 * k)]
                for a in adds
            ]
        plan = self._plan(scope, adds, stored, extraction.produced_by)
        for step, op in zip(plan, self._embedded([s.op for s in plan]), strict=True):
            step.op = op  # a remainder the plan added still needs its vector
        return self._commit(scope, idempotency_key, lambda tx: self._carry_out(tx, scope, plan))

    def _plan(
        self,
        scope: Scope,
        adds: list[Add],
        stored: list[list[Memory]],
        extracted_by: str | None,
    ) -> list[_Step]:
        """Resolve the facts one by one (model calls, no transaction). A fact's candidates are its
        related memories plus the facts this batch has already accepted (batch-local ids), so two
        wordings of one change do not both land. The model proposes; this decides (rule 5): a
        target outside the candidates is not applied, and every such downgrade says why."""
        assert self._resolver is not None
        steps: list[_Step] = []
        accepted: dict[uuid.UUID, _Step] = {}  # batch-local id -> the step that adds it
        claimed: dict[uuid.UUID, uuid.UUID] = {}  # stored memory -> batch-local id replacing it
        for add, near in zip(adds, stored, strict=True):
            pending = [  # facts this batch accepted, shown like memories (batch-local ids)
                _new_memory(step.op.fact, scope, self._now(), local)
                for local, step in accepted.items()
                if isinstance(step.op, Add)
            ]
            decision = self._resolver.resolve(add.fact.content, near + pending)
            action, target, reason = decision.action, decision.target, decision.reason
            allowed = {m.id for m in near} | set(accepted)
            if action in ("UPDATE", "DELETE") and target not in allowed:
                action, reason = "ADD", f"{action} of {target} is not among the candidates; added"
            if target in claimed:  # replaced earlier in this batch: aim at what replaced it
                target = claimed[target]
            if action == "NOOP":
                continue
            replaces: uuid.UUID | None = None
            if target is not None and target in accepted:  # a fact of this very batch
                earlier = accepted.pop(target)
                steps.remove(earlier)
                if action == "UPDATE" and isinstance(earlier.op, Add) and earlier.op.replaces:
                    replaces = earlier.op.replaces[0]  # the later wording takes over its target
            elif action == "UPDATE":
                replaces = target
            elif action == "DELETE" and target is not None:
                steps.append(_Step(Retire(target), reason, decision.produced_by))
            local = uuid7()
            by = decision.produced_by if replaces is not None else extracted_by
            held = (replaces,) if replaces is not None else ()
            step = _Step(dataclasses.replace(add, replaces=held), reason, by)
            steps.append(step)
            accepted[local] = step
            if replaces is not None:
                claimed[replaces] = local
            old = next((m for m in near if m.id == target), None)
            if (
                old is not None
                and decision.remainder
                and action in ("UPDATE", "DELETE")
                and _restates(decision.remainder, old.content)
                and not _held(decision.remainder, [add.fact.content, *(m.content for m in near)])
            ):
                steps += self._kept(old, decision.remainder, scope, decision.produced_by)
        return steps

    def _kept(self, old: Memory, remainder: str, scope: Scope, by: str | None) -> list[_Step]:
        """What still holds of a compound memory the batch replaces or retires: its own memory,
        screened like any derived fact (it needs a vector: see `_infer`)."""
        why = f"kept from {old.id}, which changed"
        return [_Step(Add(f), why, by) for f in self._screened([_inherit([old], remainder)], scope)]

    def _screened(self, facts: Sequence[Fact], scope: Scope) -> list[Fact]:
        """Derived facts as they may be written: secrets dropped or masked, then the strict gate."""
        return [_checked(f, scope) for f in self._derived(facts)]

    def _derived(self, facts: Sequence[Fact]) -> list[Fact]:
        """Facts the engine derived itself (an extractor's, a merge's, a remainder): the same
        screening as a caller's fact, every field; one that would be refused is dropped."""
        kept: list[Fact] = []
        dropped: list[str] = []
        for fact in facts:
            try:
                kept.append(self._stated_fact(fact))
            except InvalidRequest as exc:  # its message names kinds, never the secret
                dropped.append(str(exc))
        if dropped:
            log.warning("not stored: %d derived fact(s): %s", len(dropped), sorted(set(dropped)))
        return kept

    def _stated_fact(self, fact: Fact) -> Fact:
        """A fact with every text field screened: content and source masked or refused, a name
        (profile key, metadata key) refused (see `_named`)."""
        return dataclasses.replace(
            fact,
            content=self._stated(fact.content),
            key=None if fact.key is None else self._named(fact.key, "a profile key"),
            source=None if fact.source is None else self._stated(fact.source),
            metadata=self._stated_metadata(fact.metadata),
        )

    def _secrets(self, text: str) -> list[Finding]:
        return [] if self._sensitive == "keep" else find(text)

    def _stated_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """A caller's metadata, screened like content: every text value in it, at any depth."""

        def screen(value: Any) -> Any:
            if isinstance(value, str):
                found = self._secrets(value)
                if found and self._sensitive == "drop":
                    kinds = ", ".join(sorted({f.kind for f in found}))
                    raise InvalidRequest(f"not stored: metadata holds a secret ({kinds})")
                return mask(value) if found else value
            if isinstance(value, dict):
                return {self._named(str(k), "a metadata key"): screen(v) for k, v in value.items()}
            if isinstance(value, list):
                return [screen(v) for v in value]
            return value

        return screen(metadata)  # type: ignore[no-any-return]

    def _named(self, name: str, what: str) -> str:
        """A name (a profile key, a metadata key) holding a secret is refused under any policy
        but keep: a masked name would no longer name the same thing."""
        found = self._secrets(name)
        if found:
            kinds = ", ".join(sorted({f.kind for f in found}))
            raise InvalidRequest(f"not stored: {what} holds a secret ({kinds})")
        return name

    def _stated(self, content: str) -> str:
        """A fact a caller states explicitly: masked, or refused so the caller knows."""
        found = self._secrets(content)
        if not found:
            return content
        if self._sensitive == "mask":
            return mask(content)
        kinds = ", ".join(sorted({f.kind for f in found}))
        raise InvalidRequest(f"not stored: it holds a secret ({kinds}); sensitive policy is drop")

    def _carry_out(self, tx: Tx, scope: Scope, steps: list[_Step]) -> list[MemoryEvent]:
        """Apply a plan in the write transaction. A stored target that changed since it was looked
        up is not applied: the fact is added instead, and says so."""
        events: list[MemoryEvent] = []
        for step in steps:
            op, reason = step.op, step.reason
            target = op.memory_id if isinstance(op, Retire) else next(iter(op.replaces), None)
            if target is not None:
                found = tx.get_memory(self._ns, target, lock=True)
                if found is None or not found.holds_at(self._now()):
                    if isinstance(op, Retire):
                        continue  # already gone: nothing to retire
                    op = dataclasses.replace(op, replaces=())
                    reason = f"{target} changed meanwhile; added instead"
            events += self._apply(tx, scope, op, step.produced_by, reason)
        return events

    def _embedded[O: Op](self, ops: Sequence[O]) -> list[O]:
        """The ops with the vectors their new text needs, from one batch call (an op that already
        carries one keeps it)."""
        todo = [op for op in ops if isinstance(op, Add | Correct) and op.vector is None]
        texts = [op.content if isinstance(op, Correct) else op.fact.content for op in todo]
        vectors = iter(self._embedder.embed(texts) if texts else [])
        done = {id(op): dataclasses.replace(op, vector=next(vectors)) for op in todo}
        return [done.get(id(op), op) for op in ops]

    def _change(self, op: Correct | Retire, reason: str | None, produced_by: str | None) -> Memory:
        if isinstance(op, Correct):
            op = dataclasses.replace(op, content=_content(op.content))
        [ready] = self._embedded([op])
        with self._store.transaction() as tx:
            current = self._current(tx, op.memory_id)
            self._apply(tx, current.scope, ready, produced_by, reason)
            return self._existing(tx, op.memory_id)

    def _apply(
        self, tx: Tx, scope: Scope, op: Op, produced_by: str | None, reason: str | None
    ) -> list[MemoryEvent]:
        """One write operation inside the caller's transaction, with its history events; what it
        reports (a MERGE reports one event per memory it closed)."""
        now = self._now()
        if reason and self._secrets(reason):  # a reason is free text: masked, never refused
            reason = mask(reason)
        if isinstance(op, Correct):
            memory = self._current(tx, op.memory_id)
            content = op.content  # checked by _change
            assert op.vector is not None  # _write / _change embed every op before applying it
            tx.set_content(self._ns, memory.id, content, now)
            tx.index_memory(self._ns, memory.id, content, op.vector)
            self._event(tx, memory, "CORRECT", {"content": memory.content}, {"content": content},
                        reason, produced_by, now)  # fmt: skip
            return [MemoryEvent("CORRECT", memory.id, content, reason, produced_by)]
        if isinstance(op, Retire | Expire):
            kind: EventOp
            if isinstance(op, Expire):  # past its time: not "current", but still open
                memory, kind = self._existing(tx, op.memory_id, lock=True), "EXPIRE"
                due = op.at or memory.expires_at
                assert due is not None and memory.valid_to is None
                at = max(due, memory.valid_from)  # never before it began
            else:
                memory, kind, at = self._current(tx, op.memory_id), "DELETE", now
            tx.close_memory(self._ns, memory.id, at)
            self._event(tx, memory, kind, {"valid_to": None}, {"valid_to": at.isoformat()},
                        reason, produced_by, now)  # fmt: skip
            return [MemoryEvent(kind, memory.id, memory.content, reason, produced_by)]
        fact, events = op.fact, []
        memory = _new_memory(fact, scope, now, uuid7())
        replaced = [self._current(tx, i) for i in op.replaces]
        if fact.key is not None and not replaced:
            tx.lock_profile_slot(self._ns, scope, fact.key)
            holder = tx.current_profile(self._ns, scope, fact.key, lock=True)
            if holder is not None and not holder.holds_at(now):  # it ran out: that is no change
                events += self._apply(tx, scope, Expire(holder.id), None, "its expires_at passed")
            elif holder is not None:
                replaced = [holder]
        for old in replaced:  # closed first: it frees a profile key's current slot
            tx.close_memory(self._ns, old.id, now, superseded_by=memory.id)
        tx.insert_memory(self._ns, memory)
        assert op.vector is not None
        tx.index_memory(self._ns, memory.id, memory.content, op.vector)
        after: dict[str, Any] = {"content": memory.content, "type": memory.type, "key": memory.key}
        if op.closes_as == "MERGE":
            after["merged_from"] = [str(old.id) for old in replaced]
        self._event(tx, memory, "ADD" if op.closes_as == "SUPERSEDE" else "MERGE", None, after,
                    reason, produced_by, now)  # fmt: skip
        if not replaced:
            return [*events, MemoryEvent("ADD", memory.id, memory.content, reason, produced_by)]
        why = reason or f"profile key {fact.key!r} has a new value"
        for old in replaced:
            self._event(tx, old, op.closes_as, {"valid_to": None},
                        {"valid_to": now.isoformat(), "superseded_by": str(memory.id)},
                        why, produced_by, now)  # fmt: skip
            events.append(
                MemoryEvent(op.closes_as, memory.id, memory.content, why, produced_by, old.id)
            )
        return events

    def _event(
        self,
        tx: Tx,
        memory: Memory,
        op: EventOp,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        reason: str | None,
        produced_by: str | None,
        at: datetime,
    ) -> None:
        tx.add_event(
            self._ns, memory, op, before=before, after=after, reason=reason,
            produced_by=produced_by, at=at,
        )  # fmt: skip

    def _existing(self, tx: Tx, memory_id: uuid.UUID, *, lock: bool = False) -> Memory:
        memory = tx.get_memory(self._ns, memory_id, lock=lock)
        if memory is None:
            raise NotFound(f"no memory {memory_id} in namespace {self._ns!r}")
        return memory

    def _current(self, tx: Tx, memory_id: uuid.UUID) -> Memory:
        memory = self._existing(tx, memory_id, lock=True)
        if not memory.holds_at(self._now()):
            raise Conflict(
                f"memory {memory_id} is no longer current"
                + (f" (superseded by {memory.superseded_by})" if memory.superseded_by else "")
                + (" (expired)" if memory.valid_to is None else "")
            )
        return memory


def health(store: Store, embedder: Embedder, *, deep: bool) -> dict[str, str]:
    """ "ok" or "error" for the store and, when `deep`, the embedding model (one embedding: it also
    keeps the model loaded, so frequent probes should not be deep). The reason is logged
    (redacted), never returned: it can carry endpoints."""
    checks: list[tuple[str, Callable[[], object]]] = [("store", store.ping)]
    if deep:
        checks.append(("embedding", lambda: embedder.embed(["health"])))
    parts: dict[str, str] = {}
    for part, check in checks:
        try:
            check()
            parts[part] = "ok"
        except Exception as exc:  # any failure is "error"
            log.warning("health: %s failed: %s", part, redact(f"{type(exc).__name__}: {exc}"))
            parts[part] = "error"
    return parts


def _content(text: str) -> str:
    content = text.strip()
    if not content:
        raise InvalidRequest("a memory needs content")
    if len(content) > MAX_FACT_CHARS:
        raise InvalidRequest(f"a memory is at most {MAX_FACT_CHARS} characters")
    return content


def _filters(
    scope: Scope,
    include_shared: bool,
    as_of: datetime | None,
    types: Sequence[MemoryType] | None,
    metadata: dict[str, Any] | None,
    now: datetime,
) -> Filters:
    _check_scope(scope)
    _aware(as_of, "as_of")
    kinds = tuple(types) if types else None
    if kinds and any(t not in MEMORY_TYPES for t in kinds):
        raise InvalidRequest(f"types must be among {MEMORY_TYPES}")
    if metadata and not all(isinstance(k, str) for k in metadata):
        raise InvalidRequest("metadata filter keys must be strings")
    return Filters(
        scope, include_shared=include_shared, as_of=as_of, types=kinds,
        metadata=metadata or None, now=now,
    )  # fmt: skip


def _restates(part: str, whole: str) -> bool:
    """Whether `part` repeats what `whole` said (most of its words are in it): a remainder is
    what still holds of the old fact, never a fact of its own (S9: qwen3 named an unrelated,
    already stored fact as the remainder of an atomic one)."""
    words, said = set(terms(part)), set(terms(whole))
    return bool(words) and len(words & said) >= 0.5 * len(words)


def _held(text: str, contents: Sequence[str]) -> bool:
    """Whether `text` is already one of these, up to case and spacing."""
    norm = " ".join(text.lower().split())
    return any(" ".join(c.lower().split()) == norm for c in contents)


def valid_namespace(namespace: str) -> str:
    try:
        return TypeAdapter(Namespace).validate_python(namespace)
    except ValidationError as exc:
        raise InvalidRequest(f"invalid namespace {namespace!r}") from exc


def _aware(at: datetime | None, name: str) -> None:
    """A time without a zone means something different on each machine (and in each store):
    refused rather than guessed."""
    if at is not None and at.utcoffset() is None:
        raise InvalidRequest(f"{name} needs a timezone (e.g. 2026-01-01T09:00:00+00:00)")


def _check_scope(scope: Scope) -> None:
    for name in ("user_id", "agent_id", "session_id"):
        value = getattr(scope, name)
        if value is not None and not value.strip():
            raise InvalidRequest(f"scope {name} is empty: leave it out (None) instead")


def _checked(fact: Fact, scope: Scope) -> Fact:
    """The strict gate for a caller's fact (a model's output is normalised before it, S4)."""
    if fact.type not in MEMORY_TYPES:
        raise InvalidRequest(f"type must be one of {MEMORY_TYPES}")
    _aware(fact.expires_at, "expires_at")
    if (fact.type == "profile") != (fact.key is not None):
        raise InvalidRequest("a profile memory has a key, and only a profile memory has one")
    if fact.key is not None and scope.session_id is not None:
        raise InvalidRequest("a profile value belongs to a user, not a session: drop session_id")
    for name in ("importance", "confidence"):
        if not 0 <= getattr(fact, name) <= 1:
            raise InvalidRequest(f"{name} must be between 0 and 1")
    return dataclasses.replace(fact, content=_content(fact.content))
