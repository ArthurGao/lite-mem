"""`MemoryEngine`: the one core behind the library and the HTTP service (design §1, §2).

An engine is bound to one namespace and one embedding model (design §4). Every write goes through
`_apply`, one operation at a time inside one transaction: the memory rows, their history and the
idempotency record commit together."""

import dataclasses
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from lite_mem.config import Namespace, SearchConfig
from lite_mem.embedding import Embedder
from lite_mem.errors import Conflict, InvalidRequest, NotFound
from lite_mem.fusion import rrf
from lite_mem.ids import uuid7
from lite_mem.keyword import TOKENIZER
from lite_mem.stores.base import Filters, Store, Tx
from lite_mem.types import (
    MEMORY_TYPES,
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

Clock = Callable[[], datetime]
_EVENTS = TypeAdapter(list[MemoryEvent])  # idempotency results, stored as JSON


def _utcnow() -> datetime:
    return datetime.now(UTC)


# The write operations `_apply` knows (design §3). A model's UPDATE decision is an Add that
# `replaces` the memory it changes: a new id, the old row closed (SUPERSEDE).


@dataclasses.dataclass(frozen=True)
class Add:
    fact: Fact
    replaces: uuid.UUID | None = None
    vector: list[float] | None = None  # set by _write, before the transaction opens


@dataclasses.dataclass(frozen=True)
class Correct:
    memory_id: uuid.UUID
    content: str
    vector: list[float] | None = None


@dataclasses.dataclass(frozen=True)
class Retire:
    memory_id: uuid.UUID


Op = Add | Correct | Retire


class MemoryEngine:
    def __init__(
        self,
        store: Store,
        namespace: str,
        embedder: Embedder,
        *,
        search: SearchConfig | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        try:
            self._ns = TypeAdapter(Namespace).validate_python(namespace)
        except ValidationError as exc:
            raise InvalidRequest(f"invalid namespace {namespace!r}") from exc
        self._store = store
        self._embedder = embedder
        self._search = search or SearchConfig()
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
        (one semantic memory each); `infer=True` extracts facts first (design §2, S4)."""
        if infer:
            raise InvalidRequest("infer=true needs an extractor (stage S4); send infer=false")
        facts = [
            Fact(m.content, metadata=dict(metadata or {}))
            for m in messages
            if m.role == "user" and m.content.strip()
        ]
        return self._write(scope, [Add(f) for f in facts], idempotency_key)

    def remember(
        self, fact: Fact, scope: Scope, *, idempotency_key: str | None = None
    ) -> list[MemoryEvent]:
        """Store one fact as given (a profile fact supersedes the key's current value)."""
        return self._write(scope, [Add(fact)], idempotency_key)

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
        return self._change(Correct(memory_id, content), reason, produced_by)

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

    # reads

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
    ) -> list[Memory]:
        """The scope's current memories, or those in effect at `as_of`."""
        filters = _filters(scope, include_shared, as_of, types, None)
        with self._store.transaction() as tx:
            return tx.list_memories(self._ns, filters)

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
    ) -> list[MemoryHit]:
        """The memories that answer `query`, best first (design §5): vector and BM25 candidates
        fused by RRF, then weighted by importance, age (per type) and confidence."""
        if not query.strip():
            raise InvalidRequest("search needs a query")
        if not 1 <= top_k <= MAX_TOP_K:
            raise InvalidRequest(f"top_k must be between 1 and {MAX_TOP_K}")
        filters = _filters(scope, include_shared, as_of, types, metadata)
        [vector] = self._embedder.embed([query])  # before the transaction: a model call
        with self._store.transaction() as tx:
            hits = self._rank(tx, query, vector, filters, top_k)
            tx.touch(self._ns, [h.memory.id for h in hits], self._now())  # a caller's retrieval
        return hits

    def _rank(
        self, tx: Tx, query: str, vector: list[float], filters: Filters, top_k: int
    ) -> list[MemoryHit]:
        """Hybrid ranking inside `tx` with a vector already computed: vector and BM25 candidates,
        RRF, then the weights. Internal lookups (related memories, S4) use it without counting
        an access."""
        cfg, now = self._search, self._now()
        by_vector = [i for i, _ in tx.vector_search(self._ns, vector, filters, cfg.candidates)]
        by_text = tx.text_search(self._ns, query, filters, cfg.candidates)
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
        self,
        scope: Scope,
        ops: Sequence[Op],
        idempotency_key: str | None,
        *,
        produced_by: str | None = None,
        reason: str | None = None,
    ) -> list[MemoryEvent]:
        """Apply `ops` for `scope` in one transaction; a key already used replays its result."""
        _check_scope(scope)
        checked = [dataclasses.replace(op, fact=_checked(op.fact, scope)) if isinstance(op, Add)
                   else op for op in ops]  # fmt: skip
        if idempotency_key is not None:
            replay = self.replay(idempotency_key, scope)
            if replay is not None:
                return replay
        checked = self._embedded(checked)  # model calls stay outside the transaction
        with self._store.transaction() as tx:
            if idempotency_key is not None:
                held = tx.claim_idempotency(self._ns, idempotency_key, scope.user_id)
                if held is not None:  # a concurrent write with the same key got there first
                    return self._replayed(held, scope, idempotency_key)
            events = [self._apply(tx, scope, op, produced_by, reason) for op in checked]
            if idempotency_key is not None:
                result = _EVENTS.dump_python(events, mode="json")
                tx.store_idempotency(self._ns, idempotency_key, result)
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

    def _embedded(self, ops: Sequence[Op]) -> list[Op]:
        """The ops with the vectors their new text needs, from one batch call (an op that already
        carries one keeps it)."""
        todo: list[Add | Correct] = [
            op for op in ops if not isinstance(op, Retire) and op.vector is None
        ]
        texts = [op.fact.content if isinstance(op, Add) else op.content for op in todo]
        vectors = iter(self._embedder.embed(texts) if texts else [])
        done: dict[int, Op] = {id(op): dataclasses.replace(op, vector=next(vectors)) for op in todo}
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
    ) -> MemoryEvent:
        """One write operation inside the caller's transaction, with its history event(s)."""
        now = self._now()
        if isinstance(op, Correct):
            memory = self._current(tx, op.memory_id)
            content = op.content  # checked by _change
            assert op.vector is not None  # _write / _change embed every op before applying it
            tx.set_content(self._ns, memory.id, content, now)
            tx.index_memory(self._ns, memory.id, content, op.vector)
            self._event(tx, memory, "CORRECT", {"content": memory.content}, {"content": content},
                        reason, produced_by, now)  # fmt: skip
            return MemoryEvent("CORRECT", memory.id, content, reason, produced_by)
        if isinstance(op, Retire):
            memory = self._current(tx, op.memory_id)
            tx.close_memory(self._ns, memory.id, now)
            self._event(tx, memory, "DELETE", {"valid_to": None}, {"valid_to": now.isoformat()},
                        reason, produced_by, now)  # fmt: skip
            return MemoryEvent("DELETE", memory.id, memory.content, reason, produced_by)
        fact = op.fact
        memory = Memory(
            id=uuid7(), scope=scope, type=fact.type, key=fact.key, content=fact.content,
            importance=fact.importance, confidence=fact.confidence, valid_from=now,
            valid_to=None, superseded_by=None, expires_at=fact.expires_at, source=fact.source,
            metadata=fact.metadata, access_count=0, created_at=now, updated_at=now,
        )  # fmt: skip
        if op.replaces is not None:
            replaced: Memory | None = self._current(tx, op.replaces)
        elif fact.key is not None:
            tx.lock_profile_slot(self._ns, scope, fact.key)
            replaced = tx.current_profile(self._ns, scope, fact.key, lock=True)
        else:
            replaced = None
        if replaced is not None:  # close the old row first: it frees the key's current slot
            tx.close_memory(self._ns, replaced.id, now, superseded_by=memory.id)
        tx.insert_memory(self._ns, memory)
        assert op.vector is not None
        tx.index_memory(self._ns, memory.id, memory.content, op.vector)
        after = {"content": memory.content, "type": memory.type, "key": memory.key}
        self._event(tx, memory, "ADD", None, after, reason, produced_by, now)
        if replaced is None:
            return MemoryEvent("ADD", memory.id, memory.content, reason, produced_by)
        why = reason or f"profile key {fact.key!r} has a new value"
        self._event(tx, replaced, "SUPERSEDE", {"valid_to": None},
                    {"valid_to": now.isoformat(), "superseded_by": str(memory.id)},
                    why, produced_by, now)  # fmt: skip
        return MemoryEvent("SUPERSEDE", memory.id, memory.content, why, produced_by, replaced.id)

    def _event(
        self,
        tx: Tx,
        memory: Memory,
        op: Any,
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
        if not memory.current:
            raise Conflict(
                f"memory {memory_id} is no longer current"
                + (f" (superseded by {memory.superseded_by})" if memory.superseded_by else "")
            )
        return memory


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
) -> Filters:
    _check_scope(scope)
    kinds = tuple(types) if types else None
    if kinds and any(t not in MEMORY_TYPES for t in kinds):
        raise InvalidRequest(f"types must be among {MEMORY_TYPES}")
    if metadata and not all(isinstance(k, str) for k in metadata):
        raise InvalidRequest("metadata filter keys must be strings")
    return Filters(
        scope, include_shared=include_shared, as_of=as_of, types=kinds, metadata=metadata or None
    )


def _check_scope(scope: Scope) -> None:
    for name in ("user_id", "agent_id", "session_id"):
        value = getattr(scope, name)
        if value is not None and not value.strip():
            raise InvalidRequest(f"scope {name} is empty: leave it out (None) instead")


def _checked(fact: Fact, scope: Scope) -> Fact:
    """The strict gate for a caller's fact (a model's output is normalised before it, S4)."""
    if fact.type not in MEMORY_TYPES:
        raise InvalidRequest(f"type must be one of {MEMORY_TYPES}")
    if (fact.type == "profile") != (fact.key is not None):
        raise InvalidRequest("a profile memory has a key, and only a profile memory has one")
    if fact.key is not None and scope.session_id is not None:
        raise InvalidRequest("a profile value belongs to a user, not a session: drop session_id")
    for name in ("importance", "confidence"):
        if not 0 <= getattr(fact, name) <= 1:
            raise InvalidRequest(f"{name} must be between 0 and 1")
    return dataclasses.replace(fact, content=_content(fact.content))
