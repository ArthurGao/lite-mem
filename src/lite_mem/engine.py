"""`MemoryEngine`: the one core behind the library and the HTTP service (design §1, §2).

An engine is bound to one namespace and one embedding model (design §4). Every write goes through
`_apply`, one operation at a time inside one transaction: the memory rows, their history and the
idempotency record commit together."""

import dataclasses
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from lite_mem.config import Namespace, SearchConfig
from lite_mem.embedding import Embedder
from lite_mem.errors import ConfigError, Conflict, InvalidRequest, NotFound
from lite_mem.extract import Extractor
from lite_mem.fusion import rrf
from lite_mem.ids import uuid7
from lite_mem.keyword import TOKENIZER
from lite_mem.resolve import Resolver
from lite_mem.sensitive import Finding, Policy, find, mask
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
log = logging.getLogger("lite_mem")


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


@dataclasses.dataclass(eq=False)
class _Step:
    """One planned write of an inferred batch (identity matters: steps are removed by identity)."""

    op: Add | Retire
    reason: str | None
    produced_by: str | None


def _pending(local: uuid.UUID, op: Add | Retire, scope: Scope) -> Memory:
    """A fact this batch has accepted, shown to the resolver like a memory (batch-local id)."""
    assert isinstance(op, Add)
    f, now = op.fact, datetime.now(UTC)
    return Memory(
        id=local, scope=scope, type=f.type, key=f.key, content=f.content,
        importance=f.importance, confidence=f.confidence, valid_from=now, valid_to=None,
        superseded_by=None, expires_at=f.expires_at, source=f.source, metadata=f.metadata,
        access_count=0, created_at=now, updated_at=now,
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
        clock: Clock = _utcnow,
    ) -> None:
        try:
            self._ns = TypeAdapter(Namespace).validate_python(namespace)
        except ValidationError as exc:
            raise InvalidRequest(f"invalid namespace {namespace!r}") from exc
        self._store = store
        self._embedder = embedder
        self._extractor = extractor
        self._resolver = resolver
        self._related = related  # existing memories each new fact is compared with
        self._search = search or SearchConfig()
        self._sensitive = sensitive  # what happens to a secret (lite_mem.sensitive)
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
        if infer:
            return self._infer(messages, scope, metadata, idempotency_key)
        facts = [
            Fact(m.content, metadata=dict(metadata or {}))
            for m in messages
            if m.role == "user" and m.content.strip()
        ]
        return self._write(scope, [Add(f) for f in self._derived(facts)], idempotency_key)

    def remember(
        self, fact: Fact, scope: Scope, *, idempotency_key: str | None = None
    ) -> list[MemoryEvent]:
        """Store one fact as given (a profile fact supersedes the key's current value)."""
        fact = dataclasses.replace(fact, content=self._stated(fact.content))
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
        ready = self._embedded(checked)  # model calls stay outside the transaction
        return self._commit(
            scope,
            idempotency_key,
            lambda tx: [self._apply(tx, scope, op, produced_by, reason) for op in ready],
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
        facts = [
            _checked(dataclasses.replace(f, metadata=dict(metadata or {})), scope)
            for f in self._derived(extraction.facts)
        ]
        adds = [op for op in self._embedded([Add(f) for f in facts]) if isinstance(op, Add)]
        exact, k = Filters(scope, mode="exact"), self._related
        with self._store.transaction() as tx:  # read-only: what each fact may conflict with
            stored = [
                [h.memory for h in self._rank(tx, a.fact.content, a.vector or [], exact, k, 4 * k)]
                for a in adds
            ]
        plan = self._plan(scope, adds, stored, extraction.produced_by)
        return self._commit(scope, idempotency_key, lambda tx: self._carry_out(tx, scope, plan))

    def _plan(
        self,
        scope: Scope,
        adds: list[Add],
        stored: list[list[Memory]],
        extracted_by: str | None,
    ) -> list["_Step"]:
        """Resolve the facts one by one (model calls, no transaction). A fact's candidates are its
        related memories plus the facts this batch has already accepted (batch-local ids), so two
        wordings of one change do not both land. The model proposes; this decides (rule 5): a
        target outside the candidates is not applied, and every such downgrade says why."""
        assert self._resolver is not None
        steps: list[_Step] = []
        accepted: dict[uuid.UUID, _Step] = {}  # batch-local id -> the step that adds it
        claimed: dict[uuid.UUID, uuid.UUID] = {}  # stored memory -> batch-local id replacing it
        for add, near in zip(adds, stored, strict=True):
            pending = [_pending(local, step.op, scope) for local, step in accepted.items()]
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
                if action == "UPDATE":  # the later wording wins and takes over what it replaced
                    replaces = earlier.op.replaces if isinstance(earlier.op, Add) else None
            elif action == "UPDATE":
                replaces = target
            elif action == "DELETE" and target is not None:
                steps.append(_Step(Retire(target), reason, decision.produced_by))
            local = uuid7()
            by = decision.produced_by if replaces is not None else extracted_by
            step = _Step(dataclasses.replace(add, replaces=replaces), reason, by)
            steps.append(step)
            accepted[local] = step
            if replaces is not None:
                claimed[replaces] = local
        return steps

    def _derived(self, facts: Sequence[Fact]) -> list[Fact]:
        """Facts the engine derived itself: a secret is dropped or masked, never stored."""
        kept: list[Fact] = []
        dropped: list[str] = []
        for fact in facts:
            found = self._secrets(fact.content)
            if not found:
                kept.append(fact)
            elif self._sensitive == "mask":
                kept.append(dataclasses.replace(fact, content=mask(fact.content)))
            else:
                dropped += [f.kind for f in found]
        if dropped:  # kinds only: the log must not hold the secret either
            log.warning("not stored: %d sensitive value(s) %s", len(dropped), sorted(set(dropped)))
        return kept

    def _secrets(self, text: str) -> list[Finding]:
        return [] if self._sensitive == "keep" else find(text)

    def _stated(self, content: str) -> str:
        """A fact a caller states explicitly: masked, or refused so the caller knows."""
        found = self._secrets(content)
        if not found:
            return content
        if self._sensitive == "mask":
            return mask(content)
        kinds = ", ".join(sorted({f.kind for f in found}))
        raise InvalidRequest(f"not stored: it holds a secret ({kinds}); sensitive policy is drop")

    def _carry_out(self, tx: Tx, scope: Scope, steps: list["_Step"]) -> list[MemoryEvent]:
        """Apply a plan in the write transaction. A stored target that changed since it was looked
        up is not applied: the fact is added instead, and says so."""
        events: list[MemoryEvent] = []
        for step in steps:
            op, reason = step.op, step.reason
            target = op.memory_id if isinstance(op, Retire) else op.replaces
            if target is not None:
                found = tx.get_memory(self._ns, target, lock=True)
                if found is None or not found.current:
                    if isinstance(op, Retire):
                        continue  # already gone: nothing to retire
                    op = dataclasses.replace(op, replaces=None)
                    reason = f"{target} changed meanwhile; added instead"
            events.append(self._apply(tx, scope, op, step.produced_by, reason))
        return events

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
