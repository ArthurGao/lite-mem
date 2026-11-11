"""The Store Protocol (design §4): the engine reads and writes only through a transaction, so a
write's row, index entries and history commit together or not at all."""

import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

from lite_mem.errors import Conflict
from lite_mem.types import EventOp, HistoryEntry, Memory, MemoryType, Scope

# The columns a Memory is read from and written to, in one order for every store.
MEMORY_COLUMNS = (
    "id, user_id, agent_id, session_id, type, key, content, importance, confidence, valid_from, "
    "valid_to, superseded_by, expires_at, source, metadata, access_count, created_at, updated_at"
)
# Rows shared within a namespace (design §3): no user, no agent.
SHARED = "m.user_id IS NULL AND m.agent_id IS NULL"


@dataclass(frozen=True)
class Filters:
    """Which memories a read may return (design §5)."""

    scope: Scope
    # "visible": what the scope can see (its own rows, any of its sessions when no session is
    # given, plus the namespace's shared rows when include_shared). "exact": rows of exactly
    # this scope (session included, NULL matching NULL), never shared ones.
    mode: Literal["visible", "exact"] = "visible"
    include_shared: bool = False
    as_of: datetime | None = None  # None: current memories only
    # The engine's clock for "current": a memory past its expires_at is no longer current even
    # before `maintain` records the EXPIRE. The engine always sets it; None (the database's
    # clock) is only for callers of a store on its own.
    now: datetime | None = None
    types: tuple[MemoryType, ...] | None = None
    metadata: dict[str, Any] | None = None  # equality on each key


def own_rows(f: Filters, mark: Callable[[str], str]) -> str:
    """The SQL for the filter's own scope on `memories m` — the isolation rule every store applies
    (design §3): user and agent always; the session when exact or given. `mark(name)` writes a
    bound parameter in the store's style; the values are f.scope's fields."""
    fields = [("user_id", f.scope.user_id), ("agent_id", f.scope.agent_id)]
    if f.mode == "exact" or f.scope.session_id is not None:
        fields.append(("session_id", f.scope.session_id))
    return " AND ".join(
        f"m.{column} = {mark(column)}" if value is not None else f"m.{column} IS NULL"
        for column, value in fields
    )


def with_shared(f: Filters) -> bool:
    return f.include_shared and f.mode == "visible"


def check_binding(
    namespace: str, row: Mapping[str, Any], *, provider: str, model: str, dim: int, tokenizer: str
) -> None:
    """Conflict when a namespace's stored index configuration differs from the one asked for."""
    wanted = {"embedding_provider": provider, "embedding_model": model, "dim": dim,
              "tokenizer": tokenizer}  # fmt: skip
    differs = {k: (row[k], v) for k, v in wanted.items() if row[k] != v}
    if differs:
        said = ", ".join(f"{k} {had!r} (asked {got!r})" for k, (had, got) in differs.items())
        raise Conflict(
            f"namespace {namespace!r} was built with another index configuration: {said}; "
            "open it with that configuration, or use a new namespace"
        )


class Tx(Protocol):
    """One transaction. Every method is confined to the namespace it is given."""

    def bind_namespace(
        self, namespace: str, *, provider: str, model: str, dim: int, tokenizer: str
    ) -> None:
        """Record what builds this namespace's indexes, or check it matches (Conflict if not)."""
        ...

    def lookup_idempotency(self, namespace: str, key: str) -> tuple[str | None, Any] | None:
        """(user_id, result) a committed write with this key stored, or None. Read-only: run it
        before any expensive work (model calls) so a retry skips that work."""
        ...

    def claim_idempotency(
        self, namespace: str, key: str, user_id: str | None
    ) -> tuple[str | None, Any] | None:
        """None: this key is new and now claimed by this transaction. Otherwise (user_id, result)
        of the committed write that holds it (a concurrent one is waited for)."""
        ...

    def lock_profile_slot(self, namespace: str, scope: Scope, key: str) -> None:
        """Serialise writers of one profile key and scope until the transaction ends, even when
        no current value exists yet to lock."""
        ...

    def store_idempotency(self, namespace: str, key: str, result: Any) -> None: ...

    def insert_memory(self, namespace: str, memory: Memory) -> None: ...

    def get_memory(
        self, namespace: str, memory_id: uuid.UUID, *, lock: bool = False
    ) -> Memory | None: ...

    def current_profile(
        self, namespace: str, scope: Scope, key: str, *, lock: bool = False
    ) -> Memory | None: ...

    def close_memory(
        self,
        namespace: str,
        memory_id: uuid.UUID,
        at: datetime,
        superseded_by: uuid.UUID | None = None,
    ) -> None: ...

    def set_content(
        self, namespace: str, memory_id: uuid.UUID, content: str, at: datetime
    ) -> None: ...

    def add_event(
        self,
        namespace: str,
        memory: Memory,
        op: EventOp,
        *,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        reason: str | None,
        produced_by: str | None,
        at: datetime,
    ) -> None: ...

    def list_memories(self, namespace: str, filters: Filters) -> list[Memory]: ...

    def get_memories(self, namespace: str, ids: list[uuid.UUID]) -> list[Memory]: ...

    def index_memory(
        self, namespace: str, memory_id: uuid.UUID, content: str, vector: list[float]
    ) -> None:
        """(Re)build the memory's search entries — its vector and its text index — in this
        transaction, so they commit with the row (design §4)."""
        ...

    def vector_search(
        self, namespace: str, vector: list[float], filters: Filters, limit: int
    ) -> list[tuple[uuid.UUID, float]]:
        """(id, cosine similarity), nearest first, exact among the rows `filters` lets through."""
        ...

    def text_search(
        self, namespace: str, query: str, filters: Filters, limit: int
    ) -> list[uuid.UUID]:
        """Best first. PgStore: BM25 with the namespace's statistics, equal to keyword.py's formula
        (a PG-only test checks it); any store: a ranked keyword match (design §5)."""
        ...

    def expired(self, namespace: str, now: datetime, user_id: str | None) -> list[Memory]:
        """Current memories whose expires_at has passed (all users when `user_id` is None)."""
        ...

    def scopes(self, namespace: str, user_id: str | None) -> list[Scope]:
        """The scopes holding current memories (one user's when `user_id` is given)."""
        ...

    def vectors(self, namespace: str, ids: list[uuid.UUID]) -> dict[uuid.UUID, list[float]]: ...

    def idle(self, namespace: str, filters: Filters, before: datetime) -> list[Memory]:
        """The memories `filters` lets through that nothing retrieved (nor created) since
        `before`, oldest first."""
        ...

    def touch(self, namespace: str, ids: list[uuid.UUID], at: datetime) -> None:
        """Count a retrieval of these memories (access_count, last_accessed_at)."""
        ...

    def history(self, namespace: str, memory_id: uuid.UUID) -> list[HistoryEntry]: ...

    def erase_user(self, namespace: str, user_id: str) -> int: ...


class Store(Protocol):
    def transaction(self) -> AbstractContextManager[Tx]: ...

    def ping(self) -> None:
        """Raise unless the database answers."""
        ...

    def close(self) -> None: ...
