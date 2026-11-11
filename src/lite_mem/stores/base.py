"""The Store Protocol (design §4): the engine reads and writes only through a transaction, so a
write's row, index entries and history commit together or not at all."""

import uuid
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any, Protocol

from lite_mem.types import EventOp, HistoryEntry, Memory, MemoryType, Scope


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

    def list_memories(
        self,
        namespace: str,
        scope: Scope,
        *,
        include_shared: bool,
        as_of: datetime | None,
        types: tuple[MemoryType, ...] | None,
    ) -> list[Memory]: ...

    def history(self, namespace: str, memory_id: uuid.UUID) -> list[HistoryEntry]: ...

    def erase_user(self, namespace: str, user_id: str) -> int: ...


class Store(Protocol):
    def transaction(self) -> AbstractContextManager[Tx]: ...

    def close(self) -> None: ...
