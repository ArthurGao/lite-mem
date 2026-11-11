"""The engine's data shapes (design §3)."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, get_args

MemoryType = Literal["semantic", "episodic", "procedural", "profile"]
# ADD a new memory; CORRECT its wording in place (same id); SUPERSEDE it by a new memory (the fact
# changed: new id, the old row closed); DELETE (retire) it; MERGE several into one; EXPIRE it.
EventOp = Literal["ADD", "CORRECT", "DELETE", "SUPERSEDE", "MERGE", "EXPIRE"]
MEMORY_TYPES: tuple[MemoryType, ...] = get_args(MemoryType)


@dataclass(frozen=True)
class Scope:
    """Whose memory: a user, an agent, a session. All None: shared within the namespace."""

    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class Message:
    role: str  # user | assistant | system
    content: str


@dataclass(frozen=True)
class Fact:
    """A memory to write, before it has an id."""

    content: str
    type: MemoryType = "semantic"
    key: str | None = None  # profile only: one current value per key and scope
    importance: float = 0.5
    confidence: float = 0.8
    expires_at: datetime | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Memory:
    id: uuid.UUID
    scope: Scope
    type: MemoryType
    key: str | None
    content: str
    importance: float
    confidence: float
    valid_from: datetime
    valid_to: datetime | None
    superseded_by: uuid.UUID | None
    expires_at: datetime | None
    source: str | None
    metadata: dict[str, Any]
    access_count: int
    created_at: datetime
    updated_at: datetime

    @property
    def current(self) -> bool:
        return self.valid_to is None


@dataclass(frozen=True)
class MemoryEvent:
    """One change, as the write that made it reports it (and as the history keeps it)."""

    op: EventOp
    memory_id: uuid.UUID
    content: str | None = None
    reason: str | None = None
    produced_by: str | None = None  # provider:model when a model made the change
    replaced: uuid.UUID | None = None  # SUPERSEDE / MERGE: the memory this one closed


@dataclass(frozen=True)
class HistoryEntry:
    op: EventOp
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    reason: str | None
    produced_by: str | None
    at: datetime


@dataclass(frozen=True)
class MemoryHit:
    """One search result. `score` is only comparable within one search of lite-mem."""

    memory: Memory
    score: float  # relevance x weight
    relevance: float  # fused (RRF) relevance, before the weights
