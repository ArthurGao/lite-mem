"""Request and response bodies. Responses reuse the engine's dataclasses (pydantic serialises
them), so the HTTP shapes cannot drift from the library's."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lite_mem.types import HistoryEntry, Memory, MemoryEvent, MemoryHit, MemoryType, Scope


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopeIn(_Body):
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None

    def to_scope(self) -> Scope:
        return Scope(self.user_id, self.agent_id, self.session_id)


class MessageIn(_Body):
    role: Literal["user", "assistant", "system"]
    content: str


class AddIn(_Body):
    messages: list[MessageIn] = Field(min_length=1)
    scope: ScopeIn
    infer: bool = True
    metadata: dict[str, Any] | None = None


class SearchIn(_Body):
    query: str
    scope: ScopeIn
    top_k: int = 10
    include_shared: bool = True
    types: list[MemoryType] | None = None
    metadata: dict[str, Any] | None = None
    as_of: datetime | None = None


class UpdateIn(_Body):
    content: str
    reason: str | None = None


class EventsOut(BaseModel):
    events: list[MemoryEvent]


class HitsOut(BaseModel):
    hits: list[MemoryHit]


class MemoriesOut(BaseModel):
    memories: list[Memory]


class HistoryOut(BaseModel):
    entries: list[HistoryEntry]


class ErasedOut(BaseModel):
    erased: int
