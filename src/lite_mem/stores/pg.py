"""PostgreSQL store (design §4): one schema of lite-mem's own, a pooled connection per transaction
with `search_path` set once per connection."""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from lite_mem.config import own_schema
from lite_mem.db import require_url
from lite_mem.errors import Conflict, StoreError
from lite_mem.types import EventOp, HistoryEntry, Memory, MemoryType, Scope

_COLUMNS = (
    "id, user_id, agent_id, session_id, type, key, content, importance, confidence, valid_from, "
    "valid_to, superseded_by, expires_at, source, metadata, access_count, created_at, updated_at"
)

_INSERT = (
    f"INSERT INTO memories (namespace, {_COLUMNS}) "
    f"VALUES ({', '.join(['%s'] * (1 + len(_COLUMNS.split(','))))})"
).encode()


def _memory(row: dict[str, Any]) -> Memory:
    return Memory(
        id=row["id"],
        scope=Scope(row["user_id"], row["agent_id"], row["session_id"]),
        type=row["type"],
        key=row["key"],
        content=row["content"],
        importance=row["importance"],
        confidence=row["confidence"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
        expires_at=row["expires_at"],
        source=row["source"],
        metadata=row["metadata"],
        access_count=row["access_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PgTx:
    def __init__(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        self._conn = conn

    def _one(self, query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        return self._conn.execute(query.encode(), params).fetchone()

    def bind_namespace(
        self, namespace: str, *, provider: str, model: str, dim: int, tokenizer: str
    ) -> None:
        self._conn.execute(
            b"INSERT INTO namespaces (name, embedding_provider, embedding_model, dim, tokenizer) "
            b"VALUES (%s, %s, %s, %s, %s) ON CONFLICT (name) DO NOTHING",
            (namespace, provider, model, dim, tokenizer),
        )
        row = self._one(
            "SELECT embedding_provider, embedding_model, dim, tokenizer FROM namespaces "
            "WHERE name = %s",
            (namespace,),
        )
        assert row is not None
        wanted = {
            "embedding_provider": provider,
            "embedding_model": model,
            "dim": dim,
            "tokenizer": tokenizer,
        }
        differs = {k: (row[k], v) for k, v in wanted.items() if row[k] != v}
        if differs:
            said = ", ".join(f"{k} {had!r} (asked {got!r})" for k, (had, got) in differs.items())
            raise Conflict(
                f"namespace {namespace!r} was built with another index configuration: {said}; "
                "open it with that configuration, or use a new namespace"
            )

    def lookup_idempotency(self, namespace: str, key: str) -> tuple[str | None, Any] | None:
        row = self._one(
            "SELECT user_id, result FROM idempotency WHERE namespace = %s AND key = %s "
            "AND result <> 'null'",
            (namespace, key),
        )
        return (row["user_id"], row["result"]) if row else None

    def claim_idempotency(
        self, namespace: str, key: str, user_id: str | None
    ) -> tuple[str | None, Any] | None:
        claimed = self._one(
            "INSERT INTO idempotency (namespace, key, user_id, result, created_at) "
            "VALUES (%s, %s, %s, 'null', now()) ON CONFLICT DO NOTHING RETURNING key",
            (namespace, key, user_id),
        )
        if claimed is not None:
            return None
        # Taken: by a committed write (its result), or by one that was in flight — the insert
        # above waited for it, so it has committed by now.
        row = self._one(
            "SELECT user_id, result FROM idempotency WHERE namespace = %s AND key = %s",
            (namespace, key),
        )
        if row is None:  # the other write rolled back between our insert and this read
            raise StoreError("idempotency key released concurrently: retry the write")
        return row["user_id"], row["result"]

    def lock_profile_slot(self, namespace: str, scope: Scope, key: str) -> None:
        slot = "\x1f".join((namespace, scope.user_id or "", scope.agent_id or "", key))
        self._conn.execute(b"SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (slot,))

    def store_idempotency(self, namespace: str, key: str, result: Any) -> None:
        self._conn.execute(
            b"UPDATE idempotency SET result = %s WHERE namespace = %s AND key = %s",
            (Jsonb(result), namespace, key),
        )

    def insert_memory(self, namespace: str, memory: Memory) -> None:
        m, s = memory, memory.scope
        self._conn.execute(
            _INSERT,
            (
                namespace, m.id, s.user_id, s.agent_id, s.session_id, m.type, m.key, m.content,
                m.importance, m.confidence, m.valid_from, m.valid_to, m.superseded_by,
                m.expires_at, m.source, Jsonb(m.metadata), m.access_count, m.created_at,
                m.updated_at,
            ),
        )  # fmt: skip

    def get_memory(
        self, namespace: str, memory_id: uuid.UUID, *, lock: bool = False
    ) -> Memory | None:
        row = self._one(
            f"SELECT {_COLUMNS} FROM memories WHERE namespace = %s AND id = %s"
            + (" FOR UPDATE" if lock else ""),
            (namespace, memory_id),
        )
        return _memory(row) if row else None

    def current_profile(
        self, namespace: str, scope: Scope, key: str, *, lock: bool = False
    ) -> Memory | None:
        row = self._one(  # the predicate matches memories_profile_current exactly
            f"SELECT {_COLUMNS} FROM memories WHERE namespace = %s "
            "AND coalesce(user_id, '') = %s AND coalesce(agent_id, '') = %s AND key = %s "
            "AND valid_to IS NULL" + (" FOR UPDATE" if lock else ""),
            (namespace, scope.user_id or "", scope.agent_id or "", key),
        )
        return _memory(row) if row else None

    def close_memory(
        self,
        namespace: str,
        memory_id: uuid.UUID,
        at: datetime,
        superseded_by: uuid.UUID | None = None,
    ) -> None:
        self._conn.execute(
            b"UPDATE memories SET valid_to = %s, superseded_by = %s, updated_at = %s "
            b"WHERE namespace = %s AND id = %s",
            (at, superseded_by, at, namespace, memory_id),
        )

    def set_content(self, namespace: str, memory_id: uuid.UUID, content: str, at: datetime) -> None:
        self._conn.execute(
            b"UPDATE memories SET content = %s, updated_at = %s WHERE namespace = %s AND id = %s",
            (content, at, namespace, memory_id),
        )

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
    ) -> None:
        self._conn.execute(
            b"INSERT INTO memory_events "
            b"(namespace, memory_id, user_id, op, before, after, reason, produced_by, at) "
            b"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                namespace,
                memory.id,
                memory.scope.user_id,
                op,
                None if before is None else Jsonb(before),
                None if after is None else Jsonb(after),
                reason,
                produced_by,
                at,
            ),
        )

    def list_memories(
        self,
        namespace: str,
        scope: Scope,
        *,
        include_shared: bool,
        as_of: datetime | None,
        types: tuple[MemoryType, ...] | None,
    ) -> list[Memory]:
        # Plain predicates per field (no "x IS NOT DISTINCT FROM y" or "%s IS NULL OR …"), so
        # the planner can use the scope indexes.
        own = " AND ".join(
            f"{column} = %({column})s" if value is not None else f"{column} IS NULL"
            for column, value in (("user_id", scope.user_id), ("agent_id", scope.agent_id))
        )
        if scope.session_id is not None:
            own += " AND session_id = %(session_id)s"
        own = f"({own})"
        shared = " OR (user_id IS NULL AND agent_id IS NULL)" if include_shared else ""
        when = (
            "valid_from <= %(at)s AND (valid_to IS NULL OR valid_to > %(at)s)"
            if as_of is not None
            else "valid_to IS NULL"
        )
        kinds = " AND type = ANY(%(types)s)" if types else ""
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM memories WHERE namespace = %(ns)s AND ({own}{shared}) "
            f"AND {when}{kinds} ORDER BY valid_from, id".encode(),
            {
                "ns": namespace,
                "user_id": scope.user_id,
                "agent_id": scope.agent_id,
                "session_id": scope.session_id,
                "at": as_of,
                "types": list(types or ()),
            },
        ).fetchall()
        return [_memory(r) for r in rows]

    def history(self, namespace: str, memory_id: uuid.UUID) -> list[HistoryEntry]:
        rows = self._conn.execute(
            b"SELECT op, before, after, reason, produced_by, at FROM memory_events "
            b"WHERE namespace = %s AND memory_id = %s ORDER BY id",
            (namespace, memory_id),
        ).fetchall()
        return [HistoryEntry(**r) for r in rows]

    def erase_user(self, namespace: str, user_id: str) -> int:
        self._conn.execute(
            b"DELETE FROM idempotency WHERE namespace = %s AND user_id = %s", (namespace, user_id)
        )
        # A merged or superseded row may point at one of these: clear the links first.
        self._conn.execute(
            b"UPDATE memories SET superseded_by = NULL WHERE namespace = %s AND superseded_by IN "
            b"(SELECT id FROM memories WHERE namespace = %s AND user_id = %s)",
            (namespace, namespace, user_id),
        )
        erased = self._conn.execute(
            b"DELETE FROM memories WHERE namespace = %s AND user_id = %s", (namespace, user_id)
        )
        return erased.rowcount


class PgStore:
    def __init__(self, database_url: str | None, schema: str, *, max_size: int = 4) -> None:
        own_schema(schema)

        def configure(conn: psycopg.Connection[dict[str, Any]]) -> None:
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
            conn.commit()

        try:
            self._pool: ConnectionPool[psycopg.Connection[dict[str, Any]]] = ConnectionPool(
                require_url(database_url),
                min_size=1,
                max_size=max_size,
                kwargs={"row_factory": dict_row},
                configure=configure,
                open=True,
            )
        except psycopg.Error as exc:
            raise StoreError(f"cannot open the PostgreSQL pool: {exc}") from exc

    @contextmanager
    def transaction(self) -> Iterator[PgTx]:
        try:
            with self._pool.connection() as conn, conn.transaction():
                yield PgTx(conn)
        except psycopg.OperationalError as exc:
            raise StoreError(f"PostgreSQL failed: {exc}") from exc

    def close(self) -> None:
        self._pool.close()
