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
from lite_mem.errors import ConfigError, StoreError
from lite_mem.keyword import K1, B, term_counts, terms
from lite_mem.stores.base import (
    MEMORY_COLUMNS,
    SHARED,
    Filters,
    check_binding,
    own_rows,
    with_shared,
)
from lite_mem.types import EventOp, HistoryEntry, Memory, Scope

MAX_DIM = 16_000  # pgvector's limit for `vector` (no HNSW index since S8: its limit was 2000)


_INSERT = (
    f"INSERT INTO memories (namespace, {MEMORY_COLUMNS}) "
    f"VALUES ({', '.join(['%s'] * (1 + len(MEMORY_COLUMNS.split(','))))})"
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


def _where(f: Filters) -> tuple[str, dict[str, Any]]:
    """The SQL condition on `memories m` and its parameters for `f` (plain predicates the planner
    can index)."""
    own = own_rows(f, lambda name: f"%({name})s")
    shared = f" OR ({SHARED})" if with_shared(f) else ""
    when = (
        "m.valid_from <= %(as_of)s AND (m.valid_to IS NULL OR m.valid_to > %(as_of)s)"
        " AND (m.expires_at IS NULL OR m.expires_at > %(as_of)s)"
        if f.as_of is not None
        else "m.valid_to IS NULL"
        " AND (m.expires_at IS NULL OR m.expires_at > coalesce(%(now)s, now()))"
    )
    sql_ = f"m.namespace = %(ns)s AND (({own}){shared}) AND {when}"
    if f.types:
        sql_ += " AND m.type = ANY(%(types)s)"
    if f.metadata:
        sql_ += " AND m.metadata @> %(metadata)s"
    return sql_, {
        "user_id": f.scope.user_id,
        "agent_id": f.scope.agent_id,
        "session_id": f.scope.session_id,
        "as_of": f.as_of,
        "now": f.now,
        "types": list(f.types or ()),
        "metadata": Jsonb(f.metadata or {}),
    }


def _vectors(dim: int) -> sql.Identifier:
    return sql.Identifier(f"memory_vectors_{dim}")


class PgTx:
    def __init__(self, conn: psycopg.Connection[dict[str, Any]], dims: dict[str, int]) -> None:
        self._conn = conn
        self._dims = dims  # namespace -> dim: the store's cache (fixed once a namespace is bound)

    def _dim(self, namespace: str) -> int:
        if namespace not in self._dims:
            row = self._one("SELECT dim FROM namespaces WHERE name = %s", (namespace,))
            if row is None:
                raise StoreError(f"namespace {namespace!r} is not bound")
            self._dims[namespace] = row["dim"]
        return self._dims[namespace]

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
        check_binding(namespace, row, provider=provider, model=model, dim=dim, tokenizer=tokenizer)
        if dim > MAX_DIM:
            raise ConfigError(f"embedding dim {dim} is over pgvector's limit ({MAX_DIM})")
        self._dims[namespace] = dim
        table = _vectors(dim)
        exists = self._one("SELECT to_regclass(%s) AS t", (f"memory_vectors_{dim}",))
        if exists and exists["t"] is not None:
            return  # the usual case: no DDL, no lock
        # One vector table per dimension; serialise its creation across concurrent first binds.
        self._conn.execute(b"SELECT pg_advisory_xact_lock(hashtext(%s))", (f"vectors:{dim}",))
        self._conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (memory_id uuid PRIMARY KEY REFERENCES memories (id)"
                " ON DELETE CASCADE, namespace text NOT NULL, embedding vector({}) NOT NULL)"
            ).format(table, sql.Literal(dim))
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
            f"SELECT {MEMORY_COLUMNS} FROM memories WHERE namespace = %s AND id = %s"
            + (" FOR UPDATE" if lock else ""),
            (namespace, memory_id),
        )
        return _memory(row) if row else None

    def current_profile(
        self, namespace: str, scope: Scope, key: str, *, lock: bool = False
    ) -> Memory | None:
        row = self._one(  # the predicate matches memories_profile_current exactly
            f"SELECT {MEMORY_COLUMNS} FROM memories WHERE namespace = %s "
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

    def expired(self, namespace: str, now: datetime, user_id: str | None) -> list[Memory]:
        rows = self._conn.execute(
            f"SELECT {MEMORY_COLUMNS} FROM memories WHERE namespace = %s AND valid_to IS NULL "
            "AND expires_at <= %s AND (%s::text IS NULL OR user_id = %s) "
            "ORDER BY expires_at, id".encode(),
            (namespace, now, user_id, user_id),
        ).fetchall()
        return [_memory(r) for r in rows]

    def scopes(self, namespace: str, user_id: str | None) -> list[Scope]:
        rows = self._conn.execute(
            b"SELECT DISTINCT user_id, agent_id, session_id FROM memories "
            b"WHERE namespace = %s AND valid_to IS NULL AND (%s::text IS NULL OR user_id = %s) "
            b"ORDER BY user_id NULLS FIRST, agent_id NULLS FIRST, session_id NULLS FIRST",
            (namespace, user_id, user_id),
        ).fetchall()
        return [Scope(r["user_id"], r["agent_id"], r["session_id"]) for r in rows]

    def vectors(self, namespace: str, ids: list[uuid.UUID]) -> dict[uuid.UUID, list[float]]:
        rows = self._conn.execute(
            sql.SQL(
                "SELECT memory_id, embedding::real[] AS vector FROM {} "
                "WHERE namespace = %s AND memory_id = ANY(%s)"
            ).format(_vectors(self._dim(namespace))),
            (namespace, ids),
        ).fetchall()
        return {r["memory_id"]: list(r["vector"]) for r in rows}

    def idle(self, namespace: str, filters: Filters, before: datetime) -> list[Memory]:
        where, params = _where(filters)
        rows = self._conn.execute(
            f"SELECT {MEMORY_COLUMNS} FROM memories m WHERE {where} "
            "AND coalesce(m.last_accessed_at, m.valid_from) < %(before)s "
            "ORDER BY valid_from, id".encode(),
            {**params, "ns": namespace, "before": before},
        ).fetchall()
        return [_memory(r) for r in rows]

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
        self, namespace: str, filters: Filters, *, limit: int | None = None, offset: int = 0
    ) -> list[Memory]:
        where, params = _where(filters)
        rows = self._conn.execute(
            f"SELECT {MEMORY_COLUMNS} FROM memories m WHERE {where} ORDER BY valid_from, id "
            "LIMIT %(limit)s OFFSET %(offset)s".encode(),
            {**params, "ns": namespace, "limit": limit, "offset": offset},
        ).fetchall()
        return [_memory(r) for r in rows]

    def get_memories(self, namespace: str, ids: list[uuid.UUID]) -> list[Memory]:
        rows = self._conn.execute(
            f"SELECT {MEMORY_COLUMNS} FROM memories WHERE namespace = %s AND id = ANY(%s)".encode(),
            (namespace, ids),
        ).fetchall()
        return [_memory(r) for r in rows]

    def index_memory(
        self, namespace: str, memory_id: uuid.UUID, content: str, vector: list[float]
    ) -> None:
        dim = self._dim(namespace)
        if len(vector) != dim:
            raise StoreError(f"namespace {namespace!r} holds {dim}-d vectors, got {len(vector)}")
        counts = term_counts(content)
        self._conn.execute(b"DELETE FROM memory_postings WHERE memory_id = %s", (memory_id,))
        with self._conn.cursor() as cur:
            cur.executemany(
                b"INSERT INTO memory_postings (namespace, memory_id, term, tf) "
                b"VALUES (%s, %s, %s, %s)",
                [(namespace, memory_id, term, tf) for term, tf in counts.items()],
            )
        self._conn.execute(
            b"UPDATE memories SET terms = %s WHERE id = %s", (sum(counts.values()), memory_id)
        )
        self._conn.execute(
            sql.SQL(
                "INSERT INTO {} (memory_id, namespace, embedding) "
                "VALUES (%s, %s, %s::float8[]::vector) "
                "ON CONFLICT (memory_id) DO UPDATE SET embedding = EXCLUDED.embedding"
            ).format(_vectors(dim)),
            (memory_id, namespace, vector),
        )

    def vector_search(
        self, namespace: str, vector: list[float], filters: Filters, limit: int
    ) -> list[tuple[uuid.UUID, float]]:
        where, params = _where(filters)
        # Exact among the rows the filter lets through: first those rows, then their distances.
        # An HNSW scan filtered after the join can run out of candidates when many nearer
        # vectors belong to other scopes; a scope holds thousands of memories, cheap to rank.
        rows = self._conn.execute(
            sql.SQL(
                "WITH visible AS MATERIALIZED (SELECT m.id FROM memories m WHERE {}) "
                "SELECT v.memory_id AS id, 1 - (v.embedding <=> %(q)s::float8[]::vector) AS sim "
                "FROM visible JOIN {} v ON v.memory_id = visible.id "
                "ORDER BY v.embedding <=> %(q)s::float8[]::vector, v.memory_id LIMIT %(limit)s"
            ).format(sql.SQL(where), _vectors(self._dim(namespace))),
            {**params, "ns": namespace, "q": vector, "limit": limit},
        ).fetchall()
        return [(r["id"], float(r["sim"])) for r in rows]

    def text_search(
        self, namespace: str, query: str, filters: Filters, limit: int
    ) -> list[uuid.UUID]:
        wanted = sorted(set(terms(query)))
        if not wanted:
            return []
        where, params = _where(filters)
        # N, avgdl and n(t) are the namespace's current memories' (keyword.py has the formula).
        rows = self._conn.execute(
            f"""
            WITH cur AS (SELECT id, terms FROM memories
                         WHERE namespace = %(ns)s AND valid_to IS NULL),
                 stats AS (SELECT count(*) AS n, coalesce(avg(terms), 0) AS avgdl FROM cur),
                 df AS (SELECT p.term, count(*) AS df FROM memory_postings p
                        JOIN cur ON cur.id = p.memory_id
                        WHERE p.namespace = %(ns)s AND p.term = ANY(%(terms)s) GROUP BY p.term)
            SELECT id FROM (
                SELECT m.id, sum(
                           ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5))
                           * p.tf * (%(k1)s + 1)
                           / (p.tf + %(k1)s * (1 - %(b)s + %(b)s * m.terms
                                               / nullif(stats.avgdl, 0)))
                       ) AS score
                FROM memory_postings p
                JOIN memories m ON m.id = p.memory_id
                JOIN df ON df.term = p.term
                CROSS JOIN stats
                WHERE p.namespace = %(ns)s AND p.term = ANY(%(terms)s) AND {where}
                GROUP BY m.id
            ) scored
            ORDER BY round(score::numeric, 9) DESC, id
            LIMIT %(limit)s
            """.encode(),
            {**params, "ns": namespace, "terms": wanted, "k1": K1, "b": B, "limit": limit},
        ).fetchall()
        return [r["id"] for r in rows]

    def touch(self, namespace: str, ids: list[uuid.UUID], at: datetime) -> None:
        if ids:
            self._conn.execute(
                b"UPDATE memories SET access_count = access_count + 1, last_accessed_at = %s "
                b"WHERE namespace = %s AND id = ANY(%s)",
                (at, namespace, ids),
            )

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
        self._dims: dict[str, int] = {}

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
                yield PgTx(conn, self._dims)
        except psycopg.OperationalError as exc:
            raise StoreError(f"PostgreSQL failed: {exc}") from exc

    def ping(self) -> None:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1")
        except psycopg.Error as exc:
            raise StoreError(f"PostgreSQL failed: {exc}") from exc

    def close(self) -> None:
        self._pool.close()
