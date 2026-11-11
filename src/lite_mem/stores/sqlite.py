"""`SqliteStore` (S7b): the same Store contract on one SQLite file — embedded, no server.

- Vectors: float32 blobs, ranked exactly with sqlite-vec's `vec_distance_cosine` among the rows
  the filters let through (as PgStore does: an ANN index filtered afterwards can miss in-scope
  memories).
- Keywords: the same postings and the same BM25 as PgStore, scored with `keyword.py`'s functions,
  so both stores rank text identically (not FTS5: its bm25 is another formula).
- Transactions: one connection per thread, WAL, every transaction `BEGIN IMMEDIATE`. Writers are
  serialised by the database itself, so PgStore's row locks and advisory locks have nothing to do
  here; readers wait at most `busy_timeout` for a writer.

Times are stored as UTC ISO-8601 text with a fixed width, so text order is time order."""

import json
import sqlite3
import struct
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lite_mem.errors import ConfigError, StoreError
from lite_mem.keyword import idf, term_counts, term_score, terms
from lite_mem.stores.base import (
    MEMORY_COLUMNS,
    SHARED,
    Filters,
    check_binding,
    own_rows,
    with_shared,
)
from lite_mem.types import EventOp, HistoryEntry, Memory, Scope

_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS namespaces (
    name TEXT PRIMARY KEY,
    embedding_provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dim INTEGER NOT NULL CHECK (dim > 0),
    tokenizer TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL REFERENCES namespaces (name) ON DELETE CASCADE,
    user_id TEXT,
    agent_id TEXT,
    session_id TEXT,
    type TEXT NOT NULL CHECK (type IN ('semantic', 'episodic', 'procedural', 'profile')),
    key TEXT,
    content TEXT NOT NULL CHECK (content <> ''),
    importance REAL NOT NULL CHECK (importance BETWEEN 0 AND 1),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    superseded_by TEXT REFERENCES memories (id) DEFERRABLE INITIALLY DEFERRED,
    expires_at TEXT,
    source TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    terms INTEGER NOT NULL DEFAULT 0,
    CHECK ((type = 'profile') = (key IS NOT NULL)),
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);
CREATE INDEX IF NOT EXISTS memories_current ON memories (namespace, user_id, agent_id)
    WHERE valid_to IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS memories_profile_current
    ON memories (namespace, coalesce(user_id, ''), coalesce(agent_id, ''), key)
    WHERE key IS NOT NULL AND valid_to IS NULL;
CREATE INDEX IF NOT EXISTS memories_user ON memories (namespace, user_id);
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    user_id TEXT,
    op TEXT NOT NULL CHECK (op IN ('ADD', 'CORRECT', 'DELETE', 'SUPERSEDE', 'MERGE', 'EXPIRE')),
    before TEXT,
    after TEXT,
    reason TEXT,
    produced_by TEXT,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_events_memory ON memory_events (memory_id, id);
CREATE TABLE IF NOT EXISTS idempotency (
    namespace TEXT NOT NULL REFERENCES namespaces (name) ON DELETE CASCADE,
    key TEXT NOT NULL,
    user_id TEXT,
    result TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idempotency_user ON idempotency (namespace, user_id);
CREATE TABLE IF NOT EXISTS memory_postings (
    namespace TEXT NOT NULL,
    memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    tf INTEGER NOT NULL CHECK (tf > 0),
    PRIMARY KEY (memory_id, term)
);
CREATE INDEX IF NOT EXISTS memory_postings_term ON memory_postings (namespace, term);
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT PRIMARY KEY REFERENCES memories (id) ON DELETE CASCADE,
    namespace TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""

_INSERT = (
    f"INSERT INTO memories (namespace, {MEMORY_COLUMNS}) "
    f"VALUES ({', '.join(['?'] * (1 + len(MEMORY_COLUMNS.split(','))))})"
)


def _t(at: datetime | None) -> str | None:
    """Fixed-width UTC text: text order is time order."""
    return None if at is None else at.astimezone(UTC).isoformat(timespec="microseconds")


def _at(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


def _id(value: str | None) -> uuid.UUID | None:
    return None if value is None else uuid.UUID(value)


def _text(value: uuid.UUID | None) -> str | None:
    return None if value is None else str(value)


def _dumps(value: dict[str, Any] | None) -> str | None:
    return None if value is None else json.dumps(value)


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _contains(doc: Any, part: Any) -> bool:
    """PostgreSQL's jsonb containment (`doc @> part`), which PgStore's metadata filter uses: an
    object contains every key of `part` with a contained value; an array contains each element of
    `part` somewhere; scalars are equal with their JSON type (true is not 1)."""
    if isinstance(part, dict):
        return isinstance(doc, dict) and all(
            k in doc and _contains(doc[k], v) for k, v in part.items()
        )
    if isinstance(part, list):
        return isinstance(doc, list) and all(any(_contains(d, p) for d in doc) for p in part)
    return type(doc) is type(part) and doc == part


def _json_contains(doc: str, part: str) -> int:
    return int(_contains(json.loads(doc), json.loads(part)))


def _blob(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _floats(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _memory(row: sqlite3.Row) -> Memory:
    at = _at(row["valid_from"])
    created, updated = _at(row["created_at"]), _at(row["updated_at"])
    assert at is not None and created is not None and updated is not None
    return Memory(
        id=uuid.UUID(row["id"]),
        scope=Scope(row["user_id"], row["agent_id"], row["session_id"]),
        type=row["type"], key=row["key"], content=row["content"],
        importance=row["importance"], confidence=row["confidence"], valid_from=at,
        valid_to=_at(row["valid_to"]), superseded_by=_id(row["superseded_by"]),
        expires_at=_at(row["expires_at"]), source=row["source"],
        metadata=json.loads(row["metadata"]), access_count=row["access_count"],
        created_at=created, updated_at=updated,
    )  # fmt: skip


def _where(f: Filters) -> tuple[str, dict[str, Any]]:
    """PgStore's `_where`, in SQLite: the same rows for the same filters."""
    params: dict[str, Any] = {
        "user_id": f.scope.user_id,
        "agent_id": f.scope.agent_id,
        "session_id": f.scope.session_id,
    }
    own = own_rows(f, lambda name: f":{name}")
    current = " AND m.valid_to IS NULL" if f.as_of is None else ""
    if with_shared(f):
        # Two indexed lookups: SQLite does not split `ns AND (own OR shared)` into them, and
        # scans the whole namespace instead (checked with EXPLAIN QUERY PLAN).
        scope = (
            f"m.id IN (SELECT m.id FROM memories m WHERE m.namespace = :ns AND {own}{current} "
            f"UNION ALL SELECT m.id FROM memories m WHERE m.namespace = :ns AND {SHARED}{current})"
        )
    else:
        scope = f"m.namespace = :ns AND {own}"
    if f.as_of is not None:
        params["as_of"] = _t(f.as_of)
        when = (
            "m.valid_from <= :as_of AND (m.valid_to IS NULL OR m.valid_to > :as_of)"
            " AND (m.expires_at IS NULL OR m.expires_at > :as_of)"
        )
    else:
        params["now"] = _t(f.now or datetime.now(UTC))
        when = "m.valid_to IS NULL AND (m.expires_at IS NULL OR m.expires_at > :now)"
    sql = f"{scope} AND {when}"
    if f.types:
        names = [f"type{i}" for i in range(len(f.types))]
        sql += f" AND m.type IN ({', '.join(':' + n for n in names)})"
        params.update(zip(names, f.types, strict=True))
    if f.metadata:  # jsonb containment, as PgStore's `@>`
        sql += " AND json_contains(m.metadata, :metadata)"
        params["metadata"] = json.dumps(f.metadata)
    return sql, params


class SqliteTx:
    def __init__(self, conn: sqlite3.Connection, dims: dict[str, int]) -> None:
        self._conn = conn
        self._dims = dims

    def _rows(self, query: str, params: Any = ()) -> list[sqlite3.Row]:
        return self._conn.execute(query, params).fetchall()

    def _one(self, query: str, params: Any = ()) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._conn.execute(query, params).fetchone()
        return row

    def _dim(self, namespace: str) -> int:
        if namespace not in self._dims:
            row = self._one("SELECT dim FROM namespaces WHERE name = ?", (namespace,))
            if row is None:
                raise StoreError(f"namespace {namespace!r} is not bound")
            self._dims[namespace] = row["dim"]
        return self._dims[namespace]

    def bind_namespace(
        self, namespace: str, *, provider: str, model: str, dim: int, tokenizer: str
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO namespaces "
            "(name, embedding_provider, embedding_model, dim, tokenizer, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (namespace, provider, model, dim, tokenizer, _t(datetime.now(UTC))),
        )
        row = self._one(
            "SELECT embedding_provider, embedding_model, dim, tokenizer FROM namespaces "
            "WHERE name = ?",
            (namespace,),
        )
        assert row is not None
        check_binding(
            namespace, dict(row), provider=provider, model=model, dim=dim, tokenizer=tokenizer
        )
        self._dims[namespace] = dim

    def lookup_idempotency(self, namespace: str, key: str) -> tuple[str | None, Any] | None:
        row = self._one(
            "SELECT user_id, result FROM idempotency WHERE namespace = ? AND key = ? "
            "AND result IS NOT NULL",
            (namespace, key),
        )
        return (row["user_id"], json.loads(row["result"])) if row else None

    def claim_idempotency(
        self, namespace: str, key: str, user_id: str | None
    ) -> tuple[str | None, Any] | None:
        claimed = self._conn.execute(
            "INSERT OR IGNORE INTO idempotency (namespace, key, user_id, result, created_at) "
            "VALUES (?, ?, ?, NULL, ?)",
            (namespace, key, user_id, _t(datetime.now(UTC))),
        )
        if claimed.rowcount == 1:
            return None
        # Writers are serialised: whoever holds the key has committed, with its result.
        row = self._one(
            "SELECT user_id, result FROM idempotency WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        assert row is not None and row["result"] is not None
        return row["user_id"], json.loads(row["result"])

    def lock_profile_slot(self, namespace: str, scope: Scope, key: str) -> None:
        """Nothing to do: `BEGIN IMMEDIATE` already serialises every writer."""

    def store_idempotency(self, namespace: str, key: str, result: Any) -> None:
        self._conn.execute(
            "UPDATE idempotency SET result = ? WHERE namespace = ? AND key = ?",
            (json.dumps(result), namespace, key),
        )

    def insert_memory(self, namespace: str, memory: Memory) -> None:
        m, s = memory, memory.scope
        self._conn.execute(
            _INSERT,
            (
                namespace, str(m.id), s.user_id, s.agent_id, s.session_id, m.type, m.key,
                m.content, m.importance, m.confidence, _t(m.valid_from), _t(m.valid_to),
                _text(m.superseded_by), _t(m.expires_at), m.source, json.dumps(m.metadata),
                m.access_count, _t(m.created_at), _t(m.updated_at),
            ),
        )  # fmt: skip

    def get_memory(
        self, namespace: str, memory_id: uuid.UUID, *, lock: bool = False
    ) -> Memory | None:
        row = self._one(
            f"SELECT {MEMORY_COLUMNS} FROM memories WHERE namespace = ? AND id = ?",
            (namespace, str(memory_id)),
        )
        return _memory(row) if row else None

    def current_profile(
        self, namespace: str, scope: Scope, key: str, *, lock: bool = False
    ) -> Memory | None:
        row = self._one(
            f"SELECT {MEMORY_COLUMNS} FROM memories WHERE namespace = ? "
            "AND coalesce(user_id, '') = ? AND coalesce(agent_id, '') = ? AND key = ? "
            "AND valid_to IS NULL",
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
            "UPDATE memories SET valid_to = ?, superseded_by = ?, updated_at = ? "
            "WHERE namespace = ? AND id = ?",
            (_t(at), _text(superseded_by), _t(at), namespace, str(memory_id)),
        )

    def set_content(self, namespace: str, memory_id: uuid.UUID, content: str, at: datetime) -> None:
        self._conn.execute(
            "UPDATE memories SET content = ?, updated_at = ? WHERE namespace = ? AND id = ?",
            (content, _t(at), namespace, str(memory_id)),
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
            "INSERT INTO memory_events "
            "(namespace, memory_id, user_id, op, before, after, reason, produced_by, at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                namespace, str(memory.id), memory.scope.user_id, op, _dumps(before),
                _dumps(after), reason, produced_by, _t(at),
            ),
        )  # fmt: skip

    def list_memories(
        self, namespace: str, filters: Filters, *, limit: int | None = None, offset: int = 0
    ) -> list[Memory]:
        where, params = _where(filters)
        rows = self._rows(
            f"SELECT {MEMORY_COLUMNS} FROM memories m WHERE {where} ORDER BY valid_from, id "
            "LIMIT :limit OFFSET :offset",  # -1: no limit
            {**params, "ns": namespace, "limit": -1 if limit is None else limit, "offset": offset},
        )
        return [_memory(r) for r in rows]

    def get_memories(self, namespace: str, ids: list[uuid.UUID]) -> list[Memory]:
        if not ids:
            return []
        marks = ", ".join("?" * len(ids))
        rows = self._rows(
            f"SELECT {MEMORY_COLUMNS} FROM memories WHERE namespace = ? AND id IN ({marks})",
            (namespace, *(str(i) for i in ids)),
        )
        return [_memory(r) for r in rows]

    def index_memory(
        self, namespace: str, memory_id: uuid.UUID, content: str, vector: list[float]
    ) -> None:
        dim = self._dim(namespace)
        if len(vector) != dim:
            raise StoreError(f"namespace {namespace!r} holds {dim}-d vectors, got {len(vector)}")
        counts = term_counts(content)
        mid = str(memory_id)
        self._conn.execute("DELETE FROM memory_postings WHERE memory_id = ?", (mid,))
        self._conn.executemany(
            "INSERT INTO memory_postings (namespace, memory_id, term, tf) VALUES (?, ?, ?, ?)",
            [(namespace, mid, term, tf) for term, tf in counts.items()],
        )
        self._conn.execute(
            "UPDATE memories SET terms = ? WHERE id = ?", (sum(counts.values()), mid)
        )
        self._conn.execute(
            "INSERT INTO memory_vectors (memory_id, namespace, embedding) VALUES (?, ?, ?) "
            "ON CONFLICT (memory_id) DO UPDATE SET embedding = excluded.embedding",
            (mid, namespace, _blob(vector)),
        )

    def vector_search(
        self, namespace: str, vector: list[float], filters: Filters, limit: int
    ) -> list[tuple[uuid.UUID, float]]:
        where, params = _where(filters)
        rows = self._rows(
            "SELECT m.id, vec_distance_cosine(v.embedding, :q) AS d FROM memories m "
            f"JOIN memory_vectors v ON v.memory_id = m.id WHERE {where} "
            "ORDER BY d, m.id LIMIT :limit",
            {**params, "ns": namespace, "q": _blob(vector), "limit": limit},
        )
        return [(uuid.UUID(r["id"]), 1 - float(r["d"])) for r in rows]

    def text_search(
        self, namespace: str, query: str, filters: Filters, limit: int
    ) -> list[uuid.UUID]:
        wanted = sorted(set(terms(query)))
        if not wanted:
            return []
        # N, avgdl and n(t) are the rows the filters let through, as in PgStore.
        where, params = _where(filters)
        params = {**params, "ns": namespace}
        stats = self._one(
            "SELECT count(*) AS n, coalesce(avg(m.terms), 0) AS avgdl FROM memories m "
            f"WHERE {where}",
            params,
        )
        assert stats is not None
        names = {f"t{i}": t for i, t in enumerate(wanted)}
        rows = self._rows(
            "SELECT m.id, m.terms, p.term, p.tf FROM memories m "
            "JOIN memory_postings p ON p.memory_id = m.id "
            f"AND p.term IN ({', '.join(':' + n for n in names)}) WHERE {where}",
            {**params, **names},
        )
        df: dict[str, int] = {}
        for r in rows:
            df[r["term"]] = df.get(r["term"], 0) + 1
        scores: dict[str, float] = {}
        for r in rows:
            score = idf(stats["n"], df[r["term"]]) * term_score(r["tf"], r["terms"], stats["avgdl"])
            scores[r["id"]] = scores.get(r["id"], 0.0) + score
        ranked = sorted(scores.items(), key=lambda item: (-round(item[1], 9), item[0]))
        return [uuid.UUID(i) for i, _ in ranked[:limit]]

    def expired(self, namespace: str, now: datetime, user_id: str | None) -> list[Memory]:
        rows = self._rows(
            f"SELECT {MEMORY_COLUMNS} FROM memories WHERE namespace = ? AND valid_to IS NULL "
            "AND expires_at <= ? AND (? IS NULL OR user_id = ?) ORDER BY expires_at, id",
            (namespace, _t(now), user_id, user_id),
        )
        return [_memory(r) for r in rows]

    def scopes(self, namespace: str, user_id: str | None) -> list[Scope]:
        rows = self._rows(
            "SELECT DISTINCT user_id, agent_id, session_id FROM memories "
            "WHERE namespace = ? AND valid_to IS NULL AND (? IS NULL OR user_id = ?) "
            "ORDER BY user_id, agent_id, session_id",  # NULLs first, as in PgStore
            (namespace, user_id, user_id),
        )
        return [Scope(r["user_id"], r["agent_id"], r["session_id"]) for r in rows]

    def vectors(self, namespace: str, ids: list[uuid.UUID]) -> dict[uuid.UUID, list[float]]:
        if not ids:
            return {}
        marks = ", ".join("?" * len(ids))
        rows = self._rows(
            "SELECT memory_id, embedding FROM memory_vectors "
            f"WHERE namespace = ? AND memory_id IN ({marks})",
            (namespace, *(str(i) for i in ids)),
        )
        return {uuid.UUID(r["memory_id"]): _floats(r["embedding"]) for r in rows}

    def idle(self, namespace: str, filters: Filters, before: datetime) -> list[Memory]:
        where, params = _where(filters)
        rows = self._rows(
            f"SELECT {MEMORY_COLUMNS} FROM memories m WHERE {where} "
            "AND coalesce(m.last_accessed_at, m.valid_from) < :before ORDER BY valid_from, id",
            {**params, "ns": namespace, "before": _t(before)},
        )
        return [_memory(r) for r in rows]

    def touch(self, namespace: str, ids: list[uuid.UUID], at: datetime) -> None:
        if ids:
            marks = ", ".join("?" * len(ids))
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
                f"WHERE namespace = ? AND id IN ({marks})",
                (_t(at), namespace, *(str(i) for i in ids)),
            )

    def history(self, namespace: str, memory_id: uuid.UUID) -> list[HistoryEntry]:
        rows = self._rows(
            "SELECT op, before, after, reason, produced_by, at FROM memory_events "
            "WHERE namespace = ? AND memory_id = ? ORDER BY id",
            (namespace, str(memory_id)),
        )
        return [
            HistoryEntry(
                op=r["op"],
                before=_loads(r["before"]),
                after=_loads(r["after"]),
                reason=r["reason"],
                produced_by=r["produced_by"],
                at=datetime.fromisoformat(r["at"]),  # NOT NULL
            )
            for r in rows
        ]

    def erase_user(self, namespace: str, user_id: str) -> int:
        self._conn.execute(
            "DELETE FROM idempotency WHERE namespace = ? AND user_id = ?", (namespace, user_id)
        )
        self._conn.execute(
            "UPDATE memories SET superseded_by = NULL WHERE namespace = ? AND superseded_by IN "
            "(SELECT id FROM memories WHERE namespace = ? AND user_id = ?)",
            (namespace, namespace, user_id),
        )
        erased = self._conn.execute(
            "DELETE FROM memories WHERE namespace = ? AND user_id = ?", (namespace, user_id)
        )
        return erased.rowcount


class SqliteStore:
    """One SQLite file (created with its schema when missing). Needs the `sqlite` extra."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 10_000) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover - the extra is part of the test setup
            raise ConfigError("the SQLite store needs the `sqlite` extra (sqlite-vec)") from exc
        self._path = str(path)
        self._busy = busy_timeout_ms
        self._load = sqlite_vec.load
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)  # notified when a transaction ends
        self._active = 0  # transactions running now: close() waits for them
        self._closed = False
        self._dims: dict[str, int] = {}
        conn = self._connection()
        if conn.execute("PRAGMA user_version").fetchone()[0] < _SCHEMA_VERSION:
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _connection(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False)
                conn.enable_load_extension(True)
                self._load(conn)
                conn.enable_load_extension(False)
            except sqlite3.Error as exc:
                raise StoreError(f"cannot open the SQLite database: {exc}") from exc
            conn.row_factory = sqlite3.Row
            conn.create_function("json_contains", 2, _json_contains, deterministic=True)
            conn.execute(f"PRAGMA busy_timeout = {self._busy}")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
            with self._lock:
                self._all.append(conn)
        return conn

    @contextmanager
    def transaction(self) -> Iterator[SqliteTx]:
        with self._lock:
            if self._closed:
                raise StoreError("the SQLite store is closed")
            self._active += 1
        try:
            conn = self._connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                raise StoreError(f"SQLite failed: {exc}") from exc
            try:
                yield SqliteTx(conn, self._dims)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            try:
                conn.execute("COMMIT")
            except sqlite3.Error as exc:  # e.g. a deferred foreign key, checked at commit
                conn.execute("ROLLBACK")
                raise StoreError(f"SQLite failed: {exc}") from exc
        finally:
            with self._lock:
                self._active -= 1
                self._idle.notify_all()

    def ping(self) -> None:
        try:
            self._connection().execute("SELECT 1")
        except sqlite3.Error as exc:
            raise StoreError(f"SQLite failed: {exc}") from exc

    def close(self) -> None:
        """Wait for running transactions, then close every thread's connection."""
        with self._lock:
            self._closed = True
            self._idle.wait_for(lambda: self._active == 0)
            for conn in self._all:
                conn.close()
            self._all.clear()
        self._local = threading.local()
