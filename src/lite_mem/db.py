"""PostgreSQL access and migrations (design §4).

Migrations are the `NNNN_*.sql` files in `migrations/`, applied in order, each in its own
transaction, recorded in `<schema>.schema_migrations`. They run with
`search_path = <schema>, public`, so one set of files serves any schema name (the app's and the
tests')."""

from importlib import resources

import psycopg
from psycopg import sql

from lite_mem.config import own_schema
from lite_mem.errors import StoreError

MIGRATIONS = "lite_mem.migrations"
MIN_PGVECTOR = (0, 8)  # the oldest version tested (the `vector` type and `<=>`; no ANN index)


def require_url(database_url: str | None) -> str:
    if not database_url:
        raise StoreError("no database configured: set LITEMEM_DATABASE_URL")
    return database_url


def connect(database_url: str | None) -> psycopg.Connection:
    try:
        return psycopg.connect(require_url(database_url))
    except psycopg.OperationalError as exc:
        raise StoreError(f"cannot connect to PostgreSQL: {exc}") from exc


def migration_files(package: str = MIGRATIONS) -> list[tuple[str, str]]:
    """(version, SQL) of each `NNNN_*.sql` file in `package`, in order."""
    files = resources.files(package)
    names = sorted(f.name for f in files.iterdir() if f.name.endswith(".sql"))
    return [(name.removesuffix(".sql"), files.joinpath(name).read_text("utf-8")) for name in names]


def use_schema(conn: psycopg.Connection, schema: str) -> None:
    """Point the current transaction at `schema` (`vector` stays reachable through public)."""
    conn.execute(sql.SQL("SET LOCAL search_path TO {}, public").format(sql.Identifier(schema)))


def migrate(conn: psycopg.Connection, schema: str) -> list[str]:
    """Apply pending migrations to `schema`; returns the versions applied (empty: up to date)."""
    try:
        own_schema(schema)
    except ValueError as exc:
        raise StoreError(str(exc)) from exc
    with conn.transaction():
        # in public, never the search path's first schema: that is "$user" when a schema is named
        # like the role (llm-steward's bundled database), out of reach of `<schema>, public`
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.schema_migrations "
                "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            ).format(sql.Identifier(schema))
        )
    applied: list[str] = []
    for version, script in migration_files():
        with conn.transaction():
            use_schema(conn, schema)
            # Serialise concurrent migrators on this schema; the check below then sees their work.
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"lite_mem:{schema}",))
            done = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s", (version,)
            ).fetchone()
            if done:
                continue
            conn.execute(script.encode())  # trusted SQL file shipped in the package
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
    check_vector(conn)
    return applied


def check_vector(conn: psycopg.Connection) -> None:
    """pgvector is new enough, and a real vector operation runs: an extension can look installed
    while its library is missing, so presence alone proves nothing."""
    try:
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        conn.execute("SELECT '[1,2]'::vector <=> '[1,2]'").fetchone()
    except psycopg.Error as exc:
        raise StoreError(f"pgvector is not usable: {exc}") from exc
    version = tuple(int(p) for p in row[0].split(".")[:2]) if row else (0, 0)
    if version < MIN_PGVECTOR:
        found = row[0] if row else "not installed"
        raise StoreError(f"pgvector >= 0.8 is required (found {found})")
