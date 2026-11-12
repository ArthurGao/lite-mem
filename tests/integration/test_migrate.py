from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

from lite_mem.cli import app
from lite_mem.db import migrate, migration_files
from lite_mem.errors import StoreError
from tests.conftest import TEST_SCHEMA


def test_migrations_apply_once_and_record_themselves(pg: psycopg.Connection) -> None:
    versions = [v for v, _ in migration_files()]
    assert migrate(pg, TEST_SCHEMA) == versions
    assert migrate(pg, TEST_SCHEMA) == []  # up to date: nothing applied twice
    recorded = pg.execute(
        f"SELECT version FROM {TEST_SCHEMA}.schema_migrations ORDER BY version"
    ).fetchall()
    assert [r[0] for r in recorded] == versions
    tables = pg.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
        (TEST_SCHEMA,),
    ).fetchall()
    assert {"namespaces", "schema_migrations"} <= {t[0] for t in tables}


def test_the_cli_migrates_and_never_prints_the_password(
    pg: psycopg.Connection, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITEMEM_DATABASE_URL", pg_url)
    monkeypatch.setenv("LITEMEM_DB_SCHEMA", TEST_SCHEMA)
    first = CliRunner().invoke(app, ["migrate"])
    again = CliRunner().invoke(app, ["migrate"])
    assert first.exit_code == 0, first.output
    assert "applied 0001_namespaces" in first.output and "up to date" in again.output
    password = psycopg.conninfo.conninfo_to_dict(pg_url).get("password")
    if password:
        assert password not in first.output + again.output


def test_the_cli_refuses_without_a_database() -> None:
    result = CliRunner().invoke(app, ["migrate"])
    assert result.exit_code == 1 and "LITEMEM_DATABASE_URL" in result.output


def test_migrate_itself_refuses_a_shared_schema(pg: psycopg.Connection) -> None:
    with pytest.raises(StoreError, match="public"):
        migrate(pg, "public")


def test_an_error_quoting_a_connection_string_is_printed_without_its_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever a driver or server puts in an error, the CLI redacts it on the way out."""
    from lite_mem import cli

    def leaky(url: str | None) -> psycopg.Connection:
        raise StoreError("cannot connect to postgresql://u:hunter2@db/x: timeout")

    monkeypatch.setenv("LITEMEM_DATABASE_URL", "postgresql://u:hunter2@db/x")
    monkeypatch.setattr(cli, "connect", leaky)
    result = CliRunner().invoke(app, ["migrate"])
    assert result.exit_code == 1 and "cannot connect to postgresql://" in result.output
    assert "hunter2" not in result.output


def test_pgvector_is_installed_in_public_whatever_the_role_is_called(
    pg: psycopg.Connection,
) -> None:
    """Found running lite-mem in-process under llm-steward (2026-11-12): `CREATE EXTENSION vector`
    without a schema lands in the first existing schema of the search path, `"$user"` — in
    llm-steward's bundled database the role and its governance schema are both `steward` — where
    lite-mem's `search_path = <schema>, public` cannot see the `vector` type. The test database
    already has the extension, so this checks the statement migrate sends."""
    sent: list[str] = []

    class Recording:
        def __init__(self, conn: psycopg.Connection) -> None:
            self._conn = conn

        def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
            text = query.as_string(self._conn) if hasattr(query, "as_string") else query
            sent.append(text.decode() if isinstance(text, bytes) else text)
            return self._conn.execute(query, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

    migrate(Recording(pg), TEST_SCHEMA)  # type: ignore[arg-type]
    [extension] = [s for s in sent if "EXTENSION" in s]
    assert extension == "CREATE EXTENSION IF NOT EXISTS vector SCHEMA public"
