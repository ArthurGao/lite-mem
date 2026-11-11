"""`litemem` command line. Commands are added as their stage lands (design §12)."""

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any, NoReturn

import typer

from lite_mem import __version__, factory
from lite_mem.config import Settings
from lite_mem.db import connect, migrate
from lite_mem.engine import MemoryEngine
from lite_mem.errors import LiteMemError, redact
from lite_mem.types import Message, Scope

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {redact(message)}", err=True)
    raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command("migrate")
def migrate_command() -> None:
    """Create or update the schema on LITEMEM_DATABASE_URL."""
    settings = Settings()
    try:
        with connect(settings.database_url) as conn:
            applied = migrate(conn, settings.db_schema)
    except LiteMemError as exc:
        _fail(str(exc))
    where = f"{redact(settings.database_url or '')} schema {settings.db_schema}"
    typer.echo(f"applied {', '.join(applied)} on {where}" if applied else f"{where} is up to date")


User = Annotated[str | None, typer.Option("--user", help="user_id of the scope")]
Agent = Annotated[str | None, typer.Option("--agent", help="agent_id of the scope")]


@contextmanager
def _engine() -> Iterator[MemoryEngine]:
    """An engine from the settings; any lite-mem error ends the command with its (redacted)
    message."""
    try:
        engine = factory.open_engine(Settings(), max_size=1)  # one command, one connection
        try:
            yield engine
        finally:
            engine.close()
    except LiteMemError as exc:
        _fail(str(exc))


def _show(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        typer.echo(json.dumps(row, ensure_ascii=False, default=str))


@app.command("add")
def add_command(
    text: Annotated[list[str], typer.Argument(help="what the user said (one message each)")],
    user: User = None,
    agent: Agent = None,
    raw: Annotated[bool, typer.Option("--raw", help="store as said (infer=false)")] = False,
) -> None:
    """Remember what the user said: extract and resolve facts (or store them as said)."""
    with _engine() as engine:
        events = engine.add([Message("user", t) for t in text], Scope(user, agent), infer=not raw)
        _show([{"op": e.op, "id": e.memory_id, "content": e.content} for e in events])


@app.command("search")
def search_command(query: str, user: User = None, agent: Agent = None, top_k: int = 10) -> None:
    """The memories that answer QUERY, best first."""
    with _engine() as engine:
        hits = engine.search(query, Scope(user, agent), top_k=top_k)
        _show([{"score": round(h.score, 6), "id": h.memory.id, "content": h.memory.content}
               for h in hits])  # fmt: skip


@app.command("list")
def list_command(user: User = None, agent: Agent = None) -> None:
    """The scope's current memories."""
    with _engine() as engine:
        memories = engine.list_memories(Scope(user, agent))
        _show([{"id": m.id, "type": m.type, "content": m.content} for m in memories])


@app.command("history")
def history_command(memory_id: uuid.UUID) -> None:
    """Every change to one memory, oldest first."""
    with _engine() as engine:
        _show([{"op": h.op, "at": h.at, "after": h.after, "reason": h.reason,
                "by": h.produced_by} for h in engine.history(memory_id)])  # fmt: skip


@app.command("erase")
def erase_command(
    user: Annotated[str, typer.Option("--user", help="the user to erase")],
    yes: Annotated[bool, typer.Option("--yes", help="confirm: this cannot be undone")] = False,
) -> None:
    """Delete everything about one user (the only hard delete)."""
    if not yes:
        _fail("erase deletes everything about the user and cannot be undone: add --yes")
    with _engine() as engine:
        typer.echo(f"erased {engine.erase(user)} memories of {user}")
