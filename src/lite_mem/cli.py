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

# A crash never prints locals: they can hold the database URL or a key.
app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)

# What an eval run records about each system's settings (never the database URL).
_MODEL_FACING = {"llm", "embedding", "pipeline", "search"}


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
def _engine(settings: Settings | None = None) -> Iterator[MemoryEngine]:
    """An engine from the settings; any lite-mem error ends the command with its (redacted)
    message."""
    try:
        engine = factory.open_engine(settings or Settings(), max_size=1)  # one connection
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


key_app = typer.Typer(no_args_is_help=True, help="API keys of the HTTP service")
app.add_typer(key_app, name="key")


@key_app.command("new")
def key_new(
    key_id: Annotated[str, typer.Argument(help="a name for the key: a-z, 0-9, -")],
    namespace: Annotated[
        list[str], typer.Option("--namespace", "-n", help="namespace glob it reaches (repeatable)")
    ],
) -> None:
    """Make a key: prints the token (shown once) and the entry for LITEMEM_SERVER__API_KEYS."""
    from pydantic import ValidationError

    from lite_mem.server.auth import new_key

    try:
        token, entry = new_key(key_id, namespace)
    except ValidationError as exc:
        _fail(f"invalid key: {exc.errors()[0]['msg']}")
    typer.echo(f"token (shown once): {token}")
    typer.echo("add this entry to LITEMEM_SERVER__API_KEYS (a JSON list):")
    typer.echo(entry.model_dump_json())


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="bind address (default 127.0.0.1)")] = None,
    port: Annotated[int | None, typer.Option(help="port (default 8710)")] = None,
) -> None:
    """Run the HTTP service (extra: server)."""
    import uvicorn

    from lite_mem.server.app import create_app
    from lite_mem.server.auth import is_loopback

    settings = Settings()
    host = host or settings.server.host
    if not settings.server.api_keys and not is_loopback(host):
        _fail(f"no API keys configured: refusing to listen on {host}; set LITEMEM_SERVER__API_KEYS")
    uvicorn.run(create_app(settings), host=host, port=port or settings.server.port)


@app.command("eval")
def eval_command(
    variant: Annotated[
        list[str] | None, typer.Option("--variant", "-v", help="lite-mem variant (repeatable)")
    ] = None,
    mem0: Annotated[bool, typer.Option("--mem0", help="also run mem0 2.2.1 (extra: eval)")] = False,
    dataset: Annotated[str, typer.Option(help="the conversations")] = "eval/dataset.json",
    k: int = 3,
    schema: Annotated[str, typer.Option(help="where eval data lives")] = "lite_mem_eval",
    out: Annotated[
        str | None, typer.Option(help="also save the run as JSON (results + what produced them)")
    ] = None,
) -> None:
    """Run the synthetic conversations through lite-mem variants (and mem0); print the table."""
    from pathlib import Path

    from lite_mem import eval as ev

    names = variant or ["resolve", "add_only"]
    unknown = [n for n in names if n not in ev.VARIANTS]
    if unknown:
        _fail(f"unknown variant(s) {unknown}; known: {sorted(ev.VARIANTS)}")
    data, base, reports = ev.load(Path(dataset)), Settings(), []
    used: dict[str, Any] = {}  # per system, the model-facing settings (secrets masked)
    try:
        with connect(base.database_url) as conn:
            migrate(conn, schema)
    except LiteMemError as exc:
        _fail(str(exc))
    mem0_system = None
    if mem0:  # set up first: a missing dependency fails before the long lite-mem runs
        from lite_mem.eval_mem0 import Mem0System

        try:
            mem0_system = Mem0System(base)
        except Exception as exc:  # third-party: its message may carry the URL; _fail redacts
            _fail(f"mem0 setup failed: {type(exc).__name__}: {exc}")
    try:
        for name in names:
            settings = ev.variant_settings(base, name, schema)
            used[name] = ev.describe(settings, _MODEL_FACING)
            with _engine(settings) as engine:
                reports.append(ev.run(ev.LiteMemSystem(name, engine), data, k=k))
        if mem0_system:
            used[mem0_system.name] = ev.describe(base, {"llm", "embedding"})
            try:
                reports.append(ev.run(mem0_system, data, k=k))
            except Exception as exc:  # as above; the lite-mem results are still printed
                typer.echo(
                    f"error: mem0 run failed: {redact(f'{type(exc).__name__}: {exc}')}", err=True
                )
    finally:
        if mem0_system:
            mem0_system.close()
    if out:
        ev.save(Path(out), reports, dataset=Path(dataset), settings=used)
    typer.echo(ev.table(reports))
    for r in reports:
        for miss in r.misses:
            typer.echo(f"  {r.system}: {miss}")
