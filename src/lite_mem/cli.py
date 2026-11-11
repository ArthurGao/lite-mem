"""`litemem` command line. Commands are added as their stage lands (design §12)."""

from typing import NoReturn

import typer

from lite_mem import __version__
from lite_mem.config import Settings
from lite_mem.db import connect, migrate
from lite_mem.errors import LiteMemError, redact

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
