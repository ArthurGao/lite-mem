"""Errors the engine raises. A failure is never an empty result.

Each class says what it means to an HTTP client (`status`) and whether its message may be shown
(`expose`): upstream failures can carry endpoints or provider messages, so they are logged, not
returned (design §7)."""

import re
from typing import ClassVar

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

# (pattern, replacement) for free text: a URL's user:password@; a password-like parameter in a
# URL query (it ends at the next parameter); a key=value password, any prefix (PGPASSWORD,
# db_password, sslpassword), quoted or up to the next space (it may itself contain '&').
_CREDENTIALS = (
    (re.compile(r"(://)[^\s/@]+@"), r"\1***@"),
    (re.compile(r"([?&]\w*password=)[^\s&]+", re.IGNORECASE), r"\1***"),
    (re.compile(r"((?<![?&\w])\w*password\s*=\s*)('[^']*'|\S+)", re.IGNORECASE), r"\1***"),
)


def redact(text: str) -> str:
    """`text` with credentials removed, for anything that leaves the process (the CLI's output,
    later the server's logs). A whole connection string is parsed the way libpq does; free text
    (an error message quoting one) is scrubbed by pattern."""
    try:
        params = conninfo_to_dict(text)
    except psycopg.ProgrammingError:
        params = None
    if params is not None and ("=" in text or "://" in text):
        for name in params:  # password, sslpassword, and any other secret libpq may add
            if "password" in name and params[name]:
                params[name] = "***"
        return make_conninfo("", **params)
    for pattern, replacement in _CREDENTIALS:
        text = pattern.sub(replacement, text)
    return text


class LiteMemError(Exception):
    """Base class for every lite-mem error."""

    status: ClassVar[int] = 500
    expose: ClassVar[bool] = False


class InvalidRequest(LiteMemError):
    """The caller asked for something invalid; the caller can fix it, so the message is shown."""

    status = 400
    expose = True


class NotFound(LiteMemError):
    """No such memory in this namespace (another namespace's memory is not found either)."""

    status = 404
    expose = True


class Conflict(LiteMemError):
    """The request conflicts with the stored state (e.g. a namespace bound to another model)."""

    status = 409
    expose = True


class ConfigError(LiteMemError):
    """The service's own configuration is wrong or incomplete: the operator's problem, logged."""


class StoreError(LiteMemError):
    """The database failed or is unreachable."""

    status = 503


class ModelError(LiteMemError):
    """An embedding or LLM call failed, or answered outside its schema (no fallback: it raises)."""

    status = 502
