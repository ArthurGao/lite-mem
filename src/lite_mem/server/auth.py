"""API keys (design §7): `Authorization: Bearer lm_<id>_<secret>`. Only sha256(secret) is
configured; a key reaches the namespaces its globs match and no other.

Deliberately unguarded: `GET /health` (it returns "ok"/"error" per part, nothing stored)."""

import hashlib
import hmac
import ipaddress
import secrets
from fnmatch import fnmatchcase

from lite_mem.config import ApiKey

PREFIX = "lm_"


def new_key(key_id: str, namespaces: list[str]) -> tuple[str, ApiKey]:
    """A new token (shown once) and the entry to configure for it."""
    secret = secrets.token_urlsafe(32)  # may hold "_": the id never does, so parsing is unambiguous
    token = f"{PREFIX}{key_id}_{secret}"
    return token, ApiKey(id=key_id, sha256=_digest(secret), namespaces=namespaces)


def authenticate(header: str | None, keys: list[ApiKey]) -> ApiKey | None:
    """The configured key the `Authorization` header holds, or None."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.startswith(PREFIX):
        return None
    key_id, _, secret = token[len(PREFIX) :].partition("_")
    if not secret:
        return None
    digest = _digest(secret)
    for key in keys:  # compare every digest in constant time; ids are not secret
        if key.id == key_id and hmac.compare_digest(key.sha256, digest):
            return key
    return None


def allows(key: ApiKey, namespace: str) -> bool:
    return any(fnmatchcase(namespace, pattern) for pattern in key.namespaces)


def is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()
