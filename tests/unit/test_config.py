import json

import pytest
from lite_model import Target
from pydantic import ValidationError

from lite_mem.config import Settings
from lite_mem.errors import redact
from lite_mem.llm import build_model


def _target(settings: Settings, role: str) -> Target:
    return build_model(settings.llm).target(role)


def test_defaults_point_at_local_models() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (s.db_schema, s.namespace) == ("lite_mem", None)
    assert (s.embedding.provider, s.embedding.model) == ("ollama", "bge-m3:567m")
    t = _target(s, "extract")
    # Ollama's /v1 as a keyless `openai` backend: the history keeps `openai:<model>`, no
    # temperature is added, and retries are the OpenAI SDK's own 2 (lite-mem before lite-model)
    assert (t.backend, t.type, t.model) == ("openai", "openai", "qwen3:8b")
    assert (t.base_url, t.api_key_env, t.temperature) == ("http://127.0.0.1:11434/v1", None, None)
    assert t.max_retries == 2


def test_env_maps_roles_to_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITEMEM_LLM__MODEL", "base")
    monkeypatch.setenv("LITEMEM_LLM__ROLES", '{"resolve": {"model": "careful"}}')
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    extract, resolve = _target(s, "extract"), _target(s, "resolve")
    assert (extract.model, resolve.model) == ("base", "careful")
    assert extract.base_url == resolve.base_url == "http://127.0.0.1:11434/v1"


def test_an_own_endpoint_keeps_the_built_in_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    # a whole `openai` backend given (with its type) replaces lite-mem's local one
    monkeypatch.setenv("LITEMEM_LLM__BACKENDS", json.dumps({"openai": {
        "type": "openai", "base_url": "http://host.docker.internal:11434/v1", "keyless": True
    }}))  # fmt: skip
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LITEMEM_LLM__ROLES", json.dumps(
        {"resolve": {"backend": "anthropic", "model": "claude-x", "max_output_tokens": 2048}}
    ))  # fmt: skip
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert _target(s, "extract").base_url == "http://host.docker.internal:11434/v1"
    # a role on another backend uses that backend's endpoint and key, never the default's
    resolve = _target(s, "resolve")
    assert (resolve.type, resolve.model, resolve.api_key_env) == (
        "anthropic", "claude-x", "ANTHROPIC_API_KEY"
    )  # fmt: skip
    assert resolve.base_url != "http://host.docker.internal:11434/v1"


@pytest.mark.parametrize("bad", ["App", "a//b", "", "a/" * 5 + "a", "a b"])
def test_namespace_is_validated(bad: str) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        Settings(_env_file=None, namespace=bad)  # type: ignore[call-arg]


def test_schema_must_be_an_identifier() -> None:
    with pytest.raises(ValidationError, match="schema"):
        Settings(_env_file=None, db_schema="lite-mem")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("text", "secret", "kept"),
    [
        ("postgresql://u:secret@h/db", "secret", "host=h"),  # parsed: rendered as key=value
        ("host=h password=secret dbname=x", "secret", "dbname=x"),
        ("host=h password='a b c' dbname=x", "a b c", "dbname=x"),
        ("postgresql://h/db?password=secret", "secret", "host=h"),
        ("host=h sslpassword=secret dbname=x", "secret", "dbname=x"),
        ("postgresql://h/db?sslpassword=secret", "secret", "host=h"),
        ("failed: host=h password=ab&cd dbname=x", "cd", "dbname=x"),
        ("failed: postgresql://h/db?password=ab&sslmode=require", "ab", "sslmode=require"),
        ("env PGPASSWORD=secret psql", "secret", "psql"),
        ("db_password=secret in config", "secret", "in config"),
        ("postgresql://h/db?db_password=secret&x=1", "secret", "x=1"),
        ("connection to postgresql://u:secret@h/db failed: timeout", "secret", "timeout"),
    ],
)
def test_redact_hides_passwords(text: str, secret: str, kept: str) -> None:
    assert secret not in redact(text)
    assert kept in redact(text)  # the rest stays readable


@pytest.mark.parametrize("reserved", ["public", "information_schema", "pg_catalog", "pg_temp"])
def test_a_shared_or_system_schema_is_refused(reserved: str) -> None:
    with pytest.raises(ValidationError, match="schema"):
        Settings(_env_file=None, db_schema=reserved)  # type: ignore[call-arg]


def test_the_schema_comes_from_its_own_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITEMEM_DB_SCHEMA", "elsewhere")
    assert Settings(_env_file=None).db_schema == "elsewhere"  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [0, -30])
def test_a_half_life_must_be_positive(bad: float) -> None:
    from lite_mem.config import SearchConfig

    with pytest.raises(ValidationError, match="half"):
        SearchConfig(half_lives={"episodic": bad})


def test_the_pipeline_is_switchable_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from lite_mem import factory
    from lite_mem.resolve import AddOnlyResolver

    monkeypatch.setenv("LITEMEM_PIPELINE__RESOLVER", "add_only")
    monkeypatch.setenv("LITEMEM_PIPELINE__RELATED", "3")
    monkeypatch.setenv("LITEMEM_EMBEDDING__DIM", "8")
    monkeypatch.setenv("LITEMEM_PIPELINE__SENSITIVE", "mask")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (s.pipeline.resolver, s.pipeline.related, s.embedding.dim) == ("add_only", 3, 8)

    class Store:  # the factory's store boundary; binding needs no model call when dim is set
        def transaction(self) -> object:
            from contextlib import nullcontext
            from unittest.mock import MagicMock

            return nullcontext(MagicMock())

        def close(self) -> None: ...

    s = s.model_copy(update={"namespace": "app"})
    engine = factory.open_engine(s, Store())  # type: ignore[arg-type]
    assert isinstance(engine._resolver, AddOnlyResolver) and engine._related == 3
    assert engine._sensitive == "mask"


def test_compose_changes_only_the_local_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Codex S12: compose's `environment` beats `.env`, so it must not replace the whole backends
    # map; it sets one field, merged with the .env's own backends and lite-mem's local default.
    monkeypatch.setenv("LITEMEM_LLM__BACKENDS", json.dumps({"vendor": {
        "type": "openai", "base_url": "https://vendor.test/v1", "api_key_env": "VENDOR_KEY"
    }}))  # fmt: skip
    monkeypatch.setenv(
        "LITEMEM_LLM__BACKENDS__OPENAI__BASE_URL", "http://host.docker.internal:11434/v1"
    )
    monkeypatch.setenv("VENDOR_KEY", "k")
    monkeypatch.setenv("LITEMEM_LLM__ROLES", '{"resolve": {"backend": "vendor", "model": "big"}}')
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    extract, resolve = _target(s, "extract"), _target(s, "resolve")
    assert (extract.type, extract.base_url, extract.api_key_env) == (
        "openai", "http://host.docker.internal:11434/v1", None
    )  # fmt: skip
    assert (resolve.base_url, resolve.model) == ("https://vendor.test/v1", "big")


@pytest.mark.parametrize(
    "legacy",
    [
        ("LITEMEM_LLM__BASE_URL", "http://elsewhere/v1"),
        ("LITEMEM_LLM__API_KEY", "sk-x"),
        ("LITEMEM_LLM__PROVIDER", "anthropic"),
        ("LITEMEM_LLM__REASONING", "none"),
        ("LITEMEM_LLM__ROLES", '{"resolve": {"provider": "anthropic"}}'),
    ],
)
def test_the_settings_from_before_lite_model_are_refused(
    monkeypatch: pytest.MonkeyPatch, legacy: tuple[str, str]
) -> None:
    # S12 renamed them; one still set must stop the start, never be ignored (Codex S12)
    monkeypatch.setenv(*legacy)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings(_env_file=None)  # type: ignore[call-arg]
