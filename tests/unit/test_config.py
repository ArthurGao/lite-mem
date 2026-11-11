import pytest
from pydantic import ValidationError

from lite_mem.config import Settings
from lite_mem.errors import redact


def test_defaults_point_at_local_models() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (s.db_schema, s.namespace) == ("lite_mem", None)
    assert (s.embedding.provider, s.embedding.model) == ("ollama", "bge-m3:567m")
    assert (s.llm.base_url, s.llm.model) == ("http://127.0.0.1:11434/v1", "qwen3:8b")


def test_env_maps_roles_to_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITEMEM_LLM__MODEL", "base")
    monkeypatch.setenv("LITEMEM_LLM__ROLES", '{"resolve": {"model": "careful"}}')
    llm = Settings(_env_file=None).llm  # type: ignore[call-arg]
    extract, resolve = llm.target("extract"), llm.target("resolve")
    assert (extract.model, resolve.model) == ("base", "careful")
    assert extract.base_url == resolve.base_url == "http://127.0.0.1:11434/v1"


def test_a_role_on_another_provider_does_not_inherit_the_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITEMEM_LLM__API_KEY", "ollama-key")
    monkeypatch.setenv(
        "LITEMEM_LLM__ROLES", '{"resolve": {"provider": "anthropic", "model": "claude-x"}}'
    )
    resolve = Settings(_env_file=None).llm.target("resolve")  # type: ignore[call-arg]
    assert (resolve.provider, resolve.model) == ("anthropic", "claude-x")
    assert (resolve.base_url, resolve.api_key) == (None, None)  # not Ollama's URL or key


@pytest.mark.parametrize("bad", ["App", "a//b", "", "a/" * 5 + "a", "a b"])
def test_namespace_is_validated(bad: str) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        Settings(_env_file=None, namespace=bad)  # type: ignore[call-arg]


def test_schema_must_be_an_identifier() -> None:
    with pytest.raises(ValidationError, match="schema"):
        Settings(_env_file=None, db_schema="lite-mem")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("postgresql://u:secret@h/db", "secret"),
        ("host=h password=secret dbname=x", "secret"),
        ("host=h password='a b c' dbname=x", "a b c"),
        ("postgresql://h/db?password=secret", "secret"),
        ("host=h sslpassword=secret dbname=x", "secret"),
        ("postgresql://h/db?sslpassword=secret", "secret"),
        ("failed: host=h password=ab&cd dbname=x", "cd"),
        ("env PGPASSWORD=secret psql", "secret"),
        ("db_password=secret in config", "secret"),
        ("postgresql://h/db?db_password=secret&x=1", "secret"),
        ("failed: postgresql://h/db?password=ab&sslmode=require", "ab"),
        ("connection to postgresql://u:secret@h/db failed: timeout", "secret"),
    ],
)
def test_redact_hides_passwords(text: str, secret: str) -> None:
    assert secret not in redact(text)
    assert "h" in redact(text)  # the rest stays readable


@pytest.mark.parametrize("reserved", ["public", "information_schema", "pg_catalog", "pg_temp"])
def test_a_shared_or_system_schema_is_refused(reserved: str) -> None:
    with pytest.raises(ValidationError, match="schema"):
        Settings(_env_file=None, db_schema=reserved)  # type: ignore[call-arg]


def test_the_schema_comes_from_its_own_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITEMEM_DB_SCHEMA", "elsewhere")
    assert Settings(_env_file=None).db_schema == "elsewhere"  # type: ignore[call-arg]
