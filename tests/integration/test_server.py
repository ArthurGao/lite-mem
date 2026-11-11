"""S6 acceptance (design §7, §12): the HTTP service over the same engine. No key → 401, a key for
another namespace → 403; add → search → history works and matches the library; errors are Problem
Details that never expose an upstream message; `Idempotency-Key` replays."""

import uuid
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from lite_mem.config import ServerConfig, Settings
from lite_mem.errors import ModelError
from lite_mem.factory import Components
from lite_mem.server.app import PROBLEM, create_app
from lite_mem.server.auth import new_key
from lite_mem.stores.pg import PgStore
from lite_mem.testing import FakeEmbedder
from lite_mem.types import Scope
from tests.conftest import fake_components

STEWARD_TOKEN, STEWARD_KEY = new_key("steward", ["steward/*"])
APP_TOKEN, APP_KEY = new_key("app", ["app"])
LOCAL = "http://127.0.0.1:8710"


def _settings(keys: bool = True) -> Settings:
    api_keys = [STEWARD_KEY, APP_KEY] if keys else []
    return Settings(namespace="app", server=ServerConfig(api_keys=api_keys))


@pytest.fixture
def components(store: PgStore) -> Components:
    return fake_components(store)


def _client(components: Components, *, keys: bool = True, **kw: Any) -> TestClient:
    return TestClient(create_app(_settings(keys), components), **kw)


def _auth(token: str = APP_TOKEN, namespace: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if namespace:
        headers["X-LiteMem-Namespace"] = namespace
    return headers


def _add(client: TestClient, text: str, user: str = "ada", **headers: str) -> Any:
    body = {"messages": [{"role": "user", "content": text}], "scope": {"user_id": user},
            "infer": False}  # fmt: skip
    return client.post("/v1/memories:add", json=body, headers={**_auth(), **headers})


def _list(client: TestClient, user: str = "ada", **headers: str) -> httpx.Response:
    return client.get("/v1/memories", params={"user_id": user}, headers=headers or _auth())


def _is_problem(r: httpx.Response, status: int) -> bool:
    return (
        r.status_code == status
        and r.headers["content-type"].startswith(PROBLEM)
        and r.json()["status"] == status
    )


def test_no_key_or_a_wrong_key_is_401(components: Components) -> None:
    client = _client(components)
    for headers in ({}, {"Authorization": "Bearer lm_app_wrong"}, {"Authorization": "Basic x"}):
        r = client.get("/v1/memories", params={"user_id": "ada"}, headers=headers)
        assert _is_problem(r, 401) and r.headers["www-authenticate"] == "Bearer"


def test_a_key_reaches_only_its_namespaces(components: Components) -> None:
    client = _client(components)
    assert _list(client).status_code == 200
    assert _is_problem(_list(client, **_auth(namespace="steward/1")), 403)
    assert _list(client, **_auth(STEWARD_TOKEN, "steward/1")).status_code == 200
    # the steward key does not reach the default namespace ("app")
    assert _is_problem(_list(client, **_auth(STEWARD_TOKEN)), 403)


def test_a_memory_of_another_namespace_is_not_found(components: Components) -> None:
    client = _client(components)
    [event] = _add(client, "住在北京").json()["events"]
    path = f"/v1/memories/{event['memory_id']}"
    assert _is_problem(client.get(path, headers=_auth(STEWARD_TOKEN, "steward/1")), 404)


def test_add_search_history_and_the_library_agree(components: Components) -> None:
    client = _client(components)
    r = _add(client, "住在北京")
    assert r.status_code == 200
    [added] = r.json()["events"]
    assert added["op"] == "ADD" and added["content"] == "住在北京"
    memory_id = added["memory_id"]
    hits = client.post(
        "/v1/memories:search", json={"query": "北京", "scope": {"user_id": "ada"}}, headers=_auth()
    ).json()["hits"]
    assert [h["memory"]["id"] for h in hits] == [memory_id]
    fixed = client.patch(
        f"/v1/memories/{memory_id}", json={"content": "住在北京市"}, headers=_auth()
    )
    assert fixed.status_code == 200 and fixed.json()["content"] == "住在北京市"
    history = client.get(f"/v1/memories/{memory_id}/history", headers=_auth()).json()["entries"]
    assert [e["op"] for e in history] == ["ADD", "CORRECT"]
    library = components.engine("app")
    assert library.get(uuid.UUID(memory_id)).content == "住在北京市"
    assert [m.id for m in library.list_memories(Scope(user_id="ada"))] == [uuid.UUID(memory_id)]
    assert [m["id"] for m in _list(client).json()["memories"]] == [memory_id]
    gone = client.delete(f"/v1/memories/{memory_id}", params={"reason": "moved"}, headers=_auth())
    assert gone.status_code == 200 and gone.json()["valid_to"] is not None
    assert _list(client).json() == {"memories": []}


def test_erase_deletes_one_user_only(components: Components) -> None:
    client = _client(components)
    _add(client, "ada's fact")
    _add(client, "bob's fact", user="bob")
    r = client.delete("/v1/users/ada", headers=_auth())
    assert r.status_code == 200 and r.json() == {"erased": 1}
    assert [m["content"] for m in _list(client, "bob").json()["memories"]] == ["bob's fact"]


def test_idempotency_key_replays_and_refuses_another_user(components: Components) -> None:
    client = _client(components)
    first = _add(client, "住在北京", **{"Idempotency-Key": "ep-1"})
    again = _add(client, "住在北京", **{"Idempotency-Key": "ep-1"})
    assert first.json() == again.json()
    assert len(_list(client).json()["memories"]) == 1
    assert _is_problem(_add(client, "x", user="bob", **{"Idempotency-Key": "ep-1"}), 409)


def test_errors_are_problem_details(components: Components) -> None:
    client = _client(components)
    missing = client.get(f"/v1/memories/{uuid.uuid4()}", headers=_auth())
    assert _is_problem(missing, 404) and {"type", "title", "detail"} <= set(missing.json())
    invalid = client.post("/v1/memories:search", json={"query": " ", "scope": {}}, headers=_auth())
    assert _is_problem(invalid, 400) and "query" in invalid.json()["detail"]
    malformed = client.post("/v1/memories:search", json={"scope": {}}, headers=_auth())
    assert _is_problem(malformed, 422) and malformed.json()["errors"][0]["loc"][-1] == "query"
    echoed = client.post(  # a rejected value is not sent back (it may be a secret)
        "/v1/memories:search", json={"query": "q", "scope": {}, "top_k": "sk-not-a-number"},
        headers=_auth(),
    )  # fmt: skip
    assert _is_problem(echoed, 422) and "sk-not-a-number" not in echoed.text
    assert _is_problem(client.get("/v1/nowhere", headers=_auth()), 404)


def test_an_upstream_failure_never_shows_its_message(
    components: Components, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_embed(self: FakeEmbedder, texts: list[str]) -> list[list[float]]:
        raise ModelError("POST http://u:topsecret@llm.test/v1 failed: invalid key sk-abc")

    monkeypatch.setattr(FakeEmbedder, "embed", failing_embed)
    r = _client(components).post(
        "/v1/memories:search", json={"query": "x", "scope": {"user_id": "ada"}}, headers=_auth()
    )
    assert _is_problem(r, 502)
    assert "topsecret" not in r.text and "sk-abc" not in r.text and "llm.test" not in r.text


def test_an_unexpected_error_is_a_problem_without_its_message(
    components: Components, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("bug near postgresql://u:topsecret@db/x")

    monkeypatch.setattr(PgStore, "transaction", broken)
    client = _client(components, raise_server_exceptions=False)
    r = client.get("/v1/memories", params={"user_id": "ada"}, headers=_auth())
    assert _is_problem(r, 500) and "topsecret" not in r.text and "bug" not in r.text


def test_health_needs_no_key_and_is_cheap_unless_deep(
    components: Components, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(components)
    embedder = components.embedder
    assert isinstance(embedder, FakeEmbedder)
    before = len(embedder.calls)
    assert client.get("/health").json() == {"status": "ok", "store": "ok"}
    assert len(embedder.calls) == before  # a frequent probe never loads the model
    assert client.get("/health", params={"deep": "true"}).json() == {
        "status": "ok", "store": "ok", "embedding": "ok"
    }  # fmt: skip

    def down(self: FakeEmbedder, texts: list[str]) -> list[list[float]]:
        raise ModelError("connection refused")

    monkeypatch.setattr(FakeEmbedder, "embed", down)
    assert client.get("/health").status_code == 200  # the store is fine
    r = client.get("/health", params={"deep": "true"})
    assert r.status_code == 503 and r.json()["embedding"] == "error" and "refused" not in r.text

    def no_db(self: PgStore) -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(PgStore, "ping", no_db)
    assert client.get("/health").json() == {"status": "error", "store": "error"}


def test_without_keys_only_direct_loopback_clients_are_served(components: Components) -> None:
    local = _client(components, keys=False, client=("127.0.0.1", 50000), base_url=LOCAL)
    assert local.get("/v1/memories", params={"user_id": "ada"}).status_code == 200
    remote = _client(components, keys=False, client=("10.0.0.5", 50000), base_url=LOCAL)
    assert _is_problem(remote.get("/v1/memories", params={"user_id": "ada"}), 403)
    # a local reverse proxy makes every client look local: a forwarded request is refused
    proxied = local.get(
        "/v1/memories", params={"user_id": "ada"}, headers={"X-Forwarded-For": "203.0.113.9"}
    )
    assert _is_problem(proxied, 403)


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "attacker.example:8710"},  # DNS rebinding: a web page's name now points here
        {"Origin": "https://attacker.example"},  # any browser page, even same-host
    ],
)
def test_without_keys_a_browser_page_cannot_reach_the_service(
    components: Components, headers: dict[str, str]
) -> None:
    local = _client(components, keys=False, client=("127.0.0.1", 50000), base_url=LOCAL)
    r = local.get("/v1/memories", params={"user_id": "ada"}, headers=headers)
    assert _is_problem(r, 403)


def test_a_missing_namespace_is_400(components: Components) -> None:
    app = create_app(_settings().model_copy(update={"namespace": None}), components)
    r = TestClient(app).get("/v1/memories", params={"user_id": "ada"}, headers=_auth())
    assert _is_problem(r, 400) and "X-LiteMem-Namespace" in r.json()["detail"]


def test_the_default_app_shares_its_components_across_namespaces(
    migrated: Any, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lite_mem import factory
    from tests.conftest import TEST_SCHEMA

    opened: list[Components] = []
    built: list[str] = []

    class Tracked(Components):
        def engine(self, namespace: str) -> Any:
            built.append(namespace)
            return super().engine(namespace)

    def fake_open(settings: Settings, *_: Any, **__: Any) -> Components:
        found = fake_components(PgStore(pg_url, TEST_SCHEMA))
        tracked = Tracked(found.store, found.embedder, found.extractor, found.resolver)
        opened.append(tracked)
        return tracked

    monkeypatch.setattr(factory, "open_components", fake_open)
    settings = _settings().model_copy(update={"database_url": pg_url, "db_schema": TEST_SCHEMA})
    with TestClient(create_app(settings)) as client:
        for _ in range(2):
            _add(client, "住在北京")
        _list(client, **_auth(STEWARD_TOKEN, "steward/1"))
        assert client.get("/health").json()["status"] == "ok"
    assert len(opened) == 1  # one store, one embedder, one set of LLM clients
    assert built == ["app", "steward/1"]  # one engine per namespace; health needs none
