"""The embedder over lite-model's `embed()` (design §3, §4). Batching, retries and the reply's
checks are lite-model's and tested there (its `tests/contract/test_embed.py`); these tests pin
what lite-mem relies on: the index identity, the request, the dim, and its own errors."""

import json
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from lite_mem.config import EmbeddingConfig, Settings
from lite_mem.embedding import LiteModelEmbedder, build_embedder
from lite_mem.errors import ConfigError, ModelError


def _vectors(body: dict[str, Any], vector: list[float]) -> dict[str, Any]:
    data = [
        {"object": "embedding", "index": i, "embedding": vector} for i in range(len(body["input"]))
    ]
    return {"object": "list", "data": data, "model": body["model"], "usage": {"prompt_tokens": 3}}


class Endpoint:
    """An OpenAI-compatible /embeddings; `statuses` are answered first, then vectors."""

    def __init__(self, vector: list[float], *statuses: int) -> None:
        self.vector, self.statuses = vector, list(statuses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.statuses:
            return httpx2.Response(self.statuses.pop(0), json={"error": {"message": "busy"}})
        return httpx2.Response(200, json=_vectors(json.loads(request.content), self.vector))

    def embedder(self, config: EmbeddingConfig | None = None) -> LiteModelEmbedder:
        client = httpx2.Client(transport=httpx2.MockTransport(self))
        return LiteModelEmbedder.build(config or EmbeddingConfig(), http_client=client)


def test_the_default_keeps_every_namespace_s_identity() -> None:
    # A namespace records provider / model / dim (rule 8); these are the values every existing
    # namespace was created with, before lite-model served embeddings.
    up = Endpoint([0.1, 0.2, 0.3])
    e = up.embedder()
    assert (e.provider, e.model) == ("ollama", "bge-m3:567m")
    assert e.embed(["a", "b"]) == [[0.1, 0.2, 0.3]] * 2
    assert e.dim == 3 and len(up.requests) == 1  # learnt from the first call, no probe
    [sent] = up.requests
    assert str(sent.url) == "http://127.0.0.1:11434/v1/embeddings"
    assert json.loads(sent.content) == {
        "model": "bge-m3:567m", "input": ["a", "b"], "encoding_format": "float",
    }  # fmt: skip


def test_dim_is_probed_once_when_not_configured_and_checked_when_it_is() -> None:
    up = Endpoint([1.0, 0.0])
    assert up.embedder().dim == 2
    assert json.loads(up.requests[0].content)["input"] == ["dimension probe"]
    with pytest.raises(ModelError, match="expected 3-d vectors, got 2"):
        Endpoint([1.0, 0.0]).embedder(EmbeddingConfig(dim=3)).embed(["x"])


def test_no_texts_send_nothing() -> None:
    up = Endpoint([1.0])
    assert up.embedder().embed([]) == [] and up.requests == []


def test_a_keyed_backend_reads_its_key_from_the_variable_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_TEST_KEY", "k-123")
    config = EmbeddingConfig.model_validate(
        {
            "backend": "vendor",
            "model": "text-embedding-3-small",
            "backends": {
                "vendor": {
                    "type": "openai",
                    "base_url": "http://e.test/v1",
                    "api_key_env": "EMBED_TEST_KEY",
                }
            },
        }
    )
    up = Endpoint([0.5])
    e = up.embedder(config)
    assert (e.provider, e.model) == ("vendor", "text-embedding-3-small")
    e.embed(["x"])
    assert up.requests[0].headers["Authorization"] == "Bearer k-123"
    assert str(up.requests[0].url) == "http://e.test/v1/embeddings"


def test_a_missing_key_is_refused_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        build_embedder(EmbeddingConfig(backend="openai", model="text-embedding-3-small"))


def test_a_busy_server_is_retried_on_the_same_model_and_a_client_error_is_not() -> None:
    busy = Endpoint([1.0], 503)
    assert busy.embedder(EmbeddingConfig(max_retries=1)).embed(["x"]) == [[1.0]]
    assert [json.loads(r.content)["model"] for r in busy.requests] == ["bge-m3:567m"] * 2
    bad = Endpoint([1.0], 400)
    with pytest.raises(ModelError, match=r"role embed \(ollama:bge-m3:567m\)"):
        bad.embedder(EmbeddingConfig(max_retries=1)).embed(["x"])
    assert len(bad.requests) == 1


def test_a_failed_call_does_not_leak_upstream_text() -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("cannot reach http://user:hunter2@10.0.0.5:11434/v1")

    client = httpx2.Client(transport=httpx2.MockTransport(down))
    e = LiteModelEmbedder.build(EmbeddingConfig(max_retries=0), http_client=client)
    with pytest.raises(ModelError) as caught:
        e.embed(["x"])
    assert "hunter2" not in str(caught.value)


@pytest.mark.parametrize(
    ("key", "points_to"),
    [
        ("PROVIDER", "LITEMEM_EMBEDDING__BACKEND"),
        ("BASE_URL", "__BASE_URL"),
        ("API_KEY", "API_KEY_ENV"),
    ],
)
def test_the_pre_lite_model_keys_are_refused_with_where_they_went(
    monkeypatch: pytest.MonkeyPatch, key: str, points_to: str
) -> None:
    # Left alone they would be ignored, and a namespace would silently embed with the default.
    monkeypatch.setenv(f"LITEMEM_EMBEDDING__{key}", "x")
    with pytest.raises(ValidationError, match="embedding settings moved to lite-model's") as caught:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert points_to in str(caught.value)
