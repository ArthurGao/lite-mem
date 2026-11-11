import json
import math

import httpx2
import pytest

from lite_mem.config import EmbeddingConfig
from lite_mem.embedding import OllamaEmbedder, OpenAIEmbedder, build_embedder
from lite_mem.errors import ConfigError, ModelError
from lite_mem.fusion import rrf
from lite_mem.keyword import idf, term_score, terms


def test_terms_cut_chinese_and_keep_latin_words() -> None:
    got = terms("我住在上海浦东，喜欢 Green Tea!")
    assert {"上海", "浦东", "喜欢", "green", "tea"} <= set(got)
    assert not any(t in got for t in ("，", "!", " "))


def test_bm25_parts_match_the_formula() -> None:
    assert idf(10, 1) == pytest.approx(math.log(1 + 9.5 / 1.5))
    assert idf(10, 10) > 0  # the +1 keeps it positive
    assert term_score(1, 4, 4.0) == pytest.approx(1.0)  # an average-length doc, tf 1
    assert term_score(2, 4, 4.0) > term_score(1, 4, 4.0)
    assert term_score(1, 8, 4.0) < term_score(1, 2, 4.0)  # longer docs score lower


def test_rrf_rewards_agreement_and_uses_ranks_only() -> None:
    fused = rrf([["a", "b", "c"], ["b", "a"]], k=60)
    assert fused["a"] == fused["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63)


def _client(handler: object) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))  # type: ignore[arg-type]


def test_ollama_embedder_posts_the_batch_and_learns_its_dim() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/api/embed"
        return httpx2.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * len(body["input"])})

    e = OllamaEmbedder(EmbeddingConfig(model="bge-m3:567m"), _client(handler))
    assert e.embed(["a", "b"]) == [[0.1, 0.2, 0.3]] * 2
    assert e.dim == 3 and seen[0] == {"model": "bge-m3:567m", "input": ["a", "b"]}


def test_openai_embedder_sends_its_key_and_orders_by_index() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["Authorization"] == "Bearer k"
        data = [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]
        return httpx2.Response(200, json={"data": data})

    config = EmbeddingConfig(provider="openai", model="m", api_key="k")  # type: ignore[arg-type]
    assert OpenAIEmbedder(config, _client(handler)).embed(["x", "y"]) == [[1.0], [2.0]]


def test_a_failed_or_short_embedding_call_raises() -> None:
    down = OllamaEmbedder(EmbeddingConfig(), _client(lambda r: httpx2.Response(500)))
    with pytest.raises(ModelError, match="failed"):
        down.embed(["x"])
    short = OllamaEmbedder(
        EmbeddingConfig(), _client(lambda r: httpx2.Response(200, json={"embeddings": [[1.0]]}))
    )
    with pytest.raises(ModelError, match="asked for 2"):
        short.embed(["x", "y"])


def test_openai_without_a_key_or_endpoint_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="API_KEY"):
        build_embedder(EmbeddingConfig(provider="openai", model="m"))
