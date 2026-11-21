"""Opt-in (`-m ollama`): the embedder over lite-model sends Ollama's OpenAI-compatible
/v1/embeddings; before lite-model it called /api/embed. Every stored vector came from the old
path, so the new one must return the same vectors, or each namespace's index would mix two."""

import httpx2
import pytest

from lite_mem.config import OLLAMA_URL, EmbeddingConfig
from lite_mem.embedding import build_embedder

TEXTS = ["住在上海浦东", "Prefers green tea in the morning", "x"]


@pytest.mark.ollama
def test_the_vectors_equal_the_ones_api_embed_returned() -> None:
    e = build_embedder(EmbeddingConfig())
    assert (e.provider, e.model) == ("ollama", "bge-m3:567m")
    new = e.embed(TEXTS)
    reply = httpx2.post(
        f"{OLLAMA_URL}/api/embed", json={"model": "bge-m3:567m", "input": TEXTS}, timeout=120
    )
    reply.raise_for_status()
    old = reply.json()["embeddings"]
    assert len(new) == len(old) == len(TEXTS) and e.dim == len(old[0]) == 1024
    for got, want in zip(new, old, strict=True):
        assert got == pytest.approx(want, abs=1e-6)
