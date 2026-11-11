"""Embedders (design §3): one model per namespace. No fallback: a failed call raises ModelError."""

from typing import Any, Protocol

import httpx2

from lite_mem.config import OLLAMA_URL, EmbeddingConfig
from lite_mem.errors import ConfigError, ModelError

BATCH = 64


class Embedder(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _HTTPEmbedder:
    provider: str
    default_url: str

    def __init__(self, config: EmbeddingConfig, client: httpx2.Client | None = None) -> None:
        self._config = config
        self._base_url = (config.base_url or self.default_url).rstrip("/")
        self._client = client or httpx2.Client(timeout=config.timeout_s)
        self._dim: int | None = None

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def dim(self) -> int:
        if self._dim is None:  # learnt from the model once
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            try:
                got = self._call(batch)
            except (httpx2.HTTPError, KeyError, TypeError, ValueError) as exc:
                raise ModelError(f"embedding ({self.provider} {self.model}) failed: {exc}") from exc
            if len(got) != len(batch):
                raise ModelError(f"{self.model}: asked for {len(batch)} vectors, got {len(got)}")
            vectors.extend(got)
        if vectors and self._dim is not None and len(vectors[0]) != self._dim:
            raise ModelError(f"{self.model}: expected {self._dim}-d vectors, got {len(vectors[0])}")
        return vectors

    def _post(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> Any:
        response = self._client.post(f"{self._base_url}{path}", json=body, headers=headers)
        response.raise_for_status()
        return response.json()

    def _call(self, batch: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OllamaEmbedder(_HTTPEmbedder):
    """Ollama's `/api/embed` (local, free)."""

    provider = "ollama"
    default_url = OLLAMA_URL

    def _call(self, batch: list[str]) -> list[list[float]]:
        reply = self._post("/api/embed", {"model": self.model, "input": batch}, {})
        embeddings: list[list[float]] = reply["embeddings"]
        return embeddings


class OpenAIEmbedder(_HTTPEmbedder):
    """`POST {base_url}/embeddings`: OpenAI and the services that copy its API."""

    provider = "openai"
    default_url = "https://api.openai.com/v1"

    def __init__(self, config: EmbeddingConfig, client: httpx2.Client | None = None) -> None:
        if config.api_key is None and config.base_url is None:
            raise ConfigError("embedding provider openai needs LITEMEM_EMBEDDING__API_KEY")
        super().__init__(config, client)

    def _call(self, batch: list[str]) -> list[list[float]]:
        key = self._config.api_key
        headers = {"Authorization": f"Bearer {key.get_secret_value()}"} if key else {}
        reply = self._post("/embeddings", {"model": self.model, "input": batch}, headers)
        return [d["embedding"] for d in sorted(reply["data"], key=lambda d: d["index"])]


def build_embedder(config: EmbeddingConfig) -> Embedder:
    if config.provider == "ollama":
        return OllamaEmbedder(config)
    return OpenAIEmbedder(config)
