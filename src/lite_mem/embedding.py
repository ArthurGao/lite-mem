"""Embedders (design §3): one model per namespace. The calls are lite-model's `embed()` (its design
§3.2): it batches, retries the *same* model on 429 / 5xx / transport errors, checks the vectors'
count, order and length, and reads which model answered. No fallback: a failed call raises
ModelError. This module keeps lite-mem's own `Embedder` Protocol (rule 1) and the adapter."""

from typing import TYPE_CHECKING, Protocol

import lite_model
from lite_model import RolePolicy, RoleSpec
from lite_model.config import resolve

from lite_mem.config import EmbeddingConfig
from lite_mem.errors import ConfigError, ModelError
from lite_mem.llm import build_model

if TYPE_CHECKING:
    import httpx2

ROLE = "embed"
# Never routable (lite-model refuses it anyway): an index is bound to one model (rule 8).
ROLES = RoleSpec({ROLE: RolePolicy(kind="embed")})


class Embedder(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LiteModelEmbedder:
    """lite-mem's `Embedder` over a lite-model embed role. `provider` is the lite-model backend's
    name and `model` the configured one: together they are the index's identity (design §4),
    fixed by the config before any call, like the model that writes a memory."""

    def __init__(
        self, model: lite_model.EmbedModel, target: lite_model.Target, dim: int | None = None
    ) -> None:
        self._model, self.target, self._dim = model, target, dim

    @classmethod
    def build(
        cls, config: EmbeddingConfig, http_client: "httpx2.Client | None" = None
    ) -> "LiteModelEmbedder":
        """The whole config is checked before any call (an unknown backend, a missing key, a
        backend that serves no embeddings) and refused as lite-mem's ConfigError."""
        model = build_model(config, http_client, ROLES)
        return cls(model, model.target(ROLE), config.dim)

    @property
    def provider(self) -> str:
        return self.target.backend

    @property
    def model(self) -> str:
        return self.target.model

    @property
    def dim(self) -> int:
        if self._dim is None:  # learnt from the model once
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            reply = self._model.embed(ROLE, texts)
        except lite_model.ModelError as exc:
            raise ModelError(str(exc)) from exc
        if self._dim is None:  # the first reply says it: no probe call needed after one
            self._dim = reply.dim
        elif reply.dim != self._dim:
            raise ModelError(f"{self.model}: expected {self._dim}-d vectors, got {reply.dim}")
        return reply.vectors


def embed_target(config: EmbeddingConfig) -> lite_model.Target:
    """Where the embed role's calls go (backend, model, endpoint), checked as `build()` checks
    it, without building a client."""
    try:
        return resolve(config, ROLES)[ROLE]
    except lite_model.ConfigError as exc:
        raise ConfigError(str(exc)) from exc


def build_embedder(config: EmbeddingConfig) -> Embedder:
    return LiteModelEmbedder.build(config)
