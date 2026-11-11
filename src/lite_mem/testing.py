"""Deterministic fakes for tests, here and in projects that use lite-mem (e.g. llm-steward's
contract tests). They never touch the network or a model (AGENTS.md rule 10)."""

import hashlib
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel

from lite_mem.config import Role
from lite_mem.llm import Provenance

# Latin words, digits, and each CJK character on its own.
_TOKEN = re.compile(r"[a-z0-9]+|[㐀-鿿]")

Handler = Callable[[dict[str, Any]], dict[str, Any]]


class FakeEmbedder:
    """Bag-of-hashed-tokens vectors, L2-normalised: texts that share words are close, so search
    tests behave like the real thing without a model. Each CJK character counts as one token."""

    provider = "fake"

    def __init__(self, dim: int = 64, model: str = "fake-embedder") -> None:
        self._dim = dim
        self._model = model
        self.calls: list[list[str]] = []  # every batch embedded, for assertions

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeLLM:
    """Scripted structured answers per role; records every call (role, the JSON payload sent)."""

    provider = "fake"
    model = "fake-llm"

    def __init__(self, **handlers: Handler) -> None:
        self._handlers: dict[str, Handler] = handlers
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def structured[T: BaseModel](
        self, role: Role, system: str, payload: Mapping[str, Any], schema: type[T]
    ) -> tuple[T, Provenance]:
        sent = dict(payload)
        self.calls.append((role, sent))
        if role not in self._handlers:
            raise AssertionError(f"FakeLLM has no answer scripted for role {role!r}")
        answer = self._handlers[role](sent)
        return schema.model_validate(answer), Provenance(role, self.provider, self.model)

    def roles(self) -> list[str]:
        return [role for role, _ in self.calls]


def user_lines(payload: dict[str, Any]) -> dict[str, Any]:
    """An `extract` handler: each user message is one semantic fact, as said."""
    said = [m["content"] for m in payload["MESSAGES"] if m["role"] == "user"]
    return {"facts": [{"content": c, "type": "semantic", "importance": 0.5} for c in said]}


def update_when(marker: str) -> Handler:
    """A `resolve` handler: UPDATE the first existing memory when the new fact contains `marker`,
    otherwise ADD."""

    def decide(payload: dict[str, Any]) -> dict[str, Any]:
        if marker in payload["NEW"] and payload["EXISTING"]:
            return {
                "action": "UPDATE",
                "target_id": payload["EXISTING"][0]["id"],
                "reason": "moved",
            }
        return {"action": "ADD", "target_id": None, "reason": "unrelated"}

    return decide
