"""Configuration (design §9): environment variables with the `LITEMEM_` prefix and a `.env` file.
Nested keys use `__`, e.g. `LITEMEM_LLM__MODEL=qwen3:8b`.

Precedence, highest first: explicit arguments, environment, `.env`, defaults."""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

from lite_mem.sensitive import Policy

OLLAMA_URL = "http://127.0.0.1:11434"  # the local Ollama server (embeddings: /api, LLM: /v1)

# A namespace: lowercase segments separated by "/", e.g. "app" or "steward/<uuid>".
Namespace = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,62}(/[a-z0-9][a-z0-9_.-]{0,62}){0,3}$"),
]


def own_schema(name: str) -> str:
    """lite-mem keeps its tables in a schema of its own: never a shared or system one."""
    if name in ("public", "information_schema") or name.startswith("pg_"):
        raise ValueError(f"schema {name!r} is shared or a system schema: use one of lite-mem's own")
    return name


# A schema lite-mem owns: a lowercase SQL identifier, not public or a system schema.
OwnSchema = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z_][a-z0-9_]{0,62}$"),
    AfterValidator(own_schema),
]

Role = Literal["extract", "resolve", "consolidate"]
LLMProvider = Literal["openai", "anthropic"]
# Sent as `reasoning_effort` to OpenAI-compatible endpoints. "none" turns qwen3's thinking off on
# Ollama /v1: 20 s -> 1 s for one short extraction on this machine (measured 2026-11-12).
# "default" (in a role's override) clears an inherited setting: send nothing, the endpoint decides.
Reasoning = Literal["none", "low", "medium", "high", "default"]


class EmbeddingConfig(BaseModel):
    """One embedding model per namespace (design §4: an index is bound to its model)."""

    provider: Literal["ollama", "openai"] = "ollama"
    model: str = "bge-m3:567m"
    # The API root; None = the provider's default (Ollama on the host; OpenAI's API).
    base_url: str | None = None
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=60.0, gt=0)
    # The model's vector size, when known: skips the one probe call that learns it.
    dim: int | None = Field(default=None, gt=0)


class PipelineConfig(BaseModel):
    """How add(infer=true) runs (design §2), switchable for the eval (design §8)."""

    resolver: Literal["llm", "add_only"] = "llm"  # add_only: mem0 2.2.1's behaviour
    related: int = Field(default=5, ge=1, le=20)  # existing memories each fact is compared with
    extract_prompt: str = "extract_facts.j2"  # a file in lite_mem/prompts
    resolve_prompt: str = "resolve_conflict.j2"
    # Secrets (passwords, card and ID numbers, API keys; lite_mem.sensitive): drop (default),
    # mask the value, or keep. Applies to every write, not only add(infer=true).
    sensitive: Policy = "drop"


class RoleOverride(BaseModel):
    """What one role uses instead of the defaults (any field left out keeps the default)."""

    provider: LLMProvider | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    reasoning: Reasoning | None = None


@dataclass(frozen=True)
class ModelTarget:
    """Where one role's calls go: resolved from the defaults and the role's override."""

    provider: LLMProvider
    base_url: str | None
    api_key: SecretStr | None
    model: str
    timeout_s: float
    reasoning: Reasoning | None = None  # None: the endpoint's default


def _reasoning(value: Reasoning | None) -> Reasoning | None:
    return None if value == "default" else value


class LLMConfig(BaseModel):
    """Every LLM call names a role; this maps each role to a provider, endpoint and model
    (design §3). The top-level fields are the defaults; `roles` overrides them per role."""

    # openai: any OpenAI-compatible endpoint (Ollama /v1, llm-gateway, a vendor); anthropic: extra.
    provider: LLMProvider = "openai"
    base_url: str | None = f"{OLLAMA_URL}/v1"
    api_key: SecretStr | None = None
    model: str = "qwen3:8b"
    roles: dict[Role, RoleOverride] = Field(default_factory=dict)
    timeout_s: float = Field(default=120.0, gt=0)
    reasoning: Reasoning | None = None

    def target(self, role: Role) -> ModelTarget:
        o = self.roles.get(role, RoleOverride())
        provider = o.provider or self.provider
        switched = provider != self.provider  # another provider: never reuse this one's endpoint
        return ModelTarget(
            provider=provider,
            base_url=o.base_url if (o.base_url or switched) else self.base_url,
            api_key=o.api_key if (o.api_key or switched) else self.api_key,
            model=o.model or self.model,
            timeout_s=self.timeout_s,
            reasoning=_reasoning(o.reasoning if o.reasoning is not None else self.reasoning),
        )


class SearchConfig(BaseModel):
    """Hybrid search and its weights (design §5). Scores are only comparable within lite-mem."""

    candidates: int = Field(default=50, ge=1, le=1000)  # per list, before fusion
    rrf_k: int = Field(default=60, ge=1)
    importance_boost: float = Field(default=0.3, ge=0)  # weight * (1 + boost * importance)
    # Half-life in days per memory type; a type not listed does not decay.
    half_lives: dict[str, Annotated[float, Field(gt=0)]] = Field(
        default_factory=lambda: {"episodic": 30.0}
    )
    low_confidence: float = Field(default=0.5, ge=0, le=1)
    low_confidence_factor: float = Field(default=0.8, ge=0, le=1)


class ApiKey(BaseModel):
    """One API key of the HTTP service (design §7). Only sha256(secret) is configured; the token
    `lm_<id>_<secret>` is shown once by `litemem key new`. `namespaces` are globs
    (`steward/*`): the key reaches those namespaces and no other."""

    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]{1,32}$")]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    namespaces: list[str] = Field(min_length=1)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8710, gt=0, lt=65536)
    # LITEMEM_SERVER__API_KEYS='[{"id": …, "sha256": …, "namespaces": [...]}]'. None configured:
    # the service only serves loopback clients and `litemem serve` refuses any other bind address.
    api_keys: list[ApiKey] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LITEMEM_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    database_url: str | None = None
    # LITEMEM_DB_SCHEMA. No alias: pydantic-settings reads an aliased field from the bare alias
    # (`SCHEMA`), not from the prefixed name, so a test's LITEMEM_SCHEMA once missed it.
    db_schema: OwnSchema = "lite_mem"
    namespace: Namespace | None = None  # required by the engine (design §4)
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    search: SearchConfig = SearchConfig()
    pipeline: PipelineConfig = PipelineConfig()
    server: ServerConfig = ServerConfig()
