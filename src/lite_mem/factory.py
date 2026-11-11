"""Assembling an engine from Settings (design §9): the store, the embedder, and the LLM-backed
extractor and resolver. The library can also build a MemoryEngine by hand."""

from lite_mem.config import Settings
from lite_mem.embedding import build_embedder
from lite_mem.engine import MemoryEngine
from lite_mem.errors import ConfigError
from lite_mem.extract import LlmExtractor
from lite_mem.llm import RoutedLLM
from lite_mem.resolve import AddOnlyResolver, LlmResolver, Resolver
from lite_mem.stores.base import Store
from lite_mem.stores.pg import PgStore


def open_store(settings: Settings, *, max_size: int = 4) -> PgStore:
    return PgStore(settings.database_url, settings.db_schema, max_size=max_size)


def open_engine(
    settings: Settings, store: Store | None = None, *, max_size: int = 4
) -> MemoryEngine:
    if settings.namespace is None:
        raise ConfigError("no namespace configured: set LITEMEM_NAMESPACE")
    llm, pipe = RoutedLLM(settings.llm), settings.pipeline
    resolver: Resolver = (
        LlmResolver(llm, pipe.resolve_prompt) if pipe.resolver == "llm" else AddOnlyResolver()
    )
    return MemoryEngine(
        store or open_store(settings, max_size=max_size),
        settings.namespace,
        build_embedder(settings.embedding),
        extractor=LlmExtractor(llm, pipe.extract_prompt),
        resolver=resolver,
        related=pipe.related,
        search=settings.search,
        sensitive=pipe.sensitive,
    )
