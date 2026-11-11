"""Assembling an engine from Settings (design §9): the store, the embedder, and the LLM-backed
extractor and resolver. The library can also build a MemoryEngine by hand."""

import threading
from collections import defaultdict
from dataclasses import dataclass, field

from lite_mem.config import MaintainConfig, SearchConfig, Settings
from lite_mem.consolidate import Consolidator, LlmConsolidator
from lite_mem.embedding import Embedder, build_embedder
from lite_mem.engine import MemoryEngine, health, valid_namespace
from lite_mem.errors import ConfigError, Refused
from lite_mem.extract import Extractor, LlmExtractor
from lite_mem.llm import RoutedLLM
from lite_mem.resolve import AddOnlyResolver, LlmResolver, Resolver
from lite_mem.sensitive import Policy
from lite_mem.stores.base import Store
from lite_mem.stores.pg import PgStore

SQLITE = "sqlite:///"


def is_sqlite(url: str | None) -> bool:
    return (url or "").startswith("sqlite:")


def open_store(settings: Settings, *, max_size: int = 4) -> Store:
    """`sqlite:///<file>`: the embedded store (extra `sqlite`); anything else: PostgreSQL."""
    url = settings.database_url or ""
    if is_sqlite(url):
        if not url.startswith(SQLITE) or len(url) == len(SQLITE):
            raise ConfigError("a SQLite database is named sqlite:///<path to the file>")
        from lite_mem.stores.sqlite import SqliteStore

        return SqliteStore(url[len(SQLITE) :])
    return PgStore(settings.database_url, settings.db_schema, max_size=max_size)


@dataclass(frozen=True)
class Components:
    """What every namespace's engine shares: the store, the embedding model, the extractor and
    resolver (their LLM clients). Built once; an engine per namespace is then cheap (the HTTP
    service keeps one per namespace on the same pool and clients)."""

    store: Store
    embedder: Embedder
    extractor: Extractor
    resolver: Resolver
    related: int = 5
    search: SearchConfig = field(default_factory=SearchConfig)
    sensitive: Policy = "drop"
    consolidator: Consolidator | None = None
    maintenance: MaintainConfig = field(default_factory=MaintainConfig)

    def engine(self, namespace: str) -> MemoryEngine:
        return MemoryEngine(
            self.store, namespace, self.embedder, extractor=self.extractor,
            resolver=self.resolver, related=self.related, search=self.search,
            sensitive=self.sensitive, consolidator=self.consolidator,
            maintenance=self.maintenance,
        )  # fmt: skip

    def health(self, *, deep: bool) -> dict[str, str]:
        return health(self.store, self.embedder, deep=deep)


class Engines:
    """One engine per namespace on shared components, made on first use (S9's in-process
    provider and the HTTP service both keep one). Building one binds the namespace (a DB
    transaction, maybe a model probe), so it holds only that namespace's lock: other namespaces
    are served meanwhile, and a built engine is read without locking. `max_namespaces` bounds how
    many one process opens (a broad key glob could otherwise bind names without end)."""

    def __init__(self, components: Components, *, max_namespaces: int | None = None) -> None:
        self.components = components
        self._max = max_namespaces
        self._built: dict[str, MemoryEngine] = {}
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._guard = threading.Lock()  # for _locks and the count

    def __call__(self, namespace: str) -> MemoryEngine:
        found = self._built.get(namespace)
        if found is not None:
            return found
        valid_namespace(namespace)  # before any lock: an invalid name leaves no trace
        with self._guard:  # only engines that opened count: a failed open takes no slot
            if self._max is not None and len(self._built) >= self._max:
                raise Refused(f"this service opens at most {self._max} namespaces")
            lock = self._locks[namespace]  # never removed: a waiter always shares it
        with lock:
            if namespace not in self._built:
                self._built[namespace] = self.components.engine(namespace)
            return self._built[namespace]


def open_components(
    settings: Settings, store: Store | None = None, *, max_size: int = 4
) -> Components:
    llm, pipe = RoutedLLM(settings.llm), settings.pipeline
    resolver: Resolver = (
        LlmResolver(llm, pipe.resolve_prompt) if pipe.resolver == "llm" else AddOnlyResolver()
    )
    return Components(
        store or open_store(settings, max_size=max_size),
        build_embedder(settings.embedding),
        LlmExtractor(llm, pipe.extract_prompt),
        resolver,
        related=pipe.related,
        search=settings.search,
        sensitive=pipe.sensitive,
        consolidator=LlmConsolidator(
            llm, settings.maintain.merge_prompt, settings.maintain.summarise_prompt
        ),
        maintenance=settings.maintain,
    )


def open_engine(
    settings: Settings, store: Store | None = None, *, max_size: int = 4
) -> MemoryEngine:
    if settings.namespace is None:
        raise ConfigError("no namespace configured: set LITEMEM_NAMESPACE")
    return open_components(settings, store, max_size=max_size).engine(settings.namespace)
