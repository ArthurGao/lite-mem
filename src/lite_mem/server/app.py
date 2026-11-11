"""FastAPI app over `MemoryEngine` (design §7). The store, embedding model and LLM clients are
built once (`factory.Components`); each namespace gets a light engine on them, made on first use.

Every route but `/health` needs an API key that reaches the request's namespace (header
`X-LiteMem-Namespace`, default `LITEMEM_NAMESPACE`). With no key configured, only loopback clients
are served: never a proxied request (a local proxy makes every client look local) nor a browser's
(an Origin header, or a Host that is not a loopback name: DNS rebinding).
Errors are Problem Details (RFC 9457); an upstream failure's message is logged (redacted), never
returned."""

import logging
import threading
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from lite_mem import __version__, factory
from lite_mem.config import Settings
from lite_mem.engine import MemoryEngine
from lite_mem.errors import LiteMemError, redact
from lite_mem.factory import Components
from lite_mem.server import auth
from lite_mem.server.schemas import (
    AddIn,
    ErasedOut,
    EventsOut,
    HistoryOut,
    HitsOut,
    MemoriesOut,
    SearchIn,
    UpdateIn,
)
from lite_mem.types import Memory, MemoryType, Message, Scope

log = logging.getLogger("lite_mem.server")
PROBLEM = "application/problem+json"


def problem(
    status: int, detail: str, headers: dict[str, str] | None = None, **extra: Any
) -> JSONResponse:
    body = {"type": "about:blank", "title": HTTPStatus(status).phrase, "status": status,
            "detail": detail, **extra}  # fmt: skip
    return JSONResponse(body, status_code=status, headers=headers, media_type=PROBLEM)


class _Engines:
    """One engine per namespace on shared components. Building one binds the namespace (a DB
    transaction, maybe a model probe), so it holds only that namespace's lock: other namespaces
    are served meanwhile, and a built engine is read without locking."""

    def __init__(self, components: Components) -> None:
        self._components = components
        self._built: dict[str, MemoryEngine] = {}
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._guard = threading.Lock()  # for _locks only

    def __call__(self, namespace: str) -> MemoryEngine:
        found = self._built.get(namespace)
        if found is not None:
            return found
        with self._guard:
            lock = self._locks[namespace]
        with lock:
            if namespace not in self._built:
                self._built[namespace] = self._components.engine(namespace)
            return self._built[namespace]


def create_app(settings: Settings, components: Components | None = None) -> FastAPI:
    """The app. `components` are opened from `settings` at startup unless given (tests); given
    components are not closed by the app."""
    held: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if components is not None:
            yield
            return
        opened = factory.open_components(settings)
        held["components"], held["engines"] = opened, _Engines(opened)
        try:
            yield
        finally:
            opened.store.close()

    if components is not None:
        held["components"], held["engines"] = components, _Engines(components)
    app = FastAPI(title="lite-mem", version=__version__, lifespan=lifespan)
    keys = settings.server.api_keys

    def engine(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        x_litemem_namespace: Annotated[str | None, Header()] = None,
    ) -> MemoryEngine:
        key = None
        if keys:
            key = auth.authenticate(authorization, keys)
            if key is None:
                raise HTTPException(
                    401, "a valid API key is required", {"WWW-Authenticate": "Bearer"}
                )
        elif not auth.local_request(
            request.client.host if request.client else None, request.headers
        ):
            raise HTTPException(403, "no API keys are configured: only direct loopback clients")
        namespace = x_litemem_namespace or settings.namespace
        if namespace is None:
            raise HTTPException(400, "no namespace: send X-LiteMem-Namespace")
        if key is not None and not auth.allows(key, namespace):
            raise HTTPException(403, "this key does not reach that namespace")
        engines: _Engines = held["engines"]
        return engines(namespace)

    Engine = Annotated[MemoryEngine, Depends(engine)]

    @app.exception_handler(LiteMemError)
    async def _engine_error(_: Request, exc: LiteMemError) -> JSONResponse:
        if exc.expose:
            return problem(exc.status, str(exc))
        log.error("%s: %s", type(exc).__name__, redact(str(exc)))
        return problem(exc.status, HTTPStatus(exc.status).description)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        # loc / msg / type only: the input itself is not echoed back
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return problem(422, "the request does not match the schema", errors=errors)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem(exc.status_code, str(exc.detail), dict(exc.headers or {}))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.error("unexpected %s: %s", type(exc).__name__, redact(str(exc)))
        return problem(500, HTTPStatus.INTERNAL_SERVER_ERROR.description)

    @app.get("/health")
    def health(deep: bool = False) -> JSONResponse:
        """The store (cheap: for frequent probes); `deep=true` also embeds once, so a down model
        shows (design §10: what llm-steward's provider health asks)."""
        found: Components = held["components"]
        parts = found.health(deep=deep)
        ok = all(v == "ok" for v in parts.values())
        return JSONResponse({"status": "ok" if ok else "error", **parts}, 200 if ok else 503)

    @app.post("/v1/memories:add", response_model=EventsOut)
    def add(
        body: AddIn,
        engine: Engine,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> EventsOut:
        messages = [Message(m.role, m.content) for m in body.messages]
        events = engine.add(messages, body.scope.to_scope(), infer=body.infer,
                            metadata=body.metadata, idempotency_key=idempotency_key)  # fmt: skip
        return EventsOut(events=events)

    @app.post("/v1/memories:search", response_model=HitsOut)
    def search(body: SearchIn, engine: Engine) -> HitsOut:
        hits = engine.search(
            body.query, body.scope.to_scope(), top_k=body.top_k,
            include_shared=body.include_shared, types=body.types, metadata=body.metadata,
            as_of=body.as_of,
        )  # fmt: skip
        return HitsOut(hits=hits)

    @app.get("/v1/memories", response_model=MemoriesOut)
    def list_memories(
        engine: Engine,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        include_shared: bool = False,
        as_of: datetime | None = None,
        types: Annotated[list[MemoryType] | None, Query()] = None,
    ) -> MemoriesOut:
        scope = Scope(user_id, agent_id, session_id)
        found = engine.list_memories(scope, include_shared=include_shared, as_of=as_of, types=types)
        return MemoriesOut(memories=found)

    @app.get("/v1/memories/{memory_id}", response_model=Memory)
    def get(memory_id: uuid.UUID, engine: Engine) -> Memory:
        return engine.get(memory_id)

    @app.patch("/v1/memories/{memory_id}", response_model=Memory)
    def update(memory_id: uuid.UUID, body: UpdateIn, engine: Engine) -> Memory:
        return engine.update(memory_id, body.content, reason=body.reason)

    @app.delete("/v1/memories/{memory_id}", response_model=Memory)
    def delete(memory_id: uuid.UUID, engine: Engine, reason: str | None = None) -> Memory:
        return engine.delete(memory_id, reason=reason)

    @app.get("/v1/memories/{memory_id}/history", response_model=HistoryOut)
    def history(memory_id: uuid.UUID, engine: Engine) -> HistoryOut:
        return HistoryOut(entries=engine.history(memory_id))

    @app.delete("/v1/users/{user_id}", response_model=ErasedOut)
    def erase(user_id: str, engine: Engine) -> ErasedOut:
        return ErasedOut(erased=engine.erase(user_id))

    return app
