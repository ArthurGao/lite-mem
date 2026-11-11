"""mem0 as a System for the eval (design §8), on the same models as lite-mem (extra `eval`).

Written the way llm-steward's contract-tested mem0 adapter calls 2.2.1 (KB global/mem0):
telemetry off before the import; the pgvector store takes its schema through the connection
string's search_path; search and get_all filter by user_id; `add` only ever adds. mem0 builds its
own prompts and cannot pass `reasoning_effort`, so a reasoning model thinks on every call: compare
its quality with the lite-mem variants, and its latency with lite-mem's `thinking` variant.

Each run is isolated: its own collection (table) in schema `mem0_eval` and its own history
directory, both removed by `close()`. Nothing another run or person stored there is touched."""

import os
import shutil
import tempfile
import uuid
from typing import Any
from urllib.parse import quote

import psycopg
from psycopg import sql

from lite_mem.config import OLLAMA_URL, Settings
from lite_mem.embedding import build_embedder

SCHEMA = "mem0_eval"
LIST_LIMIT = 1000  # far above one eval user's memories; the collection is fresh per run


class Mem0System:
    name = "mem0-2.2.1"

    def __init__(self, settings: Settings, *, schema: str = SCHEMA) -> None:
        if settings.embedding.provider != "ollama":
            raise ValueError("the mem0 comparison runs on an ollama embedding model")
        os.environ["MEM0_TELEMETRY"] = "false"  # before the import (KB global/mem0 §2.1)
        from mem0 import Memory  # the eval extra

        self._url = settings.database_url or ""
        self._dir = tempfile.mkdtemp(prefix="mem0-eval-")
        self._schema, self._collection = schema, f"run_{uuid.uuid4().hex[:12]}"
        with psycopg.connect(self._url, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        llm = settings.llm.target("extract")
        dim = build_embedder(settings.embedding).dim  # probes once when not configured
        options = quote(f"-csearch_path={schema},public")
        joiner = "&" if "?" in self._url else "?"
        config: dict[str, Any] = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm.model,
                    "openai_base_url": llm.base_url or f"{OLLAMA_URL}/v1",
                    "api_key": llm.api_key.get_secret_value() if llm.api_key else "unused",
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": settings.embedding.model,
                    "ollama_base_url": settings.embedding.base_url or OLLAMA_URL,
                    "embedding_dims": dim,
                },
            },
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "connection_string": f"{self._url}{joiner}options={options}",
                    "collection_name": self._collection,
                    "embedding_model_dims": dim,
                    "hnsw": True,
                },
            },
            "history_db_path": os.path.join(self._dir, "history.db"),
        }
        self._memory = Memory.from_config(config)

    def add(self, user_id: str, messages: list[dict[str, str]]) -> None:
        self._memory.add(messages, user_id=user_id, infer=True)

    def current(self, user_id: str) -> list[str]:
        found = self._memory.get_all(filters={"user_id": user_id}, top_k=LIST_LIMIT)
        return [m["memory"] for m in _results(found)]

    def search(self, user_id: str, query: str, k: int) -> list[str]:
        found = self._memory.search(query, top_k=k, filters={"user_id": user_id})
        return [m["memory"] for m in _results(found)]

    def reset(self, user_ids: list[str]) -> None:
        """Nothing to do: the collection is new for this run."""

    def close(self) -> None:
        """Drop this run's collection and history."""
        with psycopg.connect(self._url, autocommit=True) as conn:
            table = sql.Identifier(self._schema, self._collection)
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
        shutil.rmtree(self._dir, ignore_errors=True)


def _results(found: Any) -> list[dict[str, Any]]:
    items = found.get("results", found) if isinstance(found, dict) else found
    return list(items or [])
