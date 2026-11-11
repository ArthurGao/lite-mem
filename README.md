# lite-mem

A small memory engine for AI agents, built to learn from and to use on its own. It turns conversations into
long-term memories about a person — extracting facts and resolving them against what is already known
(add, update, delete, duplicate) — keeps every change in a history you can query back in time, and finds the
right memories again with hybrid vector + BM25 search weighted by importance and age.

- Library and optional HTTP service over one `MemoryEngine`.
- PostgreSQL + pgvector, or one SQLite file; local models through Ollama by default (no paid API
  needed).
- Switchable LLM per role (Ollama, any OpenAI-compatible endpoint, Anthropic); no silent fallbacks.
- Built-in evaluation, including a side-by-side comparison with mem0.
- A memory engine for [llm-steward](https://github.com/ArthurGao/llm-steward); sister of
  [lite-rag](https://github.com/ArthurGao/lite-rag).

Status: in development (stages S0–S6 done). The design is in [docs/lite-mem-design.md](docs/lite-mem-design.md);
stages, acceptance and what each stage's review found are in its §12.

## Quick start

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), PostgreSQL with pgvector, and [Ollama](https://ollama.com)
with `bge-m3:567m` (embeddings) and `qwen3:8b` (extraction and conflict resolution).

```sh
cp .env.example .env              # set LITEMEM_DATABASE_URL
uv sync --all-extras
uv run litemem migrate
uv run litemem add --user emma "I live in Wellington and work as a nurse"
uv run litemem search --user emma "where does Emma live"
uv run litemem eval -v resolve -v add_only --mem0   # the synthetic comparison (about an hour on a laptop)
```

Secrets (passwords, PINs, card and ID numbers, API keys) are never stored as memories: they are dropped by
default (`LITEMEM_PIPELINE__SENSITIVE=drop|mask|keep`).

### Without a database server

The same engine runs on one SQLite file (extra `sqlite`; the same tests pass on both stores):

```sh
LITEMEM_DATABASE_URL=sqlite:///./memory.db uv run litemem add --user emma "I live in Wellington"
```

Writes are serialised by SQLite: fine on one machine; for many concurrent writers use PostgreSQL.

## HTTP service

```sh
uv run litemem key new app -n app     # prints a token once, and the entry for LITEMEM_SERVER__API_KEYS
uv run litemem serve                  # http://127.0.0.1:8710
curl -H "Authorization: Bearer lm_app_…" -H 'Content-Type: application/json' \
  http://127.0.0.1:8710/v1/memories:add \
  -d '{"messages": [{"role": "user", "content": "I moved to Auckland"}], "scope": {"user_id": "emma"}}'
```

Routes: `POST /v1/memories:add` (header `Idempotency-Key` makes a retry replay the first result),
`POST /v1/memories:search`, `GET /v1/memories?user_id=…`, `GET|PATCH|DELETE /v1/memories/{id}`,
`GET /v1/memories/{id}/history`, `DELETE /v1/users/{user_id}` (erase everything about a user), `GET /health`.
A key reaches the namespaces its globs match (`X-LiteMem-Namespace`, default `LITEMEM_NAMESPACE`); without
any key the service serves loopback clients only. Errors are Problem Details (`application/problem+json`).

With Docker (lite-mem + its own PostgreSQL; Ollama stays on the host):

```sh
# in .env: POSTGRES_PASSWORD and LITEMEM_SERVER__API_KEYS (see .env.example)
docker compose -f deploy/compose.yaml --env-file .env up -d --build --wait
```

## License

[Apache-2.0](LICENSE)
