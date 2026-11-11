# lite-mem — Agent Guide

lite-mem is a **small memory engine for AI agents, built to learn from and to use on its own**: it turns
conversations into long-term memories about a person (extract facts, resolve them against what is already
known — add, update, delete, duplicate), stores them with their full history, and finds them again (hybrid
vector + BM25 search, weighted by importance and age). Consolidation and forgetting run offline. It is a
library and an optional HTTP service over one `MemoryEngine`, and a memory engine for
[llm-steward](https://github.com/ArthurGao/llm-steward) (sister project of
[lite-rag](https://github.com/ArthurGao/lite-rag)).

This file is the shared guide for every coding agent. `CLAUDE.md` imports it.

Naming: repo `lite-mem`, Python package `lite_mem`, CLI `litemem`, env prefix `LITEMEM_`, HTTP port 8710.

## Sources of truth (highest first)

1. `docs/lite-mem-design.md` — the design. §3 interfaces and the memory's shape, §4 storage and ids, §5 search,
   §6 consolidation and forgetting, §7 HTTP service, §8 evaluation, §10 llm-steward integration, §12 stages.
2. Project KB: `/Users/arthurgao/Docs/Knowledgebase/projects/lite-mem/` (created in S0) — decisions, findings,
   measured eval results, lessons. Library facts shared across projects go to `KB/global/`. Write with the
   `kb-maintain` skill.

If a change must contradict the design, update the design in the same change and say why in the commit.

## Current stage

**S9 — `LiteMemProvider` in llm-steward** (design §10, §12; first settle the S9 entry decisions in §13). S0–S8b done 2026-11-12; eval results in the KB (`projects/lite-mem/reference/eval-results.md`).

## Rules

1. **Every component is a Protocol** (`Extractor`, `Resolver`, `Consolidator`, `Embedder`, `Store`, `LLM`); the engine depends
   on Protocols only. Swapping an implementation is config.
2. **Namespaces are mandatory**: an engine is bound to one namespace; no API reads or writes across namespaces.
3. **Invalidate, never delete**: a changed fact SUPERSEDEs (a new id; the old row gets `valid_to` /
   `superseded_by`); delete, merge and expiry close the row; every change appends a history event and `as_of`
   sees the past. CORRECT (fixing a fact's wording) edits in place, same id. The only hard delete is
   `erase(user_id)`. Every write goes through `MemoryEngine._apply`.
4. **One write, one transaction**: a memory's row, vector, postings and history event commit together.
   An `idempotency_key` makes a retried write a no-op that returns the first result.
5. **The model proposes, code decides**: structured output (Pydantic) for every LLM call; an UPDATE/DELETE
   target outside the candidates is not applied; extracted facts are trimmed, de-duplicated and capped.
6. **Every LLM call names a role** (`extract`, `resolve`, `consolidate`), never a model; config maps roles to
   models. No fallback chain: a failed call raises. Every model-made change records `provider:model`.
7. **Prompts live in `src/lite_mem/prompts/*.j2`**, never inline strings.
8. **An index is bound to its model**: a namespace records its embedding model and dim; a mismatch raises.
9. **Secrets only from environment variables.** Never commit `.env*` except `.env.example`.
10. **Tests never call a real model or a paid API.** Unit and integration tests use fakes. Real Ollama runs
    only under the opt-in marker `-m ollama`. Integration tests use their own PG schema (`LITEMEM_TEST_DATABASE_URL`,
    schema `lite_mem_test`), never the app schema. Never call llm-gateway subscription profiles from tests or
    on your own initiative.
11. **Measure, don't guess**: a change meant to improve memory quality comes with an eval run before and after.
12. **Pinned dependencies** for the engines (psycopg, jieba, SDKs): exact versions.

## Tech stack

Python 3.12, uv, pydantic / pydantic-settings, Typer, PostgreSQL + pgvector (psycopg 3), jieba, OpenAI SDK
(OpenAI-compatible LLMs and Ollama `/v1`), Anthropic SDK (extra), Jinja2, FastAPI (extra), pytest, ruff, mypy.
Ollama on the host (`127.0.0.1:11434`, `bge-m3:567m`, `qwen3:8b`).

## Commands

```
uv run pytest                    # unit + integration, fakes only (needs LITEMEM_TEST_DATABASE_URL)
uv run pytest -m ollama          # opt-in: real Ollama
uv run ruff check && uv run ruff format --check && uv run mypy
uv run litemem migrate           # schema LITEMEM_DB_SCHEMA (default lite_mem) on LITEMEM_DATABASE_URL
uv run litemem add|search|history|erase ...
uv run litemem eval              # design §8
uv run --extra server litemem serve   # 127.0.0.1:8710
```

## Workflow

1. Work stage by stage (design §12). Each stage ends with every check green; every new test seen red once.
2. Definition of done: ruff, ruff format, mypy, tests pass; acceptance items covered by tests; design updated
   if behaviour changed.
3. Conventional Commits (`feat(store): …`). All code-facing text in English.
4. **Stage close-out — after every stage's implementation passes its checks, before its commit and before the
   next stage starts:**
   1. Run `/simplify` on the stage's full diff (reuse, simplification, efficiency, altitude) and apply the fixes.
   2. Independent read-only Codex review of the same diff (command in `CLAUDE.md`).
   3. Fix every finding that holds up — a bug gets a failing regression test first. A finding you reject is
      written down with the reason (design §12 notes or the KB); never silently dropped.
   4. Re-run all checks so they cover the final code, then report the review outcome with the stage.
