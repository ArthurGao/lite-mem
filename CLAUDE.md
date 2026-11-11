@AGENTS.md

# Claude Code notes

- At the start of a session: check §12 of `docs/lite-mem-design.md` for the current stage and its acceptance,
  and skim the project KB at `/Users/arthurgao/Docs/Knowledgebase/projects/lite-mem/`.
- Never answer a psycopg / pgvector / jieba / FastAPI / OpenAI or Anthropic SDK / mem0 API question from
  memory: check context7 or the pinned version's docs, and record surprises in the KB.
- Relevant user-level skills: `llm-app-architecture` (roles, structured output, no fallback chain, provenance),
  `rag-design` (hybrid search, rerank, evaluation), `llm-model-eval` (comparing models), `testing-and-ci`,
  `postgres-rls-migrations` (migrations), `kb-maintain`.
- Ollama runs on the host; check it for free with `curl -s http://127.0.0.1:11434/api/tags` before `-m ollama`.
- **Stage close-out, for every stage, in this order** (`AGENTS.md` Workflow step 4):
  1. `/simplify` on the stage's full diff; apply its fixes.
  2. Codex, from the repo root, read-only, on the stage's whole diff:
     `codex exec -s read-only -o <out.md> "Review \`git diff <commit before the stage>\` (includes uncommitted
     changes) … focus … report severity, file:line, scenario, fix" < /dev/null`.
     The `< /dev/null` matters: with a piped stdin, `codex exec` waits for stdin to close and hangs in a
     background shell.
  3. Fix every finding that holds up (a bug gets a failing regression test first); write down the ones you
     reject, with the reason.
  4. Re-run every check (read summary lines, e.g. `rg 'passed|failed'`, never a bare `| tail -1`), then report
     and commit. Do not start the next stage before this is done. After each push, read the CI result
     (`gh run list --limit 3 --json headSha,conclusion`).
