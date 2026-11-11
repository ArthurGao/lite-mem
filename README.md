# lite-mem

A small memory engine for AI agents, built to learn from and to use on its own. It turns conversations into
long-term memories about a person — extracting facts and resolving them against what is already known
(add, update, delete, duplicate) — keeps every change in a history you can query back in time, and finds the
right memories again with hybrid vector + BM25 search weighted by importance and age.

- Library and optional HTTP service over one `MemoryEngine`.
- PostgreSQL + pgvector; local models through Ollama by default (no paid API needed).
- Switchable LLM per role (Ollama, any OpenAI-compatible endpoint, Anthropic); no silent fallbacks.
- Built-in evaluation, including a side-by-side comparison with mem0.
- A memory engine for [llm-steward](https://github.com/ArthurGao/llm-steward); sister of
  [lite-rag](https://github.com/ArthurGao/lite-rag).

Status: design stage. The design is in [docs/lite-mem-design.md](docs/lite-mem-design.md); stages and acceptance
are in its §12.

## License

[Apache-2.0](LICENSE)
