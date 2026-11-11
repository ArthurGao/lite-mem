"""Prompt templates, one per LLM role (AGENTS.md rule 7): system prompts, loaded by name."""

from collections.abc import Sequence
from importlib import resources
from typing import Any

from lite_mem.types import Memory

# The default template of each step (config and the step classes both use these).
EXTRACT = "extract_facts.j2"
RESOLVE = "resolve_conflict.j2"
MERGE = "consolidate_merge.j2"
SUMMARISE = "consolidate_summarise.j2"


def load(name: str) -> str:
    """One template's text, by file name (e.g. `extract_facts.j2`)."""
    return (resources.files(__package__) / name).read_text(encoding="utf-8")


def listed(memories: Sequence[Memory]) -> list[dict[str, Any]]:
    """Memories as every prompt lists them: an id the model may name back, and the fact."""
    return [{"id": str(m.id), "fact": m.content} for m in memories]
