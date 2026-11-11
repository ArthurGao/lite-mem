"""Prompt templates, one per LLM role (AGENTS.md rule 7): system prompts, loaded by name."""

from collections.abc import Sequence
from importlib import resources
from typing import Any

from lite_mem.types import Memory


def load(name: str) -> str:
    """One template's text, by file name (e.g. `extract_facts.j2`)."""
    return (resources.files(__package__) / name).read_text(encoding="utf-8")


def listed(memories: Sequence[Memory]) -> list[dict[str, Any]]:
    """Memories as every prompt lists them: an id the model may name back, and the fact."""
    return [{"id": str(m.id), "fact": m.content} for m in memories]
