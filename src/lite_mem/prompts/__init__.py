"""Prompt templates, one per LLM role (AGENTS.md rule 7): system prompts, loaded by name."""

from importlib import resources


def load(name: str) -> str:
    """One template's text, by file name (e.g. `extract_facts.j2`)."""
    return (resources.files(__package__) / name).read_text(encoding="utf-8")
