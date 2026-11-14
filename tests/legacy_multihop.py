"""`eval/multihop.py` as it was before S11 (commit 7491e36), frozen for the equivalence tests
(tests/integration/test_eval_target.py): the data and the settings are parameters, the rest is
the original. Not used anywhere else; do not change it."""

import tempfile
from pathlib import Path
from typing import Any

from lite_mem import factory
from lite_mem.config import Settings
from lite_mem.types import Message, Scope


def _has(found: list[str], keywords: list[str]) -> bool:
    return any(all(k.lower() in m.lower() for k in keywords) for m in found)


def run(data: dict[str, Any], top_k: int, follow_up: int) -> dict[str, Any]:
    db = Path(tempfile.mkdtemp()) / "multihop.db"
    engine = factory.open_engine(Settings(database_url=f"sqlite:///{db}", namespace="multihop"))
    rows, both, first, second = [], 0, 0, 0
    for user in data["users"]:
        scope = Scope(user_id=user["user_id"])
        for memory in user["memories"]:
            engine.add([Message("user", memory)], scope, infer=False)
        for q in user["questions"]:
            hits = engine.search(q["q"], scope, top_k=top_k, follow_up=follow_up)
            found = [h.memory.content for h in hits]
            h1, h2 = (_has(found, hop) for hop in q["hops"])
            first, second, both = first + h1, second + h2, both + (h1 and h2)
            rows.append({"user": user["user_id"], "q": q["q"], "hop1": h1, "hop2": h2,
                         "found": found})  # fmt: skip
    engine.close()
    summary = {"top_k": top_k, "follow_up": follow_up, "questions": len(rows),
               "hop1": first, "hop2": second, "both": both}  # fmt: skip
    return {"summary": summary, "rows": rows}
