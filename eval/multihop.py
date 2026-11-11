"""S10 gate: do two-hop questions find both memories they need, without a graph?

Stores `eval/multihop.json`'s memories as said (infer=false: no model call) in a throwaway SQLite
database with the configured embedder (bge-m3 on the local Ollama), searches every question and
reports, per question, whether each hop's memory is in the top k (`--follow-up N`: with the
engine's follow-up search through the top N hits).

    uv run python eval/multihop.py [--top-k 5] [--follow-up 1]
"""

import argparse
import json
import tempfile
from pathlib import Path

from lite_mem import factory
from lite_mem.config import Settings
from lite_mem.types import Message, Scope

DATA = Path(__file__).with_name("multihop.json")


def _has(found: list[str], keywords: list[str]) -> bool:
    return any(all(k.lower() in m.lower() for k in keywords) for m in found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--follow-up", type=int, default=0)
    args = parser.parse_args()
    data = json.loads(DATA.read_text())
    db = Path(tempfile.mkdtemp()) / "multihop.db"
    engine = factory.open_engine(Settings(database_url=f"sqlite:///{db}", namespace="multihop"))
    rows, both, first, second = [], 0, 0, 0
    for user in data["users"]:
        scope = Scope(user_id=user["user_id"])
        for memory in user["memories"]:
            engine.add([Message("user", memory)], scope, infer=False)
        for q in user["questions"]:
            hits = engine.search(q["q"], scope, top_k=args.top_k, follow_up=args.follow_up)
            found = [h.memory.content for h in hits]
            h1, h2 = (_has(found, hop) for hop in q["hops"])
            first, second, both = first + h1, second + h2, both + (h1 and h2)
            rows.append({"user": user["user_id"], "q": q["q"], "hop1": h1, "hop2": h2,
                         "found": found})  # fmt: skip
    summary = {"top_k": args.top_k, "follow_up": args.follow_up, "questions": len(rows),
               "hop1": first, "hop2": second, "both": both}  # fmt: skip
    print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
