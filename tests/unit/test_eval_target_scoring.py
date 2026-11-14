"""lite-eval design §18.16, layer 1: lite-eval's hop scorers decide every hop exactly as the old
`eval/multihop.py` `_has` did (case-insensitive substring, a hop's keywords all in one memory)."""

import random

from lite_eval.contracts import Case, Output
from lite_eval.scorers.memory import HopScorer

from tests import legacy_multihop as legacy

# Mixed case, CJK, punctuation, whitespace variants and keywords that overlap one another.
WORDS = [
    *("Lily", "lily", "LILY", "sister", "older sister", "Older  Sister", "nurse", "护士"),
    *("我的姐姐", "姐姐", "Dunedin", "dunedin.", "İstanbul", "straße", "STRASSE", " ", "-"),
]


def _memory(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(0, 5)))


def test_every_hop_is_decided_as_the_old_rule_decided_it() -> None:
    rng = random.Random(20261114)
    decided = {True: 0, False: 0}
    for n in range(4000):
        found = [_memory(rng) for _ in range(rng.randint(0, 6))]
        hops = [[rng.choice(WORDS) for _ in range(rng.randint(1, 3))] for _ in range(2)]
        case = Case(id=f"c{n}", input=None, expected={"hops": hops})
        output = Output(case_id=case.id, value=found)
        old = [legacy._has(found, hop) for hop in hops]
        new = [HopScorer(hop=i).score(case, output).value for i in (1, 2)]
        assert new == [float(h) for h in old], (found, hops)
        assert HopScorer(hop=None).score(case, output).value == float(all(old))
        decided[old[0]] += 1
    assert min(decided.values()) > 500  # both outcomes are exercised, not only misses
