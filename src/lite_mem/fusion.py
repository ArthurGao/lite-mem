"""Reciprocal rank fusion (design §5): score(d) = Σ_lists 1 / (k + rank_list(d)), ranks from 1.
It uses ranks only, so the vector distances and BM25 scores never have to share a scale."""

from collections.abc import Hashable, Iterable, Sequence


def rrf[K: Hashable](rankings: Iterable[Sequence[K]], k: int = 60) -> dict[K, float]:
    fused: dict[K, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused
