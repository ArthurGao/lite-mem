import math

import pytest

from lite_mem.fusion import rrf
from lite_mem.keyword import idf, term_score, terms


def test_terms_cut_chinese_and_keep_latin_words() -> None:
    got = terms("我住在上海浦东，喜欢 Green Tea!")
    assert {"上海", "浦东", "喜欢", "green", "tea"} <= set(got)
    assert not any(t in got for t in ("，", "!", " "))


def test_bm25_parts_match_the_formula() -> None:
    assert idf(10, 1) == pytest.approx(math.log(1 + 9.5 / 1.5))
    assert idf(10, 10) > 0  # the +1 keeps it positive
    assert term_score(1, 4, 4.0) == pytest.approx(1.0)  # an average-length doc, tf 1
    assert term_score(2, 4, 4.0) > term_score(1, 4, 4.0)
    assert term_score(1, 8, 4.0) < term_score(1, 2, 4.0)  # longer docs score lower


def test_rrf_rewards_agreement_and_uses_ranks_only() -> None:
    fused = rrf([["a", "b", "c"], ["b", "a"]], k=60)
    assert fused["a"] == fused["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63)
