"""Terms and BM25 (design §5), written out so they can be read and changed.

    score(q, d) = Σ_t IDF(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))
    IDF(t)      = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

t runs over the distinct query terms; f(t,d) is the term's count in memory d, |d| its length in
terms, N the number of memories the search may return (its scope and filters), n(t) those
containing t, avgdl their mean length. The statistics are the scope's, not the namespace's (S8b):
one user's ranking does not depend on what other users said, and the work grows with the user.
Both stores compute this formula (PgStore in SQL); the tests check them against these functions."""

import logging
import math
import re
from collections import Counter

import jieba

TOKENIZER = "jieba-0.42.1"  # recorded per namespace: postings built with another are not mixed
K1, B = 1.2, 0.75
_WORDLIKE = re.compile(r"\w")

jieba.setLogLevel(logging.WARNING)


def terms(text: str) -> list[str]:
    """Lower-cased search terms: jieba's search-mode cut (Chinese words and their parts; Latin words
    and numbers as they are), without whitespace and punctuation."""
    return [t.strip().lower() for t in jieba.cut_for_search(text) if _WORDLIKE.search(t)]


def term_counts(text: str) -> Counter[str]:
    return Counter(terms(text))


def idf(n_docs: int, doc_freq: int) -> float:
    return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def term_score(tf: int, length: int, avg_length: float, k1: float = K1, b: float = B) -> float:
    norm = 1 - b + b * (length / avg_length) if avg_length else 1.0
    return tf * (k1 + 1) / (tf + k1 * norm)
