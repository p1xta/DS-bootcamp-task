from __future__ import annotations

from src.text_utils import tokenize


def build_features(
    query_text: str,
    candidate_id: int,
    bm25_score: float,
    bm25_rank: int,
    dense_score: float,
    dense_rank: int,
    title_by_id: dict[int, str],
    bm25_score_norm: float = 0.0,
    dense_score_norm: float = 0.0,
    title_dense_score: float = 0.0,
    title_dense_score_norm: float = 0.0,
    title_dense_rank: int = 999,
) -> dict[str, float]:
    query_tokens = set(tokenize(query_text))
    title_tokens = set(tokenize(title_by_id.get(candidate_id, "")))

    title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)

    return {
        "bm25_score": bm25_score,
        "bm25_score_norm": bm25_score_norm,
        "bm25_rank_inv": 1.0 / (bm25_rank + 1),
        "dense_score": dense_score,
        "dense_score_norm": dense_score_norm,
        "dense_rank_inv": 1.0 / (dense_rank + 1),
        "title_dense_score": title_dense_score,
        "title_dense_score_norm": title_dense_score_norm,
        "title_dense_rank_inv": 1.0 / (title_dense_rank + 1),
        "title_overlap": title_overlap,
        "in_both_lists": float(bm25_rank < 999 and dense_rank < 999),
        "query_len_tokens": float(len(query_tokens)),
    }


FEATURE_NAMES = [
    "bm25_score",
    "bm25_score_norm",
    "bm25_rank_inv",
    "dense_score",
    "dense_score_norm",
    "dense_rank_inv",
    "title_dense_score",
    "title_dense_score_norm",
    "title_dense_rank_inv",
    "title_overlap",
    "in_both_lists",
    "query_len_tokens",
]


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max нормализация скоров одного запроса."""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]
