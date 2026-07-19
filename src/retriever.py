from __future__ import annotations

import pickle
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

import src.config as config
from src.features import build_features
from src.text_utils import tokenize


def _rrf_merge(
    rank_lists: list[list[int]],
    k: int = 60,
    top_n: int | None = None,
    weights: list[float] | None = None,
) -> list[int]:
    if weights is None:
        weights = [1.0] * len(rank_lists)
    scores: dict[int, float] = {}
    for weight, ranked_ids in zip(weights, rank_lists):
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    merged = sorted(scores.items(), key=lambda x: -x[1])
    ids = [doc_id for doc_id, _ in merged]
    return ids[:top_n] if top_n else ids


def _minmax_normalize(scores: list[float]) -> list[float]:
    """Приводит скоры одного запроса к диапазону [0, 1]."""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _score_fusion_merge(
    bm25_pairs: list[tuple[int, float]],
    dense_pairs: list[tuple[int, float]],
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    top_n: int | None = None,
) -> list[int]:
    """
    Взвешенная сумма нормализованных скоров.
    """
    bm25_ids = [doc_id for doc_id, _ in bm25_pairs]
    bm25_scores_norm = _minmax_normalize([s for _, s in bm25_pairs])

    dense_ids = [doc_id for doc_id, _ in dense_pairs]
    dense_scores_norm = _minmax_normalize([s for _, s in dense_pairs])

    scores: dict[int, float] = {}
    for doc_id, s in zip(bm25_ids, bm25_scores_norm):
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight * s
    for doc_id, s in zip(dense_ids, dense_scores_norm):
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight * s

    merged = sorted(scores.items(), key=lambda x: -x[1])
    ids = [doc_id for doc_id, _ in merged]
    return ids[:top_n] if top_n else ids


class RetrievalEngine:
    """
    Держит все тяжёлые артефакты (индексы, модели) в памяти.
    """
    def __init__(self, load_reranker: bool = True, load_learned_reranker: bool = True) -> None:
        self._load_bm25()
        self._load_dense_body()
        self._load_dense_title()
        self._load_articles_text()

        print(f"Загрузка эмбеддера: {config.EMBEDDING_MODEL_NAME} (CPU)")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
        self.embedder.max_seq_length = config.EMBEDDING_MAX_SEQ_LENGTH

        self.reranker = None
        if load_reranker and config.RERANKER_MODEL_NAME:
            print(f"Загрузка реранкера: {config.RERANKER_MODEL_NAME} (CPU)")
            self.reranker = CrossEncoder(config.RERANKER_MODEL_NAME, device="cpu")

        self.learned_reranker_model = None
        self.learned_reranker_feature_names: list[str] = []
        self.learned_reranker_model_type = "sklearn"
        if load_learned_reranker and config.LEARNED_RERANKER_PATH.exists():
            print(f"Загрузка learned reranker: {config.LEARNED_RERANKER_PATH}")
            with open(config.LEARNED_RERANKER_PATH, "rb") as f:
                payload = pickle.load(f)
            self.learned_reranker_model = payload["model"]
            self.learned_reranker_feature_names = payload["feature_names"]
            self.learned_reranker_model_type = payload.get("model_type", "sklearn")

    def _load_bm25(self) -> None:
        with open(config.BM25_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.bm25_article_ids = payload["article_ids"]

    def _load_dense_body(self) -> None:
        self.dense_body_index = faiss.read_index(str(config.DENSE_BODY_INDEX_PATH))
        with open(config.DENSE_BODY_META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.dense_body_article_ids = meta["article_ids"]

    def _load_dense_title(self) -> None:
        self.dense_title_index = faiss.read_index(str(config.DENSE_TITLE_INDEX_PATH))
        with open(config.DENSE_TITLE_META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.dense_title_article_ids = meta["article_ids"]

    def _load_articles_text(self) -> None:
        with open(config.ARTICLES_CLEAN_PATH, "rb") as f:
            df = pickle.load(f)
        self.article_text_by_id = dict(zip(df["article_id"], df["indexed_text"]))
        self.title_by_id = dict(zip(df["article_id"], df["title"]))

    def embed_query(self, query_text: str) -> np.ndarray:
        return self.embedder.encode(
            config.EMBEDDING_QUERY_PREFIX + query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_queries_batch(self, query_texts: list[str]) -> np.ndarray:
        prefixed = [config.EMBEDDING_QUERY_PREFIX + q for q in query_texts]
        return self.embedder.encode(
            prefixed,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def bm25_search_with_scores(
        self, query_text: str, top_n: int
    ) -> list[tuple[int, float]]:
        q_tokens = tokenize(query_text)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(-scores)[:top_n]
        return [(self.bm25_article_ids[i], float(scores[i])) for i in top_idx]

    def bm25_search_batch_with_scores(
        self, query_texts: list[str], top_n: int
    ) -> list[list[tuple[int, float]]]:
        return [self.bm25_search_with_scores(q, top_n) for q in query_texts]

    def dense_search_with_scores(
        self, query_text: str, top_n: int, q_emb: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        if q_emb is None:
            q_emb = self.embed_query(query_text)
        scores, indices = self.dense_body_index.search(np.expand_dims(q_emb, axis=0), top_n)
        return [
            (self.dense_body_article_ids[i], float(s))
            for i, s in zip(indices[0], scores[0])
            if i != -1
        ]

    def dense_search_batch_with_scores(
        self, query_texts: list[str], top_n: int, embs: np.ndarray | None = None
    ) -> list[list[tuple[int, float]]]:
        if embs is None:
            embs = self.embed_queries_batch(query_texts)
        all_scores, all_indices = self.dense_body_index.search(embs, top_n)
        return [
            [
                (self.dense_body_article_ids[i], float(s))
                for i, s in zip(indices, scores)
                if i != -1
            ]
            for indices, scores in zip(all_indices, all_scores)
        ]

    def title_dense_search_with_scores(
        self, query_text: str, top_n: int, q_emb: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        if q_emb is None:
            q_emb = self.embed_query(query_text)
        scores, indices = self.dense_title_index.search(np.expand_dims(q_emb, axis=0), top_n)
        return [
            (self.dense_title_article_ids[i], float(s))
            for i, s in zip(indices[0], scores[0])
            if i != -1
        ]

    def title_dense_search_batch_with_scores(
        self, query_texts: list[str], top_n: int, embs: np.ndarray | None = None
    ) -> list[list[tuple[int, float]]]:
        if embs is None:
            embs = self.embed_queries_batch(query_texts)
        all_scores, all_indices = self.dense_title_index.search(embs, top_n)
        return [
            [
                (self.dense_title_article_ids[i], float(s))
                for i, s in zip(indices, scores)
                if i != -1
            ]
            for indices, scores in zip(all_indices, all_scores)
        ]

    def rerank(self, query_text: str, candidate_ids: list[int]) -> list[int]:
        if self.reranker is None or not candidate_ids:
            return candidate_ids
        pairs = [
            (query_text, self.article_text_by_id.get(cid, "")[:config.RERANKER_MAX_CHARS])
            for cid in candidate_ids
        ]
        scores = self.reranker.predict(
            pairs, batch_size=config.RERANKER_BATCH_SIZE, show_progress_bar=False
        )
        order = np.argsort(-np.asarray(scores))
        return [candidate_ids[i] for i in order]

    def learned_rerank(
        self,
        query_text: str,
        candidate_ids: list[int],
        bm25_pairs: list[tuple[int, float]],
        dense_pairs: list[tuple[int, float]],
        title_pairs: list[tuple[int, float]],
    ) -> list[int]:
        """
        Пересортировывает candidate_ids моделью, обученной на признаках
        (bm25/body-dense/title-dense скоры и ранги, title_overlap.
        """
        if self.learned_reranker_model is None or not candidate_ids:
            return candidate_ids

        bm25_rank = {aid: i for i, (aid, _) in enumerate(bm25_pairs)}
        dense_rank = {aid: i for i, (aid, _) in enumerate(dense_pairs)}
        title_rank = {aid: i for i, (aid, _) in enumerate(title_pairs)}
        bm25_score = {aid: s for aid, s in bm25_pairs}
        dense_score = {aid: s for aid, s in dense_pairs}
        title_score = {aid: s for aid, s in title_pairs}
        bm25_score_norm = dict(zip(bm25_score, _minmax_normalize(list(bm25_score.values()))))
        dense_score_norm = dict(zip(dense_score, _minmax_normalize(list(dense_score.values()))))
        title_score_norm = dict(zip(title_score, _minmax_normalize(list(title_score.values()))))

        feats = [
            build_features(
                query_text, cid,
                bm25_score.get(cid, 0.0), bm25_rank.get(cid, 999),
                dense_score.get(cid, 0.0), dense_rank.get(cid, 999),
                self.title_by_id,
                bm25_score_norm=bm25_score_norm.get(cid, 0.0),
                dense_score_norm=dense_score_norm.get(cid, 0.0),
                title_dense_score=title_score.get(cid, 0.0),
                title_dense_score_norm=title_score_norm.get(cid, 0.0),
                title_dense_rank=title_rank.get(cid, 999),
            )
            for cid in candidate_ids
        ]
        X = pd.DataFrame(feats)[self.learned_reranker_feature_names]

        if self.learned_reranker_model_type == "lightgbm":
            scores = self.learned_reranker_model.predict(X)
        else:
            scores = self.learned_reranker_model.predict_proba(X)[:, 1]

        order = np.argsort(-scores)
        return [candidate_ids[i] for i in order]


@dataclass
class HybridRetriever:
    engine: RetrievalEngine
    bm25_top_n: int = config.BM25_TOP_N
    dense_top_n: int = config.DENSE_TOP_N
    title_dense_top_n: int = config.TITLE_DENSE_TOP_N
    rrf_k: int = config.RRF_K
    final_top_k: int = config.FINAL_TOP_K
    rerank_top_n: int = config.RERANK_TOP_N
    use_reranker: bool = config.USE_RERANKER
    use_learned_reranker: bool = config.USE_LEARNED_RERANKER
    bm25_weight: float = config.BM25_WEIGHT
    dense_weight: float = 1.0
    fusion_method: str = config.FUSION_METHOD

    def _fuse_from_pairs(
        self,
        bm25_pairs: list[tuple[int, float]],
        dense_pairs: list[tuple[int, float]],
    ) -> list[int]:
        if self.fusion_method == "score":
            return _score_fusion_merge(
                bm25_pairs, dense_pairs,
                bm25_weight=self.bm25_weight, dense_weight=self.dense_weight,
                top_n=self.rerank_top_n,
            )
        bm25_ids = [aid for aid, _ in bm25_pairs]
        dense_ids = [aid for aid, _ in dense_pairs]
        return _rrf_merge(
            [bm25_ids, dense_ids], k=self.rrf_k, top_n=self.rerank_top_n,
            weights=[self.bm25_weight, self.dense_weight],
        )

    def _apply_reranking(
        self,
        query_text: str,
        fused_ids: list[int],
        bm25_pairs: list[tuple[int, float]],
        dense_pairs: list[tuple[int, float]],
        title_pairs: list[tuple[int, float]],
    ) -> list[int]:
        if self.use_learned_reranker:
            return self.engine.learned_rerank(
                query_text, fused_ids, bm25_pairs, dense_pairs, title_pairs
            )
        if self.use_reranker:
            return self.engine.rerank(query_text, fused_ids)
        return fused_ids

    def search(self, query_text: str) -> list[int]:
        query_text = (query_text or "").strip()
        if not query_text:
            return []

        q_emb = self.engine.embed_query(query_text)

        bm25_pairs = self.engine.bm25_search_with_scores(query_text, self.bm25_top_n)
        dense_pairs = self.engine.dense_search_with_scores(query_text, self.dense_top_n, q_emb=q_emb)
        title_pairs = self.engine.title_dense_search_with_scores(
            query_text, self.title_dense_top_n, q_emb=q_emb
        )

        fused_ids = self._fuse_from_pairs(bm25_pairs, dense_pairs)
        fused_ids = self._apply_reranking(query_text, fused_ids, bm25_pairs, dense_pairs, title_pairs)

        return fused_ids[: self.final_top_k]

    def search_as_string(self, query_text: str) -> str:
        return " ".join(str(x) for x in self.search(query_text))

    def search_batch(self, query_texts: list[str]) -> list[list[int]]:
        embs = self.engine.embed_queries_batch(query_texts)

        bm25_lists = self.engine.bm25_search_batch_with_scores(query_texts, self.bm25_top_n)
        dense_lists = self.engine.dense_search_batch_with_scores(
            query_texts, self.dense_top_n, embs=embs
        )
        title_lists = self.engine.title_dense_search_batch_with_scores(
            query_texts, self.title_dense_top_n, embs=embs
        )

        fused_lists = [
            self._fuse_from_pairs(bm25_pairs, dense_pairs)
            for bm25_pairs, dense_pairs in zip(bm25_lists, dense_lists)
        ]

        results = []
        iterator = zip(query_texts, fused_lists, bm25_lists, dense_lists, title_lists)
        if self.use_learned_reranker:
            desc = "Learned-реранкинг запросов"
        elif self.use_reranker:
            desc = "Реранкинг запросов"
        else:
            desc = "Фьюжн кандидатов"
        for query_text, fused_ids, bm25_pairs, dense_pairs, title_pairs in tqdm(
            iterator, total=len(query_texts), desc=desc
        ):
            fused_ids = self._apply_reranking(
                query_text, fused_ids, bm25_pairs, dense_pairs, title_pairs
            )
            results.append(fused_ids[: self.final_top_k])
        return results
