import argparse
import itertools
import time

import pandas as pd

from src import config
from src.retriever import HybridRetriever, RetrievalEngine
from src.seed_utils import set_all_seeds
from src.retriever import _rrf_merge


def average_precision_at_10(pred_ids: list[int], gt_ids: set[int]) -> float:
    if not gt_ids:
        return 0.0

    hits = 0
    ap_sum = 0.0
    for i, pid in enumerate(pred_ids[:10], start=1):
        if pid in gt_ids:
            hits += 1
            ap_sum += hits / i

    return ap_sum / min(len(gt_ids), 10)


def load_calibration() -> pd.DataFrame:
    df = pd.read_feather(config.CALIBRATION_PATH)
    required_cols = {"query_id", "query_text", "ground_truth"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В calibration.f отсутствуют колонки: {missing}")
    return df


def compute_map10(pred_lists: list[list[int]], gt_lists: list[set[int]]) -> float:
    ap_scores = [
        average_precision_at_10(pred, gt) for pred, gt in zip(pred_lists, gt_lists)
    ]
    return sum(ap_scores) / len(ap_scores)


def recall_at_k(candidate_lists: list[list[int]], gt_lists: list[set[int]], k: int) -> float:
    """
    Доля запросов, для которых хотя бы одна из правильных статей попала
    в top-k кандидатов.
    """
    hits = 0
    for candidates, gt in zip(candidate_lists, gt_lists):
        if gt & set(candidates[:k]):
            hits += 1
    return hits / len(gt_lists)


def run_recall_diagnostics() -> None:
    """
    Диагностика первого этапа поиска (до реранка): считает recall@10/20/30
    отдельно для BM25, dense и их RRF-объединения.
    """
    calibration = load_calibration()
    query_texts = calibration["query_text"].tolist()
    gt_lists = [
        {int(x) for x in str(gt).split()} for gt in calibration["ground_truth"]
    ]

    engine = RetrievalEngine(load_reranker=False, load_learned_reranker=False)

    max_k = 30
    print("BM25, body-dense и title-dense кандидаты для всех запросов...")
    embs = engine.embed_queries_batch(query_texts)
    bm25_pairs_lists = engine.bm25_search_batch_with_scores(query_texts, max_k)
    dense_pairs_lists = engine.dense_search_batch_with_scores(query_texts, max_k, embs=embs)
    title_pairs_lists = engine.title_dense_search_batch_with_scores(query_texts, max_k, embs=embs)

    bm25_lists = [[aid for aid, _ in pairs] for pairs in bm25_pairs_lists]
    dense_lists = [[aid for aid, _ in pairs] for pairs in dense_pairs_lists]
    title_lists = [[aid for aid, _ in pairs] for pairs in title_pairs_lists]

    fused_lists = [
        _rrf_merge([bm25, dense], k=config.RRF_K, top_n=max_k)
        for bm25, dense in zip(bm25_lists, dense_lists)
    ]

    print("\nRecall@K по источникам кандидатов (до реранка):")
    header = f"{'K':>4} | {'BM25':>8} | {'Body-D':>8} | {'Title-D':>8} | {'RRF (BM25+Body)':>16}"
    print(header)
    print("-" * len(header))
    for k in (10, 20, 30):
        r_bm25 = recall_at_k(bm25_lists, gt_lists, k)
        r_dense = recall_at_k(dense_lists, gt_lists, k)
        r_title = recall_at_k(title_lists, gt_lists, k)
        r_fused = recall_at_k(fused_lists, gt_lists, k)
        print(f"{k:>4} | {r_bm25:>8.3f} | {r_dense:>8.3f} | {r_title:>8.3f} | {r_fused:>16.3f}")


def run_single_eval() -> None:
    calibration = load_calibration()
    engine = RetrievalEngine(load_reranker=config.USE_RERANKER)
    retriever = HybridRetriever(engine=engine)

    query_texts = calibration["query_text"].tolist()
    gt_lists = [
        {int(x) for x in str(gt).split()} for gt in calibration["ground_truth"]
    ]

    t0 = time.time()
    pred_lists = retriever.search_batch(query_texts)
    elapsed = time.time() - t0

    map10 = compute_map10(pred_lists, gt_lists)
    print(f"\nMAP@10 на calibration.f: {map10:.4f}")
    print(f"Время инференса на {len(query_texts)} запросов: {elapsed:.1f} сек")


def run_tuning(skip_reranker: bool) -> None:
    """
    Grid-search по ключевым гиперпараметрам фьюжна/реранка.
    """
    calibration = load_calibration()
    query_texts = calibration["query_text"].tolist()
    gt_lists = [
        {int(x) for x in str(gt).split()} for gt in calibration["ground_truth"]
    ]

    need_reranker = not skip_reranker
    engine = RetrievalEngine(load_reranker=need_reranker)

    rrf_k_options = [20, 60, 100]
    reranker_options = [False] if skip_reranker else [True, False]
    rerank_top_n_options = [10, 20, 30]

    results = []
    combos = list(itertools.product(rrf_k_options, reranker_options, rerank_top_n_options))
    print(f"Всего комбинаций для перебора: {len(combos)}")

    for rrf_k, use_reranker, rerank_top_n in combos:
        retriever = HybridRetriever(
            engine=engine,
            rrf_k=rrf_k,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
        )
        t0 = time.time()
        pred_lists = retriever.search_batch(query_texts)
        elapsed = time.time() - t0

        map10 = compute_map10(pred_lists, gt_lists)
        results.append(
            {
                "rrf_k": rrf_k,
                "use_reranker": use_reranker,
                "rerank_top_n": rerank_top_n,
                "map@10": map10,
                "seconds": round(elapsed, 1),
            }
        )
        print(results[-1])

    results_df = pd.DataFrame(results).sort_values("map@10", ascending=False)
    print("\nЛучшие конфигурации:")
    print(results_df.head(10).to_string(index=False))


def main() -> None:
    set_all_seeds(config.RANDOM_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tune", action="store_true", help="Перебор гиперпараметров на calibration.f"
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Не грузить и не использовать реранкер в переборе",
    )
    parser.add_argument(
        "--recall",
        action="store_true",
        help="Посчитать Recall@10/20/30 отдельно по BM25/dense/RRF, без реранка",
    )
    args = parser.parse_args()

    if args.recall:
        run_recall_diagnostics()
    elif args.tune:
        run_tuning(skip_reranker=args.skip_reranker)
    else:
        run_single_eval()


if __name__ == "__main__":
    main()
