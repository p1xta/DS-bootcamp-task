from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.features import FEATURE_NAMES, build_features, normalize_scores
from src.retriever import RetrievalEngine
from src.seed_utils import set_all_seeds, lgbm_seed_params


CANDIDATES_TOP_N = 30 
N_FOLDS = 5


def collect_training_rows(
    engine: RetrievalEngine, calibration: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Возвращает X (признаки), y (0/1 метка релевантности), query_ids.
    """
    rows, labels, query_ids = [], [], []

    for _, row in calibration.iterrows():
        query_text = row["query_text"]
        gt_ids = {int(x) for x in str(row["ground_truth"]).split()}

        q_emb = engine.embed_query(query_text)
        bm25_pairs = engine.bm25_search_with_scores(query_text, CANDIDATES_TOP_N)
        dense_pairs = engine.dense_search_with_scores(query_text, CANDIDATES_TOP_N, q_emb=q_emb)
        title_pairs = engine.title_dense_search_with_scores(
            query_text, config.TITLE_DENSE_TOP_N, q_emb=q_emb
        )
        bm25_rank = {aid: i for i, (aid, _) in enumerate(bm25_pairs)}
        dense_rank = {aid: i for i, (aid, _) in enumerate(dense_pairs)}
        title_rank = {aid: i for i, (aid, _) in enumerate(title_pairs)}
        bm25_score = {aid: s for aid, s in bm25_pairs}
        dense_score = {aid: s for aid, s in dense_pairs}
        title_score = {aid: s for aid, s in title_pairs}

        bm25_score_norm = dict(zip(bm25_score, normalize_scores(list(bm25_score.values()))))
        dense_score_norm = dict(zip(dense_score, normalize_scores(list(dense_score.values()))))
        title_score_norm = dict(zip(title_score, normalize_scores(list(title_score.values()))))

        candidates = sorted(set(bm25_rank) | set(dense_rank))  # сортировка — для воспроизводимости
        for cid in candidates:
            feats = build_features(
                query_text,
                cid,
                bm25_score.get(cid, 0.0),
                bm25_rank.get(cid, 999),
                dense_score.get(cid, 0.0),
                dense_rank.get(cid, 999),
                engine.title_by_id,
                bm25_score_norm=bm25_score_norm.get(cid, 0.0),
                dense_score_norm=dense_score_norm.get(cid, 0.0),
                title_dense_score=title_score.get(cid, 0.0),
                title_dense_score_norm=title_score_norm.get(cid, 0.0),
                title_dense_rank=title_rank.get(cid, 999),
            )
            rows.append(feats)
            labels.append(int(cid in gt_ids))
            query_ids.append(row["query_id"])

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = np.array(labels)
    q = np.array(query_ids)
    return X, y, q


def average_precision_at_10(pred_ids: list[int], gt_ids: set[int]) -> float:
    if not gt_ids:
        return 0.0
    hits, ap_sum = 0, 0.0
    for i, pid in enumerate(pred_ids[:10], start=1):
        if pid in gt_ids:
            hits += 1
            ap_sum += hits / i
    return ap_sum / min(len(gt_ids), 10)


def train_lambdamart(
    X: pd.DataFrame,
    y: np.ndarray,
    group_sizes: list[int],
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    val_group_sizes: list[int] | None = None,
) -> lgb.Booster:
    train_data = lgb.Dataset(X, label=y, group=group_sizes)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "verbose": -1,
        **lgbm_seed_params(config.RANDOM_SEED),
    }

    if X_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, group=val_group_sizes, reference=train_data)
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    else:
        model = lgb.train(params, train_data, num_boost_round=200)
    return model


def evaluate_fold_map10(
    model: lgb.Booster,
    calibration_fold: pd.DataFrame,
    engine: RetrievalEngine,
) -> float:
    ap_scores = []
    for _, row in calibration_fold.iterrows():
        query_text = row["query_text"]
        gt_ids = {int(x) for x in str(row["ground_truth"]).split()}

        q_emb = engine.embed_query(query_text)
        bm25_pairs = engine.bm25_search_with_scores(query_text, CANDIDATES_TOP_N)
        dense_pairs = engine.dense_search_with_scores(query_text, CANDIDATES_TOP_N, q_emb=q_emb)
        title_pairs = engine.title_dense_search_with_scores(
            query_text, config.TITLE_DENSE_TOP_N, q_emb=q_emb
        )
        bm25_rank = {aid: i for i, (aid, _) in enumerate(bm25_pairs)}
        dense_rank = {aid: i for i, (aid, _) in enumerate(dense_pairs)}
        title_rank = {aid: i for i, (aid, _) in enumerate(title_pairs)}
        bm25_score = {aid: s for aid, s in bm25_pairs}
        dense_score = {aid: s for aid, s in dense_pairs}
        title_score = {aid: s for aid, s in title_pairs}
        bm25_score_norm = dict(zip(bm25_score, normalize_scores(list(bm25_score.values()))))
        dense_score_norm = dict(zip(dense_score, normalize_scores(list(dense_score.values()))))
        title_score_norm = dict(zip(title_score, normalize_scores(list(title_score.values()))))

        candidate_ids = sorted(set(bm25_rank) | set(dense_rank))
        feats = [
            build_features(
                query_text, cid,
                bm25_score.get(cid, 0.0), bm25_rank.get(cid, 999),
                dense_score.get(cid, 0.0), dense_rank.get(cid, 999),
                engine.title_by_id,
                bm25_score_norm=bm25_score_norm.get(cid, 0.0),
                dense_score_norm=dense_score_norm.get(cid, 0.0),
                title_dense_score=title_score.get(cid, 0.0),
                title_dense_score_norm=title_score_norm.get(cid, 0.0),
                title_dense_rank=title_rank.get(cid, 999),
            )
            for cid in candidate_ids
        ]
        X = pd.DataFrame(feats, columns=FEATURE_NAMES)
        scores = model.predict(X)
        order = np.argsort(-scores)
        pred_ids = [candidate_ids[i] for i in order][:10]

        ap_scores.append(average_precision_at_10(pred_ids, gt_ids))

    return float(np.mean(ap_scores))


def _ordered_group_sizes(query_ids_seq: np.ndarray) -> list[int]:
    _, first_idx, counts = np.unique(query_ids_seq, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    return list(counts[order])


def split_train_earlystop(
    X: pd.DataFrame, y: np.ndarray, query_ids: np.ndarray, holdout_frac: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray, list[int], pd.DataFrame, np.ndarray, list[int]]:
    unique_qids = np.unique(query_ids)
    shuffled = rng.permutation(unique_qids)
    n_holdout = max(1, int(len(shuffled) * holdout_frac))
    holdout_qids = set(shuffled[:n_holdout])

    train_mask = ~np.isin(query_ids, list(holdout_qids))
    val_mask = ~train_mask

    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask]
    group_train = _ordered_group_sizes(query_ids[train_mask])

    X_val = X[val_mask].reset_index(drop=True)
    y_val = y[val_mask]
    group_val = _ordered_group_sizes(query_ids[val_mask])

    return X_train, y_train, group_train, X_val, y_val, group_val


def main() -> None:
    set_all_seeds(config.RANDOM_SEED)
    calibration = pd.read_feather(config.CALIBRATION_PATH)

    print("Загрузка RetrievalEngine")
    engine = RetrievalEngine(load_reranker=False, load_learned_reranker=False)

    print("Сбор обучающих пар (query, candidate) на calibration.f")
    X, y, query_ids = collect_training_rows(engine, calibration)
    print(f"Обучающих пар: {len(X)}, положительных: {int(y.sum())}")

    rng = np.random.default_rng(42)
    unique_query_ids = calibration["query_id"].to_numpy()
    shuffled = rng.permutation(unique_query_ids)
    folds = np.array_split(shuffled, N_FOLDS)

    fold_maps = []
    for fold_idx, val_query_ids in enumerate(folds, start=1):
        val_query_ids_set = set(val_query_ids)
        train_mask = ~np.isin(query_ids, list(val_query_ids_set))

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        qids_train_full = query_ids[train_mask]

        X_tr, y_tr, group_tr, X_es, y_es, group_es = split_train_earlystop(
            X_train_full, y_train_full, qids_train_full, holdout_frac=0.15, rng=rng
        )

        model = train_lambdamart(X_tr, y_tr, group_tr, X_val=X_es, y_val=y_es, val_group_sizes=group_es)

        calibration_val = calibration[calibration["query_id"].isin(val_query_ids_set)]
        map10 = evaluate_fold_map10(model, calibration_val, engine)
        fold_maps.append(map10)
        print(
            f"  fold {fold_idx}/{N_FOLDS}: MAP@10 = {map10:.4f} "
            f"(best_iteration={model.best_iteration}, val={len(calibration_val)} запросов)"
        )

    print(f"\nCV MAP@10: {np.mean(fold_maps):.4f} ± {np.std(fold_maps):.4f}")

    X_tr, y_tr, group_tr, X_es, y_es, group_es = split_train_earlystop(
        X, y, query_ids, holdout_frac=0.15, rng=rng
    )
    final_model = train_lambdamart(X_tr, y_tr, group_tr, X_val=X_es, y_val=y_es, val_group_sizes=group_es)
    print(f"Финальная модель: best_iteration={final_model.best_iteration}")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LEARNED_RERANKER_PATH, "wb") as f:
        pickle.dump(
            {"model": final_model, "feature_names": FEATURE_NAMES, "model_type": "lightgbm"},
            f,
        )
    print(f"Сохранено: {config.LEARNED_RERANKER_PATH}")


if __name__ == "__main__":
    main()
