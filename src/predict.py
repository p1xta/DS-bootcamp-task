import pandas as pd

from src import config
from src.retriever import HybridRetriever, RetrievalEngine
from src.seed_utils import set_all_seeds


def validate_answers(answers: pd.DataFrame, valid_article_ids: set[int]) -> None:
    """Проверка перед сохранением"""
    assert answers["query_id"].is_unique, "Обнаружены дубли query_id в answer.csv"
    assert answers["answer"].notna().all(), "Есть пустые ответы"

    for _, row in answers.iterrows():
        ids = row["answer"].split()
        assert len(ids) <= 10, f"query_id={row['query_id']}: больше 10 статей"
        assert len(ids) == len(set(ids)), f"query_id={row['query_id']}: повторы article_id"
        for aid in ids:
            assert int(aid) in valid_article_ids, (
                f"query_id={row['query_id']}: article_id={aid} отсутствует в articles.f"
            )


def main() -> None:
    set_all_seeds(config.RANDOM_SEED)
    articles = pd.read_feather(config.ARTICLES_PATH)
    valid_article_ids = set(articles["article_id"].tolist())

    test_df = pd.read_feather(config.TEST_PATH)
    required_cols = {"query_id", "query_text"}
    missing = required_cols - set(test_df.columns)
    if missing:
        raise ValueError(f"В test.f отсутствуют колонки: {missing}")

    engine = RetrievalEngine(
        load_reranker=config.USE_RERANKER,
        load_learned_reranker=config.USE_LEARNED_RERANKER,
    )
    retriever = HybridRetriever(engine=engine)

    query_texts = test_df["query_text"].tolist()
    pred_lists = retriever.search_batch(query_texts)

    test_df["answer"] = [
        " ".join(str(x) for x in ids) for ids in pred_lists
    ]

    result = test_df[["query_id", "answer"]].copy()
    validate_answers(result, valid_article_ids)

    result.to_csv(config.OUTPUT_ANSWER_PATH, index=False)
    print(f"\nГотово. Сохранено: {config.OUTPUT_ANSWER_PATH}")
    print(f"Всего строк: {len(result)}")


if __name__ == "__main__":
    main()
