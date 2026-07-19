import pickle

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src import config
from src.text_utils import build_indexed_text, clean_html, tokenize
from src.seed_utils import set_all_seeds


def load_and_clean_articles() -> pd.DataFrame:
    articles = pd.read_feather(config.ARTICLES_PATH)
    required_cols = {"article_id", "title", "body"}
    missing = required_cols - set(articles.columns)
    if missing:
        raise ValueError(f"В articles.f отсутствуют колонки: {missing}")

    if config.EXCLUDED_ARTICLE_IDS:
        before = len(articles)
        articles = articles[~articles["article_id"].isin(config.EXCLUDED_ARTICLE_IDS)].copy()
        removed = before - len(articles)
        print(
            f"Исключено статей по EXCLUDED_ARTICLE_IDS: {removed} "
            f"({sorted(config.EXCLUDED_ARTICLE_IDS)})"
        )

    articles["title"] = articles["title"].fillna("").astype(str)

    tqdm.pandas(desc="Формирование indexed_text для BM25")
    articles["indexed_text"] = articles.progress_apply(
        lambda row: build_indexed_text(row["title"], row["body"], config.TITLE_REPEAT),
        axis=1,
    )

    tqdm.pandas(desc="Очистка body для dense-индекса")
    articles["clean_body"] = articles["body"].progress_apply(clean_html)

    return articles


def build_bm25_index(articles: pd.DataFrame) -> None:
    print("Токенизация корпуса для BM25...")
    corpus_tokens = [
        tokenize(text) for text in tqdm(articles["indexed_text"], desc="BM25 tokenize")
    ]
    bm25 = BM25Okapi(corpus_tokens)

    payload = {
        "bm25": bm25,
        "article_ids": articles["article_id"].tolist(),
    }
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"BM25 индекс сохранён: {config.BM25_INDEX_PATH}")


def _build_single_dense_index(
    model: SentenceTransformer,
    texts: list[str],
    article_ids: list[int],
    index_path,
    meta_path,
    label: str,
) -> None:
    passages = [config.EMBEDDING_PASSAGE_PREFIX + t for t in texts]
    print(f"Вычисление эмбеддингов ({label})...")
    embeddings = model.encode(
        passages,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"article_ids": article_ids}, f)

    print(f"Dense-индекс ({label}) сохранён: {index_path}")


def build_dense_indexes(articles: pd.DataFrame) -> None:
    print(f"Загрузка модели эмбеддингов: {config.EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    model.max_seq_length = config.EMBEDDING_MAX_SEQ_LENGTH

    article_ids = articles["article_id"].tolist()

    _build_single_dense_index(
        model,
        articles["clean_body"].tolist(),
        article_ids,
        config.DENSE_BODY_INDEX_PATH,
        config.DENSE_BODY_META_PATH,
        label="BODY",
    )

    _build_single_dense_index(
        model,
        articles["title"].tolist(),
        article_ids,
        config.DENSE_TITLE_INDEX_PATH,
        config.DENSE_TITLE_META_PATH,
        label="TITLE",
    )


def main() -> None:
    set_all_seeds(config.RANDOM_SEED)
    articles = load_and_clean_articles()

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.ARTICLES_CLEAN_PATH, "wb") as f:
        pickle.dump(
            articles[["article_id", "title", "clean_body", "indexed_text"]], f
        )

    build_bm25_index(articles)
    build_dense_indexes(articles)


if __name__ == "__main__":
    main()
