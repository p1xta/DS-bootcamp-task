from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "candidate_data"
ARTICLES_PATH = DATA_DIR / "articles.f"
CALIBRATION_PATH = DATA_DIR / "calibration.f"
TEST_PATH = DATA_DIR / "test.f"

INDEX_DIR = ROOT_DIR / "indexes"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

RANDOM_SEED = 42

DENSE_BODY_INDEX_PATH = INDEX_DIR / "dense_body.faiss"
DENSE_BODY_META_PATH = INDEX_DIR / "dense_body_meta.pkl"
DENSE_TITLE_INDEX_PATH = INDEX_DIR / "dense_title.faiss"
DENSE_TITLE_META_PATH = INDEX_DIR / "dense_title_meta.pkl"

ARTICLES_CLEAN_PATH = INDEX_DIR / "articles_clean.pkl"
LEARNED_RERANKER_PATH = INDEX_DIR / "learned_reranker.pkl"

OUTPUT_ANSWER_PATH = ROOT_DIR / "answer.csv"

# Модель эмбеддингов
EMBEDDING_MODEL_NAME = "deepvk/USER-bge-m3"
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_SEQ_LENGTH = 1024

# Модель cross-encoder реранкер
# Если RERANKER_MODEL_NAME = None, реранкинг отключён.
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_BATCH_SIZE = 32
RERANKER_MAX_CHARS = 1000

USE_RERANKER = False

USE_LEARNED_RERANKER = True

# Сколько заголовок весит по сравнению с телом статьи в BM25-индексе:
TITLE_REPEAT = 5
BM25_WEIGHT = 1.0

# article_id, которые нужно исключить из индекса целиком
EXCLUDED_ARTICLE_IDS = {2924}

# Сколько кандидатов брать от каждого метода до фьюжна.
BM25_TOP_N = 20
DENSE_TOP_N = 20

# Сколько кандидатов брать из TITLE dense-индекса 
TITLE_DENSE_TOP_N = 30

# Финальное число кандидатов в ответе
FINAL_TOP_K = 10

# Сколько кандидатов после фьюжна отправлять на реранкинг
RERANK_TOP_N = 30

FUSION_METHOD = "score"  # "rrf" / "score"

RRF_K = 60
