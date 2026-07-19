import pandas as pd

import src.config as config
from src.retriever import HybridRetriever, RetrievalEngine
from src.seed_utils import set_all_seeds


def load_calibration():
    df = pd.read_feather(config.CALIBRATION_PATH)
    return df


def main():
    set_all_seeds(config.RANDOM_SEED)
    calibration = load_calibration()

    engine = RetrievalEngine(load_reranker=config.USE_RERANKER)
    retriever = HybridRetriever(engine=engine)

    query_texts = calibration["query_text"].tolist()

    predictions = retriever.search_batch(query_texts)

    calibration["prediction"] = predictions

    calibration.to_feather("analysis_predictions.feather")
    calibration.to_csv("analysis_predictions.csv", index=False)

    print("Сохранено:")
    print("analysis_predictions.feather")
    print("analysis_predictions.csv")


if __name__ == "__main__":
    main()