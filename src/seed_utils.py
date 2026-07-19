from __future__ import annotations

import random


def set_all_seeds(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def lgbm_seed_params(seed: int) -> dict:
    return {
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True, 
        "force_row_wise": True, 
    }
