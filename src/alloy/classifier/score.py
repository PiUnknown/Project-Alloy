from __future__ import annotations
import numpy as np


def score_pool(clf, embeddings: np.ndarray) -> np.ndarray:
    """Returns P(positive) for every row in embeddings, in the same order."""
    return clf.predict_proba(embeddings)[:, 1]