from __future__ import annotations
import numpy as np


def combined_score(question_score: int, answer_score: int) -> int:
    return question_score + answer_score


def weak_label(scores: list[int], top_quantile: float, bottom_quantile: float) -> list[int]:
    """Returns a label per item: 1 (positive), 0 (negative), or -1 (ambiguous,
    excluded from classifier training). Thresholds are computed once across
    the whole input, not per-batch, so they stay stable."""
    scores_arr = np.array(scores)
    top_cut = np.quantile(scores_arr, top_quantile)
    bottom_cut = np.quantile(scores_arr, bottom_quantile)

    labels = []
    for s in scores:
        if s >= top_cut:
            labels.append(1)
        elif s <= bottom_cut:
            labels.append(0)
        else:
            labels.append(-1)
    return labels