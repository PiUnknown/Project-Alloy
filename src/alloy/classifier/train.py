from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, classification_report

from alloy.labeling.weak_labels import combined_score, weak_label


def load_embeddings(interim_dir: Path) -> tuple[np.ndarray, list[int]]:
    embeddings = np.load(interim_dir / "embeddings.npy")
    question_ids = json.loads((interim_dir / "embeddings_index.json").read_text())
    return embeddings, question_ids


def train_quality_classifier(
    pool: list[dict],
    embeddings: np.ndarray,
    question_ids: list[int],
    top_quantile: float,
    bottom_quantile: float,
    seed: int,
):
    id_to_row = {row["question_id"]: row for row in pool}
    scores = [
        combined_score(id_to_row[qid]["question_score"], id_to_row[qid]["answer_score"])
        for qid in question_ids
    ]
    labels = weak_label(scores, top_quantile, bottom_quantile)

    labeled_mask = np.array([label != -1 for label in labels])
    X = embeddings[labeled_mask]
    y = np.array([label for label in labels if label != -1])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    metrics = {
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred, output_dict=True),
        "n_labeled": int(labeled_mask.sum()),
        "n_ambiguous_excluded": int((~labeled_mask).sum()),
    }
    return clf, metrics