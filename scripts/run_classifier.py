from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib

from alloy.config import AlloyConfig
from alloy.classifier.train import load_embeddings, train_quality_classifier
from alloy.classifier.score import score_pool


def _load_pool(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = AlloyConfig.load(args.config)
    interim_dir = Path(cfg.paths.interim)

    pool = _load_pool(interim_dir / "pool.jsonl")
    embeddings, question_ids = load_embeddings(interim_dir)
    assert [row["question_id"] for row in pool] == question_ids, (
        "pool.jsonl and embeddings_index.json are out of sync -- re-run run_embed.py"
    )

    clf, metrics = train_quality_classifier(
        pool=pool,
        embeddings=embeddings,
        question_ids=question_ids,
        top_quantile=cfg.labeling.top_quantile,
        bottom_quantile=cfg.labeling.bottom_quantile,
        seed=cfg.seed,
    )

    print(f"Classifier trained on {metrics['n_labeled']} labeled examples "
          f"({metrics['n_ambiguous_excluded']} ambiguous excluded).")
    print(f"Validation precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}")

    model_dir = Path(cfg.paths.models_classifier)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "quality_classifier.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    scores = score_pool(clf, embeddings)
    scores_path = interim_dir / "quality_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as f:
        for qid, score in zip(question_ids, scores):
            f.write(json.dumps({"question_id": qid, "quality_score": float(score)}) + "\n")

    print(f"Wrote quality scores for {len(question_ids)} pairs -> {scores_path}")


if __name__ == "__main__":
    main()