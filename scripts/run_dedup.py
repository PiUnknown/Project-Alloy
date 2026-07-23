from __future__ import annotations
import argparse
import json
from pathlib import Path

from alloy.config import AlloyConfig
from alloy.classifier.train import load_embeddings
from alloy.dedup.near_duplicates import find_duplicate_clusters, choose_representative


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

    scores = [row["question_score"] + row["answer_score"] for row in pool]

    clusters = find_duplicate_clusters(
        embeddings, threshold=cfg.dedup.similarity_threshold, top_k=cfg.dedup.top_k_neighbors
    )

    drop_ids: set[int] = set()
    for cluster in clusters:
        keep_row = choose_representative(cluster, scores)
        for row_idx in cluster:
            if row_idx != keep_row:
                drop_ids.add(question_ids[row_idx])

    decisions_path = interim_dir / "dedup_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as f:
        for qid in question_ids:
            f.write(json.dumps({"question_id": qid, "drop_as_duplicate": qid in drop_ids}) + "\n")

    print(f"Found {len(clusters)} duplicate clusters covering {len(drop_ids)} pairs to drop "
          f"out of {len(pool)} total.")


if __name__ == "__main__":
    main()