from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from alloy.config import AlloyConfig
from alloy.embeddings import embed_texts


def _load_pool(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = AlloyConfig.load(args.config)
    pool_path = Path(cfg.paths.interim) / "pool.jsonl"
    pool = _load_pool(pool_path)

    texts = [f"{row['question_text']} {row['answer_text']}" for row in pool]
    question_ids = [row["question_id"] for row in pool]

    embeddings = embed_texts(texts)

    out_dir = Path(cfg.paths.interim)
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "embeddings_index.json").write_text(json.dumps(question_ids))

    print(f"Embedded {len(pool)} pairs -> {embeddings.shape}")


if __name__ == "__main__":
    main()