from __future__ import annotations
import argparse
import json
from pathlib import Path

from alloy.config import AlloyConfig
from alloy.ingest.parser import parse_dump
from alloy.ingest.carveout import carve_out_eval_set
from alloy.lineage.manifest import write_manifest


def _write_jsonl(pairs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.__dict__) + "\n")


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="configs/base.yaml")
    args = arg_parser.parse_args()

    cfg = AlloyConfig.load(args.config)
    pairs = parse_dump(cfg.source.dump_path)

    remaining_pool, eval_set = carve_out_eval_set(
        pairs=pairs,
        target_size=cfg.eval_carveout.target_size,
        score_bucket_edges=cfg.eval_carveout.score_bucket_edges,
        seed=cfg.seed,
    )

    _write_jsonl(remaining_pool, Path(cfg.paths.interim) / "pool.jsonl")
    _write_jsonl(eval_set, Path(cfg.paths.eval) / "held_out.jsonl")
    write_manifest(eval_set, Path(cfg.paths.eval) / "manifest.json")

    print(f"Parsed {len(pairs)} question/accepted-answer pairs.")
    print(f"Held out {len(eval_set)} for eval, {len(remaining_pool)} remain in the working pool.")


if __name__ == "__main__":
    main()