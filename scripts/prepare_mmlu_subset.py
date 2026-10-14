"""Create and freeze Alloy's deterministic MMLU evaluation sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/eval/mmlu_subset.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the evaluation extra, including datasets, to prepare MMLU") from exc

    # Cache the selected examples as a project artifact. Evaluation itself never
    # fetches or re-samples the upstream dataset.
    source = load_dataset("cais/mmlu", "all", split="test")
    by_subject: dict[str, list[dict]] = {}
    for item in source:
        row = {
            "subject": str(item["subject"]),
            "question": str(item["question"]),
            "choices": [str(choice) for choice in item["choices"]],
            "answer": int(item["answer"]),
        }
        by_subject.setdefault(row["subject"], []).append(row)

    rng = random.Random(args.seed)
    subjects = sorted(by_subject)
    selected: list[dict] = []
    cursors = {subject: 0 for subject in subjects}
    for subject in subjects:
        rng.shuffle(by_subject[subject])
    while len(selected) < min(args.count, len(source)):
        added = False
        for subject in subjects:
            if cursors[subject] < len(by_subject[subject]):
                selected.append(by_subject[subject][cursors[subject]])
                cursors[subject] += 1
                added = True
                if len(selected) == args.count:
                    break
        if not added:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    output.write_text(encoded, encoding="utf-8")
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    manifest = {
        "source": "cais/mmlu",
        "split": "test",
        "seed": args.seed,
        "requested_count": args.count,
        "count": len(selected),
        "subjects": sorted({row["subject"] for row in selected}),
        "sha256": fingerprint,
        "data_path": str(output),
    }
    manifest_path = output.with_name("mmlu_subset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} fixed questions to {output}")
    print(f"SHA-256: {fingerprint}")


if __name__ == "__main__":
    main()
