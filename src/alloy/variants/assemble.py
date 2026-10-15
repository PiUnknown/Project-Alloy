"""Materialize the unfiltered V0 training inputs without touching frozen eval data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_rows(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def load_eval_question_ids(manifest_path: str | Path) -> set[int]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {int(question_id) for question_id in manifest["question_ids"]}


def assemble_v0(
    pool_path: str | Path,
    output_dir: str | Path,
    *,
    eval_manifest_path: str | Path,
) -> dict[str, Any]:
    """Write raw CPT/SFT inputs from the Phase 0 working pool and audit them."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cpt_path = output / "cpt.jsonl"
    sft_path = output / "sft.jsonl"
    eval_ids = load_eval_question_ids(eval_manifest_path)
    count = 0
    ids_hash = hashlib.sha256()
    with cpt_path.open("w", encoding="utf-8", newline="\n") as cpt, sft_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as sft:
        for row in read_rows(pool_path):
            question_id = int(row["question_id"])
            if question_id in eval_ids:
                raise ValueError(f"Frozen evaluation question {question_id} leaked into the V0 pool")
            question = str(row["question_text"]).strip()
            answer = str(row["answer_text"]).strip()
            if not question or not answer:
                continue
            cpt_row = {
                "question_id": question_id,
                "text": f"Question: {question}\n\nAnswer: {answer}",
            }
            sft_row = {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "question_score": int(row.get("question_score", 0)),
                "answer_score": int(row.get("answer_score", 0)),
            }
            cpt.write(json.dumps(cpt_row, ensure_ascii=False) + "\n")
            sft.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            ids_hash.update(f"{question_id}\n".encode("ascii"))
            count += 1

    if not count:
        raise ValueError("V0 assembly produced no usable examples")
    manifest = {
        "variant": "v0",
        "source_pool": str(pool_path),
        "count": count,
        "question_ids_sha256_in_source_order": ids_hash.hexdigest(),
        "eval_manifest": str(eval_manifest_path),
        "eval_question_ids_checked": len(eval_ids),
        "quality_filter": False,
        "deduplication": False,
        "replay_buffer": False,
        "artifacts": {"cpt": str(cpt_path), "sft": str(sft_path)},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
