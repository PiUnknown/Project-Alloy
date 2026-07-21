from __future__ import annotations
import hashlib
import json
from pathlib import Path

from alloy.ingest.schema import QAPair


def write_manifest(eval_set: list[QAPair], out_path: Path) -> None:
    ids = sorted(p.question_id for p in eval_set)
    fingerprint = hashlib.sha256(",".join(map(str, ids)).encode("utf-8")).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"count": len(ids), "fingerprint": fingerprint, "question_ids": ids}, indent=2)
    )