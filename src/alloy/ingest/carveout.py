from __future__ import annotations
import hashlib

from alloy.ingest.schema import QAPair


def _stable_rank(question_id: int, seed: int) -> float:
    """Deterministic pseudo-random rank in [0, 1) for a given id + seed.
    Same idea as consistent hashing: which bucket an item falls into
    depends only on the item's own identity and the seed, never on the
    order or size of the rest of the collection it's sitting in."""
    digest = hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF


def assign_score_bucket(score: int, edges: list[int]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= score < edges[i + 1]:
            return i
    return len(edges) - 2  # clamp anything at/above the top edge


def carve_out_eval_set(
    pairs: list[QAPair],
    target_size: int,
    score_bucket_edges: list[int],
    seed: int,
) -> tuple[list[QAPair], list[QAPair]]:
    """Returns (remaining_pool, eval_set). remaining_pool is listed first
    since it's what every later stage actually operates on."""
    num_buckets = len(score_bucket_edges) - 1
    buckets: dict[int, list[QAPair]] = {b: [] for b in range(num_buckets)}
    for pair in pairs:
        b = assign_score_bucket(pair.question_score, score_bucket_edges)
        buckets[b].append(pair)

    eval_ids: set[int] = set()
    for members in buckets.values():
        bucket_share = len(members) / len(pairs)
        bucket_target = round(target_size * bucket_share)
        ranked = sorted(members, key=lambda p: _stable_rank(p.question_id, seed))
        eval_ids.update(p.question_id for p in ranked[:bucket_target])

    eval_set = [p for p in pairs if p.question_id in eval_ids]
    remaining_pool = [p for p in pairs if p.question_id not in eval_ids]
    return remaining_pool, eval_set