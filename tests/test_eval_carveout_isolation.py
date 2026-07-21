from alloy.ingest.carveout import carve_out_eval_set
from alloy.ingest.schema import QAPair


def _make_pair(qid: int, score: int) -> QAPair:
    return QAPair(
        question_id=qid, question_text="q", answer_id=qid + 100000, answer_text="a",
        question_score=score, answer_score=score, is_accepted=True,
        tags=(), created_at="2020-01-01",
    )


def test_eval_set_is_disjoint_from_remaining_pool():
    pairs = [_make_pair(i, score=i % 20) for i in range(500)]
    remaining, eval_set = carve_out_eval_set(
        pairs, target_size=100, score_bucket_edges=[0, 5, 10, 15, 20], seed=42
    )
    remaining_ids = {p.question_id for p in remaining}
    eval_ids = {p.question_id for p in eval_set}
    assert remaining_ids.isdisjoint(eval_ids)
    assert remaining_ids | eval_ids == {p.question_id for p in pairs}


def test_carveout_is_deterministic_across_runs():
    pairs = [_make_pair(i, score=i % 20) for i in range(500)]
    _, eval_a = carve_out_eval_set(pairs, 100, [0, 5, 10, 15, 20], seed=42)
    _, eval_b = carve_out_eval_set(pairs, 100, [0, 5, 10, 15, 20], seed=42)
    assert {p.question_id for p in eval_a} == {p.question_id for p in eval_b}