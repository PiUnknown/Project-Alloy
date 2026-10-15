from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from alloy.evaluation.general_capability import evaluate_mmlu
from alloy.evaluation.llm_judge import compare_responses
from alloy.evaluation.perplexity import evaluate_perplexity
from alloy.training.dpo import build_dpo_dataset
from alloy.training.sft import encode_sft_example
from alloy.variants.assemble import assemble_v0
import json


class _FakeTokenizer:
    eos_token_id = 9
    im_end_id = 9
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False, return_tensors=None):
        if return_tensors == "pt":
            return torch.tensor([[1, 2]])
        if len(messages) == 1:
            return [1, 2, 3]
        return [1, 2, 3, 4, 5, 9]

    def decode(self, tokens, skip_special_tokens=True):
        return "B"


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 10))
        return type("Output", (), {"logits": logits})()

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[4]], device=input_ids.device)], dim=1)


def test_perplexity_scores_response_tokens_only():
    result = evaluate_perplexity(
        _FakeModel(),
        _FakeTokenizer(),
        [{"question_text": "question", "answer_text": "answer"}],
    )
    assert result["examples"] == 1
    assert result["response_tokens"] == 2
    assert result["perplexity"] == pytest.approx(10.0)


def test_pairwise_judge_balances_answer_position_and_maps_wins():
    prompts = []
    result = compare_responses(
        [{"question_text": f"q{i}"} for i in range(4)],
        ["left answer"] * 4,
        ["right answer"] * 4,
        lambda prompt: prompts.append(prompt) or "A: better",
        seed=7,
        left_name="v0",
        right_name="v1",
    )
    assert result["left_wins"] == 2
    assert result["right_wins"] == 2
    assert result["left_win_rate"] == pytest.approx(0.5)
    assert all("v0" not in prompt and "v1" not in prompt for prompt in prompts)


def test_mmlu_accuracy_uses_answer_choice_index():
    result = evaluate_mmlu(
        _FakeModel(),
        _FakeTokenizer(),
        [{"question": "q", "choices": ["a", "b", "c", "d"], "answer": 1}],
    )
    assert result["examples"] == 1
    assert result["answered"] == 1
    assert result["correct"] == 1
    assert result["accuracy"] == 1.0


def test_sft_preprocessing_masks_prompt_and_keeps_response():
    encoded = encode_sft_example(_FakeTokenizer(), "question", "answer", max_length=10)
    assert encoded["labels"][:3] == [-100, -100, -100]
    assert encoded["labels"][3:5] == [4, 5]
    assert encoded["input_ids"][-1] == 9


def test_v0_assembly_rejects_eval_ids(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps({"question_id": 1, "question_text": "q", "answer_text": "a"}) + "\n",
        encoding="utf-8",
    )
    eval_manifest = tmp_path / "manifest.json"
    eval_manifest.write_text(json.dumps({"question_ids": [2]}), encoding="utf-8")
    manifest = assemble_v0(pool, tmp_path / "v0", eval_manifest_path=eval_manifest)
    assert manifest["count"] == 1
    assert json.loads((tmp_path / "v0" / "sft.jsonl").read_text(encoding="utf-8"))["question_id"] == 1

    pool.write_text(
        json.dumps({"question_id": 2, "question_text": "leak", "answer_text": "leak"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leaked"):
        assemble_v0(pool, tmp_path / "leaky", eval_manifest_path=eval_manifest)


def test_targeted_dpo_extraction_selects_best_lower_scored_answer(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "question_id": 1,
                "question_text": "q",
                "answer_text": "accepted",
                "answer_id": 11,
                "answer_score": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"question_ids": [99]}), encoding="utf-8")
    dump = tmp_path / "Posts.xml"
    dump.write_text(
        '<posts><row PostTypeId="2" Id="11" ParentId="1" Score="5" Body="accepted" />'
        '<row PostTypeId="2" Id="12" ParentId="1" Score="4" Body="best rejected" />'
        '<row PostTypeId="2" Id="13" ParentId="1" Score="-1" Body="worse" /></posts>',
        encoding="utf-8",
    )
    output = tmp_path / "dpo.jsonl"
    counts = build_dpo_dataset(pool, dump, output, eval_manifest_path=manifest)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert counts["preference_pairs"] == 1
    assert row["rejected"] == "best rejected"
    assert row["rejected_score"] == 4
