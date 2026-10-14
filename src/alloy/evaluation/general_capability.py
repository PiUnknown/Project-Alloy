"""Evaluation on a fixed, local MMLU-style multiple-choice JSONL sample."""

from __future__ import annotations

import re
from typing import Any, Iterable


def _format_question(row: dict[str, Any]) -> str:
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError("MMLU rows need a choices list with at least two options")
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    rendered = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    subject = row.get("subject")
    header = f"The following are multiple choice questions about {subject}." if subject else (
        "Answer the following multiple choice question."
    )
    return (
        f"{header}\n\n{row['question']}\n{rendered}\n\n"
        "Reply with only the letter of the correct answer."
    )


def _extract_choice(text: str, n_choices: int) -> int | None:
    valid = "".join(chr(ord("A") + i) for i in range(min(n_choices, 26)))
    match = re.search(rf"(?:^|\b)([{valid}])(?:\b|[.)])", text.strip().upper())
    return ord(match.group(1)) - ord("A") if match else None


def evaluate_mmlu(
    model: Any,
    tokenizer: Any,
    examples: Iterable[dict[str, Any]],
    *,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Score fixed MMLU examples by constrained-format answer generation."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MMLU evaluation requires PyTorch; install the evaluation extra") from exc

    rows = list(examples)
    if max_examples is not None:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError("MMLU subset is empty")
    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    correct = 0
    answered = 0
    per_subject: dict[str, dict[str, int]] = {}
    try:
        for row in rows:
            if "question" not in row or "answer" not in row:
                raise ValueError("MMLU rows need question and answer fields")
            prompt = _format_question(row)
            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                output = model.generate(
                    input_ids=encoded,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output[0, encoded.shape[1] :], skip_special_tokens=True)
            predicted = _extract_choice(text, len(row["choices"]))
            expected = row["answer"]
            if isinstance(expected, str):
                expected = ord(expected.strip().upper()[0]) - ord("A")
            expected = int(expected)
            answered += int(predicted is not None)
            is_correct = predicted == expected
            correct += int(is_correct)
            subject = str(row.get("subject", "all"))
            stats = per_subject.setdefault(subject, {"correct": 0, "total": 0})
            stats["correct"] += int(is_correct)
            stats["total"] += 1
    finally:
        if was_training:
            model.train()

    return {
        "metric": "mmlu_accuracy",
        "examples": len(rows),
        "correct": correct,
        "answered": answered,
        "accuracy": correct / len(rows),
        "per_subject": {
            subject: {**stats, "accuracy": stats["correct"] / stats["total"]}
            for subject, stats in sorted(per_subject.items())
        },
    }
