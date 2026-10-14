"""Response-only perplexity evaluation for frozen Q&A examples."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON Lines without modifying the frozen source artifact."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def _encode_pair(tokenizer: Any, question: str, answer: str) -> tuple[list[int], list[int]]:
    """Return full chat tokens and the answer-token start using the model template."""
    prompt_messages = [{"role": "user", "content": question}]
    full_messages = prompt_messages + [{"role": "assistant", "content": answer}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    if full_ids and isinstance(full_ids[0], list):
        full_ids = full_ids[0]

    # Templates should preserve the generation prefix. Find the common prefix to
    # handle tokenizers that alter a boundary token when answer text is appended.
    start = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        start += 1
    if start == 0 or start >= len(full_ids):
        raise ValueError("Chat template did not produce a usable assistant answer span")

    # The assistant turn delimiter is formatting, not answer content.
    ending_ids = set(
        token_id
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "im_end_id", None),
        )
        if token_id is not None
    )
    end = len(full_ids)
    while end > start and full_ids[end - 1] in ending_ids:
        end -= 1
    if end <= start:
        raise ValueError("Example has no response tokens after applying chat template")
    return full_ids, [start, end]


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    examples: Iterable[dict[str, Any]],
    *,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Compute token-weighted cross entropy and perplexity on answer tokens only.

    Examples must expose ``question_text`` and ``answer_text`` (the frozen
    Alloy held-out schema). Prompt and chat-template tokens are ignored.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional ML extra
        raise RuntimeError("Perplexity requires PyTorch; install the evaluation extra") from exc

    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    count = 0
    try:
        with torch.inference_mode():
            for row in examples:
                if max_examples is not None and count >= max_examples:
                    break
                question = str(row["question_text"])
                answer = str(row["answer_text"])
                token_ids, (begin, end) = _encode_pair(tokenizer, question, answer)
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                logits = model(input_ids=input_ids).logits[:, :-1, :].float()
                targets = input_ids[:, 1:]
                # Causal-LM logit at position t predicts target at t+1.
                first_target = max(begin, 1)
                last_target = min(end, len(token_ids))
                if last_target <= first_target:
                    continue
                selected_logits = logits[:, first_target - 1 : last_target - 1, :]
                selected_targets = targets[:, first_target - 1 : last_target - 1]
                nll = F.cross_entropy(
                    selected_logits.reshape(-1, selected_logits.shape[-1]),
                    selected_targets.reshape(-1),
                    reduction="sum",
                )
                total_nll += float(nll.item())
                total_tokens += int(selected_targets.numel())
                count += 1
    finally:
        if was_training:
            model.train()

    if total_tokens == 0:
        raise ValueError("No response tokens were scored; check the eval data and tokenizer")
    mean_loss = total_nll / total_tokens
    return {
        "metric": "response_token_perplexity",
        "examples": count,
        "response_tokens": total_tokens,
        "mean_nll": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
    }
