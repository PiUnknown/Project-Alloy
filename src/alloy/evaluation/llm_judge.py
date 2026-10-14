"""Blinded, position-balanced pairwise judging utilities."""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Callable, Iterable


def openai_compatible_judge(
    prompt: str,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: int = 90,
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint using stdlib HTTP."""
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = json.dumps(
        {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Judge API request failed: {exc}") from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Judge API returned an unexpected chat-completions response") from exc


def _verdict(text: str) -> str:
    normalized = text.strip().upper()
    # Require a standalone label to avoid matching explanations such as "A is
    # not better". The model is instructed to return the label first.
    import re

    match = re.search(r"\b(A|B|TIE)\b", normalized)
    if not match:
        return "invalid"
    return {"A": "left", "B": "right", "TIE": "tie"}[match.group(1)]


def compare_responses(
    examples: Iterable[dict[str, Any]],
    left_responses: list[str],
    right_responses: list[str],
    judge: Callable[[str], str],
    *,
    seed: int = 42,
    left_name: str = "left",
    right_name: str = "right",
) -> dict[str, Any]:
    """Compare outputs blind with deterministic randomized presentation order."""
    rows = list(examples)
    if len(rows) != len(left_responses) or len(rows) != len(right_responses):
        raise ValueError("Examples and both response lists must have equal lengths")
    rng = random.Random(seed)
    presentation = [True] * ((len(rows) + 1) // 2) + [False] * (len(rows) // 2)
    rng.shuffle(presentation)
    counts: Counter[str] = Counter()
    left_wins = 0
    right_wins = 0
    details = []
    for index, (row, left, right, left_is_a) in enumerate(
        zip(rows, left_responses, right_responses, presentation)
    ):
        response_a, response_b = (left, right) if left_is_a else (right, left)
        prompt = (
            "Compare two answers to the same question. Prefer the answer that is more "
            "correct, relevant, clear, and complete. Ignore style differences that do not "
            "affect usefulness. The answers are anonymized and their order is randomized. "
            "Return exactly one label first: A, B, or TIE, followed by a brief reason.\n\n"
            f"Question:\n{row.get('question_text', row.get('question', ''))}\n\n"
            f"Answer A:\n{response_a}\n\nAnswer B:\n{response_b}"
        )
        vote = _verdict(judge(prompt))
        counts[vote] += 1
        winner = ("left" if left_is_a else "right") if vote == "left" else (
            ("right" if left_is_a else "left") if vote == "right" else vote
        )
        if winner == "left":
            left_wins += 1
        elif winner == "right":
            right_wins += 1
        details.append({"index": index, "winner": winner})

    valid = counts["left"] + counts["right"] + counts["tie"]
    return {
        "metric": "blind_pairwise_llm_judge",
        "left_model": left_name,
        "right_model": right_name,
        "examples": len(rows),
        "position_randomization_seed": seed,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": counts["tie"],
        "invalid": counts["invalid"],
        "left_win_rate": left_wins / valid if valid else None,
        "right_win_rate": right_wins / valid if valid else None,
        "details": details,
    }
