from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QAPair:
    question_id: int
    question_text: str
    answer_id: int
    answer_text: str
    question_score: int
    answer_score: int
    is_accepted: bool
    tags: tuple[str, ...]
    created_at: str