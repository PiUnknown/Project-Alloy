from __future__ import annotations
import re
from xml.etree import ElementTree as ET

from alloy.ingest.schema import QAPair

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(body: str) -> str:
    """Lightweight HTML stripping. Known limitation: doesn't special-case
    <code> blocks or MathJax, so code/math-heavy posts read a bit mangled.
    Fine for Phase 0 -- revisit only if Phase 2 eval shows it's hurting."""
    return _TAG_RE.sub("", body).strip()


def _iter_rows(dump_path: str):
    for _, elem in ET.iterparse(dump_path, events=("end",)):
        if elem.tag == "row":
            yield elem.attrib
            elem.clear()  # release memory -- the whole point of iterparse


def parse_dump(dump_path: str) -> list[QAPair]:
    questions: dict[int, dict] = {}
    needed_answer_ids: set[int] = set()

    for attrib in _iter_rows(dump_path):
        if attrib.get("PostTypeId") != "1":
            continue
        accepted_id = attrib.get("AcceptedAnswerId")
        if accepted_id is None:
            continue  # Phase 0 scope: only questions with an accepted answer
        qid = int(attrib["Id"])
        questions[qid] = attrib
        needed_answer_ids.add(int(accepted_id))

    answers: dict[int, dict] = {}
    for attrib in _iter_rows(dump_path):
        if attrib.get("PostTypeId") != "2":
            continue
        aid = int(attrib["Id"])
        if aid in needed_answer_ids:
            answers[aid] = attrib

    pairs: list[QAPair] = []
    for qid, q in questions.items():
        accepted_id = int(q["AcceptedAnswerId"])
        answer = answers.get(accepted_id)
        if answer is None:
            continue  # accepted answer was deleted/missing from the dump

        tags = tuple(re.findall(r"<([^>]+)>", q.get("Tags", "")))
        pairs.append(
            QAPair(
                question_id=qid,
                question_text=_strip_html(q.get("Body", "")),
                answer_id=accepted_id,
                answer_text=_strip_html(answer.get("Body", "")),
                question_score=int(q.get("Score", 0)),
                answer_score=int(answer.get("Score", 0)),
                is_accepted=True,
                tags=tags,
                created_at=q.get("CreationDate", ""),
            )
        )
    return pairs