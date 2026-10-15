"""Targeted rejected-answer extraction and V0 Direct Preference Optimization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from alloy.ingest.parser import _strip_html
from alloy.training.common import (
    load_json_dataset,
    load_training_model,
    require_training_stack,
    save_final_adapter,
)
from alloy.variants.assemble import load_eval_question_ids, read_rows


def build_dpo_dataset(
    pool_path: str | Path,
    dump_path: str | Path,
    output_path: str | Path,
    *,
    eval_manifest_path: str | Path,
) -> dict[str, int]:
    """Find each V0 question's highest-scored strictly lower-scored answer.

    The raw XML is streamed, and only candidate answers for pool question IDs
    are retained. The frozen eval IDs are rejected before any extraction.
    """
    eval_ids = load_eval_question_ids(eval_manifest_path)
    chosen: dict[int, dict[str, Any]] = {}
    for row in read_rows(pool_path):
        qid = int(row["question_id"])
        if qid in eval_ids:
            raise ValueError(f"Frozen evaluation question {qid} leaked into the V0 DPO pool")
        chosen[qid] = {
            "question": str(row["question_text"]),
            "chosen": str(row["answer_text"]),
            "accepted_answer_id": int(row["answer_id"]),
            "chosen_score": int(row.get("answer_score", 0)),
        }

    rejected: dict[int, tuple[int, int, str]] = {}
    for _, elem in ET.iterparse(str(dump_path), events=("end",)):
        if elem.tag == "row" and elem.attrib.get("PostTypeId") == "2":
            parent_id = elem.attrib.get("ParentId")
            if parent_id is not None and int(parent_id) in chosen:
                qid = int(parent_id)
                answer_id = int(elem.attrib["Id"])
                score = int(elem.attrib.get("Score", 0))
                data = chosen[qid]
                if answer_id != data["accepted_answer_id"] and score < data["chosen_score"]:
                    previous = rejected.get(qid)
                    candidate = (score, answer_id, _strip_html(elem.attrib.get("Body", "")))
                    if previous is None or score > previous[0] or (
                        score == previous[0] and answer_id < previous[1]
                    ):
                        rejected[qid] = candidate
        if elem.tag == "row":
            elem.clear()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        for qid, data in chosen.items():
            candidate = rejected.get(qid)
            if candidate is None or not candidate[2].strip():
                continue
            row = {
                "question_id": qid,
                "prompt": data["question"],
                "chosen": data["chosen"],
                "rejected": candidate[2],
                "chosen_score": data["chosen_score"],
                "rejected_score": candidate[0],
                "rejected_answer_id": candidate[1],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if not count:
        raise ValueError("No lower-scored rejected answers found for V0 questions")
    return {"pool_questions": len(chosen), "preference_pairs": count}


def train_dpo(
    config: dict[str, Any], *, adapter_path: str | Path, resume: bool = True
) -> Path:
    torch, _, _, _, _, get_last_checkpoint, _ = require_training_stack()
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("DPO training requires TRL; install the training extra") from exc

    model_cfg = config["model"]
    train_cfg = config["training"]["dpo"]
    output_dir = Path(config["paths"]["checkpoints"]) / "dpo"
    dataset = load_json_dataset(Path(config["paths"]["v0_data"]) / "dpo.jsonl")
    model, tokenizer = load_training_model(
        model_cfg["name"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        adapter_path=adapter_path,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompts(batch):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question in batch["prompt"]
        ]
        return {"prompt": prompts}

    dataset = dataset.map(format_prompts, batched=True)
    bf16 = bool(torch.cuda.is_bf16_supported())
    args = DPOConfig(
        output_dir=str(output_dir),
        max_steps=int(train_cfg["max_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        warmup_steps=int(train_cfg.get("warmup_steps", 10)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=not bf16,
        bf16=bf16,
        optim="adamw_8bit",
        seed=int(config["project"]["seed"]),
        data_seed=int(config["project"]["seed"]),
        beta=float(train_cfg.get("beta", 0.1)),
        max_length=int(model_cfg["max_seq_length"]),
        max_prompt_length=int(model_cfg.get("max_prompt_length", 1024)),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    checkpoint = get_last_checkpoint(str(output_dir)) if resume else None
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir))
    return save_final_adapter(model, tokenizer, output_dir)
