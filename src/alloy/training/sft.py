"""Response-masked supervised fine-tuning for the V0 adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alloy.training.common import (
    load_json_dataset,
    load_training_model,
    require_training_stack,
    resume_if_available,
    save_final_adapter,
    training_arguments,
)


def encode_sft_example(
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize a chat pair and mask every prompt token from the loss."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if hasattr(full, "tolist"):
        full = full.tolist()
    if prompt and isinstance(prompt[0], list):
        prompt = prompt[0]
    if full and isinstance(full[0], list):
        full = full[0]
    response_start = 0
    for prompt_token, full_token in zip(prompt, full):
        if prompt_token != full_token:
            break
        response_start += 1
    if response_start == 0 or response_start >= len(full):
        raise ValueError("Tokenizer chat template did not preserve an assistant response span")

    # Keep the response intact whenever it fits. If the prompt is too long,
    # discard its oldest tokens first; if the answer itself exceeds the limit,
    # retain its final tokens rather than silently training on prompt tokens.
    if len(full) > max_length:
        if len(full) - response_start >= max_length:
            cut = len(full) - max_length
        else:
            cut = max(0, response_start - (max_length - (len(full) - response_start)))
        full = full[cut : cut + max_length]
        response_start = max(0, response_start - cut)
    labels = [-100] * response_start + full[response_start:]
    return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}


def response_masking_collator(tokenizer: Any):
    """Pad inputs and labels without allowing padding to contribute to loss."""
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        maximum = max(len(row["input_ids"]) for row in features)
        input_ids = []
        labels = []
        masks = []
        for row in features:
            padding = maximum - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            masks.append(row["attention_mask"] + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    return collate


def train_sft(config: dict[str, Any], *, adapter_path: str | Path, resume: bool = True) -> Path:
    _, _, _, Trainer, _, _, _ = require_training_stack()
    model_cfg = config["model"]
    train_cfg = config["training"]["sft"]
    output_dir = Path(config["paths"]["checkpoints"]) / "sft"
    dataset = load_json_dataset(Path(config["paths"]["v0_data"]) / "sft.jsonl")
    model, tokenizer = load_training_model(
        model_cfg["name"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        adapter_path=adapter_path,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(model_cfg["max_seq_length"])

    def encode_batch(batch):
        encoded = [
            encode_sft_example(tokenizer, q, a, max_length=max_length)
            for q, a in zip(batch["question"], batch["answer"])
        ]
        return {key: [row[key] for row in encoded] for key in encoded[0]}

    dataset = dataset.map(encode_batch, batched=True, remove_columns=dataset.column_names)
    trainer = Trainer(
        model=model,
        args=training_arguments(train_cfg, output_dir=output_dir, seed=int(config["project"]["seed"])),
        train_dataset=dataset,
        data_collator=response_masking_collator(tokenizer),
        processing_class=tokenizer,
    )
    if resume:
        resume_if_available(trainer, output_dir)
    else:
        trainer.train()
    trainer.save_model(str(output_dir))
    return save_final_adapter(model, tokenizer, output_dir)
