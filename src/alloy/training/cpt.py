"""Continued pretraining for V0 using 4-bit LoRA on the cloud GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alloy.training.common import (
    add_lora,
    load_json_dataset,
    load_training_model,
    require_training_stack,
    resume_if_available,
    save_final_adapter,
    training_arguments,
)


def train_cpt(config: dict[str, Any], *, resume: bool = True) -> Path:
    torch, _, _, Trainer, _, _, _ = require_training_stack()
    model_cfg = config["model"]
    train_cfg = config["training"]["cpt"]
    output_dir = Path(config["paths"]["checkpoints"]) / "cpt"
    dataset = load_json_dataset(Path(config["paths"]["v0_data"]) / "cpt.jsonl")
    model, tokenizer = load_training_model(
        model_cfg["name"], max_seq_length=int(model_cfg["max_seq_length"])
    )
    model = add_lora(model, seed=int(config["project"]["seed"]), lora=config["lora"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode_batch(batch):
        rows = []
        for text in batch["text"]:
            ids = tokenizer(
                text,
                truncation=True,
                max_length=int(model_cfg["max_seq_length"]),
                add_special_tokens=True,
            )["input_ids"]
            if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
                ids = ids[: int(model_cfg["max_seq_length"]) - 1] + [tokenizer.eos_token_id]
            rows.append(ids)
        return {"input_ids": rows}

    dataset = dataset.map(encode_batch, batched=True, remove_columns=dataset.column_names)
    from transformers import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_arguments(train_cfg, output_dir=output_dir, seed=int(config["project"]["seed"])),
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if resume:
        resume_if_available(trainer, output_dir)
    else:
        trainer.train()
    trainer.save_model(str(output_dir))
    return save_final_adapter(model, tokenizer, output_dir)
