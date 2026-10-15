"""Shared cloud-training setup and checkpoint/resume helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_training_stack():
    try:
        import torch
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import Trainer, TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "V0 training requires a Linux CUDA environment with Unsloth, Transformers, "
            "Datasets, PEFT, bitsandbytes, and PyTorch installed."
        ) from exc
    return torch, load_dataset, PeftModel, Trainer, TrainingArguments, get_last_checkpoint, FastLanguageModel


def load_training_model(
    base_model: str,
    *,
    max_seq_length: int,
    adapter_path: str | Path | None = None,
):
    """Load a 4-bit Unsloth model, optionally restoring a prior LoRA adapter."""
    torch, _, PeftModel, _, _, _, FastLanguageModel = require_training_stack()
    if not torch.cuda.is_available():
        raise RuntimeError("V0 full training is cloud GPU work; no CUDA device is available")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    model.config.use_cache = False
    return model, tokenizer


def add_lora(model: Any, *, seed: int, lora: dict[str, Any]):
    _, _, _, _, _, _, FastLanguageModel = require_training_stack()
    return FastLanguageModel.get_peft_model(
        model,
        r=int(lora["rank"]),
        target_modules=list(lora["target_modules"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.0)),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
    )


def load_json_dataset(path: str | Path):
    _, load_dataset, *_ = require_training_stack()
    return load_dataset("json", data_files=str(path), split="train")


def training_arguments(
    stage_config: dict[str, Any],
    *,
    output_dir: str | Path,
    seed: int,
):
    torch, _, _, _, TrainingArguments, *_ = require_training_stack()
    bf16 = bool(torch.cuda.is_bf16_supported())
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=int(stage_config["max_steps"]),
        learning_rate=float(stage_config["learning_rate"]),
        per_device_train_batch_size=int(stage_config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(stage_config["gradient_accumulation_steps"]),
        warmup_steps=int(stage_config.get("warmup_steps", 10)),
        weight_decay=float(stage_config.get("weight_decay", 0.01)),
        lr_scheduler_type=stage_config.get("lr_scheduler_type", "cosine"),
        logging_steps=int(stage_config.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(stage_config.get("save_steps", 100)),
        save_total_limit=int(stage_config.get("save_total_limit", 2)),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=not bf16,
        bf16=bf16,
        optim="adamw_8bit",
        seed=seed,
        data_seed=seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )


def resume_if_available(trainer: Any, output_dir: str | Path) -> None:
    *_, get_last_checkpoint, _ = require_training_stack()
    last = get_last_checkpoint(str(output_dir))
    trainer.train(resume_from_checkpoint=last if last else None)


def save_final_adapter(model: Any, tokenizer: Any, output_dir: str | Path) -> Path:
    final_dir = Path(output_dir) / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    return final_dir
