"""Run Alloy's frozen evaluation suite for one or more model checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from alloy.evaluation.general_capability import evaluate_mmlu
from alloy.evaluation.llm_judge import compare_responses, openai_compatible_judge
from alloy.evaluation.perplexity import evaluate_perplexity, read_jsonl


def _load_model(model_id: str, device: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model evaluation requires torch and transformers. Install the evaluation extra."
        ) from exc
    checkpoint = Path(model_id)
    adapter_config = checkpoint / "adapter_config.json"
    load_target = model_id
    is_adapter = checkpoint.is_dir() and adapter_config.is_file()
    if is_adapter:
        try:
            adapter_data = json.loads(adapter_config.read_text(encoding="utf-8"))
            load_target = adapter_data["base_model_name_or_path"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid PEFT adapter config at {adapter_config}") from exc
    tokenizer = AutoTokenizer.from_pretrained(load_target, trust_remote_code=False)
    model_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        load_target,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading a LoRA checkpoint requires peft; install evaluation extra") from exc
        model = PeftModel.from_pretrained(model, model_id, is_trainable=False)
    model.to(device)
    return model, tokenizer


def _generate(model, tokenizer, questions: list[str], *, max_new_tokens: int = 256) -> list[str]:
    import torch

    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    outputs = []
    try:
        with torch.inference_mode():
            for question in questions:
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
                generated = model.generate(
                    input_ids=encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                outputs.append(
                    tokenizer.decode(generated[0, encoded.shape[1] :], skip_special_tokens=True).strip()
                )
    finally:
        if was_training:
            model.train()
    return outputs


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.") or "model"


def _verify_manifest(data_path: Path, manifest_path: Path, *, ids_field: str | None) -> str:
    """Verify an evaluation artifact against its recorded content fingerprint."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if ids_field:
        rows = read_jsonl(data_path)
        ids = sorted(int(row[ids_field]) for row in rows)
        actual = hashlib.sha256(",".join(map(str, ids)).encode("utf-8")).hexdigest()
        expected_count = manifest.get("count")
    else:
        content = data_path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        actual = hashlib.sha256(content).hexdigest()
        expected_count = manifest.get("count")
    expected = manifest.get("fingerprint", manifest.get("sha256"))
    if expected_count is not None and expected_count != len(read_jsonl(data_path)):
        raise SystemExit(f"Evaluation artifact count does not match manifest: {data_path}")
    if actual != expected:
        raise SystemExit(f"Evaluation artifact fingerprint does not match manifest: {data_path}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--model", action="append", dest="models",
        help="Model ID or checkpoint path; repeat for pairwise checkpoint comparisons.",
    )
    parser.add_argument("--eval-set", help="Override the frozen held-out JSONL path.")
    parser.add_argument("--mmlu-set", help="Override the fixed MMLU subset JSONL path.")
    parser.add_argument("--output-dir", help="Override evaluation report directory.")
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--skip-judge", action="store_true", help="Skip pairwise judge calls.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings = config.get("evaluation", {})
    seed = int(settings.get("seed", config.get("project", {}).get("seed", 42)))
    held_out_path = Path(args.eval_set or settings.get("held_out_path", "data/eval/held_out.jsonl"))
    mmlu_path = Path(args.mmlu_set or settings.get("mmlu_path", "data/eval/mmlu_subset.jsonl"))
    output_dir = Path(args.output_dir or settings.get("output_path", "reports/evaluation"))
    model_ids = args.models or [settings.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")]
    judge_limit = int(settings.get("judge_max_examples", 100))
    ppl_limit = int(settings.get("perplexity_max_examples", 700))
    mmlu_limit = int(settings.get("mmlu_max_examples", 200))

    for required_path in (held_out_path, mmlu_path):
        if not required_path.is_file():
            raise SystemExit(f"Required evaluation artifact is missing: {required_path}")
    held_out = read_jsonl(held_out_path)
    mmlu_rows = read_jsonl(mmlu_path)
    if not held_out or not mmlu_rows:
        raise SystemExit("Evaluation inputs must not be empty")
    eval_manifest = held_out_path.with_name("manifest.json")
    mmlu_manifest = mmlu_path.with_name("mmlu_subset_manifest.json")
    if not eval_manifest.is_file() or not mmlu_manifest.is_file():
        raise SystemExit("Both eval manifest.json and mmlu_subset_manifest.json are required")
    held_out_fingerprint = _verify_manifest(held_out_path, eval_manifest, ids_field="question_id")
    mmlu_fingerprint = _verify_manifest(mmlu_path, mmlu_manifest, ids_field=None)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the evaluation dependencies (torch, transformers) first") from exc
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "project": "alloy",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "eval_set": str(held_out_path),
        "mmlu_set": str(mmlu_path),
        "held_out_sha256": held_out_fingerprint,
        "mmlu_sha256": mmlu_fingerprint,
        "models": {},
        "pairwise": [],
    }
    eval_questions = held_out[:judge_limit]
    outputs_by_model = {}
    for model_id in model_ids:
        print(f"Evaluating {model_id} on {device}...", flush=True)
        model, tokenizer = _load_model(model_id, device)
        try:
            results = {
                "perplexity": evaluate_perplexity(
                    model, tokenizer, held_out, max_examples=ppl_limit
                ),
                "general_capability": evaluate_mmlu(
                    model, tokenizer, mmlu_rows, max_examples=mmlu_limit
                ),
            }
            if not args.skip_judge and len(model_ids) > 1:
                outputs_by_model[model_id] = _generate(
                    model,
                    tokenizer,
                    [row.get("question_text", "") for row in eval_questions],
                )
            report["models"][model_id] = results
        finally:
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    api_key = os.environ.get("OPENAI_API_KEY")
    endpoint = settings.get("judge_endpoint", "https://api.openai.com/v1")
    judge_model = settings.get("judge_model", "gpt-4.1-mini")
    if args.skip_judge:
        report["pairwise_status"] = "skipped_by_request"
    elif len(model_ids) < 2:
        report["pairwise_status"] = "not_applicable_single_checkpoint"
    elif not api_key:
        report["pairwise_status"] = "skipped_missing_OPENAI_API_KEY"
    else:
        for left_id, right_id in itertools.combinations(model_ids, 2):
            verdict = compare_responses(
                eval_questions,
                outputs_by_model[left_id],
                outputs_by_model[right_id],
                lambda prompt: openai_compatible_judge(
                    prompt, endpoint=endpoint, api_key=api_key, model=judge_model
                ),
                seed=seed,
                left_name=left_id,
                right_name=right_id,
            )
            report["pairwise"].append(verdict)
        report["pairwise_status"] = "complete"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evaluation_{_safe_name(model_ids[0])}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote evaluation report: {output_path}")


if __name__ == "__main__":
    main()
