"""Prepare and run the V0 CPT -> SFT -> DPO pipeline on a cloud GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from alloy.training.cpt import train_cpt
from alloy.training.dpo import build_dpo_dataset, train_dpo
from alloy.training.sft import train_sft
from alloy.variants.assemble import assemble_v0


def _require_baseline(config: dict) -> dict:
    report_path = Path(config["paths"]["baseline_report"])
    if not report_path.is_file():
        raise RuntimeError(
            f"Phase 2 baseline report is missing: {report_path}. "
            "Run scripts/run_evaluate.py on the base model before V0 training."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_name = config["model"]["name"]
    result = report.get("models", {}).get(model_name)
    if not result or result.get("perplexity", {}).get("perplexity") is None:
        raise RuntimeError("Phase 2 baseline report does not contain a valid base-model PPL result")
    if result.get("general_capability", {}).get("accuracy") is None:
        raise RuntimeError("Phase 2 baseline report does not contain a valid MMLU result")
    eval_manifest_path = Path(config["paths"]["eval_manifest"])
    mmlu_manifest_path = Path(
        config["paths"].get("mmlu_manifest", "data/eval/mmlu_subset_manifest.json")
    )
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    mmlu_manifest = json.loads(mmlu_manifest_path.read_text(encoding="utf-8"))
    expected_ppl = min(
        int(config.get("evaluation", {}).get("perplexity_max_examples", 700)),
        int(eval_manifest["count"]),
    )
    expected_mmlu = min(
        int(config.get("evaluation", {}).get("mmlu_max_examples", 200)),
        int(mmlu_manifest["count"]),
    )
    if result["perplexity"].get("examples") != expected_ppl:
        raise RuntimeError("Phase 2 baseline report is incomplete: held-out PPL row count mismatch")
    if result["general_capability"].get("examples") != expected_mmlu:
        raise RuntimeError("Phase 2 baseline report is incomplete: MMLU row count mismatch")
    if report.get("held_out_sha256") != eval_manifest.get("fingerprint"):
        raise RuntimeError("Phase 2 report was evaluated on a different frozen held-out set")
    if report.get("mmlu_sha256") != mmlu_manifest.get("sha256"):
        raise RuntimeError("Phase 2 report was evaluated on a different MMLU subset")
    return report


def _prepare(config: dict) -> dict:
    paths = config["paths"]
    manifest = assemble_v0(
        paths["pool"], paths["v0_data"], eval_manifest_path=paths["eval_manifest"]
    )
    dpo_summary = build_dpo_dataset(
        paths["pool"],
        paths["dump"],
        Path(paths["v0_data"]) / "dpo.jsonl",
        eval_manifest_path=paths["eval_manifest"],
    )
    manifest["dpo"] = dpo_summary
    (Path(paths["v0_data"]) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared V0 inputs: {manifest['count']} QA pairs; {dpo_summary['preference_pairs']} DPO pairs")
    return manifest


def _run_evaluation(config_path: Path, base_model: str, adapter_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_evaluate.py"),
        "--config",
        str(config_path),
        "--model",
        base_model,
        "--model",
        str(adapter_path.resolve()),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0_baseline.yaml")
    parser.add_argument(
        "--stage", choices=("prepare", "cpt", "sft", "dpo", "all"), default="all"
    )
    parser.add_argument("--no-resume", action="store_true", help="Start the selected stage from scratch")
    parser.add_argument("--no-final-evaluation", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.stage == "prepare":
        _prepare(config)
        return

    run_all = args.stage == "all"
    try:
        _require_baseline(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    v0_data = Path(config["paths"]["v0_data"])
    cpt_path = Path(config["paths"]["checkpoints"]) / "cpt" / "final_adapter"
    sft_path = Path(config["paths"]["checkpoints"]) / "sft" / "final_adapter"
    resume = not args.no_resume
    result = {
        "variant": "v0",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "baseline_report": config["paths"]["baseline_report"],
        "completed_stages": [],
    }

    if run_all:
        _prepare(config)
        result["completed_stages"].append("prepare")
        args.stage = "cpt"
    if args.stage == "cpt":
        cpt_path = train_cpt(config, resume=resume)
        result["completed_stages"].append("cpt")
        if run_all:
            args.stage = "sft"
    if args.stage == "sft":
        if not cpt_path.is_dir():
            raise RuntimeError(f"CPT adapter missing: {cpt_path}")
        sft_path = train_sft(config, adapter_path=cpt_path, resume=resume)
        result["completed_stages"].append("sft")
        if run_all:
            args.stage = "dpo"
    if args.stage == "dpo":
        if not sft_path.is_dir():
            raise RuntimeError(f"SFT adapter missing: {sft_path}")
        dpo_path = train_dpo(config, adapter_path=sft_path, resume=resume)
        result["completed_stages"].append("dpo")
        if not args.no_final_evaluation:
            _run_evaluation(config_path, config["model"]["name"], dpo_path)
            result["completed_stages"].append("evaluation")

    output = Path(config["paths"]["checkpoints"]) / "phase3_v0_run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"V0 pipeline stages completed: {', '.join(result['completed_stages'])}")


if __name__ == "__main__":
    main()
