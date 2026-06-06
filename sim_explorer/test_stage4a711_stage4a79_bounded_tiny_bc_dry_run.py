#!/usr/bin/env python3
"""Validate Stage 4A-7.11 bounded tiny BC dry-run gate."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/stage4a711_stage4a79_bounded_tiny_bc_dry_run"
STAGE79 = ROOT / "outputs/stage4a79_stage4a714_compatible_no_training_import"
SCRIPT = ROOT / "sim_explorer/run_stage4a711_stage4a79_bounded_tiny_bc_dry_run.py"

REQUIRED_FILES = [
    "stage4a711_stage4a79_tiny_bc_dry_run_summary.json",
    "stage4a711_stage4a79_tiny_bc_dry_run_summary.md",
    "stage4a711_stage4a79_tiny_bc_dry_run_index.html",
    "stage4a711_tiny_bc_config.json",
    "dataset_load_report.json",
    "dataset_load_report.md",
    "split_plan_report.json",
    "forward_only_smoke_report.json",
    "tiny_train_metrics.json",
    "tiny_train_metrics.csv",
    "eval_metrics.json",
    "eval_metrics.csv",
    "label_lineage_report.json",
    "safety_no_checkpoint_report.json",
    "safety_no_checkpoint_report.md",
    "no_runtime_report.json",
    "no_runtime_report.md",
    "no_rl_gdpo_ppo_report.json",
    "no_rl_gdpo_ppo_report.md",
    "source_hash_report.json",
    "stage4a711_readiness_decision.json",
    "recommended_next_step.md",
    "git_status_before.txt",
    "git_status_after.txt",
]


def load_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def checkpoint_like_outputs() -> list[str]:
    hits = []
    if not OUT.exists():
        return hits
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(OUT))
        lower = rel.lower()
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".tar"}:
            hits.append(rel)
        elif any(token in lower for token in ["model_weights", "optimizer_state", "state_dict", "replay_buffer"]):
            hits.append(rel)
    return sorted(hits)


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def main() -> int:
    checks: dict[str, bool] = {"output_dir_exists": OUT.is_dir()}
    blockers: list[str] = []
    for name in REQUIRED_FILES:
        checks[f"required:{name}"] = (OUT / name).is_file()

    if not all(checks.values()):
        blockers = [name for name, ok in checks.items() if not ok]
        print(json.dumps({"all_passed": False, "checks": checks, "blockers": blockers}, indent=2, sort_keys=True))
        return 1

    summary = load_json("stage4a711_stage4a79_tiny_bc_dry_run_summary.json")
    config = load_json("stage4a711_tiny_bc_config.json")
    dataset = load_json("dataset_load_report.json")
    split = load_json("split_plan_report.json")
    forward = load_json("forward_only_smoke_report.json")
    tiny = load_json("tiny_train_metrics.json")
    eval_metrics = load_json("eval_metrics.json")
    lineage = load_json("label_lineage_report.json")
    safety = load_json("safety_no_checkpoint_report.json")
    no_runtime = load_json("no_runtime_report.json")
    no_rl = load_json("no_rl_gdpo_ppo_report.json")
    source_hash = load_json("source_hash_report.json")
    decision = load_json("stage4a711_readiness_decision.json")
    train_rows = read_csv("tiny_train_metrics.csv")
    eval_rows = read_csv("eval_metrics.csv")
    html = (OUT / "stage4a711_stage4a79_tiny_bc_dry_run_index.html").read_text(encoding="utf-8")
    script_text = SCRIPT.read_text(encoding="utf-8")
    expanded = np.load(STAGE79 / "stage4a79_stage4a714_compatible_expanded_dataset_55.npz", allow_pickle=False)

    labels = expanded["expert_action_index_primary"].astype(np.int64)
    valid = expanded["candidate_valid_mask"].astype(bool)
    tracked = git_files()

    checks.update(
        {
            "summary_completed": summary.get("completed") is True,
            "summary_not_blocked": summary.get("blocked") is False,
            "decision_passed": summary.get("decision") == "tiny_bc_dry_run_passed_no_checkpoint",
            "decision_not_checkpointed_training": decision.get("ready_for_checkpointed_training") is False,
            "decision_not_runtime": decision.get("ready_for_runtime") is False,
            "decision_not_rl": decision.get("ready_for_rl_gdpo_ppo") is False,
            "sample_count_55": summary.get("sample_count") == 55,
            "shape_55_64_16": list(expanded["candidate_features_model"].shape) == [55, 64, 16],
            "dataset_load_passed": dataset.get("dataset_len") == 55 and dataset.get("strict_keep_only_len") == 55,
            "finite_features": dataset.get("finite_features") is True,
            "valid_labels_dataset": dataset.get("valid_primary_labels") is True,
            "valid_labels_runtime": bool(np.all(valid[np.arange(labels.shape[0]), labels])),
            "split_counts": split.get("train_count") == 39 and split.get("val_count") == 10 and split.get("test_count") == 6,
            "stage4a710_ready_loaded": summary.get("stage4a710_readiness_decision") == "ready_for_tiny_bc_dry_run_consideration",
            "forward_before_finite": forward.get("aggregate", {}).get("metrics_finite") is True,
            "tiny_training_ran": tiny.get("ran") is True,
            "optimizer_steps_positive": int(tiny.get("optimizer_step_count", 0)) > 0,
            "optimizer_steps_bounded": int(tiny.get("optimizer_step_count", 999)) <= int(config.get("max_optimizer_steps", -1)) <= 8,
            "summary_step_counts_match": summary.get("optimizer_step_count") == tiny.get("optimizer_step_count") == safety.get("optimizer_step_count"),
            "backward_count_matches_steps": summary.get("backward_call_count") == summary.get("optimizer_step_count"),
            "tiny_losses_finite": tiny.get("all_losses_finite") is True and all(finite(row.get("loss")) for row in train_rows),
            "train_csv_step_count": len(train_rows) == int(tiny.get("optimizer_step_count", -1)),
            "eval_rows_expected": {row.get("split") for row in eval_rows} == {"train", "val", "test", "all", "imported_stage714", "original_stage613"},
            "eval_metrics_finite": all(
                report.get("aggregate", {}).get("metrics_finite") is True for report in eval_metrics.values()
            )
            and all(finite(row.get("loss")) and finite(row.get("top1")) and finite(row.get("top3")) for row in eval_rows),
            "full_training_false": summary.get("full_training") is False and config.get("full_training") is False and safety.get("full_training") is False,
            "model_saved_false": summary.get("model_saved") is False and safety.get("model_saved") is False,
            "checkpoint_created_false": summary.get("checkpoint_created") is False and safety.get("checkpoint_created") is False,
            "checkpoint_like_outputs_absent": checkpoint_like_outputs() == [],
            "script_no_torch_save": "torch.save" not in script_text,
            "script_checkpoint_suffixes_are_scan_only": "scan_checkpoint_like" in script_text and "torch.save" not in script_text,
            "no_isaac": no_runtime.get("isaac_startup") is False,
            "no_map_predict": no_runtime.get("map_predict") is False,
            "no_rollout": no_runtime.get("rollout") is False and no_runtime.get("long_rollout") is False,
            "no_runtime_execution": no_runtime.get("runtime_execution") is False,
            "no_rl_gdpo_ppo": no_rl.get("rl") is False and no_rl.get("gdpo") is False and no_rl.get("ppo") is False,
            "lambda48_shadow_only": lineage.get("lambda48_role") == "shadow/baseline only",
            "not_recomputed_from_lambda48": lineage.get("labels_recomputed_from_lambda48") is False,
            "source_hashes_unchanged": all(item.get("unchanged") is True for item in source_hash.values() if item.get("exists")),
            "html_mentions_stage": "Stage 4A-7.11" in html,
            "html_mentions_no_checkpoint": "No checkpoint" in html,
            "tracked_no_outputs": not any(path.startswith(("outputs/", "logs/", "checkpoints/")) for path in tracked),
            "tracked_no_model_artifacts": not any(path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked),
        }
    )

    blockers = [name for name, ok in checks.items() if not ok]
    result = {
        "all_passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "checkpoint_like_outputs": checkpoint_like_outputs(),
        "optimizer_step_count": summary.get("optimizer_step_count"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
