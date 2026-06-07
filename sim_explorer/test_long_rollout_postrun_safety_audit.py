#!/usr/bin/env python3
"""Validate LR-6 bounded long rollout postrun audits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "outputs/autonomous_long_rollout_bridge"
OUT = BRIDGE / "stage_lr6_postrun_safety_audit"
RUNTIME = ROOT / "outputs/stage4a_long_bounded_expert_rollout_runtime"
RESULT = OUT / "test_long_rollout_postrun_safety_audit_result.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    checks: dict[str, bool] = {}
    required = [
        "long_rollout_postrun_safety_audit_summary.json",
        "long_rollout_dataset_integrity_audit.json",
        "long_rollout_runtime_quality_audit.json",
        "close_guard_or_manual_termination_audit.json",
        "no_training_checkpoint_rl_report.json",
    ]
    for name in required:
        checks[f"{name}_exists"] = (OUT / name).is_file()

    summary = load(OUT / "long_rollout_postrun_safety_audit_summary.json")
    dataset = load(OUT / "long_rollout_dataset_integrity_audit.json")
    quality = load(OUT / "long_rollout_runtime_quality_audit.json")
    close_guard = load(OUT / "close_guard_or_manual_termination_audit.json")
    no_train = load(OUT / "no_training_checkpoint_rl_report.json")
    runtime_summary = load(RUNTIME / "long_rollout_summary.json")

    checks["summary_all_passed"] = summary.get("all_passed") is True
    checks["dataset_all_passed"] = dataset.get("all_passed") is True
    checks["quality_all_passed"] = quality.get("all_passed") is True
    checks["close_guard_all_passed"] = close_guard.get("all_passed") is True
    checks["no_training_all_passed"] = no_train.get("all_passed") is True
    checks["runtime_summary_exists"] = bool(runtime_summary)
    checks["runtime_completed_or_legacy_limit_explained"] = runtime_summary.get("completed") is True or dataset.get("legacy_short_rollout_limit_compatible") is True
    checks["expected_150_actions"] = int(dataset.get("summary_executed_action_count", -1)) == 150
    checks["expected_10_starts"] = int(dataset.get("summary_start_count", -1)) == 10
    checks["manifest_150_rows"] = int(dataset.get("manifest_rows", -1)) == 150
    checks["primary_beta8"] = quality.get("primary_formula") == "uncertainty_bonus_composite_beta8"
    checks["lambda48_shadow_only"] = quality.get("lambda48_role") == "shadow/baseline only" and no_train.get("lambda48_primary") is False
    checks["map_predict_scoring_only"] = quality.get("map_predict_role") == "bounded expert scoring path only"
    checks["no_live_runtime_process"] = not close_guard.get("live_runtime_process_matches")
    checks["close_guard_or_manual_termination"] = close_guard.get("supervisor_success") is True or close_guard.get("manual_termination_used") is True
    checks["no_training"] = no_train.get("training") is False
    checks["no_checkpoint"] = no_train.get("checkpoint") is False and not no_train.get("checkpoint_like_artifacts_under_runtime")
    checks["no_label_promotion"] = no_train.get("label_promotion") is False
    checks["no_rl_gdpo_ppo"] = no_train.get("RL_GDPO_PPO") is False

    blockers = [name for name, passed in checks.items() if not passed]
    result = {
        "stage": "LR-6 postrun safety audit validator",
        "checks": checks,
        "blockers": blockers,
        "all_passed": not blockers,
        "runtime_dir": str(RUNTIME),
        "postrun_audit_dir": str(OUT),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
