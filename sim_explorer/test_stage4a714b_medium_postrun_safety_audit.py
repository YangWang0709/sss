from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/autonomous_pre_rl_bridge/stage4a714b_medium_postrun_safety_audit"
RUNTIME = ROOT / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"
SUMMARY = OUT / "stage4a714b_medium_postrun_safety_audit_summary.json"

REQUIRED_FILES = [
    "stage4a714b_medium_postrun_safety_audit_summary.json",
    "stage4a714b_medium_postrun_safety_audit_summary.md",
    "stage4a714b_medium_postrun_gate_matrix.json",
    "stage4a714b_medium_postrun_gate_matrix.md",
    "stage4a714b_medium_postrun_safety_audit_index.html",
    "git_status_before.txt",
    "git_status_after.txt",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def process_hits() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,pgid,stat,etime,cmd"], text=True, capture_output=True)
    hits = []
    for line in proc.stdout.splitlines():
        if "run_stage4a714_medium_uncertainty_bonus_rollout.py" in line:
            hits.append(line.strip())
        elif "run_with_isaac_close_guard.py" in line and "Stage4A-7.14" in line:
            hits.append(line.strip())
    return hits


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["output_dir_exists"] = OUT.is_dir()
    for name in REQUIRED_FILES:
        checks[f"required:{name}"] = (OUT / name).is_file()
    checks["summary_exists"] = SUMMARY.is_file()
    if not SUMMARY.is_file():
        blockers.append("missing_summary")
        print(json.dumps({"all_passed": False, "checks": checks, "blockers": blockers}, indent=2, sort_keys=True))
        return 1

    summary = load_json(SUMMARY)
    gate = load_json(OUT / "stage4a714b_medium_postrun_gate_matrix.json")
    tracked = git_files()

    checks["summary_passed"] = summary.get("completed") is True and summary.get("passed") is True and summary.get("blocked") is False
    checks["no_blockers"] = summary.get("blockers") == [] and summary.get("main_blocker") == ""
    checks["all_named_checks_true"] = all(summary.get("checks", {}).values()) and all(gate.get("checks", {}).values())
    counts = summary.get("medium_runtime_counts", {})
    checks["medium_counts_exact"] = (
        counts.get("starts") == 10
        and counts.get("executed_actions") == 60
        and counts.get("decision_frames") == 60
        and counts.get("captures") == 70
        and counts.get("map_predict_calls") == 60
        and counts.get("manifest_rows") == 60
        and counts.get("primary_rows") == 60
        and line_count(RUNTIME / "short_rollout_manifest.jsonl") == 60
        and line_count(RUNTIME / "primary_uncertainty_bonus_decisions.jsonl") == 60
    )
    manifest = summary.get("manifest_audit", {})
    checks["manifest_flags_safe"] = (
        manifest.get("stage_mismatch_count") == 0
        and manifest.get("forbidden_flag_counts") == {}
        and manifest.get("missing_referenced_path_count") == 0
        and manifest.get("nonfinite_action_rows") == []
        and manifest.get("observed_ratio_regressions") == []
        and manifest.get("no_valid_candidate_count") == 0
        and manifest.get("out_of_bounds_target_count") == 0
        and manifest.get("total_newly_observed", 0) > 0
    )
    per_start = summary.get("per_start_audit", {})
    checks["per_start_bounds_safe"] = (
        per_start.get("all_start_rows_expected") is True
        and per_start.get("all_executed_action_count_6") is True
        and per_start.get("all_decision_frame_count_6") is True
        and per_start.get("all_terminal_frame_count_1") is True
        and per_start.get("all_capture_count_7") is True
        and per_start.get("any_early_stop") is False
        and per_start.get("terminal_observed_state_missing") == []
    )
    primary = summary.get("primary_policy_audit", {})
    checks["primary_policy_beta8_not_lambda48"] = (
        primary.get("all_formula_beta8") is True
        and primary.get("source_occ_free_kept_separate") is True
        and primary.get("no_invalid_candidates") is True
        and primary.get("no_low_cost_artifacts") is True
        and primary.get("no_historical_prior_basin") is True
        and primary.get("no_candidate_all_local") is True
        and summary.get("negative_scope", {}).get("lambda48_primary") is False
        and (RUNTIME / "lambda48_shadow_decisions.jsonl").is_file()
    )
    quality = summary.get("quality_audit_inputs", {})
    checks["quality_inputs_passed"] = (
        quality.get("expert_quality_passed") is True
        and quality.get("uncertainty_bonus_quality_passed") is True
        and quality.get("prediction_safety_passed") is True
        and quality.get("uncertainty_safety_passed") is True
    )
    legacy = summary.get("legacy_short_bound_audit_handling", {})
    checks["legacy_mismatch_is_expected_only"] = (
        legacy.get("legacy_rollout_safety_passed") is False
        and legacy.get("legacy_dataset_integrity_passed") is False
        and legacy.get("legacy_mismatch_expected") is True
    )
    finalization = summary.get("finalization_and_close", {})
    checks["finalization_and_manual_close_safe"] = (
        finalization.get("sentinel_required_outputs_passed") is True
        and finalization.get("sentinel_audit_checks_passed") is False
        and finalization.get("sentinel_safe_to_terminate_after_close_timeout") is False
        and finalization.get("close_hung_after_finalization") is True
        and finalization.get("manual_termination_clean") is True
        and finalization.get("live_stage4a714_processes") == []
        and process_hits() == []
    )
    negative = summary.get("negative_scope", {})
    checks["negative_scope_enforced"] = (
        negative.get("training") is False
        and negative.get("checkpoint") is False
        and negative.get("model_save") is False
        and negative.get("rl") is False
        and negative.get("gdpo") is False
        and negative.get("ppo") is False
        and negative.get("label_promotion") is False
        and negative.get("long_rollout") is False
        and negative.get("prediction_uncertainty_writeback") is False
        and summary.get("actual_rl_training_allowed") is False
    )
    checks["outputs_logs_checkpoints_not_tracked"] = (
        not any(path.startswith(("outputs/", "logs/")) for path in tracked)
        and not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked)
    )

    for key, passed in checks.items():
        if not passed:
            blockers.append(key)
    result = {"all_passed": not blockers, "checks": checks, "blockers": blockers}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
