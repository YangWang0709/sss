from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_pre_rl_bridge"
OUT = BRIDGE / "stage4a714_medium_runtime_blocker_audit"
RUNTIME = ROOT / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"
SUMMARY = OUT / "stage4a714_medium_runtime_blocker_audit_summary.json"

REQUIRED_PACKET_FILES = [
    "stage4a714_medium_runtime_blocker_audit_summary.json",
    "stage4a714_medium_runtime_blocker_audit_summary.md",
    "stage4a714_medium_runtime_artifact_manifest.json",
    "stage4a714_medium_runtime_artifact_manifest.md",
    "stage4a714_legacy_short_bound_audit_mismatch_report.json",
    "stage4a714_legacy_short_bound_audit_mismatch_report.md",
    "stage4a714_no_training_no_checkpoint_no_rl_report.json",
    "stage4a714_no_training_no_checkpoint_no_rl_report.md",
    "stage4a714_recommended_next_faithful_step.md",
    "stage4a714_medium_runtime_blocker_audit_index.html",
    "git_status_before.txt",
    "git_status_after.txt",
]

REQUIRED_RUNTIME_FILES = [
    "short_rollout_manifest.jsonl",
    "primary_uncertainty_bonus_decisions.jsonl",
    "primary_uncertainty_bonus_decisions.csv",
    "lambda48_shadow_decisions.jsonl",
    "short_rollout_dataset_uncertainty_bonus.npz",
    "short_rollout_uncertainty_bonus_index.html",
    "short_rollout_flythrough.mp4",
    "expert_data_quality_audit.json",
    "uncertainty_bonus_runtime_quality_audit.json",
    "prediction_safety_audit.json",
    "uncertainty_safety_audit.json",
    "rollout_safety_audit.json",
    "dataset_integrity_report.json",
    "stage_finalized_before_isaac_close.json",
    "isaac_shutdown_report.json",
    "manual_process_termination_report.json",
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

    checks["packet_dir_exists"] = OUT.is_dir()
    checks["runtime_dir_exists"] = RUNTIME.is_dir()
    for name in REQUIRED_PACKET_FILES:
        checks[f"packet:{name}"] = (OUT / name).is_file()
    for name in REQUIRED_RUNTIME_FILES:
        checks[f"runtime:{name}"] = (RUNTIME / name).is_file()

    if not SUMMARY.is_file():
        blockers.append("missing_summary")
        result = {"all_passed": False, "checks": checks, "blockers": blockers}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    summary = load_json(SUMMARY)
    artifact_manifest = load_json(OUT / "stage4a714_medium_runtime_artifact_manifest.json")
    legacy = load_json(OUT / "stage4a714_legacy_short_bound_audit_mismatch_report.json")
    negative = load_json(OUT / "stage4a714_no_training_no_checkpoint_no_rl_report.json")
    rollout = load_json(RUNTIME / "rollout_safety_audit.json")
    dataset = load_json(RUNTIME / "dataset_integrity_report.json")
    sentinel = load_json(RUNTIME / "stage_finalized_before_isaac_close.json")
    shutdown = load_json(RUNTIME / "isaac_shutdown_report.json")
    manual_term = load_json(RUNTIME / "manual_process_termination_report.json")
    no_training = load_json(RUNTIME / "no_training_rl_bc_report.json")
    no_long = load_json(RUNTIME / "no_long_rollout_report.json")
    checkpoint = load_json(RUNTIME / "checkpoint_hash_report.json")
    source_hash = load_json(RUNTIME / "source_hash_report.json")
    expert_quality = load_json(RUNTIME / "expert_data_quality_audit.json")
    runtime_quality = load_json(RUNTIME / "uncertainty_bonus_runtime_quality_audit.json")
    prediction_safety = load_json(RUNTIME / "prediction_safety_audit.json")
    uncertainty_safety = load_json(RUNTIME / "uncertainty_safety_audit.json")

    checks["summary_completed_but_blocked"] = (
        summary.get("completed") is True
        and summary.get("blocked") is True
        and summary.get("pre_rl_ready") is False
        and summary.get("pre_rl_readiness_decision") == "not_ready_blocked"
    )
    checks["main_blocker_expected"] = summary.get("main_blocker") == "legacy_short_rollout_safety_audit_hardcoded_for_3_step_30_action_envelope"
    checks["medium_counts_match"] = (
        summary.get("runtime_counts", {}).get("executed_actions") == 60
        and summary.get("runtime_counts", {}).get("decision_frames") == 60
        and summary.get("runtime_counts", {}).get("captures") == 70
        and summary.get("runtime_counts", {}).get("manifest_rows") == 60
        and summary.get("runtime_counts", {}).get("primary_decision_rows") == 60
        and line_count(RUNTIME / "short_rollout_manifest.jsonl") == 60
        and line_count(RUNTIME / "primary_uncertainty_bonus_decisions.jsonl") == 60
    )
    checks["required_outputs_available"] = (
        summary.get("required_outputs_available") is True
        and all(item.get("exists") is True for item in artifact_manifest.get("required_outputs", {}).values())
    )
    checks["quality_core_passed"] = (
        summary.get("quality_core_passed") is True
        and expert_quality.get("passed") is True
        and runtime_quality.get("passed") is True
        and prediction_safety.get("passed") is True
        and uncertainty_safety.get("passed") is True
    )
    checks["legacy_short_audit_is_blocker"] = (
        summary.get("legacy_short_audit_failed") is True
        and legacy.get("blocker") == summary.get("main_blocker")
        and rollout.get("passed") is False
        and rollout.get("max_decision_steps_per_start") == 3
        and rollout.get("action_execution_exceeds_3_per_start") is True
        and rollout.get("total_action_limit_respected") is False
    )
    checks["dataset_failed_only_because_rollout_safety"] = (
        summary.get("dataset_failed_only_due_rollout_safety") is True
        and dataset.get("passed") is False
        and dataset.get("rollout_safety_audit_passed") is False
        and dataset.get("expert_data_quality_audit_passed") is True
        and dataset.get("prediction_safety_audit_passed") is True
        and dataset.get("uncertainty_safety_audit_passed") is True
        and dataset.get("missing_required_outputs") == []
    )
    checks["sentinel_and_close_hang_documented"] = (
        summary.get("sentinel_unsafe_due_audit_failure") is True
        and sentinel.get("required_output_checks_passed") is True
        and sentinel.get("audit_checks_passed") is False
        and sentinel.get("safe_to_terminate_after_close_timeout") is False
        and summary.get("close_hung_after_finalization") is True
        and shutdown.get("close_hung_after_finalization") is True
        and shutdown.get("close_returned") is False
    )
    checks["manual_termination_clean"] = (
        summary.get("manual_termination_clean") is True
        and manual_term.get("still_running_after_final_check") is False
        and manual_term.get("scope", {}).get("matched_only_stage4a714_runner") is True
        and manual_term.get("scope", {}).get("unrelated_process_killed") is False
        and process_hits() == []
    )
    checks["no_training_checkpoint_rl"] = (
        summary.get("no_training_checkpoint_rl") is True
        and negative.get("passed") is True
        and no_training.get("passed") is True
        and no_training.get("training") is False
        and no_training.get("BC") is False
        and no_training.get("RL") is False
        and no_training.get("GDPO") is False
        and no_training.get("PPO") is False
        and checkpoint.get("checkpoint_sha256_before") == checkpoint.get("checkpoint_sha256_after")
        and source_hash.get("source_usd_sha256_before") == source_hash.get("source_usd_sha256_after")
        and source_hash.get("fixed_usd_sha256_before") == source_hash.get("fixed_usd_sha256_after")
        and no_long.get("long_rollout_executed") is False
    )
    checks["lambda48_shadow_only"] = (
        summary.get("lambda48_primary_use") is False
        and summary.get("lambda48_not_primary") is True
        and negative.get("lambda48_primary") is False
        and (RUNTIME / "lambda48_shadow_decisions.jsonl").is_file()
        and (RUNTIME / "primary_uncertainty_bonus_decisions.jsonl").is_file()
    )
    tracked = git_files()
    checks["outputs_logs_checkpoints_not_tracked"] = (
        not any(path.startswith(("outputs/", "logs/")) for path in tracked)
        and not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked)
    )
    context = (ROOT / ".project_context/CURRENT_STATE.md").read_text(encoding="utf-8")[:2200]
    todo = (ROOT / ".project_context/TODO.md").read_text(encoding="utf-8")[:2200]
    log = (ROOT / ".project_context/CODEX_LOG.md").read_text(encoding="utf-8")[:2600]
    checks["context_updated"] = (
        "Stage 4A-7.14 Medium Runtime Blocker Audit Complete" in context
        and "Stage 4A-7.14 Medium Runtime Blocked" in todo
        and "Stage 4A-7.14 medium runtime blocker audit actions" in log
    )

    for key, passed in checks.items():
        if not passed:
            blockers.append(key)

    result = {"all_passed": not blockers, "checks": checks, "blockers": blockers}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
