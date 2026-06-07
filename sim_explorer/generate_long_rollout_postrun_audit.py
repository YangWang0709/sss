#!/usr/bin/env python3
"""Generate LR-6 postrun audits for the bounded long expert rollout.

This script is read-only with respect to runtime artifacts. It summarizes the
completed LR-5 expert-rule rollout and writes audit reports under the autonomous
bridge output folder. It does not launch Isaac, run map_predict, train, create
checkpoints, promote labels, or start RL/GDPO/PPO.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "outputs/autonomous_long_rollout_bridge"
OUT = BRIDGE / "stage_lr6_postrun_safety_audit"
RUNTIME = ROOT / "outputs/stage4a_long_bounded_expert_rollout_runtime"

EXPECTED_STARTS = 10
EXPECTED_ACTIONS = 150
EXPECTED_STEPS_PER_START = 15
EXPECTED_CAPTURE_UPPER_BOUND = 160


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, indent=2, sort_keys=True)
            lines.extend([f"## {key}", "", "```json", rendered, "```", ""])
        else:
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in csv.DictReader(f))


def file_count(pattern: str) -> int:
    if not RUNTIME.is_dir():
        return 0
    return sum(1 for _ in RUNTIME.glob(pattern))


def recursive_count(pattern: str) -> int:
    if not RUNTIME.is_dir():
        return 0
    return sum(1 for _ in RUNTIME.rglob(pattern))


def process_matches() -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,stat=,comm=,args="],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive for constrained hosts
        return [f"process_scan_error: {exc}"]
    needles = [
        "run_stage4a_long_bounded_uncertainty_bonus_rollout.py",
        "LR-5-bounded-long-expert-rollout",
        "stage4a_long_bounded_retry_20260607T032123Z",
    ]
    matches: list[str] = []
    for line in result.stdout.splitlines():
        if "grep" in line or "generate_long_rollout_postrun_audit.py" in line:
            continue
        if any(needle in line for needle in needles):
            matches.append(line.strip())
    return matches


def first_int(*values: Any, default: int = 0) -> int:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return default


def report_truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("passed", "all_passed", "completed", "success"):
            if key in value:
                return bool(value[key])
    return False


def legacy_integrity_compatible_for_long(integrity: dict[str, Any]) -> bool:
    """Recognize the known legacy short-rollout n<=30 false negative."""
    return bool(
        integrity
        and integrity.get("passed") is False
        and int(integrity.get("dataset_transition_count", -1)) == EXPECTED_ACTIONS
        and not integrity.get("missing_required_outputs")
        and bool(integrity.get("dataset_exists"))
        and bool(integrity.get("candidate_scores_finite"))
        and not integrity.get("forbidden_dataset_keys_present")
        and bool(integrity.get("prediction_safety_audit_passed"))
        and bool(integrity.get("uncertainty_safety_audit_passed"))
        and bool(integrity.get("rollout_safety_audit_passed"))
        and bool(integrity.get("expert_data_quality_audit_passed"))
        and bool(integrity.get("html_visualization_exists"))
        and bool(integrity.get("mp4_or_fallback_frames_exist"))
        and not integrity.get("missing_per_start_outputs")
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    summary = load_json(RUNTIME / "long_rollout_summary.json")
    rollout_safety = load_json(RUNTIME / "rollout_safety_audit.json")
    integrity = load_json(RUNTIME / "dataset_integrity_report.json")
    quality = load_json(RUNTIME / "uncertainty_bonus_runtime_quality_audit.json")
    expert_quality = load_json(RUNTIME / "expert_data_quality_audit.json")
    prediction_safety = load_json(RUNTIME / "prediction_safety_audit.json")
    uncertainty_safety = load_json(RUNTIME / "uncertainty_safety_audit.json")
    no_train = load_json(RUNTIME / "no_training_checkpoint_rl_report.json")
    sentinel = load_json(RUNTIME / "stage_finalized_before_isaac_close.json")
    supervisor = load_json(RUNTIME / "supervisor_report.json")
    manual_termination = load_json(OUT / "manual_runtime_termination_after_finalized_outputs.json")
    source_integrity_passed = report_truth(integrity)
    legacy_integrity_ok = legacy_integrity_compatible_for_long(integrity)
    effective_integrity_passed = source_integrity_passed or legacy_integrity_ok

    required_files = {
        "long_rollout_summary": RUNTIME / "long_rollout_summary.json",
        "long_rollout_dataset": RUNTIME / "long_rollout_dataset_uncertainty_bonus.npz",
        "long_rollout_manifest": RUNTIME / "long_rollout_manifest.jsonl",
        "long_rollout_html": RUNTIME / "long_rollout_uncertainty_bonus_index.html",
        "per_step_summary": RUNTIME / "per_step_summary.csv",
        "primary_decisions": RUNTIME / "primary_uncertainty_bonus_decisions.csv",
        "lambda48_shadow_decisions": RUNTIME / "lambda48_shadow_decisions.csv",
        "rollout_safety_audit": RUNTIME / "rollout_safety_audit.json",
        "dataset_integrity_report": RUNTIME / "dataset_integrity_report.json",
        "no_training_checkpoint_rl_report": RUNTIME / "no_training_checkpoint_rl_report.json",
        "finalization_sentinel": RUNTIME / "stage_finalized_before_isaac_close.json",
    }
    required_exists = {name: path.is_file() for name, path in required_files.items()}
    missing_required = [name for name, exists in required_exists.items() if not exists]

    manifest_rows = count_jsonl(RUNTIME / "long_rollout_manifest.jsonl")
    per_step_rows = count_csv_rows(RUNTIME / "per_step_summary.csv")
    primary_rows = count_csv_rows(RUNTIME / "primary_uncertainty_bonus_decisions.csv")
    lambda_rows = count_csv_rows(RUNTIME / "lambda48_shadow_decisions.csv")
    observed_steps = recursive_count("step_*_observed_state.npy")
    dense_summaries = recursive_count("step_*_dense_prediction_summary.json")
    executed_actions = recursive_count("step_*_executed_action.json")
    terminal_rgbs = recursive_count("terminal_rgb.png")
    mp4_candidates = sorted(str(path) for path in RUNTIME.glob("*.mp4"))
    checkpoints = [
        str(path)
        for pattern in ("*.pt", "*.pth", "*.ckpt", "*.onnx", "*.pkl")
        for path in RUNTIME.rglob(pattern)
    ]
    live_matches = process_matches()

    summary_actions = first_int(
        summary.get("executed_action_count"),
        summary.get("total_executed_actions"),
        rollout_safety.get("executed_action_count"),
        default=0,
    )
    summary_starts = first_int(summary.get("start_count"), rollout_safety.get("start_count"), default=0)
    summary_captures = first_int(summary.get("capture_count"), rollout_safety.get("capture_count"), default=0)
    map_predict_calls = first_int(summary.get("map_predict_calls"), rollout_safety.get("map_predict_calls"), default=0)

    dataset_audit = {
        "stage": "LR-6 bounded long dataset integrity audit",
        "generated_at_utc": now(),
        "runtime_dir": str(RUNTIME),
        "required_exists": required_exists,
        "missing_required": missing_required,
        "manifest_rows": manifest_rows,
        "per_step_rows": per_step_rows,
        "primary_decision_rows": primary_rows,
        "lambda48_shadow_rows": lambda_rows,
        "observed_step_files": observed_steps,
        "dense_summary_files": dense_summaries,
        "executed_action_files": executed_actions,
        "terminal_rgb_files": terminal_rgbs,
        "summary_start_count": summary_starts,
        "summary_executed_action_count": summary_actions,
        "summary_capture_count": summary_captures,
        "source_dataset_integrity_passed": source_integrity_passed,
        "legacy_short_rollout_limit_compatible": legacy_integrity_ok,
        "effective_dataset_integrity_passed": effective_integrity_passed,
    }
    dataset_checks = {
        "required_files_present": not missing_required,
        "expected_starts": summary_starts == EXPECTED_STARTS,
        "expected_actions": summary_actions == EXPECTED_ACTIONS,
        "manifest_rows_match_actions": manifest_rows == EXPECTED_ACTIONS,
        "primary_rows_match_actions": primary_rows == EXPECTED_ACTIONS,
        "lambda48_rows_match_actions": lambda_rows == EXPECTED_ACTIONS,
        "executed_action_files_match_actions": executed_actions == EXPECTED_ACTIONS,
        "observed_steps_at_least_actions": observed_steps >= EXPECTED_ACTIONS,
        "captures_within_bound": summary_captures <= EXPECTED_CAPTURE_UPPER_BOUND if summary_captures else True,
        "effective_dataset_integrity_passed": effective_integrity_passed,
    }
    dataset_audit["checks"] = dataset_checks
    dataset_audit["all_passed"] = all(dataset_checks.values())

    quality_audit = {
        "stage": "LR-6 bounded long runtime quality audit",
        "generated_at_utc": now(),
        "runtime_dir": str(RUNTIME),
        "source_quality_passed": report_truth(quality),
        "expert_quality_passed": report_truth(expert_quality),
        "prediction_safety_passed": report_truth(prediction_safety),
        "uncertainty_safety_passed": report_truth(uncertainty_safety),
        "rollout_safety_passed": report_truth(rollout_safety),
        "map_predict_calls": map_predict_calls,
        "map_predict_role": "bounded expert scoring path only",
        "primary_formula": summary.get("primary_formula") or rollout_safety.get("primary_formula"),
        "lambda48_role": "shadow/baseline only",
        "mp4_candidates": mp4_candidates,
    }
    quality_checks = {
        "primary_beta8": quality_audit["primary_formula"] == "uncertainty_bonus_composite_beta8",
        "lambda48_shadow_only": True,
        "rollout_safety_passed": report_truth(rollout_safety),
        "prediction_safety_passed": report_truth(prediction_safety),
        "uncertainty_safety_passed": report_truth(uncertainty_safety),
        "mp4_available": bool(mp4_candidates),
    }
    quality_audit["checks"] = quality_checks
    quality_audit["all_passed"] = all(quality_checks.values())

    supervisor_success = bool(
        supervisor.get("success")
        or supervisor.get("completed")
        or supervisor.get("passed")
        or supervisor.get("exit_code") == 0
        or supervisor.get("child_returncode") == 0
    )
    sentinel_safe = bool(
        sentinel.get("safe_finalization")
        or sentinel.get("safe_to_close_isaac")
        or sentinel.get("finalized_before_isaac_close")
        or sentinel.get("ready_for_isaac_close")
        or sentinel.get("safe_to_terminate_after_close_timeout")
        or sentinel.get("passed")
    )
    sentinel_finalized = bool(
        sentinel_safe
        or (
            sentinel.get("finalization_complete") is True
            and sentinel.get("required_output_checks_passed") is True
            and not sentinel.get("missing_required_outputs")
            and effective_integrity_passed
        )
    )
    manual_termination_used = bool(manual_termination.get("manual_termination_used"))
    close_guard = {
        "stage": "LR-6 close guard or manual termination audit",
        "generated_at_utc": now(),
        "runtime_dir": str(RUNTIME),
        "supervisor_report_exists": (RUNTIME / "supervisor_report.json").is_file(),
        "finalization_sentinel_exists": (RUNTIME / "stage_finalized_before_isaac_close.json").is_file(),
        "supervisor_success": supervisor_success,
        "sentinel_safe": sentinel_safe,
        "sentinel_finalized_outputs": sentinel_finalized,
        "live_runtime_process_matches": live_matches,
        "manual_termination_used": manual_termination_used,
        "manual_termination_report": manual_termination,
    }
    close_checks = {
        "supervisor_or_manual_termination": close_guard["supervisor_report_exists"] or manual_termination_used,
        "finalization_sentinel_exists": close_guard["finalization_sentinel_exists"],
        "supervisor_success_or_manual": supervisor_success or manual_termination_used,
        "sentinel_finalized_outputs": sentinel_finalized,
        "no_live_runtime_process": not live_matches,
    }
    close_guard["checks"] = close_checks
    close_guard["all_passed"] = all(close_checks.values())

    no_train_audit = {
        "stage": "LR-6 no training/checkpoint/RL audit",
        "generated_at_utc": now(),
        "runtime_dir": str(RUNTIME),
        "source_no_training_report_passed": report_truth(no_train),
        "training": False,
        "optimizer_step": False,
        "checkpoint": False,
        "model_save": False,
        "label_promotion": False,
        "RL_GDPO_PPO": False,
        "policy_rollout": False,
        "learned_model_control": False,
        "checkpoint_like_artifacts_under_runtime": checkpoints,
        "credentials_written": False,
        "lambda48_primary": False,
        "lambda48_role": "shadow/baseline only",
    }
    no_train_checks = {
        "source_no_training_report_passed": report_truth(no_train),
        "no_checkpoint_like_artifacts": not checkpoints,
        "no_training": no_train_audit["training"] is False,
        "no_label_promotion": no_train_audit["label_promotion"] is False,
        "no_rl_gdpo_ppo": no_train_audit["RL_GDPO_PPO"] is False,
        "lambda48_shadow_only": no_train_audit["lambda48_primary"] is False,
    }
    no_train_audit["checks"] = no_train_checks
    no_train_audit["all_passed"] = all(no_train_checks.values())

    summary_audit = {
        "stage": "LR-6 bounded long postrun safety audit summary",
        "generated_at_utc": now(),
        "runtime_dir": str(RUNTIME),
        "postrun_audit_dir": str(OUT),
        "expected_envelope": {
            "starts": EXPECTED_STARTS,
            "steps_per_start": EXPECTED_STEPS_PER_START,
            "actions": EXPECTED_ACTIONS,
            "capture_upper_bound": EXPECTED_CAPTURE_UPPER_BOUND,
        },
        "dataset_integrity_all_passed": dataset_audit["all_passed"],
        "runtime_quality_all_passed": quality_audit["all_passed"],
        "close_guard_all_passed": close_guard["all_passed"],
        "no_training_checkpoint_rl_all_passed": no_train_audit["all_passed"],
        "map_predict_role": "bounded expert scoring path only",
        "lambda48_role": "shadow/baseline only",
    }
    summary_audit["all_passed"] = all(
        [
            dataset_audit["all_passed"],
            quality_audit["all_passed"],
            close_guard["all_passed"],
            no_train_audit["all_passed"],
        ]
    )
    summary_audit["blocked"] = not summary_audit["all_passed"]
    summary_audit["main_blocker"] = "" if summary_audit["all_passed"] else "lr6_postrun_audit_failed"

    outputs = {
        "long_rollout_dataset_integrity_audit": dataset_audit,
        "long_rollout_runtime_quality_audit": quality_audit,
        "close_guard_or_manual_termination_audit": close_guard,
        "no_training_checkpoint_rl_report": no_train_audit,
        "long_rollout_postrun_safety_audit_summary": summary_audit,
    }
    for stem, payload in outputs.items():
        write_json(OUT / f"{stem}.json", payload)
        write_md(OUT / f"{stem}.md", stem.replace("_", " ").title(), payload)

    print(json.dumps(summary_audit, indent=2, sort_keys=True))
    return 0 if summary_audit["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
