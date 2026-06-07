#!/usr/bin/env python3
"""Bounded long uncertainty-bonus expert rollout adapter.

This narrow adapter keeps the historical Stage 4A-6.13 runner intact while
allowing the user-approved long expert envelope:

- 10 starts
- 15 decision steps per start
- at most 150 executed expert actions
- at most 150 decision frames
- 160 captures, counting one terminal capture per start

Primary expert scoring remains uncertainty_bonus_composite_beta8. Lambda48 is
kept only as a shadow/baseline. The adapter does not train, checkpoint, save a
model, promote labels, or run RL/GDPO/PPO.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
for _path in (
    WORKSPACE / "sim_explorer",
    WORKSPACE / "ssc_exploration",
    WORKSPACE / "ssc_exploration" / "ssc_network",
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import run_stage4a613_uncertainty_bonus_short_rollout_pilot as base
from isaac_lifecycle_guard import _audit_passed, validate_required_outputs


STAGE = "Stage LR-5-bounded-long-uncertainty-bonus-expert-rollout"
OUTPUT_NAME = "stage4a_long_bounded_expert_rollout_runtime"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs" / OUTPUT_NAME

LONG_NUM_STARTS = 10
LONG_STEPS_PER_START = 15
LONG_TOTAL_ACTIONS = 150
LONG_TOTAL_DECISION_FRAMES = 150
LONG_TOTAL_CAPTURES = 160

_orig_write_datasets_and_reports = base.write_datasets_and_reports
_orig_write_summary = base.write_summary


def enforce_stage4a_long_args(args: Any) -> None:
    required_flags = {
        "terminal_capture_per_start": args.terminal_capture_per_start,
        "save_dense_uncertainty_artifacts": args.save_dense_uncertainty_artifacts,
        "save_expert_quality_viz": args.save_expert_quality_viz,
        "compare_to_measured_only_pilot": args.compare_to_measured_only_pilot,
        "compare_to_lambda48_pilot": args.compare_to_lambda48_pilot,
        "compare_to_confidence_gated_pilot": args.compare_to_confidence_gated_pilot,
        "compare_to_uncertainty_bonus_decision_pilot": args.compare_to_uncertainty_bonus_decision_pilot,
        "save_viz": args.save_viz,
        "no_long_rollout": args.no_long_rollout,
        "no_full_expert_dataset": args.no_full_expert_dataset,
        "no_training": args.no_training,
        "no_rl_gdpo": args.no_rl_gdpo,
        "write_finalization_sentinel_before_close": args.write_finalization_sentinel_before_close,
    }
    missing = [key for key, value in required_flags.items() if not bool(value)]
    if missing:
        raise ValueError(f"Missing required bounded long boundary flags: {missing}")

    if int(args.num_starts) != LONG_NUM_STARTS:
        raise ValueError("Bounded long expert rollout requires exactly 10 starts")
    if int(args.max_decision_steps_per_start) != LONG_STEPS_PER_START:
        raise ValueError("Bounded long expert rollout requires max_decision_steps_per_start=15")
    if (
        int(args.max_total_actions) != LONG_TOTAL_ACTIONS
        or int(args.max_total_decision_frames) != LONG_TOTAL_DECISION_FRAMES
        or int(args.max_total_captures) != LONG_TOTAL_CAPTURES
    ):
        raise ValueError("Bounded long totals must be actions=150, decision_frames=150, captures=160")
    if int(args.num_candidates) != base.MAX_CANDIDATES or int(args.top_n) != 16:
        raise ValueError("Bounded long rollout uses num_candidates=64 and top_n=16")
    if str(args.primary_formula) != base.PRIMARY_FORMULA:
        raise ValueError(f"Unsupported primary formula: {args.primary_formula}")
    if float(args.lambda_sc) != base.LAMBDA_SC or float(args.beta_uncertainty) != base.BETA_UNCERTAINTY:
        raise ValueError("Bounded long rollout requires lambda_sc=48 and beta_uncertainty=8")
    if [float(v) for v in args.uncertainty_composite_weights] != [0.4, 0.4, 0.2]:
        raise ValueError("Bounded long rollout requires uncertainty composite weights 0.4 0.4 0.2")
    if str(args.motion_mode) != "bounded_long_expert_rollout":
        raise ValueError("Bounded long rollout requires motion_mode=bounded_long_expert_rollout")
    if not str(args.finalization_sentinel_path):
        raise ValueError("Bounded long rollout requires an explicit close-guard finalization sentinel path")
    if not str(args.close_guard_run_id):
        raise ValueError("Bounded long rollout requires an explicit close_guard_run_id")


def _copy_alias(src: Path, dst: Path) -> str | None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)
    return None


def _write_long_aliases(output_dir: Path) -> dict[str, Any]:
    aliases = {
        "dataset": _copy_alias(output_dir / "short_rollout_dataset_uncertainty_bonus.npz", output_dir / "long_rollout_dataset_uncertainty_bonus.npz"),
        "manifest": _copy_alias(output_dir / "short_rollout_manifest.jsonl", output_dir / "long_rollout_manifest.jsonl"),
        "html_index": _copy_alias(output_dir / "short_rollout_uncertainty_bonus_index.html", output_dir / "long_rollout_uncertainty_bonus_index.html"),
    }
    base.save_json(output_dir / "long_rollout_alias_manifest.json", aliases)
    base.write_text(
        output_dir / "long_rollout_alias_manifest.md",
        "# Long Rollout Alias Manifest\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in aliases.items()),
    )
    return aliases


def _long_rollout_safety(args: Any, bundle: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    per_start_rows = bundle["per_start_rows"]
    transition_rows = bundle["transition_rows"]
    source_hash_report = reports["source_hash_report"]
    max_actions_per_start = max((int(row.get("executed_action_count", 0)) for row in per_start_rows), default=0)
    safety = {
        "stage": STAGE,
        "start_count": int(args.num_starts),
        "max_decision_steps_per_start": int(args.max_decision_steps_per_start),
        "executed_action_count": int(bundle["total_executed_actions"]),
        "decision_frame_count": int(bundle["total_decision_frames"]),
        "terminal_frame_count": int(bundle["total_terminal_frames"]),
        "capture_count": int(bundle["total_captures"]),
        "map_predict_calls": int(bundle["map_predict_calls"]),
        "max_actions_per_start": max_actions_per_start,
        "total_action_limit_respected": int(bundle["total_executed_actions"]) <= LONG_TOTAL_ACTIONS,
        "per_start_step_limit_respected": max_actions_per_start <= LONG_STEPS_PER_START,
        "bounded_long_expert_rollout_executed": True,
        "unbounded_rollout_executed": False,
        "policy_long_rollout_executed": False,
        "learned_policy_control": False,
        "full_expert_dataset_executed": False,
        "training": False,
        "bc_il_rl_gdpo_ppo": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "label_promotion": False,
        "lambda48_primary": False,
        "lambda48_role": "shadow/baseline only",
        "primary_formula": base.PRIMARY_FORMULA,
        "source_usd_modified": source_hash_report["source_usd_sha256_before"] != source_hash_report["source_usd_sha256_after"],
        "fixed_usd_modified": source_hash_report["fixed_usd_sha256_before"] != source_hash_report["fixed_usd_sha256_after"],
        "checkpoint_modified": not bool(bundle["predictor"].checkpoint_unchanged()),
        "no_valid_candidate_count": sum(1 for row in transition_rows if row.get("no_valid_candidate")),
        "stuck_revisit_count": sum(1 for row in transition_rows if row.get("same_cell_target")),
        "candidate_all_local_count": sum(1 for row in transition_rows if row.get("candidate_all_local")),
        "low_cost_artifact_count": sum(1 for row in transition_rows if row.get("low_cost_artifact")),
        "historical_prior_basin_count": sum(1 for row in transition_rows if row.get("historical_prior_basin")),
        "prediction_uncertainty_writeback": False,
        "target_ground_truth_future_observed_use": False,
        "legacy_short_rollout_audit_replaced_by_long_aware_audit": True,
    }
    safety["passed"] = bool(
        safety["start_count"] == LONG_NUM_STARTS
        and safety["max_decision_steps_per_start"] == LONG_STEPS_PER_START
        and safety["executed_action_count"] <= LONG_TOTAL_ACTIONS
        and safety["decision_frame_count"] <= LONG_TOTAL_DECISION_FRAMES
        and safety["terminal_frame_count"] <= LONG_NUM_STARTS
        and safety["capture_count"] <= LONG_TOTAL_CAPTURES
        and safety["map_predict_calls"] <= LONG_TOTAL_DECISION_FRAMES
        and safety["total_action_limit_respected"]
        and safety["per_start_step_limit_respected"]
        and safety["bounded_long_expert_rollout_executed"]
        and not safety["unbounded_rollout_executed"]
        and not safety["policy_long_rollout_executed"]
        and not safety["learned_policy_control"]
        and not safety["training"]
        and not safety["bc_il_rl_gdpo_ppo"]
        and not safety["source_usd_modified"]
        and not safety["fixed_usd_modified"]
        and not safety["checkpoint_modified"]
    )
    return safety


def dataset_integrity_long(
    output_dir: Path,
    dataset_path: Path,
    prediction_safety: dict[str, Any],
    uncertainty_safety: dict[str, Any],
    rollout_safety: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Use the base dataset checks, replacing the legacy short n<=30 limit."""
    integrity = base.dataset_integrity(output_dir, dataset_path, prediction_safety, uncertainty_safety, rollout_safety, quality)
    long_required = [
        "long_rollout_dataset_uncertainty_bonus.npz",
        "long_rollout_manifest.jsonl",
        "long_rollout_uncertainty_bonus_index.html",
        "long_rollout_alias_manifest.json",
        "bounded_long_rollout_report.json",
    ]
    missing_long = [name for name in long_required if not (output_dir / name).is_file()]
    n = int(integrity.get("dataset_transition_count", 0))
    long_checks = {
        "legacy_short_rollout_n_le_30_replaced": True,
        "expected_transition_count": LONG_TOTAL_ACTIONS,
        "dataset_transition_count_matches_long_bound": n == LONG_TOTAL_ACTIONS,
        "long_required_outputs_present": not missing_long,
        "candidate_scores_finite": bool(integrity.get("candidate_scores_finite")),
        "dataset_exists": bool(integrity.get("dataset_exists")),
        "no_forbidden_dataset_keys": not integrity.get("forbidden_dataset_keys_present"),
        "prediction_safety_audit_passed": bool(prediction_safety.get("passed")),
        "uncertainty_safety_audit_passed": bool(uncertainty_safety.get("passed")),
        "rollout_safety_audit_passed": bool(rollout_safety.get("passed")),
        "expert_data_quality_audit_passed": bool(quality.get("passed")),
        "no_missing_per_start_outputs": not integrity.get("missing_per_start_outputs"),
    }
    integrity.update(
        {
            "legacy_short_rollout_limit_ignored_for_bounded_long": True,
            "expected_transition_count": LONG_TOTAL_ACTIONS,
            "missing_long_required_outputs": missing_long,
            "long_checks": long_checks,
            "passed": all(long_checks.values()),
        }
    )
    return integrity


def write_datasets_and_reports_long(args: Any, output_dir: Path, inputs: dict[str, Any], bundle: dict[str, Any], video_report: dict[str, Any]) -> dict[str, Any]:
    reports = _orig_write_datasets_and_reports(args, output_dir, inputs, bundle, video_report)
    rollout_safety = _long_rollout_safety(args, bundle, reports)
    base.save_report_pair(output_dir, "rollout_safety_audit", rollout_safety, "Bounded Long Rollout Safety Audit")
    base.save_report_pair(
        output_dir,
        "bounded_long_rollout_report",
        {
            "passed": rollout_safety["passed"],
            "bounded_long_expert_rollout_executed": True,
            "unbounded_rollout_executed": False,
            "policy_long_rollout_executed": False,
            "max_actions": LONG_TOTAL_ACTIONS,
            "max_steps_per_start": LONG_STEPS_PER_START,
            "primary_formula": base.PRIMARY_FORMULA,
            "lambda48_role": "shadow/baseline only",
        },
        "Bounded Long Rollout Report",
    )
    base.save_report_pair(
        output_dir,
        "no_training_checkpoint_rl_report",
        {
            "passed": True,
            "training": False,
            "optimizer_step": False,
            "checkpoint": False,
            "model_save": False,
            "label_promotion": False,
            "BC": False,
            "IL": False,
            "RL": False,
            "GDPO": False,
            "PPO": False,
        },
        "No Training Checkpoint RL Report",
    )
    integrity = dataset_integrity_long(output_dir, Path(reports["dataset_path"]), reports["prediction_safety"], reports["uncertainty_safety"], rollout_safety, reports["quality"])
    base.save_report_pair(output_dir, "dataset_integrity_report", integrity, "Bounded Long Dataset Integrity Report")
    reports["rollout_safety"] = rollout_safety
    reports["integrity"] = integrity
    reports["long_aliases"] = _write_long_aliases(output_dir)
    return reports


def write_summary_long(args: Any, output_dir: Path, inputs: dict[str, Any], bundle: dict[str, Any], reports: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    summary = _orig_write_summary(args, output_dir, inputs, bundle, reports, elapsed_s)
    summary.update(
        {
            "stage": STAGE,
            "completed": bool(reports["integrity"]["passed"]),
            "blocked": not bool(reports["integrity"]["passed"]),
            "main_blocker": "" if reports["integrity"]["passed"] else "bounded_long_dataset_integrity_failed",
            "bounded_long_expert_rollout_executed": True,
            "long_rollout_executed": True,
            "unbounded_rollout_executed": False,
            "policy_long_rollout_executed": False,
            "learned_policy_control": False,
            "full_expert_dataset_executed": False,
            "training": False,
            "bc_il_rl_gdpo_ppo": False,
            "checkpoint_created": False,
            "label_promotion": False,
            "start_count": LONG_NUM_STARTS,
            "max_decision_steps_per_start": LONG_STEPS_PER_START,
            "max_total_actions": LONG_TOTAL_ACTIONS,
            "max_total_decision_frames": LONG_TOTAL_DECISION_FRAMES,
            "max_total_captures": LONG_TOTAL_CAPTURES,
            "primary_formula": base.PRIMARY_FORMULA,
            "lambda48_role": "shadow/baseline only",
            "short_rollout_dataset": str(reports["dataset_path"]),
            "long_rollout_dataset": str(output_dir / "long_rollout_dataset_uncertainty_bonus.npz"),
            "manifest": str(output_dir / "long_rollout_manifest.jsonl"),
            "html_index": str(output_dir / "long_rollout_uncertainty_bonus_index.html"),
        }
    )
    base.save_json(output_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json", summary)
    base.write_text(output_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.md", base.markdown_table("Bounded Long Expert Rollout Summary", summary))
    base.save_json(output_dir / "long_rollout_summary.json", summary)
    base.write_text(output_dir / "long_rollout_summary.md", base.markdown_table("Bounded Long Expert Rollout Summary", summary))
    return summary


def maybe_write_long_finalization_sentinel(args: Any, output_dir: Path, reports: dict[str, Any]) -> dict[str, Any] | None:
    if not bool(getattr(args, "write_finalization_sentinel_before_close", False)) and not getattr(args, "finalization_sentinel_path", None):
        return None
    sentinel_path = Path(getattr(args, "finalization_sentinel_path", None) or (output_dir / "stage_finalized_before_isaac_close.json"))
    run_id = str(getattr(args, "close_guard_run_id", "") or f"stage4a_long_{int(time.time())}")
    dataset_path = output_dir / "long_rollout_dataset_uncertainty_bonus.npz"
    if not dataset_path.is_file():
        dataset_path = Path(reports["dataset_path"])
    manifest_path = output_dir / "long_rollout_manifest.jsonl"
    if not manifest_path.is_file():
        manifest_path = output_dir / "short_rollout_manifest.jsonl"
    summary_path = output_dir / "long_rollout_summary.json"
    html_path = output_dir / "long_rollout_uncertainty_bonus_index.html"
    if not html_path.is_file():
        html_path = output_dir / "short_rollout_uncertainty_bonus_index.html"
    mp4_candidate = output_dir / "short_rollout_flythrough.mp4"
    mp4_path = mp4_candidate if mp4_candidate.is_file() else None
    audit_paths = [
        output_dir / "expert_data_quality_audit.json",
        output_dir / "uncertainty_bonus_runtime_quality_audit.json",
        output_dir / "prediction_safety_audit.json",
        output_dir / "uncertainty_safety_audit.json",
        output_dir / "rollout_safety_audit.json",
        output_dir / "dataset_integrity_report.json",
        output_dir / "no_training_checkpoint_rl_report.json",
        output_dir / "bounded_long_rollout_report.json",
    ]
    required_outputs = [
        manifest_path,
        output_dir / "short_rollout_metadata.json",
        dataset_path,
        summary_path,
        html_path,
        *audit_paths,
    ]
    if mp4_path is not None:
        required_outputs.append(mp4_path)
    required = validate_required_outputs(required_outputs)
    audit_results = [_audit_passed(path) for path in audit_paths]
    audit_checks_passed = all(row.get("passed") for row in audit_results)
    safe = bool(required["required_output_checks_passed"] and audit_checks_passed)
    sentinel = {
        "stage": STAGE,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "timestamp_utc": base.utc_now(),
        "process_pid": __import__("os").getpid(),
        "required_output_checks_passed": required["required_output_checks_passed"],
        "required_output_count": required["required_output_count"],
        "required_outputs": required["required_outputs"],
        "required_output_hashes_or_sizes": required["required_outputs"],
        "missing_required_outputs": required["missing_required_outputs"],
        "empty_required_outputs": required["empty_required_outputs"],
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "dataset_path": str(dataset_path),
        "html_path": str(html_path),
        "mp4_path": str(mp4_path) if mp4_path else None,
        "audit_paths": [str(path) for path in audit_paths],
        "audit_results": audit_results,
        "audit_checks_passed": audit_checks_passed,
        "finalization_complete": True,
        "safe_to_terminate_after_close_timeout": safe,
        "reason": "bounded_long_outputs_finalized_before_simulation_app_close" if safe else "unsafe_to_terminate:bounded_long_outputs_finalized_before_simulation_app_close",
        "bounded_long_expert_rollout": True,
        "unbounded_rollout": False,
        "policy_long_rollout": False,
        "no_training": True,
        "no_checkpoint": True,
        "no_rl_gdpo": True,
        "no_prediction_writeback": True,
        "no_uncertainty_writeback": True,
        "no_label_promotion": True,
        "primary_formula": base.PRIMARY_FORMULA,
        "lambda48_role": "shadow/baseline only",
    }
    base.save_json(sentinel_path, sentinel)
    base.save_json(output_dir / "finalization_sentinel_write_report.json", sentinel)
    return sentinel


def main() -> None:
    base.STAGE = STAGE
    base.OUTPUT_NAME = OUTPUT_NAME
    base.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    base.enforce_args = enforce_stage4a_long_args
    base.write_datasets_and_reports = write_datasets_and_reports_long
    base.write_summary = write_summary_long
    base.maybe_write_close_guard_finalization_sentinel = maybe_write_long_finalization_sentinel
    base.main()


if __name__ == "__main__":
    main()
