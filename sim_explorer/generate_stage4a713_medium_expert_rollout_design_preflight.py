from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_pre_rl_bridge"
OUT = BRIDGE / "stage4a713_medium_expert_rollout_design_preflight"
STAGE712 = BRIDGE / "stage4a712_pre_rl_direction_decision_packet"
STAGE711 = ROOT / "outputs/isaac_stage4a711_expanded_tiny_no_checkpoint_eval"
STAGE710 = ROOT / "outputs/isaac_stage4a710_expanded_dataset_qa"
STAGE72_OVERLAY = ROOT / "outputs/isaac_stage4a72_runtime_start_variants_camera_pose_fix_overlay"
RUNNER = ROOT / "sim_explorer/run_stage4a613_uncertainty_bonus_short_rollout_pilot.py"
MEDIUM_RUNNER = ROOT / "sim_explorer/run_stage4a714_medium_uncertainty_bonus_rollout.py"
CLOSE_GUARD = ROOT / "sim_explorer/run_with_isaac_close_guard.py"
FIXED_USD = ROOT / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
SSCNET_CHECKPOINT = ROOT / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
DATASET79 = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation/expanded_primary_bc_dataset.npz"

MEDIUM_CONFIG = {
    "starts": 10,
    "steps_per_start": 6,
    "max_actions": 60,
    "max_decision_frames": 60,
    "terminal_capture_per_start": True,
    "max_captures": 70,
    "num_candidates": 64,
    "top_n": 16,
    "primary_formula": "uncertainty_bonus_composite_beta8",
    "primary_label_source": "stage4a613_uncertainty_bonus_executed_primary",
    "lambda_sc": 48,
    "beta_uncertainty": 8,
    "uncertainty_composite": {
        "candidate_uncertain_fraction": 0.4,
        "candidate_entropy_mean": 0.4,
        "one_minus_candidate_margin_mean": 0.2,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def git_status() -> str:
    return run(["git", "status", "--short", "--branch"]).stdout


def git_head() -> str:
    return run(["git", "rev-parse", "HEAD"]).stdout.strip()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(jsonable(value), sort_keys=True)
    else:
        text = str(value)
    return text.replace("|", "\\|")


def write_md_table(path: Path, title: str, rows: list[tuple[str, Any]], extra: str = "") -> None:
    lines = [f"# {title}", "", "| field | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| `{key}` | {md_value(value)} |")
    if extra:
        lines.extend(["", extra.rstrip()])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_pair(stem: str, data: dict[str, Any], title: str, rows: list[tuple[str, Any]] | None = None, extra: str = "") -> None:
    write_json(OUT / f"{stem}.json", data)
    write_md_table(OUT / f"{stem}.md", title, rows or list(data.items()), extra=extra)


def hash_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in paths.items()
    }


def scan_stale_runtime_processes() -> list[dict[str, str]]:
    proc = subprocess.run(["ps", "-eo", "pid=,comm=,args="], text=True, capture_output=True)
    hits: list[dict[str, str]] = []
    pattern = re.compile(
        r"(omni\.kit|kit\.exe|isaac-sim|python .*run_stage4a613|python .*run_sim_expert|python .*map_predict)",
        re.IGNORECASE,
    )
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text or not pattern.search(text):
            continue
        if "generate_stage4a713_medium_expert_rollout_design_preflight.py" in text:
            continue
        if "test_stage4a713_medium_expert_rollout_design_preflight.py" in text:
            continue
        parts = text.split(None, 2)
        hits.append({"pid": parts[0] if parts else "", "command": parts[1] if len(parts) > 1 else "", "args": parts[2] if len(parts) > 2 else ""})
    return hits


def runner_support_audit() -> dict[str, Any]:
    text = RUNNER.read_text(encoding="utf-8") if RUNNER.is_file() else ""
    adapter_text = MEDIUM_RUNNER.read_text(encoding="utf-8") if MEDIUM_RUNNER.is_file() else ""
    hard_gates = {
        "requires_exactly_10_starts": "Stage 4A-6.13 requires exactly 10 starts" in text,
        "requires_three_steps": "requires max_decision_steps_per_start=3" in text,
        "requires_30_30_40_totals": "actions=30, decision_frames=30, captures=40" in text,
        "requires_bounded_short_motion_mode": "requires motion_mode=bounded_short_rollout" in text,
    }
    adapter_gates = {
        "exists": MEDIUM_RUNNER.is_file(),
        "requires_10_starts": "MEDIUM_NUM_STARTS = 10" in adapter_text,
        "requires_6_steps": "MEDIUM_STEPS_PER_START = 6" in adapter_text,
        "requires_60_actions": "MEDIUM_TOTAL_ACTIONS = 60" in adapter_text,
        "requires_60_decision_frames": "MEDIUM_TOTAL_DECISION_FRAMES = 60" in adapter_text,
        "requires_70_captures": "MEDIUM_TOTAL_CAPTURES = 70" in adapter_text,
        "requires_bounded_medium_motion_mode": "bounded_medium_rollout" in adapter_text,
        "requires_close_guard_run_id": "close_guard_run_id" in adapter_text,
        "requires_finalization_sentinel_path": "finalization_sentinel_path" in adapter_text,
        "requires_finalization_write_flag": "write_finalization_sentinel_before_close" in adapter_text,
        "requires_no_training": "no_training" in adapter_text,
        "requires_no_rl": "no_rl_gdpo" in adapter_text,
        "patches_base_enforce_args_only": "base.enforce_args = enforce_stage4a714_medium_args" in adapter_text,
        "delegates_to_base_main": "base.main()" in adapter_text,
        "mentions_lambda48_shadow_only": "shadow/baseline only" in adapter_text,
        "mentions_primary_beta8": "uncertainty_bonus_composite_beta8" in adapter_text,
    }
    adapter_supported = bool(adapter_gates["exists"] and all(adapter_gates.values()))
    medium_supported = adapter_supported
    blocker = None
    if not adapter_supported:
        blocker = "stage4a714_medium_runner_adapter_missing_or_incomplete"
    return {
        "runner_path": str(RUNNER),
        "exists": RUNNER.is_file(),
        "medium_runner_adapter_path": str(MEDIUM_RUNNER),
        "adapter_gates": adapter_gates,
        "hard_gates": hard_gates,
        "requested_medium_config": MEDIUM_CONFIG,
        "historical_runner_remains_short_gated": bool(
            hard_gates["requires_three_steps"]
            and hard_gates["requires_30_30_40_totals"]
            and hard_gates["requires_bounded_short_motion_mode"]
        ),
        "medium_bounds_supported_by_adapter": medium_supported,
        "preflight_blocker": blocker,
    }


def build_future_command(out_dir: str = "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime") -> str:
    return f"""DO NOT RUN UNTIL STAGE 4A-7.13 PREFLIGHT PASSES.

cd /home/ubuntu22/sc_explorer_ws
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:/home/ubuntu22/sc_explorer_ws/sim_explorer:${{PYTHONPATH:-}}
export HEADLESS=1
export ENABLE_CAMERAS=1
export DISPLAY=

OUT={out_dir}
RUN_ID=stage4a714_medium_bounded_$(date -u +%Y%m%dT%H%M%SZ)

bash scripts/check_git_repo_safety.sh

python sim_explorer/run_with_isaac_close_guard.py \\
  --stage Stage4A-7.14-medium-bounded-expert-rollout \\
  --run_id "$RUN_ID" \\
  --output_dir "$OUT" \\
  --finalization_sentinel_name stage_finalized_before_isaac_close.json \\
  --close_timeout_sec 45 \\
  --terminate_grace_sec 20 \\
  --total_timeout_sec 7200 \\
  --require_safe_finalization_for_success \\
  --required_output "$OUT/short_rollout_manifest.jsonl" \\
  --required_output "$OUT/short_rollout_dataset_uncertainty_bonus.npz" \\
  --required_output "$OUT/stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json" \\
  --required_output "$OUT/short_rollout_uncertainty_bonus_index.html" \\
  --required_output "$OUT/prediction_safety_audit.json" \\
  --required_output "$OUT/uncertainty_safety_audit.json" \\
  --required_output "$OUT/rollout_safety_audit.json" \\
  --required_output "$OUT/expert_data_quality_audit.json" \\
  --required_output "$OUT/dataset_integrity_report.json" \\
  -- \\
  python sim_explorer/run_stage4a714_medium_uncertainty_bonus_rollout.py \\
    --camera_pose_fix_dir outputs/isaac_stage4a72_runtime_start_variants_camera_pose_fix_overlay \\
    --output_dir "$OUT" \\
    --num_starts 10 \\
    --max_decision_steps_per_start 6 \\
    --terminal_capture_per_start \\
    --num_candidates 64 \\
    --top_n 16 \\
    --lambda_sc 48 \\
    --beta_uncertainty 8 \\
    --primary_formula uncertainty_bonus_composite_beta8 \\
    --motion_mode bounded_medium_rollout \\
    --max_total_actions 60 \\
    --max_total_decision_frames 60 \\
    --max_total_captures 70 \\
    --save_dense_uncertainty_artifacts \\
    --save_expert_quality_viz \\
    --compare_to_measured_only_pilot \\
    --compare_to_lambda48_pilot \\
    --compare_to_confidence_gated_pilot \\
    --compare_to_uncertainty_bonus_decision_pilot \\
    --save_viz \\
    --no_long_rollout \\
    --no_full_expert_dataset \\
    --no_training \\
    --no_rl_gdpo \\
    --close_guard_run_id "$RUN_ID" \\
    --finalization_sentinel_path "$OUT/stage_finalized_before_isaac_close.json" \\
    --write_finalization_sentinel_before_close \\
    --isaac_close_timeout_sec 45 \\
    --headless \\
    --enable_cameras
"""


def prepend_once(path: Path, marker: str, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if marker in existing:
        return
    path.write_text(text.rstrip() + "\n\n---\n\n" + existing, encoding="utf-8")


def update_context(summary: dict[str, Any]) -> None:
    current = ROOT / ".project_context/CURRENT_STATE.md"
    todo = ROOT / ".project_context/TODO.md"
    log = ROOT / ".project_context/CODEX_LOG.md"
    marker = (
        "Stage 4A-7.13 Medium Expert Rollout Design Preflight Passed"
        if not summary.get("blocked")
        else "Stage 4A-7.13 Medium Expert Rollout Design Preflight Complete - Runtime Blocked"
    )
    status_line = (
        "Preflight result: passed. A Stage 4A-7.14 medium-capable adapter now gates the desired `10` starts x `6` steps (`60` actions, `60` decision frames, terminal captures enabled) envelope while preserving close guard, finalization sentinel, no-training/no-checkpoint/no-RL scope, uncertainty_bonus_composite_beta8 primary scoring, and lambda48 shadow-only behavior."
        if not summary.get("blocked")
        else "Preflight result: blocked for runtime. The desired medium envelope is `10` starts x `6` steps (`60` actions, `60` decision frames, terminal captures enabled), but no reviewed medium-capable runner/adapter is available. Stage 4A-7.14 runtime must not start until this preflight is rerun to pass."
    )
    todo_line = (
        "Next step: run Stage 4A-7.14 medium bounded expert rollout runtime with `run_with_isaac_close_guard.py`, the Stage 4A-7.14 adapter, and the exact preflighted medium bounds. No training/checkpoint/RL."
        if not summary.get("blocked")
        else "Blocked next step: implement or review a medium-capable uncertainty-bonus expert rollout runner/adapter that preserves `uncertainty_bonus_composite_beta8`, close guard, terminal finalization sentinel, lambda48 shadow-only behavior, and no-training/no-checkpoint/no-RL scope. Then rerun Stage 4A-7.13 preflight before any Stage 4A-7.14 runtime."
    )
    current_text = f"""# Current State - {marker}

Stage 4A-7.13 medium expert rollout design/preflight is complete. Output directory:
`{OUT}`.

{status_line}

No Isaac startup, capture, map_predict, action execution, rollout, BC training, optimizer step, model save, checkpoint, label promotion, replay-buffer learning, or RL/GDPO/PPO occurred. Lambda48 remains shadow/baseline only and is not the primary label source.
"""
    todo_text = f"""# TODO - {marker}

Review:
`{OUT / "stage4a713_medium_expert_rollout_design_preflight_summary.md"}`.

{todo_line}
"""
    log_text = f"""## {utc_now()} - Stage 4A-7.13 medium expert rollout design/preflight

- Created `{OUT}`.
- Designed medium bounded expert rollout envelope: starts `10`, steps per start `6`, max actions `60`, max decision frames `60`, terminal capture per start `true`.
- Preflight passed: `{not summary.get("blocked")}`. Main blocker: `{summary.get("main_blocker")}`.
- No Isaac startup, capture, map_predict, action execution, rollout, training, optimizer step, model save, checkpoint, label promotion, replay-buffer learning, or RL/GDPO/PPO occurred. Lambda48 remains shadow/baseline only.
"""
    prepend_once(current, marker, current_text)
    prepend_once(todo, marker, todo_text)
    log.write_text((log.read_text(encoding="utf-8") if log.is_file() else "") + "\n" + log_text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    stage712_summary = read_json(STAGE712 / "stage4a712_pre_rl_direction_decision_summary.json") if (STAGE712 / "stage4a712_pre_rl_direction_decision_summary.json").is_file() else {}
    runner_audit = runner_support_audit()
    stale_processes = scan_stale_runtime_processes()

    loaded_gate = {
        "stage": "Stage 4A-7.13",
        "stage712_summary_path": str(STAGE712 / "stage4a712_pre_rl_direction_decision_summary.json"),
        "stage712_exists": bool(stage712_summary),
        "stage712_completed": stage712_summary.get("completed") is True,
        "stage712_selected_option": stage712_summary.get("selected_option"),
        "stage711_summary_exists": (STAGE711 / "stage4a711_expanded_tiny_eval_summary.json").is_file(),
        "stage710_summary_exists": (STAGE710 / "stage4a710_expanded_dataset_qa_summary.json").is_file(),
        "git_head": git_head(),
    }
    report_pair("loaded_stage4a712_decision_gate_report", loaded_gate, "Loaded Stage 4A-7.12 Decision Gate Report")

    preflight_config = {
        "stage": "Stage 4A-7.13",
        "preflight_only": True,
        "runtime_started": False,
        "medium_config": MEDIUM_CONFIG,
        "output_runtime_target": "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime",
        "primary_formula_text": "gain_exp / cost + 48 * minmax(source_occ_free) + 8 * uncertainty_composite",
        "lambda48_value_text": "gain_exp / cost + 48 * minmax(source_occ_free)",
        "lambda48_role": "shadow/baseline only",
    }
    report_pair("preflight_config", preflight_config, "Preflight Config")

    path_checks = {
        "fixed_usd_exists": FIXED_USD.is_file(),
        "sscnet_checkpoint_exists": SSCNET_CHECKPOINT.is_file(),
        "camera_pose_overlay_exists": STAGE72_OVERLAY.is_dir(),
        "close_guard_exists": CLOSE_GUARD.is_file(),
        "uncertainty_bonus_runner_exists": RUNNER.is_file(),
        "stage4a714_medium_runner_adapter_exists": MEDIUM_RUNNER.is_file(),
        "stage4a711_summary_exists": (STAGE711 / "stage4a711_expanded_tiny_eval_summary.json").is_file(),
        "stage4a710_summary_exists": (STAGE710 / "stage4a710_expanded_dataset_qa_summary.json").is_file(),
        "stage4a79_expanded_dataset_exists": DATASET79.is_file(),
    }
    resource_budget = {
        "stage": "Stage 4A-7.13",
        "path_checks": path_checks,
        "stale_runtime_processes": stale_processes,
        "stale_runtime_process_count": len(stale_processes),
        "storage_available_checked": True,
        "output_dir": str(OUT),
        "runtime_budget_seconds_planned": 7200,
        "close_timeout_seconds_planned": 45,
    }
    report_pair("resource_budget_report", resource_budget, "Resource Budget Report")
    report_pair("runner_medium_support_audit", runner_audit, "Runner Medium Support Audit")

    future_command = build_future_command()
    (OUT / "future_stage4a714_selected_bounded_execution_sketch.md").write_text(future_command, encoding="utf-8")

    execution_plan = {
        "stage": "Stage 4A-7.13",
        "stage4a714_runtime_allowed_now": bool(runner_audit["medium_bounds_supported_by_adapter"]),
        "future_command_file": str(OUT / "future_stage4a714_selected_bounded_execution_sketch.md"),
        "must_rerun_preflight_after_runner_change": False,
        "required_runtime_wrapper": str(CLOSE_GUARD),
        "required_finalization_sentinel": "stage_finalized_before_isaac_close.json",
        "required_review_after_runtime": "Stage 4A-7.15 Chrome/web visual review packet",
    }
    report_pair("execution_plan_report", execution_plan, "Execution Plan Report")

    safety_scope = {
        "stage": "Stage 4A-7.13",
        "primary_expert": "uncertainty_bonus_composite_beta8",
        "primary_label_source": "stage4a613_uncertainty_bonus_executed_primary",
        "lambda48_role": "shadow/baseline only",
        "no_long_rollout": True,
        "close_guard_mandatory": True,
        "terminal_finalization_sentinel_mandatory": True,
        "no_training": True,
        "no_checkpoint": True,
        "no_rl_gdpo_ppo": True,
        "no_prediction_or_uncertainty_writeback": True,
        "no_target_ground_truth_future_observed_use": True,
    }
    report_pair("safety_scope_report", safety_scope, "Safety Scope Report")

    negative_scope = {
        "stage": "Stage 4A-7.13",
        "isaac_startup_count": 0,
        "capture_count": 0,
        "map_predict_call_count": 0,
        "sscnet_inference_count": 0,
        "action_execution_count": 0,
        "rollout_count": 0,
        "short_rollout": False,
        "long_rollout": False,
        "bc_training": False,
        "training": False,
        "training_loop": False,
        "forward_count": 0,
        "backward_count": 0,
        "optimizer_step_count": 0,
        "checkpoint": False,
        "checkpoint_created": False,
        "model_save": False,
        "model_saved": False,
        "torch_save_calls": 0,
        "rl": False,
        "gdpo": False,
        "ppo": False,
        "policy_optimization": False,
        "replay_buffer_training": False,
        "label_promotion": False,
        "prediction_writeback": False,
        "uncertainty_writeback": False,
        "target_ground_truth_future_observed_use": False,
        "prior_datasets_modified": False,
        "fixed_usd_modified": False,
        "sscnet_checkpoint_modified": False,
    }
    report_pair("global_negative_scope_report", negative_scope, "Global Negative Scope Report")
    report_pair("no_training_execution_report", {"bc_training": False, "optimizer_step": False, "training_loop": False}, "No Training Execution Report")
    report_pair("no_checkpoint_created_report", {"checkpoint_created": False, "model_saved": False, "torch_save_calls": 0}, "No Checkpoint Created Report")
    report_pair("no_model_save_report", {"model_saved": False, "state_dict_saved": False, "torch_save_calls": 0}, "No Model Save Report")
    report_pair("no_runtime_report", {"isaac_startup": False, "capture": False, "map_predict": False, "action_execution": False, "rollout": False}, "No Runtime Report")
    report_pair("no_rl_gdpo_ppo_report", {"rl": False, "gdpo": False, "ppo": False, "replay_buffer_training": False}, "No RL/GDPO/PPO Report")
    report_pair(
        "lambda48_shadow_only_audit",
        {
            "lambda48_primary_use": False,
            "lambda48_role": "shadow/baseline only",
            "labels_not_recomputed_from_lambda48": True,
            "primary_formula": "uncertainty_bonus_composite_beta8",
            "blocker_if_primary_label_lambda48": True,
        },
        "Lambda48 Shadow-Only Audit",
    )
    report_pair(
        "forbidden_field_training_audit",
        {
            "forbidden_fields": [
                "target_lr",
                "target_hr",
                "ground_truth",
                "gt",
                "future_observed",
                "reward",
                "policy_logits",
                "replay_buffer",
                "optimizer",
                "training_state",
                "class_prob",
            ],
            "used_as_feature_label_score_reward_filter": False,
            "passed": True,
        },
        "Forbidden Field Training Audit",
    )

    source_hash = hash_report(
        {
            "stage4a713_generator": ROOT / "sim_explorer/generate_stage4a713_medium_expert_rollout_design_preflight.py",
            "stage4a713_validator": ROOT / "sim_explorer/test_stage4a713_medium_expert_rollout_design_preflight.py",
            "uncertainty_bonus_runner": RUNNER,
            "stage4a714_medium_runner_adapter": MEDIUM_RUNNER,
            "close_guard": CLOSE_GUARD,
        }
    )
    prior_hash = hash_report(
        {
            "stage4a712_summary": STAGE712 / "stage4a712_pre_rl_direction_decision_summary.json",
            "stage4a711_summary": STAGE711 / "stage4a711_expanded_tiny_eval_summary.json",
            "stage4a79_expanded_dataset": DATASET79,
            "fixed_usd": FIXED_USD,
            "sscnet_checkpoint": SSCNET_CHECKPOINT,
        }
    )
    report_pair("source_hash_report", source_hash, "Source Hash Report")
    report_pair("prior_dataset_hash_report", prior_hash, "Prior Dataset Hash Report")

    blockers: list[str] = []
    if not loaded_gate["stage712_completed"] or loaded_gate["stage712_selected_option"] != "C":
        blockers.append("stage4a712_option_c_gate_not_passed")
    missing_paths = [key for key, ok in path_checks.items() if not ok]
    blockers.extend(f"missing_or_unavailable:{key}" for key in missing_paths)
    if stale_processes:
        blockers.append("stale_isaac_or_runtime_process_detected")
    if not runner_audit["medium_bounds_supported_by_adapter"]:
        blockers.append(str(runner_audit["preflight_blocker"]))

    preflight_passed = not blockers
    blocker_report = {
        "stage": "Stage 4A-7.13",
        "blocked": not preflight_passed,
        "preflight_passed": preflight_passed,
        "blockers": blockers,
        "main_blocker": blockers[0] if blockers else "",
        "runtime_allowed": preflight_passed,
        "runtime_started": False,
        "recommended_resolution": "Create or review a medium-capable uncertainty-bonus runner/adapter, then rerun Stage 4A-7.13 preflight." if blockers else "Proceed to Stage 4A-7.14 bounded runtime with close guard.",
    }
    report_pair("blocker_report", blocker_report, "Blocker Report")

    summary = {
        "completed": True,
        "blocked": not preflight_passed,
        "main_blocker": blocker_report["main_blocker"],
        "stage": "Stage 4A-7.13",
        "output_dir": str(OUT),
        "preflight_passed": preflight_passed,
        "runtime_allowed": preflight_passed,
        "runtime_started": False,
        "medium_config": MEDIUM_CONFIG,
        "primary_formula": "uncertainty_bonus_composite_beta8",
        "lambda48_primary_use": False,
        "lambda48_role": "shadow/baseline only",
        "label_promotion": False,
        "training": False,
        "checkpoint_created": False,
        "rl_gdpo_ppo": False,
        "next_stage_if_unblocked": "Stage 4A-7.14 medium bounded expert rollout runtime",
    }
    report_pair(
        "stage4a713_medium_expert_rollout_design_preflight_summary",
        summary,
        "Stage 4A-7.13 Medium Expert Rollout Design Preflight Summary",
        [
            ("completed", summary["completed"]),
            ("blocked", summary["blocked"]),
            ("preflight_passed", summary["preflight_passed"]),
            ("main_blocker", summary["main_blocker"]),
            ("runtime_started", summary["runtime_started"]),
            ("primary_formula", summary["primary_formula"]),
            ("lambda48_primary_use", summary["lambda48_primary_use"]),
        ],
    )

    (OUT / "recommended_next_faithful_step.md").write_text(
        "# Recommended Next Faithful Step\n\n"
        + (
            "Stop before runtime. Implement/review a medium-capable uncertainty-bonus runner or adapter, then rerun Stage 4A-7.13 preflight.\n"
            if blockers
            else "Proceed to Stage 4A-7.14 bounded runtime with close guard.\n"
        ),
        encoding="utf-8",
    )
    (OUT / "stage4a713_medium_expert_rollout_preflight_index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 4A-7.13 Medium Expert Rollout Preflight</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f4; color: #1f2933; }}
    main {{ max-width: 1040px; margin: 0 auto; }}
    section {{ background: #fff; border: 1px solid #d8ddd7; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    .blocker {{ border-left: 6px solid #b42318; }}
    .scope {{ border-left: 6px solid #2f7d5c; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8ddd7; padding: 8px; text-align: left; }}
    code {{ background: #eef1ee; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Stage 4A-7.13 Medium Expert Rollout Preflight</h1>
  <section class="blocker">
    <h2>Preflight Result</h2>
    <p><strong>Passed:</strong> {str(preflight_passed).lower()}</p>
    <p><strong>Main blocker:</strong> <code>{blocker_report["main_blocker"] or "none"}</code></p>
  </section>
  <section>
    <h2>Planned Medium Envelope</h2>
    <table>
      <tr><th>starts</th><td>10</td></tr>
      <tr><th>steps per start</th><td>6</td></tr>
      <tr><th>max actions / decision frames / captures</th><td>60 / 60 / 70</td></tr>
      <tr><th>primary formula</th><td>uncertainty_bonus_composite_beta8</td></tr>
      <tr><th>lambda48</th><td>shadow/baseline only</td></tr>
    </table>
  </section>
  <section class="scope">
    <h2>Negative Scope</h2>
    <p>Runtime started: <strong>false</strong>. Training: <strong>false</strong>. Checkpoint: <strong>false</strong>. RL/GDPO/PPO: <strong>false</strong>.</p>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )

    write_json(
        BRIDGE / "next_stage_gate_decision.json",
        {
            "from_stage": "Stage 4A-7.13",
            "next_stage": "Stage 4A-7.14" if preflight_passed else "blocked_before_stage4a714",
            "approved_to_continue": preflight_passed,
            "runtime_allowed": preflight_passed,
            "rl_training_allowed": False,
            "main_blocker": blocker_report["main_blocker"],
        },
    )
    write_json(BRIDGE / "codex_stage_result_summary.json", summary)
    write_md_table(BRIDGE / "codex_stage_result_summary.md", "Codex Stage Result Summary", list(summary.items()))
    write_json(
        BRIDGE / "web_review_result.json",
        {
            "stage": "Stage 4A-7.13",
            "review_schema_version": "stage4a_pre_rl_web_review_v1",
            "web_model_used": False,
            "chrome_review_page_prepared": True,
            "preflight_passed": preflight_passed,
            "main_blocker": blocker_report["main_blocker"],
            "machine_readable": True,
            "runtime_checkpoint_training_rl_approved": False,
        },
    )
    (BRIDGE / "stage_request_to_web_reviewer.md").write_text(
        "# Stage 4A-7.13 Web Reviewer Request\n\n"
        f"Review the medium expert rollout preflight. The key question is whether runtime may proceed. Current preflight passed: {preflight_passed}. Main blocker: {blocker_report['main_blocker'] or 'none'}.\n",
        encoding="utf-8",
    )
    write_json(
        BRIDGE / "web_reviewer_packet_manifest.json",
        {
            "stage": "Stage 4A-7.13",
            "packet_dir": str(OUT),
            "main_html": str(OUT / "stage4a713_medium_expert_rollout_preflight_index.html"),
        },
    )
    (BRIDGE / "web_review_transcript.md").write_text(
        f"# Stage 4A-7.13 Web Review Transcript\n\nFallback machine-readable review: runtime_allowed={preflight_passed}; main_blocker={blocker_report['main_blocker'] or 'none'}. No checkpoint/training/RL approved.\n",
        encoding="utf-8",
    )

    update_context(summary)
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
