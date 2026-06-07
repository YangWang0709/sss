#!/usr/bin/env python3
"""Generate LR-2/LR-3 bounded long expert rollout design and preflight.

This script is offline only. It does not start Isaac, map_predict, training,
checkpointing, rollout, or RL.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_long_rollout_bridge"
DESIGN_DIR = BRIDGE / "stage_lr2_bounded_long_expert_rollout_design"
PREFLIGHT_DIR = BRIDGE / "stage_lr3_bounded_long_expert_rollout_preflight"
RUNTIME_DIR = ROOT / "outputs/stage4a_long_bounded_expert_rollout_runtime"
ADAPTER = ROOT / "sim_explorer/run_stage4a_long_bounded_uncertainty_bonus_rollout.py"
CLOSE_GUARD = ROOT / "sim_explorer/run_with_isaac_close_guard.py"
FIXED_USD = ROOT / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
CHECKPOINT = ROOT / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
MEDIUM_RUNTIME = ROOT / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"
MEDIUM_SUMMARY = MEDIUM_RUNTIME / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json"

ENVELOPE = {
    "starts": 10,
    "steps_per_start": 15,
    "max_actions": 150,
    "max_decision_frames": 150,
    "expected_terminal_frames": 10,
    "expected_captures": 160,
    "terminal_capture_per_start": True,
    "primary_expert": "uncertainty_bonus_composite_beta8",
    "beta_uncertainty": 8,
    "lambda_sc": 48,
    "lambda48_role": "shadow/baseline only",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, rows: dict[str, Any]) -> None:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if len(text) > 1600:
            text = text[:1600] + "..."
        lines.append(f"| `{key}` | `{text}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}


def process_scan() -> dict[str, Any]:
    result = run(["bash", "-lc", "ps -eo pid=,ppid=,pgid=,stat=,comm=,args= | grep -Ei 'isaac|python.*run_stage4a|SimulationApp' | grep -v grep || true"])
    ignore_tokens = (
        "generate_long_rollout_design_preflight.py",
        "test_long_rollout_design_preflight.py",
        "py_compile",
        "grep -Ei",
    )
    lines = [
        line
        for line in result["stdout"].splitlines()
        if line.strip() and not any(token in line for token in ignore_tokens)
    ]
    return {"matching_process_count": len(lines), "matching_processes": lines}


def gpu_scan() -> dict[str, Any]:
    result = run(["bash", "-lc", "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true"])
    lines = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    return {"available": bool(result["stdout"] or result["returncode"] == 0), "compute_process_count": len(lines), "compute_processes": lines}


def adapter_static_audit() -> dict[str, Any]:
    text = ADAPTER.read_text(encoding="utf-8", errors="replace") if ADAPTER.is_file() else ""
    checks = {
        "adapter_exists": ADAPTER.is_file(),
        "long_steps_constant_15": "LONG_STEPS_PER_START = 15" in text,
        "long_actions_constant_150": "LONG_TOTAL_ACTIONS = 150" in text,
        "long_captures_constant_160": "LONG_TOTAL_CAPTURES = 160" in text,
        "primary_beta8": "uncertainty_bonus_composite_beta8" in text,
        "lambda48_shadow_only": "shadow/baseline only" in text,
        "custom_long_finalization_sentinel": "maybe_write_long_finalization_sentinel" in text,
        "long_aware_rollout_safety": "legacy_short_rollout_audit_replaced_by_long_aware_audit" in text,
        "no_training_checkpoint_rl_report": "no_training_checkpoint_rl_report" in text,
        "patches_base_enforce_args": "base.enforce_args = enforce_stage4a_long_args" in text,
        "patches_reports": "base.write_datasets_and_reports = write_datasets_and_reports_long" in text,
        "patches_summary": "base.write_summary = write_summary_long" in text,
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def build_future_command() -> str:
    out = "outputs/stage4a_long_bounded_expert_rollout_runtime"
    return f"""# Future LR-5 bounded long expert rollout command
# Run only after LR-2/LR-3/LR-4 web+critic gates pass.
# This command runs expert-rule rollout only. It does not train, checkpoint,
# promote labels, or run RL.

set -euo pipefail
cd /home/ubuntu22/sc_explorer_ws
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/sim_explorer:/home/ubuntu22/sc_explorer_ws/ssc_exploration:/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network:${{PYTHONPATH:-}}
RUN_ID=stage4a_long_bounded_$(date -u +%Y%m%dT%H%M%SZ)
OUT=/home/ubuntu22/sc_explorer_ws/{out}
mkdir -p "$OUT" /home/ubuntu22/sc_explorer_ws/logs

python sim_explorer/run_with_isaac_close_guard.py \\
  --stage LR-5-bounded-long-expert-rollout \\
  --run_id "$RUN_ID" \\
  --output_dir "$OUT" \\
  --finalization_sentinel_name stage_finalized_before_isaac_close.json \\
  --close_timeout_sec 120 \\
  --terminate_grace_sec 30 \\
  --total_timeout_sec 21600 \\
  --require_safe_finalization_for_success \\
  --child_cwd /home/ubuntu22/sc_explorer_ws \\
  -- \\
  python sim_explorer/run_stage4a_long_bounded_uncertainty_bonus_rollout.py \\
    --output_dir "$OUT" \\
    --num_starts 10 \\
    --max_decision_steps_per_start 15 \\
    --terminal_capture_per_start \\
    --max_total_actions 150 \\
    --max_total_decision_frames 150 \\
    --max_total_captures 160 \\
    --num_candidates 64 \\
    --top_n 16 \\
    --primary_formula uncertainty_bonus_composite_beta8 \\
    --lambda_sc 48 \\
    --beta_uncertainty 8 \\
    --uncertainty_composite_weights 0.4 0.4 0.2 \\
    --motion_mode bounded_long_expert_rollout \\
    --prediction_mode sim_dynamic \\
    --predictor_device cuda \\
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
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/lr5_bounded_long_expert_rollout_runtime_${{RUN_ID}}.log
"""


def main() -> None:
    for path in [DESIGN_DIR, PREFLIGHT_DIR, BRIDGE / "stage_manifests"]:
        path.mkdir(parents=True, exist_ok=True)
    generated = utc_now()
    design = {
        "stage": "LR-2 bounded long expert rollout design",
        "generated_at": generated,
        "envelope": ENVELOPE,
        "start_sources": "Reuse Stage 4A-7.14/Stage 4A-6.13 corrected interior starts from Stage 4A-6.6c camera pose fix inputs.",
        "candidate_policy": {"num_candidates": 64, "top_n": 16, "sampling": "reachable_frontier", "candidate_validity": "measured geometry only; no prediction/uncertainty validity"},
        "scoring_formula": "gain_exp / cost + 48 * minmax(source_occ_free) + 8 * uncertainty_composite",
        "uncertainty_composite": "0.4 * minmax(candidate_uncertain_fraction) + 0.4 * minmax(candidate_entropy_mean) + 0.2 * minmax(1 - candidate_margin_mean)",
        "termination_conditions": ["15 steps per start", "150 total actions", "no valid candidate", "safety exception", "runtime close guard timeout after finalized outputs"],
        "safety_stops": ["outside bounds target blocker", "low cost artifact blocker", "prediction/uncertainty writeback blocker", "target/ground_truth/future_observed use blocker", "source/fixed USD/checkpoint modification blocker"],
        "stuck_revisit_policy": {"same_cell_target": "count and warn", "repeated_target": "count and show in review packet", "distance_cues": {"very_close_m": 0.25, "close_m": 0.50}},
        "required_review_packet": ["2D topdown review HTML", "action story HTML", "per-step cards", "RGB/depth thumbnails", "source-to-action arrows", "historical path", "camera swept/observed area", "export JSON/CSV controls"],
        "negative_scope": {"training": False, "checkpoint": False, "RL_GDPO_PPO": False, "label_promotion": False, "lambda48_primary": False},
    }
    write_json(DESIGN_DIR / "bounded_long_expert_rollout_design.json", design)
    write_md(DESIGN_DIR / "bounded_long_expert_rollout_design.md", "LR-2 Bounded Long Expert Rollout Design", design)
    write_json(DESIGN_DIR / "formula_policy.json", {"primary_formula": design["scoring_formula"], "uncertainty_composite": design["uncertainty_composite"], "lambda48_role": "shadow/baseline only", "lambda48_primary": False})
    write_md(DESIGN_DIR / "formula_policy.md", "Formula Policy", {"primary_formula": design["scoring_formula"], "uncertainty_composite": design["uncertainty_composite"], "lambda48_role": "shadow/baseline only"})
    write_json(DESIGN_DIR / "start_pose_plan.json", {"starts": 10, "source": design["start_sources"], "pose_validity": "checked by existing runner inputs and per-capture quality audits"})
    write_md(DESIGN_DIR / "start_pose_plan.md", "Start Pose Plan", {"starts": 10, "source": design["start_sources"], "pose_validity": "existing runner inputs plus per-capture audits"})
    write_json(DESIGN_DIR / "safety_stop_policy.json", {"termination_conditions": design["termination_conditions"], "safety_stops": design["safety_stops"], "close_guard": "mandatory"})
    write_md(DESIGN_DIR / "safety_stop_policy.md", "Safety Stop Policy", {"termination_conditions": design["termination_conditions"], "safety_stops": design["safety_stops"], "close_guard": "mandatory"})
    required_outputs = {"runtime_output_dir": str(RUNTIME_DIR), "required_outputs": ["long_rollout_dataset_uncertainty_bonus.npz", "long_rollout_manifest.jsonl", "per_step_summary.csv", "primary_uncertainty_bonus_decisions.csv", "lambda48_shadow_decisions.csv", "prediction_safety_audit.json", "uncertainty_safety_audit.json", "rollout_safety_audit.json", "dataset_integrity_report.json", "long_rollout_summary.json", "long_rollout_uncertainty_bonus_index.html", "stage_finalized_before_isaac_close.json", "supervisor_report.json"]}
    write_json(DESIGN_DIR / "required_outputs_plan.json", required_outputs)
    write_md(DESIGN_DIR / "required_outputs_plan.md", "Required Outputs Plan", required_outputs)
    negative = {"passed": True, "training": False, "optimizer_step": False, "checkpoint": False, "model_save": False, "label_promotion": False, "RL_GDPO_PPO": False, "runtime_started_in_design_preflight": False}
    write_json(DESIGN_DIR / "no_training_no_checkpoint_no_rl_report.json", negative)
    write_md(DESIGN_DIR / "no_training_no_checkpoint_no_rl_report.md", "No Training No Checkpoint No RL Report", negative)

    disk = shutil.disk_usage(ROOT)
    adapter = adapter_static_audit()
    proc = process_scan()
    gpu = gpu_scan()
    medium = json.loads(MEDIUM_SUMMARY.read_text(encoding="utf-8")) if MEDIUM_SUMMARY.is_file() else {}
    checks = {
        "fixed_usd_exists": FIXED_USD.is_file(),
        "checkpoint_exists": CHECKPOINT.is_file(),
        "medium_runtime_summary_exists": MEDIUM_SUMMARY.is_file(),
        "medium_runtime_actions_60": int(medium.get("executed_action_count", -1)) == 60,
        "medium_primary_beta8": medium.get("primary_formula") == "uncertainty_bonus_composite_beta8",
        "adapter_static_all_passed": adapter["all_passed"],
        "close_guard_exists": CLOSE_GUARD.is_file(),
        "disk_free_gib_gte_20": disk.free >= 20 * 1024**3,
        "no_live_isaac_process": proc["matching_process_count"] == 0,
        "runtime_output_dir_absent_or_reusable": not RUNTIME_DIR.exists() or not any(RUNTIME_DIR.iterdir()),
        "conda_env_python_exists": (ROOT.parent / "miniconda3/envs/env_isaaclab/bin/python").is_file(),
        "lambda48_shadow_only": True,
        "no_runtime_started": True,
        "no_training_checkpoint_rl": True,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    preflight = {
        "stage": "LR-3 bounded long expert rollout preflight",
        "generated_at": generated,
        "envelope": ENVELOPE,
        "checks": checks,
        "blockers": blockers,
        "all_passed": not blockers,
        "disk": {"total_gib": disk.total / 1024**3, "used_gib": disk.used / 1024**3, "free_gib": disk.free / 1024**3},
        "gpu_scan": gpu,
        "process_scan": proc,
        "adapter_static_audit": adapter,
        "runtime_allowed_after_lr3_alone": False,
        "runtime_requires_lr4_web_and_critic_gate": True,
    }
    write_json(PREFLIGHT_DIR / "bounded_long_expert_rollout_preflight_summary.json", preflight)
    write_md(PREFLIGHT_DIR / "bounded_long_expert_rollout_preflight_summary.md", "LR-3 Bounded Long Expert Rollout Preflight", preflight)
    write_json(PREFLIGHT_DIR / "disk_gpu_process_preflight.json", {"disk": preflight["disk"], "gpu_scan": gpu, "process_scan": proc})
    write_md(PREFLIGHT_DIR / "disk_gpu_process_preflight.md", "Disk GPU Process Preflight", {"disk": preflight["disk"], "gpu_scan": gpu, "process_scan": proc})
    write_json(PREFLIGHT_DIR / "runner_capability_report.json", adapter)
    write_md(PREFLIGHT_DIR / "runner_capability_report.md", "Runner Capability Report", adapter)
    write_json(PREFLIGHT_DIR / "preflight_blockers.json", {"blockers": blockers, "all_passed": not blockers})
    write_md(PREFLIGHT_DIR / "preflight_blockers.md", "Preflight Blockers", {"blockers": blockers, "all_passed": not blockers})
    (PREFLIGHT_DIR / "future_lr5_bounded_long_expert_rollout_runtime_command.md").write_text(build_future_command(), encoding="utf-8")
    manifest = {"stage": "LR-2/LR-3", "generated_at": generated, "all_passed": not blockers, "design_dir": str(DESIGN_DIR), "preflight_dir": str(PREFLIGHT_DIR)}
    write_json(BRIDGE / "stage_manifests/lr2_lr3_stage_manifest.json", manifest)
    print(json.dumps({"all_passed": not blockers, "blockers": blockers, "design_dir": str(DESIGN_DIR), "preflight_dir": str(PREFLIGHT_DIR)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
