#!/usr/bin/env python3
"""Validate LR-2/LR-3 bounded long expert rollout design/preflight artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_long_rollout_bridge"
DESIGN = BRIDGE / "stage_lr2_bounded_long_expert_rollout_design"
PREFLIGHT = BRIDGE / "stage_lr3_bounded_long_expert_rollout_preflight"
ADAPTER = ROOT / "sim_explorer/run_stage4a_long_bounded_uncertainty_bonus_rollout.py"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checks = {}
    blockers = []
    checks["adapter_exists"] = ADAPTER.is_file()
    adapter_text = ADAPTER.read_text(encoding="utf-8", errors="replace") if ADAPTER.is_file() else ""
    checks["adapter_enforces_10x15"] = "LONG_NUM_STARTS = 10" in adapter_text and "LONG_STEPS_PER_START = 15" in adapter_text
    checks["adapter_enforces_150_actions"] = "LONG_TOTAL_ACTIONS = 150" in adapter_text and "LONG_TOTAL_DECISION_FRAMES = 150" in adapter_text
    checks["adapter_enforces_160_captures"] = "LONG_TOTAL_CAPTURES = 160" in adapter_text
    checks["primary_beta8"] = "uncertainty_bonus_composite_beta8" in adapter_text
    checks["lambda48_shadow_only"] = "shadow/baseline only" in adapter_text and "lambda48_primary" in adapter_text
    checks["custom_long_sentinel"] = "maybe_write_long_finalization_sentinel" in adapter_text and "bounded_long_expert_rollout" in adapter_text
    checks["long_aware_audit"] = "legacy_short_rollout_audit_replaced_by_long_aware_audit" in adapter_text
    checks["no_training_checkpoint_rl_report"] = "no_training_checkpoint_rl_report" in adapter_text
    required_design = [
        "bounded_long_expert_rollout_design.json",
        "formula_policy.json",
        "start_pose_plan.json",
        "safety_stop_policy.json",
        "required_outputs_plan.json",
        "no_training_no_checkpoint_no_rl_report.json",
    ]
    for name in required_design:
        checks[f"design_{name}_exists"] = (DESIGN / name).is_file()
    required_preflight = [
        "bounded_long_expert_rollout_preflight_summary.json",
        "disk_gpu_process_preflight.json",
        "runner_capability_report.json",
        "preflight_blockers.json",
        "future_lr5_bounded_long_expert_rollout_runtime_command.md",
    ]
    for name in required_preflight:
        checks[f"preflight_{name}_exists"] = (PREFLIGHT / name).is_file()
    if (DESIGN / "bounded_long_expert_rollout_design.json").is_file():
        design = load(DESIGN / "bounded_long_expert_rollout_design.json")
        env = design.get("envelope", {})
        checks["design_envelope_10_15_150"] = env.get("starts") == 10 and env.get("steps_per_start") == 15 and env.get("max_actions") == 150
        checks["design_primary_beta8"] = env.get("primary_expert") == "uncertainty_bonus_composite_beta8"
        checks["design_lambda48_shadow"] = env.get("lambda48_role") == "shadow/baseline only"
    if (PREFLIGHT / "bounded_long_expert_rollout_preflight_summary.json").is_file():
        preflight = load(PREFLIGHT / "bounded_long_expert_rollout_preflight_summary.json")
        checks["preflight_all_passed"] = bool(preflight.get("all_passed"))
        checks["preflight_runtime_not_allowed_by_lr3_alone"] = preflight.get("runtime_allowed_after_lr3_alone") is False
        checks["preflight_requires_lr4"] = preflight.get("runtime_requires_lr4_web_and_critic_gate") is True
        checks["preflight_no_runtime_started"] = preflight.get("checks", {}).get("no_runtime_started") is True
        checks["preflight_no_training_checkpoint_rl"] = preflight.get("checks", {}).get("no_training_checkpoint_rl") is True
    command_path = PREFLIGHT / "future_lr5_bounded_long_expert_rollout_runtime_command.md"
    command_text = command_path.read_text(encoding="utf-8", errors="replace") if command_path.is_file() else ""
    checks["future_command_uses_close_guard"] = "run_with_isaac_close_guard.py" in command_text and "--require_safe_finalization_for_success" in command_text
    checks["future_command_uses_long_adapter"] = "run_stage4a_long_bounded_uncertainty_bonus_rollout.py" in command_text
    checks["future_command_has_long_bounds"] = "--max_decision_steps_per_start 15" in command_text and "--max_total_actions 150" in command_text
    checks["future_command_no_rl_training_checkpoint"] = "--no_training" in command_text and "--no_rl_gdpo" in command_text
    checks["lambda48_not_primary"] = "primary_formula uncertainty_bonus_composite_beta8" in command_text and "lambda48 primary" not in command_text.lower()
    for name, passed in checks.items():
        if not passed:
            blockers.append(name)
    print(json.dumps({"all_passed": not blockers, "blockers": blockers, "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
