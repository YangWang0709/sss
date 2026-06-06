from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_pre_rl_bridge"
OUT = BRIDGE / "stage4a713_medium_expert_rollout_design_preflight"
SUMMARY = OUT / "stage4a713_medium_expert_rollout_design_preflight_summary.json"

REQUIRED_FILES = [
    "stage4a713_medium_expert_rollout_design_preflight_summary.json",
    "stage4a713_medium_expert_rollout_design_preflight_summary.md",
    "loaded_stage4a712_decision_gate_report.json",
    "loaded_stage4a712_decision_gate_report.md",
    "preflight_config.json",
    "preflight_config.md",
    "resource_budget_report.json",
    "resource_budget_report.md",
    "runner_medium_support_audit.json",
    "runner_medium_support_audit.md",
    "execution_plan_report.json",
    "execution_plan_report.md",
    "safety_scope_report.json",
    "safety_scope_report.md",
    "global_negative_scope_report.json",
    "global_negative_scope_report.md",
    "no_training_execution_report.json",
    "no_training_execution_report.md",
    "no_checkpoint_created_report.json",
    "no_checkpoint_created_report.md",
    "no_model_save_report.json",
    "no_model_save_report.md",
    "no_runtime_report.json",
    "no_runtime_report.md",
    "no_rl_gdpo_ppo_report.json",
    "no_rl_gdpo_ppo_report.md",
    "lambda48_shadow_only_audit.json",
    "lambda48_shadow_only_audit.md",
    "forbidden_field_training_audit.json",
    "forbidden_field_training_audit.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "prior_dataset_hash_report.json",
    "prior_dataset_hash_report.md",
    "blocker_report.json",
    "blocker_report.md",
    "future_stage4a714_selected_bounded_execution_sketch.md",
    "recommended_next_faithful_step.md",
    "stage4a713_medium_expert_rollout_preflight_index.html",
    "git_status_before.txt",
    "git_status_after.txt",
]
BRIDGE_FILES = [
    "stage_request_to_web_reviewer.md",
    "web_reviewer_packet_manifest.json",
    "web_review_result.json",
    "web_review_transcript.md",
    "codex_stage_result_summary.json",
    "codex_stage_result_summary.md",
    "next_stage_gate_decision.json",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["output_dir_exists"] = OUT.is_dir()
    checks["bridge_dir_exists"] = BRIDGE.is_dir()
    for name in REQUIRED_FILES:
        checks[f"required:{name}"] = (OUT / name).is_file()
    for name in BRIDGE_FILES:
        checks[f"bridge:{name}"] = (BRIDGE / name).is_file()

    if not SUMMARY.is_file():
        blockers.append("missing_summary")
        result = {"all_passed": False, "checks": checks, "blockers": blockers}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    summary = load_json(SUMMARY)
    gate = load_json(OUT / "loaded_stage4a712_decision_gate_report.json")
    config = load_json(OUT / "preflight_config.json")
    resources = load_json(OUT / "resource_budget_report.json")
    runner = load_json(OUT / "runner_medium_support_audit.json")
    execution = load_json(OUT / "execution_plan_report.json")
    safety = load_json(OUT / "safety_scope_report.json")
    negative = load_json(OUT / "global_negative_scope_report.json")
    no_training = load_json(OUT / "no_training_execution_report.json")
    no_checkpoint = load_json(OUT / "no_checkpoint_created_report.json")
    no_model_save = load_json(OUT / "no_model_save_report.json")
    no_runtime = load_json(OUT / "no_runtime_report.json")
    no_rl = load_json(OUT / "no_rl_gdpo_ppo_report.json")
    lambda_audit = load_json(OUT / "lambda48_shadow_only_audit.json")
    forbidden = load_json(OUT / "forbidden_field_training_audit.json")
    blocker = load_json(OUT / "blocker_report.json")
    next_gate = load_json(BRIDGE / "next_stage_gate_decision.json")
    web_review = load_json(BRIDGE / "web_review_result.json")
    tracked = git_files()

    medium = config.get("medium_config", {})
    adapter_available = bool(runner.get("medium_bounds_supported_by_adapter"))
    checks["summary_completed"] = summary.get("completed") is True
    checks["preflight_state_matches_adapter"] = (
        (summary.get("blocked") is False and summary.get("preflight_passed") is True and summary.get("runtime_allowed") is True and blocker.get("runtime_allowed") is True)
        if adapter_available
        else (summary.get("blocked") is True and summary.get("preflight_passed") is False and summary.get("runtime_allowed") is False and blocker.get("runtime_allowed") is False)
    )
    checks["main_blocker_expected"] = (
        summary.get("main_blocker", "") == ""
        if adapter_available
        else "adapter_missing_or_incomplete" in str(summary.get("main_blocker", ""))
    )
    checks["stage712_gate_loaded"] = gate.get("stage712_completed") is True and gate.get("stage712_selected_option") == "C"
    checks["medium_config_expected"] = (
        medium.get("starts") == 10
        and medium.get("steps_per_start") == 6
        and medium.get("max_actions") == 60
        and medium.get("max_decision_frames") == 60
        and medium.get("terminal_capture_per_start") is True
        and medium.get("primary_formula") == "uncertainty_bonus_composite_beta8"
    )
    checks["required_paths_exist"] = all(resources.get("path_checks", {}).values())
    checks["runner_audit_detects_short_gate"] = (
        runner.get("exists") is True
        and runner.get("historical_runner_remains_short_gated") is True
        and runner.get("hard_gates", {}).get("requires_three_steps") is True
        and runner.get("hard_gates", {}).get("requires_30_30_40_totals") is True
    )
    adapter_gates = runner.get("adapter_gates", {})
    checks["adapter_medium_support"] = (
        adapter_available
        and adapter_gates.get("requires_10_starts") is True
        and adapter_gates.get("requires_6_steps") is True
        and adapter_gates.get("requires_60_actions") is True
        and adapter_gates.get("requires_60_decision_frames") is True
        and adapter_gates.get("requires_70_captures") is True
        and adapter_gates.get("requires_bounded_medium_motion_mode") is True
        and adapter_gates.get("requires_finalization_sentinel_path") is True
        and adapter_gates.get("requires_close_guard_run_id") is True
        and adapter_gates.get("requires_no_training") is True
        and adapter_gates.get("requires_no_rl") is True
        and adapter_gates.get("patches_base_enforce_args_only") is True
    )
    checks["execution_plan_runtime_flag_matches_adapter"] = execution.get("stage4a714_runtime_allowed_now") is adapter_available
    checks["safety_scope_primary_beta8"] = (
        safety.get("primary_expert") == "uncertainty_bonus_composite_beta8"
        and safety.get("lambda48_role") == "shadow/baseline only"
        and safety.get("close_guard_mandatory") is True
        and safety.get("no_training") is True
        and safety.get("no_checkpoint") is True
        and safety.get("no_rl_gdpo_ppo") is True
    )
    checks["lambda48_not_primary"] = (
        summary.get("lambda48_primary_use") is False
        and lambda_audit.get("lambda48_primary_use") is False
        and lambda_audit.get("lambda48_role") == "shadow/baseline only"
        and lambda_audit.get("labels_not_recomputed_from_lambda48") is True
    )
    checks["no_training"] = (
        no_training.get("bc_training") is False
        and no_training.get("optimizer_step") is False
        and negative.get("bc_training") is False
        and negative.get("optimizer_step_count") == 0
        and negative.get("backward_count") == 0
    )
    checks["no_checkpoint_or_model_save"] = (
        no_checkpoint.get("checkpoint_created") is False
        and no_checkpoint.get("torch_save_calls") == 0
        and no_model_save.get("model_saved") is False
        and negative.get("checkpoint_created") is False
        and negative.get("model_saved") is False
    )
    checks["no_runtime"] = (
        no_runtime.get("isaac_startup") is False
        and no_runtime.get("capture") is False
        and no_runtime.get("map_predict") is False
        and no_runtime.get("action_execution") is False
        and no_runtime.get("rollout") is False
        and negative.get("isaac_startup_count") == 0
        and negative.get("map_predict_call_count") == 0
        and negative.get("action_execution_count") == 0
    )
    checks["no_rl"] = (
        no_rl.get("rl") is False
        and no_rl.get("gdpo") is False
        and no_rl.get("ppo") is False
        and negative.get("rl") is False
        and negative.get("replay_buffer_training") is False
    )
    checks["forbidden_fields_not_used"] = forbidden.get("passed") is True and forbidden.get("used_as_feature_label_score_reward_filter") is False
    checks["next_gate_matches_adapter"] = (
        next_gate.get("approved_to_continue") is adapter_available
        and next_gate.get("runtime_allowed") is adapter_available
        and next_gate.get("rl_training_allowed") is False
    )
    checks["web_review_json_machine_readable"] = (
        web_review.get("machine_readable") is True
        and web_review.get("runtime_checkpoint_training_rl_approved") is False
    )
    context_head = (ROOT / ".project_context/CURRENT_STATE.md").read_text(encoding="utf-8")[:1800]
    checks["context_front_updated"] = (
        "Stage 4A-7.13 Medium Expert Rollout Design Preflight Passed" in context_head
        if adapter_available
        else "Stage 4A-7.13 Medium Expert Rollout Design Preflight Complete - Runtime Blocked" in context_head
    )
    checks["future_command_guarded"] = (OUT / "future_stage4a714_selected_bounded_execution_sketch.md").read_text(encoding="utf-8").startswith("DO NOT RUN UNTIL STAGE 4A-7.13 PREFLIGHT PASSES.")
    checks["no_outputs_tracked"] = not any(path.startswith("outputs/") or path.startswith("logs/") for path in tracked)
    checks["no_checkpoint_files_tracked"] = not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked)

    for key, ok in checks.items():
        if not ok:
            blockers.append(key)

    result = {
        "all_passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "stage": "Stage 4A-7.13",
        "output_dir": str(OUT),
        "preflight_passed": summary.get("preflight_passed"),
        "runtime_allowed": summary.get("runtime_allowed"),
        "main_blocker": summary.get("main_blocker"),
    }
    qa_dir = OUT / "subagent_reports"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "qa_validator_agent_report.md").write_text(
        "# QA Validator Agent Report\n\n"
        + f"- all_passed: `{result['all_passed']}`\n"
        + f"- adapter_available: `{adapter_available}`\n"
        + f"- preflight_state_matches_adapter: `{checks['preflight_state_matches_adapter']}`\n"
        + f"- blockers: `{', '.join(blockers) or 'none'}`\n"
        + f"- main_blocker: `{result['main_blocker']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
