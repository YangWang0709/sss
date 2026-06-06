from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_pre_rl_bridge"
OUT = BRIDGE / "stage4a712_pre_rl_direction_decision_packet"
SUMMARY = OUT / "stage4a712_pre_rl_direction_decision_summary.json"

REQUIRED_FILES = [
    "stage4a712_pre_rl_direction_decision_summary.json",
    "stage4a712_pre_rl_direction_decision_summary.md",
    "loaded_stage4a711_evidence.json",
    "loaded_stage4a711_evidence.md",
    "decision_option_matrix.json",
    "decision_option_matrix.md",
    "selected_next_step_decision.json",
    "selected_next_step_decision.md",
    "approval_scope_report.json",
    "approval_scope_report.md",
    "global_negative_scope_report.json",
    "global_negative_scope_report.md",
    "why_no_execution_yet.json",
    "why_no_execution_yet.md",
    "no_training_report.json",
    "no_training_report.md",
    "no_checkpoint_report.json",
    "no_checkpoint_report.md",
    "no_runtime_report.json",
    "no_runtime_report.md",
    "no_rl_gdpo_ppo_report.json",
    "no_rl_gdpo_ppo_report.md",
    "lambda48_shadow_only_audit.json",
    "lambda48_shadow_only_audit.md",
    "forbidden_field_audit.json",
    "forbidden_field_audit.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "prior_dataset_hash_report.json",
    "prior_dataset_hash_report.md",
    "future_stage4a713_medium_expert_rollout_design_preflight_sketch.md",
    "recommended_next_faithful_step.md",
    "stage4a712_pre_rl_direction_decision_index.html",
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
    evidence = load_json(OUT / "loaded_stage4a711_evidence.json")
    option_matrix = load_json(OUT / "decision_option_matrix.json")
    selected = load_json(OUT / "selected_next_step_decision.json")
    negative = load_json(OUT / "global_negative_scope_report.json")
    no_training = load_json(OUT / "no_training_report.json")
    no_checkpoint = load_json(OUT / "no_checkpoint_report.json")
    no_runtime = load_json(OUT / "no_runtime_report.json")
    no_rl = load_json(OUT / "no_rl_gdpo_ppo_report.json")
    lambda_audit = load_json(OUT / "lambda48_shadow_only_audit.json")
    forbidden = load_json(OUT / "forbidden_field_audit.json")
    bridge_review = load_json(BRIDGE / "web_review_result.json")
    next_gate = load_json(BRIDGE / "next_stage_gate_decision.json")
    tracked = git_files()

    checks["summary_completed"] = summary.get("completed") is True and summary.get("blocked") is False
    checks["selected_option_c"] = summary.get("selected_option") == "C" and option_matrix.get("selected_option") == "C"
    checks["next_stage_713"] = selected.get("selected_next_stage", "").startswith("Stage 4A-7.13")
    checks["design_preflight_only"] = selected.get("approved_to_create_design_preflight") is True and selected.get("approved_to_run_medium_runtime_now") is False
    checks["stage711_evidence_loaded"] = (
        evidence.get("stage711_samples") == 47
        and evidence.get("stage711_candidate_count") == 64
        and evidence.get("stage711_d_model") == 16
        and evidence.get("stage711_valid_labels") is True
    )
    checks["stage711_safety_loaded"] = (
        evidence.get("stage711_checkpoint_created") is False
        and evidence.get("stage711_model_saved") is False
        and evidence.get("stage711_isaac_startup") is False
        and evidence.get("stage711_map_predict") is False
        and evidence.get("stage711_rollout") is False
        and evidence.get("stage711_rl_gdpo_ppo") is False
    )
    checks["lambda48_not_primary"] = (
        summary.get("lambda48_primary_use") is False
        and evidence.get("stage711_lambda48_primary_use") is False
        and lambda_audit.get("lambda48_primary_use") is False
        and lambda_audit.get("lambda48_role") == "shadow/baseline only"
        and lambda_audit.get("labels_not_recomputed_from_lambda48") is True
    )
    checks["no_label_promotion"] = selected.get("no_label_promotion_this_stage") is True and negative.get("label_promotion") is False
    checks["no_training"] = (
        no_training.get("bc_training") is False
        and no_training.get("optimizer_step") is False
        and negative.get("bc_training") is False
        and negative.get("optimizer_step_count") == 0
        and negative.get("backward_count") == 0
    )
    checks["no_checkpoint"] = (
        no_checkpoint.get("checkpoint_created") is False
        and no_checkpoint.get("model_saved") is False
        and negative.get("checkpoint_created") is False
        and negative.get("model_saved") is False
        and negative.get("torch_save_calls") == 0
    )
    checks["no_runtime"] = (
        no_runtime.get("isaac_startup") is False
        and no_runtime.get("capture") is False
        and no_runtime.get("map_predict") is False
        and no_runtime.get("rollout") is False
        and negative.get("isaac_startup_count") == 0
        and negative.get("map_predict_call_count") == 0
        and negative.get("action_execution_count") == 0
        and negative.get("rollout_count") == 0
    )
    checks["no_rl"] = (
        no_rl.get("rl") is False
        and no_rl.get("gdpo") is False
        and no_rl.get("ppo") is False
        and negative.get("rl") is False
        and negative.get("gdpo") is False
        and negative.get("ppo") is False
        and negative.get("replay_buffer_training") is False
    )
    checks["forbidden_fields_not_used"] = forbidden.get("passed") is True and forbidden.get("target_ground_truth_future_observed_use") is False
    checks["web_review_json_machine_readable"] = bridge_review.get("machine_readable") is True
    checks["next_gate_no_runtime"] = next_gate.get("approved_to_continue") is True and next_gate.get("runtime_allowed") is False and next_gate.get("rl_training_allowed") is False
    checks["context_front_updated"] = "Stage 4A-7.12 Pre-RL Direction Decision Packet Complete" in (ROOT / ".project_context/CURRENT_STATE.md").read_text(encoding="utf-8")[:1200]
    checks["no_outputs_tracked"] = not any(path.startswith("outputs/") or path.startswith("logs/") for path in tracked)
    checks["no_checkpoint_files_tracked"] = not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked)

    for key, ok in checks.items():
        if not ok:
            blockers.append(key)

    result = {
        "all_passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "stage": "Stage 4A-7.12",
        "output_dir": str(OUT),
        "selected_option": summary.get("selected_option"),
        "recommended_next": summary.get("recommended_next"),
    }
    qa_dir = OUT / "subagent_reports"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "qa_validator_agent_report.md").write_text(
        "# QA Validator Agent Report\n\n"
        + f"- all_passed: `{result['all_passed']}`\n"
        + f"- blockers: `{', '.join(blockers) or 'none'}`\n"
        + f"- selected_option: `{result['selected_option']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
