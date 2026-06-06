#!/usr/bin/env python3
"""Validate Stage 4A-7.12 review/readiness decision packet."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/stage4a712_stage4a79_review_readiness_decision_packet"

REQUIRED_FILES = [
    "loaded_evidence_report.json",
    "loaded_evidence_report.md",
    "decision_option_matrix.json",
    "decision_option_matrix.csv",
    "decision_option_matrix.md",
    "selected_next_step_decision.json",
    "selected_next_step_decision.md",
    "risk_register.json",
    "risk_register.csv",
    "risk_register.md",
    "negative_scope_report.json",
    "negative_scope_report.md",
    "no_training_report.json",
    "no_checkpoint_report.json",
    "no_runtime_report.json",
    "no_rl_gdpo_ppo_report.json",
    "lambda48_shadow_only_audit.json",
    "source_hash_report.json",
    "packet_manifest.json",
    "stage4a712_review_readiness_decision_summary.json",
    "stage4a712_review_readiness_decision_summary.md",
    "stage4a712_review_readiness_decision_index.html",
    "stage4a712_web_review_request.md",
    "handoff_to_stage4a713_design_preflight.md",
    "git_status_before.txt",
    "git_status_after.txt",
]


def load_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def checkpoint_like_outputs() -> list[str]:
    hits = []
    if not OUT.exists():
        return hits
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(OUT))
        lower = rel.lower()
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".tar"}:
            hits.append(rel)
        elif any(token in lower for token in ["model_weights", "optimizer_state", "state_dict", "replay_buffer"]):
            hits.append(rel)
    return sorted(hits)


def main() -> int:
    checks: dict[str, bool] = {"output_dir_exists": OUT.is_dir()}
    for name in REQUIRED_FILES:
        checks[f"required:{name}"] = (OUT / name).is_file()

    blockers = [key for key, ok in checks.items() if not ok]
    if blockers:
        result = {"all_passed": False, "checks": checks, "blockers": blockers}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    summary = load_json("stage4a712_review_readiness_decision_summary.json")
    evidence = load_json("loaded_evidence_report.json")
    option_matrix = load_json("decision_option_matrix.json")
    selected = load_json("selected_next_step_decision.json")
    negative = load_json("negative_scope_report.json")
    no_training = load_json("no_training_report.json")
    no_checkpoint = load_json("no_checkpoint_report.json")
    no_runtime = load_json("no_runtime_report.json")
    no_rl = load_json("no_rl_gdpo_ppo_report.json")
    lambda_audit = load_json("lambda48_shadow_only_audit.json")
    source_hash = load_json("source_hash_report.json")
    risk_register = load_json("risk_register.json")
    html = (OUT / "stage4a712_review_readiness_decision_index.html").read_text(encoding="utf-8")
    tracked = git_files()

    checks.update(
        {
            "summary_completed": summary.get("completed") is True,
            "summary_not_blocked": summary.get("blocked") is False,
            "decision_design_preflight_only": summary.get("decision") == "ready_for_stage4a713_checkpointed_bc_design_preflight_only",
            "selected_option_b": summary.get("selected_option") == "B" and option_matrix.get("selected_option") == "B",
            "next_stage_713_design": selected.get("selected_next_stage") == "Stage 4A-7.13 bounded checkpointed BC experiment design/preflight only",
            "design_preflight_allowed": selected.get("approved_to_create_design_preflight") is True,
            "checkpointed_bc_execution_not_allowed": selected.get("approved_to_run_checkpointed_bc_now") is False,
            "checkpoint_save_not_allowed": selected.get("approved_to_save_checkpoint_now") is False,
            "runtime_not_allowed": selected.get("approved_to_run_runtime_now") is False and selected.get("approved_to_run_rollout_now") is False,
            "rl_not_allowed": selected.get("approved_to_run_rl_gdpo_ppo_now") is False,
            "stage79_shape_loaded": evidence.get("stage4a79_shape") == [55, 64, 16],
            "stage79_valid_labels": evidence.get("stage4a79_valid_labels") is True,
            "stage710_ready": evidence.get("stage4a710_readiness_decision") == "ready_for_tiny_bc_dry_run_consideration",
            "stage710_no_blockers": evidence.get("stage4a710_blocker_count") == 0,
            "stage711_passed": evidence.get("stage4a711_decision") == "tiny_bc_dry_run_passed_no_checkpoint",
            "stage711_steps_recorded": evidence.get("stage4a711_optimizer_steps") == 8,
            "stage711_no_checkpoint": evidence.get("stage4a711_checkpoint_created") is False,
            "stage711_no_model_saved": evidence.get("stage4a711_model_saved") is False,
            "stage711_not_full_training": evidence.get("stage4a711_full_training") is False,
            "stage711_no_runtime": all(value is False for value in evidence.get("stage4a711_no_runtime", {}).values()),
            "stage711_no_rl": all(value is False for value in evidence.get("stage4a711_no_rl_gdpo_ppo", {}).values()),
            "stage712_no_training": no_training.get("training") is False and no_training.get("optimizer_step_count") == 0 and no_training.get("backward_count") == 0,
            "stage712_negative_no_optimizer": negative.get("stage4a712_optimizer_step_count") == 0 and negative.get("stage4a712_backward_count") == 0,
            "stage712_no_checkpoint": no_checkpoint.get("checkpoint_created") is False and no_checkpoint.get("model_saved") is False and no_checkpoint.get("torch_save_calls") == 0,
            "stage712_no_runtime": no_runtime.get("isaac_startup") is False and no_runtime.get("map_predict") is False and no_runtime.get("rollout") is False,
            "stage712_no_rl": no_rl.get("rl") is False and no_rl.get("gdpo") is False and no_rl.get("ppo") is False,
            "lambda48_shadow_only": lambda_audit.get("lambda48_role") == "shadow/baseline only",
            "not_recomputed_from_lambda48": lambda_audit.get("labels_recomputed_from_lambda48") is False,
            "source_hashes_exist": all(item.get("exists") is True for item in source_hash.values()),
            "risk_register_present": len(risk_register.get("risks", [])) >= 3,
            "html_mentions_stage": "Stage 4A-7.12" in html,
            "html_mentions_no_training_boundary": "did not train" in html,
            "checkpoint_like_outputs_absent": checkpoint_like_outputs() == [],
            "tracked_no_outputs": not any(path.startswith(("outputs/", "logs/")) for path in tracked),
            "tracked_no_checkpoints": not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked),
        }
    )

    blockers = [key for key, ok in checks.items() if not ok]
    result = {
        "all_passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "checkpoint_like_outputs": checkpoint_like_outputs(),
        "selected_next_stage": selected.get("selected_next_stage"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
