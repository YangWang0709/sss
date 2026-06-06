#!/usr/bin/env python3
"""Validate Stage 4A-7.13 checkpointed BC design/preflight packet."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/stage4a713_checkpointed_bc_design_preflight"
DATASET = ROOT / "outputs/stage4a79_stage4a714_compatible_no_training_import/stage4a79_stage4a714_compatible_expanded_dataset_55.npz"

REQUIRED_FILES = [
    "stage4a713_checkpointed_bc_design_preflight_summary.json",
    "stage4a713_checkpointed_bc_design_preflight_summary.md",
    "loaded_stage4a712_decision_evidence.json",
    "loaded_stage4a712_decision_evidence.md",
    "dataset_preflight_report.json",
    "dataset_preflight_report.md",
    "model_design_report.json",
    "model_design_report.md",
    "training_plan_report.json",
    "training_plan_report.md",
    "checkpoint_policy_report.json",
    "checkpoint_policy_report.md",
    "split_policy_report.json",
    "split_policy_report.md",
    "metric_plan_report.json",
    "metric_plan_report.md",
    "subgroup_metric_plan_report.json",
    "subgroup_metric_plan_report.md",
    "risk_control_report.json",
    "risk_control_report.md",
    "no_training_report.json",
    "no_training_report.md",
    "no_checkpoint_report.json",
    "no_checkpoint_report.md",
    "no_runtime_report.json",
    "no_runtime_report.md",
    "no_rl_gdpo_ppo_report.json",
    "no_rl_gdpo_ppo_report.md",
    "forbidden_field_preflight_report.json",
    "forbidden_field_preflight_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "exact_approval_phrase_for_stage4a714_checkpointed_bc_execution.md",
    "future_stage4a714_checkpointed_bc_execution_sketch.md",
    "recommended_next_faithful_step.md",
    "stage4a713_checkpointed_bc_design_preflight_index.html",
    "git_status_before.txt",
    "git_status_after.txt",
]


def load_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def git_check_ignored(path: str) -> bool:
    rel = str(Path(path).relative_to(ROOT))
    result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
    return result.returncode == 0


def checkpoint_like_outputs() -> list[str]:
    hits: list[str] = []
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
        print(json.dumps({"all_passed": False, "checks": checks, "blockers": blockers}, indent=2, sort_keys=True))
        return 1

    summary = load_json("stage4a713_checkpointed_bc_design_preflight_summary.json")
    evidence = load_json("loaded_stage4a712_decision_evidence.json")
    dataset = load_json("dataset_preflight_report.json")
    model = load_json("model_design_report.json")
    training = load_json("training_plan_report.json")
    checkpoint = load_json("checkpoint_policy_report.json")
    split = load_json("split_policy_report.json")
    metrics = load_json("metric_plan_report.json")
    subgroup = load_json("subgroup_metric_plan_report.json")
    risk = load_json("risk_control_report.json")
    no_training = load_json("no_training_report.json")
    no_checkpoint = load_json("no_checkpoint_report.json")
    no_runtime = load_json("no_runtime_report.json")
    no_rl = load_json("no_rl_gdpo_ppo_report.json")
    forbidden = load_json("forbidden_field_preflight_report.json")
    source_hash = load_json("source_hash_report.json")
    approval = (OUT / "exact_approval_phrase_for_stage4a714_checkpointed_bc_execution.md").read_text(encoding="utf-8")
    future = (OUT / "future_stage4a714_checkpointed_bc_execution_sketch.md").read_text(encoding="utf-8")
    html = (OUT / "stage4a713_checkpointed_bc_design_preflight_index.html").read_text(encoding="utf-8")
    tracked = git_files()

    future_checkpoint_path = checkpoint.get("future_checkpoint_path", "")
    future_checkpoint_dir = checkpoint.get("future_checkpoint_dir", "")

    checks.update(
        {
            "summary_completed": summary.get("completed") is True,
            "summary_not_blocked": summary.get("blocked") is False,
            "summary_decision_design_preflight": summary.get("decision")
            == "design_preflight_complete_ready_for_separate_stage4a714_checkpointed_bc_execution_review",
            "stage712_decision_loaded": evidence.get("all_expected_decision_checks_passed") is True,
            "stage712_selected_option_b": evidence.get("selected_next_step_decision", {}).get("selected_option") == "B",
            "stage712_checkpointed_bc_not_approved_now": evidence.get("approved_to_run_checkpointed_bc_now") is False,
            "stage712_checkpoint_save_not_approved_now": evidence.get("approved_to_save_checkpoint_now") is False,
            "stage712_runtime_not_approved_now": evidence.get("approved_to_run_runtime_now") is False,
            "stage712_rollout_not_approved_now": evidence.get("approved_to_run_rollout_now") is False,
            "stage712_rl_not_approved_now": evidence.get("approved_to_run_rl_gdpo_ppo_now") is False,
            "dataset_points_to_stage79_artifact": dataset.get("dataset_path") == str(DATASET),
            "dataset_sample_count_55": dataset.get("sample_count") == 55,
            "dataset_shape_55_64_16": dataset.get("candidate_features_model_shape") == [55, 64, 16],
            "dataset_candidate_count_64": dataset.get("candidate_count") == 64,
            "dataset_d_model_16": dataset.get("d_model") == 16,
            "dataset_provenance_30_25": dataset.get("provenance", {}).get("stage4a70_original_primary_rows") == 30
            and dataset.get("provenance", {}).get("stage4a714_compatible_imported_rows") == 25,
            "dataset_no_rejected_held_recheck_rows": dataset.get("no_rejected_held_recheck_rows_included") is True,
            "dataset_forbidden_fields_absent": dataset.get("forbidden_fields_absent") is True,
            "dataset_valid_labels": dataset.get("valid_primary_labels") is True,
            "model_is_candidate_mlp": model.get("model_family") == "CandidateMLPPolicy",
            "model_input_dim_16": model.get("input_dim") == 16,
            "model_hidden_candidates": {row.get("hidden_dim") for row in model.get("allowed_variants", [])} == {64, 128},
            "model_invalid_mask_required": model.get("invalid_candidate_mask_required") is True,
            "training_design_only": training.get("status") == "design_only_not_executed",
            "training_experimental_only": training.get("checkpointed_bc_experiment_is_experimental_only") is True,
            "training_no_runtime_no_rl": training.get("runtime_during_training") is False and training.get("rl_during_training") is False,
            "training_caps_present": int(training.get("max_epochs_cap", 0)) <= 20
            and int(training.get("max_optimizer_steps_cap", 0)) <= 200,
            "training_optimizer_steps_zero": training.get("this_stage_optimizer_steps") == 0,
            "split_uses_existing_split_id": split.get("primary_split") == "use existing split_id from Stage 4A-7.9 artifact",
            "split_expected_counts": split.get("expected_counts") == {"train": 39, "val": 10, "test": 6},
            "metric_plan_contains_topk_mrr_ce": set(metrics.get("primary_metrics", [])) == {"CE loss", "top1", "top3", "top5", "MRR"},
            "subgroup_plan_contains_original_imported": set(subgroup.get("required_subgroups", {}).keys())
            == {"original_stage4a70_primary_30", "imported_stage4a714_clean_candidate_25"},
            "risk_policy_no_runtime_rl": risk.get("safety_policy", {}).get("no_runtime") is True
            and risk.get("safety_policy", {}).get("no_rl_gdpo_ppo") is True,
            "no_training_occurred": no_training.get("training") is False
            and no_training.get("bc_training") is False
            and no_training.get("optimizer_step_count") == 0
            and no_training.get("backward_count") == 0,
            "no_replay_buffer_training": no_training.get("replay_buffer_training") is False and no_rl.get("replay_buffer_learning") is False,
            "no_dataset_modification": no_training.get("dataset_modified") is False,
            "no_label_promotion": no_training.get("label_promotion") is False,
            "no_checkpoint_created": no_checkpoint.get("checkpoint_created") is False
            and no_checkpoint.get("model_saved") is False
            and no_checkpoint.get("torch_save_calls") == 0,
            "future_checkpoint_dir_not_created": Path(future_checkpoint_dir).exists() is False,
            "checkpoint_like_outputs_absent": checkpoint_like_outputs() == [] and no_checkpoint.get("checkpoint_like_outputs") == [],
            "future_checkpoint_path_ignored": git_check_ignored(future_checkpoint_path),
            "checkpoint_requires_future_approval": checkpoint.get("future_approval_required") is True,
            "checkpoint_metadata_required_fields_present": set(checkpoint.get("metadata_required_fields", {}).keys())
            == {
                "dataset_path",
                "dataset_sha256",
                "feature_names",
                "git_commit",
                "label_policy",
                "lambda48_shadow_only_declaration",
                "model_config",
                "no_rl_declaration",
            },
            "no_isaac": no_runtime.get("isaac_startup") is False,
            "no_map_predict": no_runtime.get("map_predict") is False,
            "no_rollout": no_runtime.get("rollout") is False and no_runtime.get("long_rollout") is False,
            "no_runtime": no_runtime.get("runtime") is False,
            "no_action_execution": no_runtime.get("action_execution") is False,
            "no_rl_gdpo_ppo": no_rl.get("rl") is False and no_rl.get("gdpo") is False and no_rl.get("ppo") is False,
            "lambda48_shadow_only": summary.get("lambda48_role") == "shadow/baseline only"
            and dataset.get("lambda48_role") == "shadow/baseline only",
            "not_recomputed_from_lambda48": summary.get("labels_recomputed_from_lambda48") is False
            and dataset.get("labels_recomputed_from_lambda48") is False,
            "forbidden_report_absent": forbidden.get("forbidden_fields_absent") is True
            and forbidden.get("prediction_uncertainty_writeback") is False,
            "source_hashes_exist": all(item.get("exists") is True for item in source_hash.values()),
            "future_sketch_guard": future.startswith("# Future Stage 4A-7.14 Checkpointed BC Execution Sketch\n\nDO NOT RUN IN STAGE 4A-7.13."),
            "approval_phrase_contains_run": "run bounded checkpointed BC experiment" in approval,
            "approval_phrase_contains_stage79": "Stage 4A-7.9 compatible expanded dataset" in approval,
            "approval_phrase_contains_ignored_checkpoint_dir": "ignored checkpoint dir" in approval,
            "approval_phrase_contains_no_runtime_rollout_rl": "do not run runtime, rollout, or RL/GDPO/PPO" in approval,
            "approval_phrase_contains_lambda48": "lambda48 shadow-only" in approval,
            "html_mentions_no_training_boundary": "No BC training" in html,
            "tracked_no_outputs_logs_checkpoints": not any(
                path.startswith(("outputs/", "logs/", "checkpoints/")) for path in tracked
            ),
            "tracked_no_large_artifact_suffixes": not any(
                path.endswith((".npz", ".png", ".mp4", ".usd", ".pt", ".pth", ".ckpt", ".tar")) for path in tracked
            ),
        }
    )

    blockers = [key for key, ok in checks.items() if not ok]
    result = {
        "all_passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "checkpoint_like_outputs": checkpoint_like_outputs(),
        "dataset_path": dataset.get("dataset_path"),
        "future_checkpoint_path": future_checkpoint_path,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
