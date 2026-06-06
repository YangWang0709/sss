from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a710_expanded_dataset_qa"
SUBAGENT_OUT = OUT / "subagent_reports"
STAGE79 = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation"
STAGE78 = ROOT / "outputs/isaac_stage4a78_promotion_candidate_decision_packet"
STAGE70 = ROOT / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
STAGE73 = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_no_training"

REQUIRED_REPORTS = [
    "stage4a710_expanded_dataset_qa_summary.json",
    "stage4a710_expanded_dataset_qa_summary.md",
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a79_evidence.json",
    "loaded_stage4a79_evidence.md",
    "expanded_dataset_npz_inventory.json",
    "expanded_dataset_npz_inventory.md",
    "expanded_dataset_shape_audit.json",
    "expanded_dataset_shape_audit.md",
    "expanded_label_validity_audit.json",
    "expanded_label_validity_audit.md",
    "expanded_lineage_audit.json",
    "expanded_lineage_audit.md",
    "rejected_unsure_conflict_exclusion_audit.json",
    "rejected_unsure_conflict_exclusion_audit.md",
    "lambda48_shadow_only_audit.json",
    "lambda48_shadow_only_audit.md",
    "forbidden_field_audit.json",
    "forbidden_field_audit.md",
    "SimExpertBCDataset_load_audit.json",
    "SimExpertBCDataset_load_audit.md",
    "optional_forward_only_smoke_report.json",
    "optional_forward_only_smoke_report.md",
    "no_training_report.json",
    "no_training_report.md",
    "no_runtime_report.json",
    "no_runtime_report.md",
    "no_checkpoint_report.json",
    "no_checkpoint_report.md",
    "no_rl_gdpo_ppo_report.json",
    "no_rl_gdpo_ppo_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "prior_dataset_hash_report.json",
    "prior_dataset_hash_report.md",
    "expanded_dataset_hash_report.json",
    "expanded_dataset_hash_report.md",
    "git_sync_report.json",
    "git_sync_report.md",
    "git_status_before.txt",
    "git_status_after.txt",
    "future_stage4a711_tiny_eval_on_expanded_dataset_sketch.md",
    "recommended_next_faithful_step.md",
]
SUBAGENT_REPORTS = [
    "context_git_sync_agent_report.md",
    "dataset_shape_agent_report.md",
    "label_lineage_agent_report.md",
    "safety_forbidden_field_agent_report.md",
    "loader_smoke_agent_report.md",
    "qa_validator_agent_report.md",
]
REQUIRED_NPZ_KEYS = {
    "sample_id",
    "candidate_features_model",
    "candidate_valid_mask",
    "expert_action_index_primary",
    "expert_action_index_measured_shadow",
    "expert_action_index_lambda48_shadow",
    "expert_action_index_confidence_gated_shadow",
    "quality_keep_mask",
    "missing_feature_mask",
}
FORBIDDEN_FIELDS = {
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
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def write_validator_report(result: dict) -> None:
    rows = [
        ("all_passed", result["all_passed"]),
        ("blockers", ", ".join(result["blockers"]) or "none"),
        ("expanded_samples", result.get("expanded_samples")),
        ("promoted_samples", result.get("promoted_samples")),
    ]
    lines = ["# QA Validator Agent Report", "", "| field | value |", "| --- | --- |"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    (SUBAGENT_OUT / "qa_validator_agent_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["output_dir_exists"] = OUT.is_dir()
    checks["subagent_dir_exists"] = SUBAGENT_OUT.is_dir()
    for report in REQUIRED_REPORTS:
        checks[f"required_report:{report}"] = (OUT / report).is_file()
    for report in SUBAGENT_REPORTS:
        checks[f"subagent_report:{report}"] = (SUBAGENT_OUT / report).is_file()
    checks["qa_index_exists"] = (OUT / "expanded_dataset_qa_index.html").is_file()

    if not all(v for k, v in checks.items() if k.startswith(("required_report:", "subagent_report:"))):
        for key, ok in checks.items():
            if not ok:
                blockers.append(key)
        result = {"all_passed": False, "blockers": blockers, "checks": checks}
        print(json.dumps(result, indent=2, sort_keys=True))
        write_validator_report(result)
        return 1

    summary = load_json(OUT / "stage4a710_expanded_dataset_qa_summary.json")
    shape = load_json(OUT / "expanded_dataset_shape_audit.json")
    label = load_json(OUT / "expanded_label_validity_audit.json")
    lineage = load_json(OUT / "expanded_lineage_audit.json")
    exclusion = load_json(OUT / "rejected_unsure_conflict_exclusion_audit.json")
    lambda48 = load_json(OUT / "lambda48_shadow_only_audit.json")
    forbidden = load_json(OUT / "forbidden_field_audit.json")
    loader = load_json(OUT / "SimExpertBCDataset_load_audit.json")
    smoke = load_json(OUT / "optional_forward_only_smoke_report.json")
    no_training = load_json(OUT / "no_training_report.json")
    no_runtime = load_json(OUT / "no_runtime_report.json")
    no_checkpoint = load_json(OUT / "no_checkpoint_report.json")
    no_rl = load_json(OUT / "no_rl_gdpo_ppo_report.json")
    source_hash = load_json(OUT / "source_hash_report.json")
    prior_hash = load_json(OUT / "prior_dataset_hash_report.json")
    expanded_hash = load_json(OUT / "expanded_dataset_hash_report.json")
    git_sync = load_json(OUT / "git_sync_report.json")

    expanded_path = STAGE79 / "expanded_primary_bc_dataset.npz"
    original_path = STAGE70 / "bc_dataset_primary_short_rollout.npz"
    adapter_path = STAGE73 / "stage4a72_compact_v1_candidate_feature_adapter.npz"
    checks["expanded_dataset_exists"] = expanded_path.is_file()
    expanded = np.load(expanded_path, allow_pickle=False)
    original = np.load(original_path, allow_pickle=False)
    adapter = np.load(adapter_path, allow_pickle=False)

    checks["summary_completed"] = summary.get("completed") is True and summary.get("blocked") is False
    checks["summary_counts"] = (
        summary.get("expanded_samples") == 47
        and summary.get("original_primary_samples") == 30
        and summary.get("promoted_stage4a72_samples") == 17
        and summary.get("candidate_count") == 64
        and summary.get("D_model") == 16
    )
    checks["required_npz_keys"] = REQUIRED_NPZ_KEYS.issubset(set(expanded.files))
    checks["forbidden_npz_keys_absent"] = not (FORBIDDEN_FIELDS & {key.lower() for key in expanded.files})
    checks["sample_count_47"] = expanded["sample_id"].shape[0] == 47
    checks["candidate_features_shape"] = tuple(expanded["candidate_features_model"].shape) == (47, 64, 16)
    checks["candidate_valid_mask_shape"] = tuple(expanded["candidate_valid_mask"].shape) == (47, 64)
    checks["missing_feature_mask_shape"] = tuple(expanded["missing_feature_mask"].shape) == (47, 64, 16)
    checks["D_model_16"] = expanded["candidate_features_model"].shape[2] == 16
    checks["features_finite"] = bool(np.isfinite(expanded["candidate_features_model"]).all())
    primary = expanded["expert_action_index_primary"].astype(np.int64)
    valid = expanded["candidate_valid_mask"].astype(bool)
    checks["all_primary_labels_valid"] = bool(np.all((primary >= 0) & (primary < valid.shape[1])))
    checks["candidate_valid_mask_true_at_label"] = bool(np.all(valid[np.arange(primary.shape[0]), primary]))
    checks["original_30_retained"] = (
        np.array_equal(expanded["sample_id"][:30], original["sample_id"])
        and np.array_equal(expanded["expert_action_index_primary"][:30], original["expert_action_index_primary"])
        and np.allclose(expanded["candidate_features_model"][:30], original["candidate_features_model"])
    )
    clean_rows = []
    with (STAGE78 / "clean_promotion_candidates.csv").open(encoding="utf-8") as f:
        import csv

        clean_rows = list(csv.DictReader(f))
    clean_indices = np.asarray([int(row["adapter_sample_index"]) for row in clean_rows], dtype=np.int64)
    checks["promoted_17_retained"] = (
        expanded["sample_id"][30:].shape[0] == 17
        and np.array_equal(expanded["sample_id"][30:], adapter["sample_id"][clean_indices])
        and np.array_equal(
            expanded["expert_action_index_primary"][30:],
            adapter["candidate_action_index_uncertainty_bonus_executed"][clean_indices],
        )
    )
    checks["no_rejected_unsure_conflict_promoted"] = (
        exclusion.get("rejected_rows_promoted") == 0
        and exclusion.get("unsure_rows_promoted") == 0
        and exclusion.get("conflict_rows_promoted") == 0
    )
    checks["lambda48_not_primary"] = (
        lambda48.get("passed") is True
        and lambda48.get("lambda48_primary_use") is False
        and summary.get("lambda48_primary_use") is False
    )
    checks["audit_reports_passed"] = all(
        report.get("passed") is True
        for report in [shape, label, lineage, exclusion, lambda48, forbidden, loader, smoke]
    )
    checks["SimExpertBCDataset_load"] = loader.get("passed") is True and loader.get("dataset_len") == 47
    checks["forward_only_smoke_safe"] = (
        smoke.get("passed") is True
        and smoke.get("forward_only_smoke") is True
        and smoke.get("backward") is False
        and smoke.get("optimizer_step") is False
        and smoke.get("model_saved") is False
        and smoke.get("checkpoint") is False
    )
    checks["no_training"] = (
        no_training.get("bc_training") is False
        and no_training.get("optimizer_step") is False
        and no_training.get("backward") is False
    )
    checks["no_checkpoint"] = no_checkpoint.get("checkpoint") is False and no_checkpoint.get("model_save") is False
    checks["no_isaac_map_predict_rollout"] = (
        no_runtime.get("isaac_startup") is False
        and no_runtime.get("map_predict") is False
        and no_runtime.get("rollout") is False
    )
    checks["no_rl_gdpo_ppo"] = (
        no_rl.get("rl") is False and no_rl.get("gdpo") is False and no_rl.get("ppo") is False
    )
    checks["future_sketch_first_line"] = (
        (OUT / "future_stage4a711_tiny_eval_on_expanded_dataset_sketch.md").read_text(encoding="utf-8").splitlines()[0]
        == "DO NOT RUN IN STAGE 4A-7.10."
    )
    checks["source_hashes_unchanged"] = all(item.get("unchanged") is True for item in source_hash.values() if item.get("exists"))
    checks["prior_hashes_unchanged"] = all(item.get("unchanged") is True for item in prior_hash.values() if item.get("exists"))
    checks["expanded_hash_unchanged"] = all(item.get("unchanged") is True for item in expanded_hash.values() if item.get("exists"))
    checks["git_sync_passed"] = git_sync.get("passed") is True

    tracked = git_files()
    forbidden_tracked = [
        path
        for path in tracked
        if path.startswith(("outputs/", "logs/", "checkpoints/", "assets/"))
        or path.lower().endswith((".npz", ".npy", ".png", ".mp4", ".usd", ".usda", ".usdc", ".pt", ".pth", ".ckpt"))
    ]
    large_tracked = [
        path
        for path in tracked
        if (ROOT / path).is_file() and (ROOT / path).stat().st_size > 50 * 1024 * 1024
    ]
    checks["git_large_artifact_policy_preserved"] = not forbidden_tracked and not large_tracked

    for key, ok in checks.items():
        if not ok:
            blockers.append(key)

    result = {
        "all_passed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "expanded_samples": int(expanded["sample_id"].shape[0]),
        "promoted_samples": 17,
        "forbidden_tracked": forbidden_tracked,
        "large_tracked": large_tracked,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    write_validator_report(result)
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
