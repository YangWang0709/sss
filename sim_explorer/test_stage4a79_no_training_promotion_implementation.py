from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation"
STAGE70 = ROOT / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
STAGE73 = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_no_training"
STAGE78 = ROOT / "outputs/isaac_stage4a78_promotion_candidate_decision_packet"

FORBIDDEN_KEYS = {
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
REQUIRED_FILES = [
    "stage4a79_no_training_promotion_summary.json",
    "stage4a79_no_training_promotion_summary.md",
    "expanded_primary_bc_dataset.npz",
    "expanded_primary_bc_dataset_metadata.json",
    "expanded_primary_bc_dataset_manifest.jsonl",
    "expanded_sample_index_table.csv",
    "promoted_stage4a72_rows.csv",
    "excluded_stage4a72_rows.csv",
    "promotion_mapping_report.json",
    "promotion_mapping_report.md",
    "primary_label_lineage_report.json",
    "primary_label_lineage_report.md",
    "forbidden_field_audit.json",
    "forbidden_field_audit.md",
    "expanded_dataset_integrity_report.json",
    "expanded_dataset_integrity_report.md",
    "split_policy_report.json",
    "split_policy_report.md",
    "feature_schema_compatibility_report.json",
    "feature_schema_compatibility_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "prior_dataset_hash_report.json",
    "prior_dataset_hash_report.md",
    "no_training_report.json",
    "no_training_report.md",
    "no_runtime_report.json",
    "no_runtime_report.md",
    "no_rl_gdpo_ppo_report.json",
    "no_rl_gdpo_ppo_report.md",
    "future_stage4a710_expanded_dataset_qa_sketch.md",
    "recommended_next_faithful_step.md",
    "git_status_before.txt",
    "git_status_after.txt",
]
REQUIRED_ARRAYS = [
    "sample_id",
    "candidate_features_model",
    "candidate_valid_mask",
    "expert_action_index_primary",
    "expert_action_index_measured_shadow",
    "expert_action_index_lambda48_shadow",
    "expert_action_index_confidence_gated_shadow",
    "quality_keep_mask",
    "missing_feature_mask",
    "source_stage",
    "source_sample_id",
    "source_start_id",
    "source_step_id",
    "promotion_status",
    "human_review_status",
    "human_promote_candidate_yes_no",
    "primary_label_policy",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []
    checks["output_dir_exists"] = OUT.is_dir()
    for name in REQUIRED_FILES:
        checks[f"required:{name}"] = (OUT / name).is_file()

    if not all(checks[f"required:{name}"] for name in REQUIRED_FILES):
        result = {"all_passed": False, "checks": checks, "blockers": [k for k, v in checks.items() if not v]}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    summary = load_json(OUT / "stage4a79_no_training_promotion_summary.json")
    metadata = load_json(OUT / "expanded_primary_bc_dataset_metadata.json")
    mapping = load_json(OUT / "promotion_mapping_report.json")
    lineage = load_json(OUT / "primary_label_lineage_report.json")
    forbidden = load_json(OUT / "forbidden_field_audit.json")
    integrity = load_json(OUT / "expanded_dataset_integrity_report.json")
    feature = load_json(OUT / "feature_schema_compatibility_report.json")
    source_hash = load_json(OUT / "source_hash_report.json")
    prior_hash = load_json(OUT / "prior_dataset_hash_report.json")
    no_training = load_json(OUT / "no_training_report.json")
    no_runtime = load_json(OUT / "no_runtime_report.json")
    no_rl = load_json(OUT / "no_rl_gdpo_ppo_report.json")
    clean_rows = read_csv(STAGE78 / "clean_promotion_candidates.csv")
    rejected_rows = read_csv(STAGE78 / "rejected_rows.csv")
    unsure_rows = read_csv(STAGE78 / "unsure_rows.csv")
    conflict_rows = read_csv(STAGE78 / "conflict_manual_recheck_rows.csv")
    promoted_rows = read_csv(OUT / "promoted_stage4a72_rows.csv")
    sample_rows = read_csv(OUT / "expanded_sample_index_table.csv")

    expanded = np.load(OUT / "expanded_primary_bc_dataset.npz", allow_pickle=False)
    original = np.load(STAGE70 / "bc_dataset_primary_short_rollout.npz", allow_pickle=False)
    adapter = np.load(STAGE73 / "stage4a72_compact_v1_candidate_feature_adapter.npz", allow_pickle=False)

    checks["summary_completed"] = summary.get("completed") is True and summary.get("blocked") is False
    checks["metadata_completed"] = metadata.get("completed") is True and metadata.get("blocked") is False
    checks["expanded_sample_count_47"] = int(expanded["sample_id"].shape[0]) == 47
    checks["summary_counts"] = (
        summary.get("original_primary_samples") == 30
        and summary.get("clean_candidates_requested") == 17
        and summary.get("promoted_stage4a72_samples") == 17
        and summary.get("expanded_primary_samples") == 47
    )
    checks["required_arrays_present"] = all(key in expanded.files for key in REQUIRED_ARRAYS)
    checks["forbidden_fields_absent"] = not FORBIDDEN_KEYS.intersection(expanded.files) and forbidden.get("passed") is True
    checks["candidate_features_model_shape"] = tuple(expanded["candidate_features_model"].shape) == (47, 64, 16)
    checks["D_model_16"] = int(expanded["candidate_features_model"].shape[2]) == 16 and summary.get("D_model") == 16
    checks["original_30_sample_ids_retained"] = np.array_equal(expanded["sample_id"][:30], original["sample_id"])
    checks["original_30_features_retained"] = np.allclose(expanded["candidate_features_model"][:30], original["candidate_features_model"])
    checks["original_30_primary_retained"] = np.array_equal(
        expanded["expert_action_index_primary"][:30], original["expert_action_index_primary"]
    )
    clean_indices = np.asarray([int(row["adapter_sample_index"]) for row in clean_rows], dtype=np.int64)
    checks["promoted_sample_ids_match_clean"] = np.array_equal(
        expanded["sample_id"][30:], adapter["sample_id"][clean_indices]
    )
    checks["promoted_features_match_adapter"] = np.allclose(
        expanded["candidate_features_model"][30:], adapter["candidate_features_model"][clean_indices]
    )
    checks["promoted_primary_from_uncertainty_bonus"] = np.array_equal(
        expanded["expert_action_index_primary"][30:],
        adapter["candidate_action_index_uncertainty_bonus_executed"][clean_indices],
    )
    lambda48 = adapter["expert_action_index_lambda48_shadow"][clean_indices]
    primary = adapter["candidate_action_index_uncertainty_bonus_executed"][clean_indices]
    distinguish = primary != lambda48
    checks["lambda48_not_used_as_primary"] = (
        lineage.get("lambda48_primary_use") is False
        and summary.get("lambda48_primary_use") is False
        and bool(np.any(distinguish))
        and np.array_equal(expanded["expert_action_index_primary"][30:][distinguish], primary[distinguish])
        and not np.array_equal(expanded["expert_action_index_primary"][30:][distinguish], lambda48[distinguish])
    )
    checks["shadow_lambda48_retained"] = np.array_equal(expanded["expert_action_index_lambda48_shadow"][30:], lambda48)
    valid_mask = expanded["candidate_valid_mask"]
    expert = expanded["expert_action_index_primary"].astype(np.int64)
    valid_range = (expert >= 0) & (expert < valid_mask.shape[1])
    checks["all_primary_labels_valid"] = bool(np.all(valid_range))
    checks["candidate_valid_mask_true_at_primary"] = bool(np.all(valid_mask[np.arange(expert.shape[0]), expert]))
    checks["quality_keep_mask_all_true"] = bool(np.all(expanded["quality_keep_mask"]))
    checks["manifest_47_rows"] = sum(1 for _ in (OUT / "expanded_primary_bc_dataset_manifest.jsonl").open(encoding="utf-8")) == 47
    checks["sample_index_47_rows"] = len(sample_rows) == 47
    checks["promoted_rows_17"] = len(promoted_rows) == 17

    promoted_set = {row["sample_id"] for row in promoted_rows}
    checks["no_rejected_promoted"] = not promoted_set.intersection({row["sample_id"] for row in rejected_rows})
    checks["no_unsure_promoted"] = not promoted_set.intersection({row["sample_id"] for row in unsure_rows})
    checks["no_conflict_promoted"] = not promoted_set.intersection({row["sample_id"] for row in conflict_rows})
    checks["excluded_rows_12_unique"] = summary.get("excluded_stage4a72_samples") == 12
    checks["mapping_report_clean"] = (
        mapping.get("all_clean_candidates_mapped_exactly") is True
        and mapping.get("promoted_stage4a72_samples") == 17
        and mapping.get("rejected_rows_promoted") == []
        and mapping.get("unsure_rows_promoted") == []
        and mapping.get("conflict_rows_promoted") == []
    )
    checks["integrity_report_passes"] = (
        integrity.get("candidate_valid_mask_true_at_primary") is True
        and integrity.get("all_primary_labels_valid") is True
        and integrity.get("D_model") == 16
    )
    checks["feature_schema_compatible"] = feature.get("compatible") is True
    checks["source_hashes_unchanged"] = all(item.get("unchanged") is True for item in source_hash.values() if item.get("exists"))
    checks["prior_dataset_hashes_unchanged"] = all(item.get("unchanged") is True for item in prior_hash.values() if item.get("exists"))
    checks["no_training"] = (
        summary.get("bc_training") is False
        and summary.get("optimizer_step") is False
        and no_training.get("training") is False
        and no_training.get("optimizer_step") is False
    )
    checks["no_checkpoint"] = summary.get("checkpoint") is False and no_training.get("checkpoint") is False
    checks["no_isaac"] = summary.get("isaac_startup") is False and no_runtime.get("isaac_startup") is False
    checks["no_map_predict"] = summary.get("map_predict") is False and no_runtime.get("map_predict") is False
    checks["no_rollout"] = summary.get("rollout") is False and no_runtime.get("rollout") is False
    checks["no_rl_gdpo_ppo"] = summary.get("rl_gdpo_ppo") is False and all(v is False for v in no_rl.values())
    checks["prior_datasets_not_modified"] = summary.get("prior_datasets_modified") is False
    checks["future_sketch_first_line"] = (
        (OUT / "future_stage4a710_expanded_dataset_qa_sketch.md").read_text(encoding="utf-8").splitlines()[0]
        == "DO NOT RUN IN STAGE 4A-7.9."
    )

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
        "promoted_samples": len(promoted_rows),
        "forbidden_tracked": forbidden_tracked,
        "large_tracked": large_tracked,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
