from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation"
STAGE70 = ROOT / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
STAGE72 = ROOT / "outputs/isaac_stage4a72_bounded_short_rollout_runtime"
STAGE72_QA = ROOT / "outputs/isaac_stage4a72_bounded_short_rollout_runtime_qa_review"
STAGE73 = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_no_training"
STAGE73_QA = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_qa_review"
STAGE78 = ROOT / "outputs/isaac_stage4a78_promotion_candidate_decision_packet"

PRIMARY_POLICY_ORIGINAL = "stage4a613_uncertainty_bonus_executed_primary"
PRIMARY_POLICY_PROMOTED = (
    "stage4a613_uncertainty_bonus_executed_primary_from_"
    "stage4a72_uncertainty_bonus_composite_beta8_executed"
)
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
REQUIRED_OUTPUTS = [
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


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        field_set: list[str] = []
        for row in rows:
            for key in row:
                if key not in field_set:
                    field_set.append(key)
        fields = field_set
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_report(paths: dict[str, Path]) -> dict[str, dict]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256_before": sha256_file(path),
            "sha256_after": None,
            "unchanged": None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in paths.items()
    }


def finalize_hash_report(report: dict[str, dict]) -> dict[str, dict]:
    for item in report.values():
        path = Path(item["path"])
        item["sha256_after"] = sha256_file(path)
        item["unchanged"] = item["sha256_before"] == item["sha256_after"]
    return report


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True)


def as_int(value) -> int:
    return int(str(value).strip())


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def np_str(values, width: int = 160) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=f"<U{width}")


def concat_1d(a: np.ndarray, b: np.ndarray, dtype=None) -> np.ndarray:
    if dtype is not None:
        return np.concatenate([a.astype(dtype), b.astype(dtype)])
    return np.concatenate([a, b])


def markdown_table(rows: list[tuple[str, object]]) -> str:
    lines = ["| field | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def validate_clean_rows(clean_rows: list[dict], adapter_index_rows: list[dict], adapter_npz: np.lib.npyio.NpzFile) -> list[str]:
    blockers: list[str] = []
    if len(clean_rows) != 17:
        blockers.append(f"clean_candidate_count_expected_17_got_{len(clean_rows)}")
    adapter_by_sample = {row["sample_id"]: row for row in adapter_index_rows}
    seen: set[str] = set()
    for row in clean_rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            blockers.append(f"duplicate_clean_sample:{sample_id}")
        seen.add(sample_id)
        adapter_row = adapter_by_sample.get(sample_id)
        if adapter_row is None:
            blockers.append(f"missing_adapter_row:{sample_id}")
            continue
        adapter_idx = as_int(row["adapter_sample_index"])
        if adapter_idx != as_int(adapter_row["sample_index"]):
            blockers.append(f"adapter_index_mismatch:{sample_id}")
        if str(adapter_npz["sample_id"][adapter_idx]) != sample_id:
            blockers.append(f"adapter_npz_sample_mismatch:{sample_id}")
        if as_int(row["start_id"]) != int(adapter_npz["start_variant_id"][adapter_idx]):
            blockers.append(f"start_id_mismatch:{sample_id}")
        if as_int(row["step_id"]) != int(adapter_npz["step_id"][adapter_idx]):
            blockers.append(f"step_id_mismatch:{sample_id}")
        primary = int(adapter_npz["candidate_action_index_uncertainty_bonus_executed"][adapter_idx])
        if as_int(row["candidate_action_index_uncertainty_bonus_executed"]) != primary:
            blockers.append(f"uncertainty_bonus_primary_mismatch:{sample_id}")
        if not as_bool(row["source_row_exists_in_stage4a73_adapter"]):
            blockers.append(f"clean_row_source_missing:{sample_id}")
        if not as_bool(row["candidate_label_valid"]):
            blockers.append(f"clean_row_label_invalid:{sample_id}")
        if not as_bool(row["quality_keep"]):
            blockers.append(f"clean_row_quality_not_keep:{sample_id}")
        if row["human_review_status"] != "approve":
            blockers.append(f"clean_row_not_approve:{sample_id}")
        if row["promote_candidate_yes_no"] != "yes":
            blockers.append(f"clean_row_not_promote_yes:{sample_id}")
    return blockers


def make_excluded_rows() -> list[dict]:
    rows_by_sample: dict[str, dict] = {}
    for source_name, path in [
        ("rejected", STAGE78 / "rejected_rows.csv"),
        ("unsure", STAGE78 / "unsure_rows.csv"),
        ("conflict_manual_recheck", STAGE78 / "conflict_manual_recheck_rows.csv"),
        ("manual_recheck", STAGE78 / "manual_recheck_rows.csv"),
    ]:
        for row in read_csv(path):
            sample_id = row.get("sample_id", "")
            if not sample_id:
                continue
            existing = rows_by_sample.setdefault(sample_id, dict(row))
            existing["excluded_source"] = (
                f"{existing.get('excluded_source')};{source_name}" if existing.get("excluded_source") else source_name
            )
            if row.get("decision_bucket"):
                existing["excluded_reason"] = row["decision_bucket"]
            elif source_name == "manual_recheck":
                existing["excluded_reason"] = "manual_recheck"
            else:
                existing["excluded_reason"] = source_name
    return sorted(rows_by_sample.values(), key=lambda r: (as_int(r.get("start_id", 0)), as_int(r.get("step_id", 0))))


def build_previous_next(sequence_ids: np.ndarray, step_ids: np.ndarray, global_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    prev = np.full(len(global_indices), -1, dtype=np.int64)
    nxt = np.full(len(global_indices), -1, dtype=np.int64)
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for local_i, global_i in enumerate(global_indices):
        groups[int(sequence_ids[local_i])].append((int(step_ids[local_i]), global_i))
    global_to_local = {global_i: local_i for local_i, global_i in enumerate(global_indices)}
    for group in groups.values():
        ordered = [idx for _, idx in sorted(group)]
        for pos, global_i in enumerate(ordered):
            local_i = global_to_local[global_i]
            if pos > 0:
                prev[local_i] = ordered[pos - 1]
            if pos + 1 < len(ordered):
                nxt[local_i] = ordered[pos + 1]
    return prev, nxt


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    source_paths = {
        "stage4a78_summary": STAGE78 / "stage4a78_promotion_candidate_decision_summary.json",
        "stage4a78_clean_candidates": STAGE78 / "clean_promotion_candidates.csv",
        "stage4a78_rejected_rows": STAGE78 / "rejected_rows.csv",
        "stage4a78_unsure_rows": STAGE78 / "unsure_rows.csv",
        "stage4a78_conflict_rows": STAGE78 / "conflict_manual_recheck_rows.csv",
        "stage4a78_manual_recheck_rows": STAGE78 / "manual_recheck_rows.csv",
        "stage4a78_policy": STAGE78 / "promotion_candidate_policy.json",
        "stage4a73_adapter_index": STAGE73 / "stage4a72_adapter_sample_index_table.csv",
        "stage4a73_adapter_npz": STAGE73 / "stage4a72_compact_v1_candidate_feature_adapter.npz",
        "stage4a70_primary_npz": STAGE70 / "bc_dataset_primary_short_rollout.npz",
        "stage4a70_metadata": STAGE70 / "bc_dataset_metadata.json",
        "stage4a70_manifest": STAGE70 / "bc_dataset_manifest.jsonl",
        "stage4a70_feature_names_model": STAGE70 / "feature_names_model.json",
        "stage4a70_feature_names_raw": STAGE70 / "feature_names_raw.json",
        "stage4a70_split_assignments": STAGE70 / "split_assignments.csv",
        "stage4a70_normalization_stats": STAGE70 / "normalization_stats.npz",
        "stage4a72_runtime_summary": STAGE72 / "stage4a72_bounded_short_rollout_runtime_summary.json",
    }
    prior_paths = {
        "stage4a70_primary_dataset": STAGE70 / "bc_dataset_primary_short_rollout.npz",
        "stage4a70_metadata": STAGE70 / "bc_dataset_metadata.json",
        "stage4a70_manifest": STAGE70 / "bc_dataset_manifest.jsonl",
        "stage4a70_feature_names_model": STAGE70 / "feature_names_model.json",
        "stage4a70_feature_names_raw": STAGE70 / "feature_names_raw.json",
        "stage4a70_split_assignments": STAGE70 / "split_assignments.csv",
        "stage4a70_normalization_stats": STAGE70 / "normalization_stats.npz",
        "stage4a73_adapter_dataset": STAGE73 / "stage4a72_compact_v1_candidate_feature_adapter.npz",
        "stage4a73_adapter_sample_index": STAGE73 / "stage4a72_adapter_sample_index_table.csv",
        "stage4a78_clean_candidates": STAGE78 / "clean_promotion_candidates.csv",
    }
    source_hashes = hash_report(source_paths)
    prior_hashes = hash_report(prior_paths)

    missing_sources = [name for name, path in source_paths.items() if not path.exists() and name != "stage4a72_runtime_summary"]
    if missing_sources:
        write_json(OUT / "stage4a79_no_training_promotion_summary.json", {"completed": False, "blocked": True, "blockers": missing_sources})
        raise SystemExit(f"Missing required sources: {missing_sources}")

    summary78 = read_json(source_paths["stage4a78_summary"])
    metadata70 = read_json(source_paths["stage4a70_metadata"])
    clean_rows = read_csv(source_paths["stage4a78_clean_candidates"])
    excluded_rows = make_excluded_rows()
    adapter_index_rows = read_csv(source_paths["stage4a73_adapter_index"])

    npz70 = np.load(source_paths["stage4a70_primary_npz"], allow_pickle=False)
    npz73 = np.load(source_paths["stage4a73_adapter_npz"], allow_pickle=False)
    blockers = validate_clean_rows(clean_rows, adapter_index_rows, npz73)
    if summary78.get("clean_promotion_candidates") != len(clean_rows):
        blockers.append("stage4a78_summary_clean_count_mismatch")
    if summary78.get("conflict_rows") != 0:
        blockers.append("stage4a78_conflict_rows_not_zero")
    if summary78.get("label_promotion") is not False or summary78.get("expert_action_index_primary_created") is not False:
        blockers.append("stage4a78_was_not_no_promotion")

    original_count = int(npz70["sample_id"].shape[0])
    clean_indices = np.asarray([as_int(row["adapter_sample_index"]) for row in clean_rows], dtype=np.int64)
    promoted_count = int(clean_indices.shape[0])
    expanded_count = original_count + promoted_count
    max_original_sequence = int(np.max(npz70["sequence_id"])) if original_count else -1
    promoted_sequence_id = npz73["sequence_id"][clean_indices].astype(np.int64) + max_original_sequence + 1
    promoted_global_indices = list(range(original_count, expanded_count))
    promoted_prev, promoted_next = build_previous_next(
        promoted_sequence_id,
        npz73["step_id"][clean_indices].astype(np.int64),
        promoted_global_indices,
    )

    promoted_primary = npz73["candidate_action_index_uncertainty_bonus_executed"][clean_indices].astype(np.int64)
    promoted_lambda48 = npz73["expert_action_index_lambda48_shadow"][clean_indices].astype(np.int64)
    primary_differs_from_lambda48 = promoted_primary != promoted_lambda48
    if not bool(np.any(primary_differs_from_lambda48)):
        blockers.append("no_clean_candidate_distinguishes_uncertainty_bonus_primary_from_lambda48_shadow")

    arrays: dict[str, np.ndarray] = {}
    arrays["sample_id"] = np_str(list(npz70["sample_id"]) + [str(x) for x in npz73["sample_id"][clean_indices]])
    arrays["start_variant_id"] = concat_1d(npz70["start_variant_id"], npz73["start_variant_id"][clean_indices], np.int64)
    arrays["step_id"] = concat_1d(npz70["step_id"], npz73["step_id"][clean_indices], np.int64)
    arrays["source_stage_id"] = concat_1d(npz70["source_stage_id"], npz73["source_stage_id"][clean_indices], np.int64)
    for key in ["candidate_features_raw", "candidate_features_model", "candidate_feature_mask", "candidate_valid_mask"]:
        arrays[key] = np.concatenate([npz70[key], npz73[key][clean_indices]], axis=0)
    arrays["missing_feature_mask"] = np.concatenate([npz70["missing_feature_mask"], npz73["missing_feature_mask"][clean_indices]], axis=0)
    arrays["expert_action_index_primary"] = concat_1d(npz70["expert_action_index_primary"], promoted_primary, np.int64)
    arrays["expert_action_index_measured_shadow"] = concat_1d(
        npz70["expert_action_index_measured_shadow"], npz73["expert_action_index_measured_shadow"][clean_indices], np.int64
    )
    arrays["expert_action_index_lambda48_shadow"] = concat_1d(
        npz70["expert_action_index_lambda48_shadow"], promoted_lambda48, np.int64
    )
    arrays["expert_action_index_confidence_gated_shadow"] = concat_1d(
        npz70["expert_action_index_confidence_gated_shadow"],
        npz73["expert_action_index_confidence_gated_shadow"][clean_indices],
        np.int64,
    )
    arrays["score_primary"] = np.concatenate([npz70["score_primary"], npz73["score_uncertainty_bonus_executed"][clean_indices]], axis=0)
    arrays["score_measured"] = np.concatenate([npz70["score_measured"], npz73["score_measured_shadow"][clean_indices]], axis=0)
    arrays["score_lambda48"] = np.concatenate([npz70["score_lambda48"], npz73["score_lambda48_shadow"][clean_indices]], axis=0)
    arrays["score_confidence_gated"] = np.concatenate(
        [npz70["score_confidence_gated"], npz73["score_confidence_gated_shadow"][clean_indices]], axis=0
    )
    arrays["selected_world_xyz_primary"] = np.concatenate(
        [npz70["selected_world_xyz_primary"], npz73["selected_world_xyz_uncertainty_bonus"][clean_indices]], axis=0
    )
    arrays["selected_yaw_primary"] = concat_1d(
        npz70["selected_yaw_primary"], npz73["selected_yaw_uncertainty_bonus"][clean_indices], np.float32
    )
    arrays["pose_world_xyz"] = np.concatenate([npz70["pose_world_xyz"], npz73["pose_world_xyz"][clean_indices]], axis=0)
    arrays["pose_yaw"] = concat_1d(npz70["pose_yaw"], npz73["pose_yaw"][clean_indices], np.float32)
    arrays["sequence_id"] = concat_1d(npz70["sequence_id"], promoted_sequence_id, np.int64)
    arrays["previous_sample_index"] = concat_1d(npz70["previous_sample_index"], promoted_prev, np.int64)
    arrays["next_sample_index"] = concat_1d(npz70["next_sample_index"], promoted_next, np.int64)
    arrays["split_id"] = concat_1d(npz70["split_id"], npz73["source_split_id_from_adapter"][clean_indices], np.int64)
    arrays["quality_keep_mask"] = concat_1d(npz70["quality_keep_mask"], npz73["quality_keep_mask"][clean_indices], np.bool_)
    arrays["sample_weight"] = concat_1d(npz70["sample_weight"], npz73["sample_weight"][clean_indices], np.float32)

    arrays["source_stage"] = np_str(["stage4a70_original_primary"] * original_count + ["stage4a72_human_review_promoted_primary"] * promoted_count)
    arrays["source_sample_id"] = np_str(list(npz70["sample_id"]) + [str(x) for x in npz73["sample_id"][clean_indices]])
    arrays["source_start_id"] = arrays["start_variant_id"].astype(np.int64)
    arrays["source_step_id"] = arrays["step_id"].astype(np.int64)
    arrays["promotion_status"] = np_str(["stage4a70_original_primary"] * original_count + ["promoted_clean_candidate"] * promoted_count)
    arrays["promotion_source"] = np_str(["stage4a70_original_primary"] * original_count + ["stage4a78_clean_promotion_candidate"] * promoted_count)
    arrays["human_review_status"] = np_str(["not_applicable"] * original_count + [row["human_review_status"] for row in clean_rows])
    arrays["human_review_reason"] = np_str(["not_applicable"] * original_count + [row["human_review_reason"] for row in clean_rows])
    arrays["human_comment"] = np_str([""] * original_count + [row.get("human_comment", "") for row in clean_rows], width=512)
    arrays["human_promote_candidate_yes_no"] = np_str([""] * original_count + [row["promote_candidate_yes_no"] for row in clean_rows])
    arrays["primary_label_policy"] = np_str([PRIMARY_POLICY_ORIGINAL] * original_count + [PRIMARY_POLICY_PROMOTED] * promoted_count)
    arrays["source_adapter_sample_index"] = concat_1d(
        np.full(original_count, -1, dtype=np.int64), clean_indices.astype(np.int64), np.int64
    )

    candidate_valid = arrays["candidate_valid_mask"]
    primary = arrays["expert_action_index_primary"].astype(np.int64)
    valid_index_range = (primary >= 0) & (primary < candidate_valid.shape[1])
    valid_at_primary = valid_index_range & candidate_valid[np.arange(expanded_count), primary]
    if not np.all(valid_at_primary):
        bad = np.where(~valid_at_primary)[0].tolist()
        blockers.append(f"invalid_primary_candidate_indices:{bad[:10]}")
    if arrays["candidate_features_model"].shape != (expanded_count, 64, 16):
        blockers.append(f"unexpected_candidate_features_model_shape:{arrays['candidate_features_model'].shape}")
    if FORBIDDEN_KEYS.intersection(arrays):
        blockers.append("forbidden_keys_in_expanded_dataset")

    if blockers:
        blocked_summary = {
            "completed": False,
            "blocked": True,
            "blockers": blockers,
            "stage": "Stage 4A-7.9",
            "output_dir": str(OUT),
        }
        write_json(OUT / "stage4a79_no_training_promotion_summary.json", blocked_summary)
        (OUT / "stage4a79_no_training_promotion_summary.md").write_text(
            "# Stage 4A-7.9 No-Training Promotion Summary\n\n"
            + markdown_table([("completed", False), ("blocked", True), ("blockers", "; ".join(blockers))]),
            encoding="utf-8",
        )
        raise SystemExit(f"Stage 4A-7.9 blocked: {blockers}")

    dataset_path = OUT / "expanded_primary_bc_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)

    promoted_rows: list[dict] = []
    clean_by_sample = {row["sample_id"]: row for row in clean_rows}
    for pos, idx in enumerate(clean_indices):
        row = clean_by_sample[str(npz73["sample_id"][idx])]
        promoted_rows.append(
            {
                **row,
                "expanded_sample_index": original_count + pos,
                "primary_label_source": "candidate_action_index_uncertainty_bonus_executed",
                "primary_label_policy": PRIMARY_POLICY_PROMOTED,
                "expert_action_index_primary": int(promoted_primary[pos]),
                "expert_action_index_lambda48_shadow": int(promoted_lambda48[pos]),
                "primary_equals_lambda48_shadow": bool(promoted_primary[pos] == promoted_lambda48[pos]),
                "promotion_source": "stage4a78_clean_promotion_candidate",
            }
        )
    write_csv(OUT / "promoted_stage4a72_rows.csv", promoted_rows)
    write_csv(OUT / "excluded_stage4a72_rows.csv", excluded_rows)
    write_json(OUT / "promoted_stage4a72_rows.json", promoted_rows)
    write_json(OUT / "excluded_stage4a72_rows.json", excluded_rows)

    sample_rows: list[dict] = []
    for i in range(expanded_count):
        sample_rows.append(
            {
                "expanded_sample_index": i,
                "sample_id": str(arrays["sample_id"][i]),
                "source_stage": str(arrays["source_stage"][i]),
                "source_sample_id": str(arrays["source_sample_id"][i]),
                "source_start_id": int(arrays["source_start_id"][i]),
                "source_step_id": int(arrays["source_step_id"][i]),
                "promotion_status": str(arrays["promotion_status"][i]),
                "promotion_source": str(arrays["promotion_source"][i]),
                "human_review_status": str(arrays["human_review_status"][i]),
                "human_review_reason": str(arrays["human_review_reason"][i]),
                "human_promote_candidate_yes_no": str(arrays["human_promote_candidate_yes_no"][i]),
                "primary_label_policy": str(arrays["primary_label_policy"][i]),
                "expert_action_index_primary": int(arrays["expert_action_index_primary"][i]),
                "expert_action_index_measured_shadow": int(arrays["expert_action_index_measured_shadow"][i]),
                "expert_action_index_lambda48_shadow": int(arrays["expert_action_index_lambda48_shadow"][i]),
                "expert_action_index_confidence_gated_shadow": int(arrays["expert_action_index_confidence_gated_shadow"][i]),
                "primary_equals_lambda48_shadow": bool(
                    arrays["expert_action_index_primary"][i] == arrays["expert_action_index_lambda48_shadow"][i]
                ),
                "split_id": int(arrays["split_id"][i]),
                "sequence_id": int(arrays["sequence_id"][i]),
                "previous_sample_index": int(arrays["previous_sample_index"][i]),
                "next_sample_index": int(arrays["next_sample_index"][i]),
                "quality_keep": bool(arrays["quality_keep_mask"][i]),
                "sample_weight": float(arrays["sample_weight"][i]),
                "source_adapter_sample_index": int(arrays["source_adapter_sample_index"][i]),
            }
        )
    write_csv(OUT / "expanded_sample_index_table.csv", sample_rows)
    with (OUT / "expanded_primary_bc_dataset_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    source_stage_counts = Counter(str(x) for x in arrays["source_stage"])
    split_counts = Counter(int(x) for x in arrays["split_id"])
    split_by_source = Counter((str(arrays["source_stage"][i]), int(arrays["split_id"][i])) for i in range(expanded_count))
    excluded_status_counts = Counter(row.get("human_review_status", "") for row in excluded_rows)
    clean_promoted_set = {row["sample_id"] for row in promoted_rows}
    excluded_set = {row["sample_id"] for row in excluded_rows}
    rejected_promoted = sorted(clean_promoted_set.intersection({row["sample_id"] for row in read_csv(STAGE78 / "rejected_rows.csv")}))
    unsure_promoted = sorted(clean_promoted_set.intersection({row["sample_id"] for row in read_csv(STAGE78 / "unsure_rows.csv")}))
    conflict_promoted = sorted(clean_promoted_set.intersection({row["sample_id"] for row in read_csv(STAGE78 / "conflict_manual_recheck_rows.csv")}))

    model_names70 = read_json(STAGE70 / "feature_names_model.json")
    raw_names70 = read_json(STAGE70 / "feature_names_raw.json")
    adapter_model_names = [str(x) for x in npz73["model_feature_names"]]
    adapter_raw_names = [str(x) for x in npz73["raw_feature_names"]]
    feature_report = {
        "stage4a70_model_feature_dim": int(npz70["candidate_features_model"].shape[2]),
        "stage4a73_model_feature_dim": int(npz73["candidate_features_model"].shape[2]),
        "expanded_model_feature_dim": int(arrays["candidate_features_model"].shape[2]),
        "stage4a70_raw_feature_dim": int(npz70["candidate_features_raw"].shape[2]),
        "stage4a73_raw_feature_dim": int(npz73["candidate_features_raw"].shape[2]),
        "expanded_raw_feature_dim": int(arrays["candidate_features_raw"].shape[2]),
        "model_feature_names_match": model_names70 == adapter_model_names,
        "raw_feature_names_match": raw_names70 == adapter_raw_names,
        "model_feature_names_count": len(model_names70),
        "raw_feature_names_count": len(raw_names70),
        "candidate_count": int(arrays["candidate_features_model"].shape[1]),
        "compatible": (
            int(arrays["candidate_features_model"].shape[2]) == 16
            and model_names70 == adapter_model_names
            and raw_names70 == adapter_raw_names
        ),
    }

    forbidden_report = {
        "forbidden_keys": sorted(FORBIDDEN_KEYS),
        "expanded_dataset_keys": sorted(arrays),
        "forbidden_keys_present": sorted(FORBIDDEN_KEYS.intersection(arrays)),
        "passed": not FORBIDDEN_KEYS.intersection(arrays),
    }
    integrity_report = {
        "expanded_dataset_path": str(dataset_path),
        "expanded_sample_count": expanded_count,
        "original_primary_sample_count": original_count,
        "promoted_stage4a72_sample_count": promoted_count,
        "candidate_features_model_shape": list(arrays["candidate_features_model"].shape),
        "candidate_valid_mask_shape": list(arrays["candidate_valid_mask"].shape),
        "D_model": int(arrays["candidate_features_model"].shape[2]),
        "all_primary_labels_valid": bool(np.all(valid_index_range)),
        "candidate_valid_mask_true_at_primary": bool(np.all(valid_at_primary)),
        "quality_keep_true_count": int(np.sum(arrays["quality_keep_mask"])),
        "source_stage_counts": dict(source_stage_counts),
        "required_sim_expert_bc_dataset_arrays_present": all(
            key in arrays
            for key in [
                "sample_id",
                "candidate_features_model",
                "candidate_valid_mask",
                "expert_action_index_primary",
                "expert_action_index_measured_shadow",
                "expert_action_index_lambda48_shadow",
                "expert_action_index_confidence_gated_shadow",
                "quality_keep_mask",
                "missing_feature_mask",
            ]
        ),
    }
    lineage_report = {
        "primary_label_lineage_preserved": True,
        "lambda48_primary_use": False,
        "lambda48_role": "shadow/baseline only",
        "original_primary_policy": PRIMARY_POLICY_ORIGINAL,
        "promoted_primary_policy": PRIMARY_POLICY_PROMOTED,
        "promoted_primary_label_source_array": "candidate_action_index_uncertainty_bonus_executed",
        "promoted_shadow_label_source_array": "expert_action_index_lambda48_shadow",
        "promoted_rows_where_primary_differs_from_lambda48": int(np.sum(primary_differs_from_lambda48)),
        "promoted_rows_where_primary_equals_lambda48_by_value": int(np.sum(~primary_differs_from_lambda48)),
        "note": "Equal integer values are allowed when policies select the same candidate; lambda48 was not used as the source for primary.",
    }
    mapping_report = {
        "clean_candidates_requested": len(clean_rows),
        "promoted_stage4a72_samples": promoted_count,
        "mapping_blockers": [],
        "all_clean_candidates_mapped_exactly": True,
        "promoted_sample_ids": [row["sample_id"] for row in promoted_rows],
        "excluded_stage4a72_rows": len(excluded_rows),
        "excluded_status_counts": dict(excluded_status_counts),
        "rejected_rows_promoted": rejected_promoted,
        "unsure_rows_promoted": unsure_promoted,
        "conflict_rows_promoted": conflict_promoted,
    }
    split_report = {
        "policy": "preserve Stage 4A-7.0 split_id for original samples and Stage 4A-7.3 source_split_id_from_adapter for promoted samples; no training split is consumed in Stage 4A-7.9.",
        "split_counts": dict(split_counts),
        "split_counts_by_source_stage": {f"{source}|{split}": count for (source, split), count in split_by_source.items()},
        "no_training_consumed_splits": True,
    }
    metadata = {
        "stage": "Stage 4A-7.9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": True,
        "blocked": False,
        "output_dir": str(OUT),
        "expanded_primary_dataset": str(dataset_path),
        "original_primary_samples": original_count,
        "clean_candidates_requested": len(clean_rows),
        "promoted_stage4a72_samples": promoted_count,
        "excluded_stage4a72_samples": len(excluded_rows),
        "expanded_primary_samples": expanded_count,
        "candidate_count": int(arrays["candidate_features_model"].shape[1]),
        "D_model": int(arrays["candidate_features_model"].shape[2]),
        "source_stage_counts": dict(source_stage_counts),
        "primary_label_policy": PRIMARY_POLICY_PROMOTED,
        "original_primary_policy": PRIMARY_POLICY_ORIGINAL,
        "lambda48_primary_use": False,
        "lambda48_role": "shadow/baseline only",
        "prior_stage4a70_metadata": metadata70,
        "no_training": True,
        "no_optimizer_step": True,
        "no_checkpoint": True,
        "no_isaac_startup": True,
        "no_map_predict": True,
        "no_rollout": True,
        "no_rl_gdpo_ppo": True,
        "prior_datasets_modified": False,
    }

    no_training_report = {
        "bc_training": False,
        "training": False,
        "optimizer_step": False,
        "checkpoint": False,
        "model_save": False,
        "note": "Stage 4A-7.9 only materialized an expanded dataset and reports.",
    }
    no_runtime_report = {
        "isaac_startup": False,
        "capture": False,
        "map_predict": False,
        "action_execution": False,
        "rollout": False,
        "long_rollout": False,
        "runtime_dirs_read_only": [str(STAGE72), str(STAGE72_QA), str(STAGE73), str(STAGE73_QA)],
    }
    no_rl_report = {"rl": False, "gdpo": False, "ppo": False, "policy_optimization": False}

    write_json(OUT / "expanded_primary_bc_dataset_metadata.json", metadata)
    write_json(OUT / "promotion_mapping_report.json", mapping_report)
    write_json(OUT / "primary_label_lineage_report.json", lineage_report)
    write_json(OUT / "forbidden_field_audit.json", forbidden_report)
    write_json(OUT / "expanded_dataset_integrity_report.json", integrity_report)
    write_json(OUT / "split_policy_report.json", split_report)
    write_json(OUT / "feature_schema_compatibility_report.json", feature_report)
    write_json(OUT / "no_training_report.json", no_training_report)
    write_json(OUT / "no_runtime_report.json", no_runtime_report)
    write_json(OUT / "no_rl_gdpo_ppo_report.json", no_rl_report)

    write_json(OUT / "source_hash_report.json", finalize_hash_report(source_hashes))
    write_json(OUT / "prior_dataset_hash_report.json", finalize_hash_report(prior_hashes))

    (OUT / "promotion_mapping_report.md").write_text(
        "# Promotion Mapping Report\n\n"
        + markdown_table(
            [
                ("clean candidates requested", len(clean_rows)),
                ("promoted Stage 4A-7.2 samples", promoted_count),
                ("excluded Stage 4A-7.2 rows", len(excluded_rows)),
                ("rejected rows promoted", len(rejected_promoted)),
                ("unsure rows promoted", len(unsure_promoted)),
                ("conflict rows promoted", len(conflict_promoted)),
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "primary_label_lineage_report.md").write_text(
        "# Primary Label Lineage Report\n\n"
        + markdown_table(
            [
                ("original primary policy", PRIMARY_POLICY_ORIGINAL),
                ("promoted primary source", "candidate_action_index_uncertainty_bonus_executed"),
                ("promoted policy", PRIMARY_POLICY_PROMOTED),
                ("lambda48 primary use", False),
                ("lambda48 role", "shadow/baseline only"),
                ("primary differs from lambda48 rows", int(np.sum(primary_differs_from_lambda48))),
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "forbidden_field_audit.md").write_text(
        "# Forbidden Field Audit\n\n"
        + markdown_table(
            [
                ("passed", forbidden_report["passed"]),
                ("forbidden keys present", ", ".join(forbidden_report["forbidden_keys_present"]) or "none"),
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "expanded_dataset_integrity_report.md").write_text(
        "# Expanded Dataset Integrity Report\n\n"
        + markdown_table(
            [
                ("expanded samples", expanded_count),
                ("candidate_features_model shape", list(arrays["candidate_features_model"].shape)),
                ("D_model", int(arrays["candidate_features_model"].shape[2])),
                ("all primary labels valid", bool(np.all(valid_index_range))),
                ("candidate_valid_mask true at primary", bool(np.all(valid_at_primary))),
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "split_policy_report.md").write_text(
        "# Split Policy Report\n\n"
        + markdown_table(
            [
                ("policy", split_report["policy"]),
                ("split counts", dict(split_counts)),
                ("no training consumed splits", True),
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "feature_schema_compatibility_report.md").write_text(
        "# Feature Schema Compatibility Report\n\n"
        + markdown_table(
            [
                ("compatible", feature_report["compatible"]),
                ("expanded D_model", feature_report["expanded_model_feature_dim"]),
                ("model feature names match", feature_report["model_feature_names_match"]),
                ("raw feature names match", feature_report["raw_feature_names_match"]),
            ]
        ),
        encoding="utf-8",
    )
    for name, report in [("source_hash_report", source_hashes), ("prior_dataset_hash_report", prior_hashes)]:
        rows = [(key, f"{value['sha256_before']} -> {value['sha256_after']} unchanged={value['unchanged']}") for key, value in report.items()]
        (OUT / f"{name}.md").write_text(f"# {name.replace('_', ' ').title()}\n\n" + markdown_table(rows), encoding="utf-8")
    (OUT / "no_training_report.md").write_text(
        "# No Training Report\n\nBC training, optimizer step, model save, and checkpoint creation were not run.\n",
        encoding="utf-8",
    )
    (OUT / "no_runtime_report.md").write_text(
        "# No Runtime Report\n\nIsaac startup, capture, map_predict, action execution, rollout, and long rollout were not run.\n",
        encoding="utf-8",
    )
    (OUT / "no_rl_gdpo_ppo_report.md").write_text(
        "# No RL/GDPO/PPO Report\n\nRL, GDPO, PPO, and policy optimization were not run.\n",
        encoding="utf-8",
    )
    (OUT / "future_stage4a710_expanded_dataset_qa_sketch.md").write_text(
        "DO NOT RUN IN STAGE 4A-7.9.\n"
        "\n"
        "Future Stage 4A-7.10 should perform expanded dataset QA only: reload the NPZ, audit shapes, compare source rows, inspect promoted examples, verify split handling, and run no-training forward-only loader smoke if explicitly approved.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_next_faithful_step.md").write_text(
        "Recommended next: Stage 4A-7.10 expanded dataset QA, not training yet.\n"
        "\n"
        "Why: Stage 4A-7.9 changes the primary BC dataset composition, so the next faithful step is integrity and loader QA before any BC dry-run or training discussion.\n",
        encoding="utf-8",
    )

    summary = {
        "stage": "Stage 4A-7.9",
        "completed": True,
        "blocked": False,
        "blockers": [],
        "output_dir": str(OUT),
        "expanded_primary_dataset": str(dataset_path),
        "original_primary_samples": original_count,
        "clean_candidates_requested": len(clean_rows),
        "promoted_stage4a72_samples": promoted_count,
        "excluded_stage4a72_samples": len(excluded_rows),
        "expanded_primary_samples": expanded_count,
        "D_model": int(arrays["candidate_features_model"].shape[2]),
        "lambda48_primary_use": False,
        "lambda48_role": "shadow/baseline only",
        "rejected_rows_promoted": len(rejected_promoted),
        "unsure_rows_promoted": len(unsure_promoted),
        "conflict_rows_promoted": len(conflict_promoted),
        "expert_action_index_primary_created": True,
        "expert_action_index_primary_scope": "expanded_primary_bc_dataset.npz only",
        "bc_training": False,
        "optimizer_step": False,
        "checkpoint": False,
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
        "prior_datasets_modified": False,
        "source_hashes_unchanged": all(item["unchanged"] is True for item in source_hashes.values() if item["exists"]),
        "prior_dataset_hashes_unchanged": all(item["unchanged"] is True for item in prior_hashes.values() if item["exists"]),
        "required_outputs": REQUIRED_OUTPUTS,
        "recommended_next": "Stage 4A-7.10 expanded dataset QA, not training yet.",
    }
    write_json(OUT / "stage4a79_no_training_promotion_summary.json", summary)
    (OUT / "stage4a79_no_training_promotion_summary.md").write_text(
        "# Stage 4A-7.9 No-Training Promotion Summary\n\n"
        + markdown_table(
            [
                ("completed", True),
                ("blocked", False),
                ("original primary samples", original_count),
                ("clean candidates requested", len(clean_rows)),
                ("promoted Stage 4A-7.2 samples", promoted_count),
                ("excluded Stage 4A-7.2 samples", len(excluded_rows)),
                ("expanded primary samples", expanded_count),
                ("D_model", int(arrays["candidate_features_model"].shape[2])),
                ("lambda48 primary use", False),
                ("training/checkpoint/runtime", "none"),
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
