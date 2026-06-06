#!/usr/bin/env python3
"""Validate Stage 4A-7.9 compatible no-training import artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
OUT = WORKSPACE / "outputs/stage4a79_stage4a714_compatible_no_training_import"
STAGE70_NPZ = WORKSPACE / "outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_primary_short_rollout.npz"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
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


def main() -> int:
    summary_path = OUT / "stage4a79_stage4a714_compatible_import_summary.json"
    summary_md = OUT / "stage4a79_stage4a714_compatible_import_summary.md"
    adapter_path = OUT / "stage4a79_stage4a714_adapter_dataset_25.npz"
    expanded_path = OUT / "stage4a79_stage4a714_compatible_expanded_dataset_55.npz"
    manifest_csv = OUT / "stage4a79_stage4a714_compatible_import_manifest.csv"
    integrity_path = OUT / "stage4a79_stage4a714_integrity_report.json"
    lineage_path = OUT / "stage4a79_stage4a714_label_lineage_report.json"
    no_training_path = OUT / "stage4a79_stage4a714_no_training_report.json"
    no_runtime_path = OUT / "stage4a79_stage4a714_no_runtime_report.json"
    checks: dict[str, bool] = {
        "output_dir_exists": OUT.is_dir(),
        "summary_json_exists": summary_path.is_file(),
        "summary_md_exists": summary_md.is_file(),
        "adapter_npz_exists": adapter_path.is_file(),
        "expanded_npz_exists": expanded_path.is_file(),
        "manifest_csv_exists": manifest_csv.is_file(),
        "integrity_exists": integrity_path.is_file(),
        "lineage_exists": lineage_path.is_file(),
        "no_training_exists": no_training_path.is_file(),
        "no_runtime_exists": no_runtime_path.is_file(),
    }
    summary = read_json(summary_path) or {}
    integrity = read_json(integrity_path) or {}
    lineage = read_json(lineage_path) or {}
    no_training = read_json(no_training_path) or {}
    no_runtime = read_json(no_runtime_path) or {}
    manifest = read_csv(manifest_csv)

    adapter = np.load(adapter_path, allow_pickle=False) if adapter_path.is_file() else None
    expanded = np.load(expanded_path, allow_pickle=False) if expanded_path.is_file() else None
    source_hashes = summary.get("source_hashes", {})
    stage70_hash_recorded = source_hashes.get("stage4a70_primary_npz", {}).get("sha256")
    stage70_hash_current = sha256_file(STAGE70_NPZ)

    if adapter is not None and expanded is not None:
        primary = expanded["expert_action_index_primary"].astype(int)
        valid = expanded["candidate_valid_mask"].astype(bool)
        valid_primary = (primary >= 0) & (primary < valid.shape[1]) & valid[np.arange(valid.shape[0]), primary]
        new_primary = adapter["expert_action_index_primary"].astype(int)
        new_valid = adapter["candidate_valid_mask"].astype(bool)
        new_valid_primary = (new_primary >= 0) & (new_primary < new_valid.shape[1]) & new_valid[
            np.arange(new_valid.shape[0]), new_primary
        ]
        forbidden_keys = {
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
        forbidden_present = sorted(forbidden_keys.intersection(expanded.files))
    else:
        valid_primary = np.asarray([False])
        new_valid_primary = np.asarray([False])
        forbidden_present = ["missing_npz"]

    negative = summary.get("negative_scope", {})
    checks.update(
        {
            "completed_true": summary.get("completed") is True,
            "blocked_false": summary.get("blocked") is False,
            "no_blockers": summary.get("blockers") == [],
            "adapter_count_25": summary.get("compatible_import_sample_count") == 25
            and adapter is not None
            and adapter["sample_id"].shape == (25,),
            "expanded_count_55": summary.get("expanded_compatible_sample_count") == 55
            and expanded is not None
            and expanded["sample_id"].shape == (55,),
            "adapter_model_shape": adapter is not None and adapter["candidate_features_model"].shape == (25, 64, 16),
            "expanded_model_shape": expanded is not None and expanded["candidate_features_model"].shape == (55, 64, 16),
            "expanded_raw_shape": expanded is not None and expanded["candidate_features_raw"].shape[0:2] == (55, 64),
            "manifest_count_25": len(manifest) == 25,
            "new_primary_labels_valid": bool(np.all(new_valid_primary)),
            "all_primary_labels_valid": bool(np.all(valid_primary)),
            "new_quality_keep_all_true": adapter is not None and bool(np.all(adapter["quality_keep_mask"])),
            "source_dataset_not_overwritten": summary.get("existing_primary_dataset_overwritten") is False
            and stage70_hash_recorded == stage70_hash_current,
            "lineage_lambda48_shadow_only": "shadow" in str(lineage.get("lambda48_role", "")).lower()
            and lineage.get("labels_recomputed_from_lambda48") is False,
            "summary_lambda48_shadow_only": "shadow" in str(summary.get("label_lineage", {}).get("lambda48_role", "")).lower()
            and summary.get("label_lineage", {}).get("labels_recomputed_from_lambda48") is False,
            "new_policy_present": summary.get("label_lineage", {}).get("new_primary_label_policy")
            == "stage4a714_uncertainty_bonus_executed_clean_candidate_compatible_import",
            "no_forbidden_keys": not forbidden_present and integrity.get("forbidden_keys_present") == [],
            "no_training": negative.get("training") is False and no_training.get("training") is False,
            "no_optimizer_step": negative.get("optimizer_step") is False and no_training.get("optimizer_step") is False,
            "no_checkpoint": negative.get("checkpoint") is False and no_training.get("checkpoint") is False,
            "no_model_save": negative.get("model_save") is False and no_training.get("model_save") is False,
            "no_isaac": negative.get("isaac_startup") is False and no_runtime.get("isaac_startup") is False,
            "no_map_predict": negative.get("map_predict") is False and no_runtime.get("map_predict") is False,
            "no_rollout": negative.get("rollout") is False and no_runtime.get("rollout") is False,
            "no_rl_gdpo_ppo": negative.get("rl_gdpo_ppo") is False and no_runtime.get("rl_gdpo_ppo") is False,
        }
    )
    blockers = [name for name, ok in sorted(checks.items()) if not ok]
    result = {"all_passed": not blockers, "blockers": blockers, "checks": checks, "forbidden_present": forbidden_present}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
