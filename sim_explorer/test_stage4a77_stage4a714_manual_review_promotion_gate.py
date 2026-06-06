#!/usr/bin/env python3
"""Validate the Stage 4A-7.7 gate for Stage 4A-7.14 manual review results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
OUT = WORKSPACE / "outputs/stage4a77_stage4a714_manual_review_promotion_gate"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    summary_path = OUT / "stage4a77_stage4a714_promotion_gate_summary.json"
    summary_md = OUT / "stage4a77_stage4a714_promotion_gate_summary.md"
    all_rows_csv = OUT / "stage4a77_stage4a714_all_review_rows_with_gate.csv"
    clean_csv = OUT / "stage4a77_stage4a714_clean_promotion_candidates.csv"
    distance_csv = OUT / "stage4a77_stage4a714_distance_manual_recheck_rows.csv"
    rejected_csv = OUT / "stage4a77_stage4a714_rejected_rows.csv"
    approved_no_csv = OUT / "stage4a77_stage4a714_approved_no_promote_rows.csv"
    manual_recheck_csv = OUT / "stage4a77_stage4a714_manual_recheck_rows.csv"
    checks: dict[str, bool] = {
        "output_dir_exists": OUT.is_dir(),
        "summary_json_exists": summary_path.is_file(),
        "summary_md_exists": summary_md.is_file(),
        "all_rows_csv_exists": all_rows_csv.is_file(),
        "clean_csv_exists": clean_csv.is_file(),
        "distance_csv_exists": distance_csv.is_file(),
        "rejected_csv_exists": rejected_csv.is_file(),
        "approved_no_csv_exists": approved_no_csv.is_file(),
        "manual_recheck_csv_exists": manual_recheck_csv.is_file(),
    }
    summary = read_json(summary_path) or {}
    all_rows = read_csv(all_rows_csv)
    clean = read_csv(clean_csv)
    distance = read_csv(distance_csv)
    rejected = read_csv(rejected_csv)
    approved_no = read_csv(approved_no_csv)
    manual_recheck = read_csv(manual_recheck_csv)
    negative = summary.get("negative_scope", {})
    clean_ids = {row.get("sample_id") for row in clean}
    distance_ids = {row.get("sample_id") for row in distance}
    expected_distance_ids = {"start_002_step_002", "start_004_step_001", "start_005_step_000"}

    checks.update(
        {
            "completed_true": summary.get("completed") is True,
            "blocked_false": summary.get("blocked") is False,
            "no_blockers": summary.get("blockers") == [],
            "row_count_60": summary.get("row_count") == 60 and len(all_rows) == 60,
            "status_counts_31_29": summary.get("status_counts", {}).get("approve") == 31
            and summary.get("status_counts", {}).get("reject") == 29,
            "promote_yes_28": summary.get("human_promote_yes_count") == 28,
            "clean_count_25": summary.get("clean_promotion_candidate_count") == 25 and len(clean) == 25,
            "distance_recheck_count_3": summary.get("distance_manual_recheck_count") == 3 and len(distance) == 3,
            "distance_recheck_ids": distance_ids == expected_distance_ids,
            "clean_excludes_very_close_promote_yes": clean_ids.isdisjoint(expected_distance_ids)
            and all(row.get("action_distance_flag") != "very_close" for row in clean),
            "approved_no_promote_count_3": summary.get("approved_no_promote_count") == 3 and len(approved_no) == 3,
            "rejected_count_29": summary.get("rejected_count") == 29 and len(rejected) == 29,
            "manual_recheck_count_3": summary.get("manual_recheck_count") == 3 and len(manual_recheck) == 3,
            "lambda48_shadow_only": "lambda48" in str(summary.get("policy", {}).get("lambda48_role", "")).lower(),
            "no_label_promotion": negative.get("label_promotion") is False,
            "no_dataset_modified": negative.get("dataset_modified") is False,
            "no_expert_action_index_primary_created": negative.get("expert_action_index_primary_created") is False,
            "no_training": negative.get("training") is False,
            "no_checkpoint": negative.get("checkpoint") is False,
            "no_isaac": negative.get("isaac_startup") is False,
            "no_map_predict": negative.get("map_predict") is False,
            "no_rollout": negative.get("rollout") is False,
            "no_rl_gdpo_ppo": negative.get("rl_gdpo_ppo") is False,
        }
    )
    blockers = [name for name, ok in sorted(checks.items()) if not ok]
    result = {"all_passed": not blockers, "blockers": blockers, "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
