#!/usr/bin/env python3
"""Validate the Stage 4A-7.14 2D manual review export audit output."""

from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
OUT = WORKSPACE / "outputs/stage4a714_2d_manual_review_import_audit"


def main() -> int:
    audit_path = OUT / "stage4a714_2d_manual_review_import_audit.json"
    md_path = OUT / "stage4a714_2d_manual_review_import_audit.md"
    decisions_csv = OUT / "stage4a714_2d_manual_review_decisions.csv"
    unreviewed_csv = OUT / "stage4a714_2d_manual_review_unreviewed_rows.csv"
    export_copy = OUT / "stage4a714_2d_manual_review_export.json"
    checks: dict[str, bool] = {
        "audit_json_exists": audit_path.is_file(),
        "audit_md_exists": md_path.is_file(),
        "decisions_csv_exists": decisions_csv.is_file(),
        "unreviewed_csv_exists": unreviewed_csv.is_file(),
        "export_copy_exists": export_copy.is_file(),
    }

    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    status_counts = audit.get("status_counts", {})
    promote_counts = audit.get("promote_candidate_counts", {})
    blockers = audit.get("blockers", [])
    negative = audit.get("negative_scope", {})

    checks.update(
        {
            "audit_only": audit.get("audit_only") is True,
            "row_count_60": audit.get("row_count") == 60,
            "approve_count_28": status_counts.get("approve") == 28,
            "reject_count_27": status_counts.get("reject") == 27,
            "unreviewed_count_5": status_counts.get("unreviewed") == 5 and audit.get("unreviewed_count") == 5,
            "promote_yes_count_26": promote_counts.get("yes") == 26,
            "no_illegal_promote_yes": audit.get("illegal_promote_yes") == [],
            "manual_review_incomplete_blocked": "manual_review_incomplete_unreviewed_rows" in blockers,
            "promotion_not_ready": audit.get("promotion_ready") is False,
            "manual_review_not_complete": audit.get("manual_review_complete") is False,
            "no_dataset_modified": negative.get("dataset_modified") is False,
            "no_expert_action_index_primary_created": negative.get("expert_action_index_primary_created") is False,
            "no_label_promotion": negative.get("label_promotion") is False,
            "no_training": negative.get("training") is False,
            "no_checkpoint": negative.get("checkpoint") is False,
            "no_isaac": negative.get("isaac_startup") is False,
            "no_map_predict": negative.get("map_predict") is False,
            "no_rollout": negative.get("rollout") is False,
            "no_rl_gdpo_ppo": negative.get("rl_gdpo_ppo") is False,
        }
    )
    blockers_out = [name for name, ok in sorted(checks.items()) if not ok]
    result = {"all_passed": not blockers_out, "blockers": blockers_out, "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
