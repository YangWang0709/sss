from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a77_manual_review_import_audit"
REVIEW_PACKET = ROOT / "outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    summary_path = OUT / "stage4a77_manual_review_import_audit_summary.json"
    rows_path = OUT / "stage4a77_imported_manual_review_rows.json"
    promote_path = OUT / "stage4a77_human_requested_promote_yes_rows.json"
    warnings_path = OUT / "stage4a77_manual_review_import_warnings.json"

    checks["summary_exists"] = summary_path.is_file()
    checks["rows_exists"] = rows_path.is_file()
    checks["promote_yes_rows_exists"] = promote_path.is_file()
    checks["warnings_exists"] = warnings_path.is_file()

    summary = load_json(summary_path) if checks["summary_exists"] else {}
    rows = load_json(rows_path) if checks["rows_exists"] else []
    promote_rows = load_json(promote_path) if checks["promote_yes_rows_exists"] else []

    checks["summary_completed"] = summary.get("completed") is True and summary.get("blocked") is False
    checks["schema_ok"] = summary.get("review_schema_version") == "stage4a76c_manual_review_export_v1"
    checks["source_stage_ok"] = summary.get("source_stage") == "Stage 4A-7.2"
    checks["packet_stage_ok"] = summary.get("packet_stage") == "Stage 4A-7.6"
    checks["row_count_30"] = len(rows) == 30 and summary.get("row_count") == 30
    checks["unique_rows_30"] = len({row.get("sample_id") for row in rows}) == 30
    checks["promote_yes_only_approve"] = all(
        row.get("human_review_status") == "approve"
        for row in rows
        if row.get("promote_candidate_yes_no") == "yes"
    )
    checks["promote_yes_count_matches"] = len(promote_rows) == summary.get("human_requested_promote_yes_count")
    checks["expected_counts_present"] = (
        summary.get("status_counts", {}).get("approve") == 18
        and summary.get("status_counts", {}).get("reject") == 11
        and summary.get("status_counts", {}).get("unsure") == 1
        and summary.get("human_requested_promote_yes_count") == 17
    )

    checks["no_promotion"] = summary.get("stage4a77_promotion_performed") is False
    checks["no_expert_action_index_primary_created"] = summary.get("expert_action_index_primary_created") is False
    checks["no_training"] = summary.get("training") is False
    checks["no_checkpoint"] = summary.get("checkpoint") is False
    checks["no_isaac"] = summary.get("isaac_startup") is False
    checks["no_map_predict"] = summary.get("map_predict") is False
    checks["no_rollout"] = summary.get("rollout") is False
    checks["no_rl_gdpo_ppo"] = summary.get("rl_gdpo_ppo") is False
    checks["lambda48_shadow_only"] = "shadow" in str(summary.get("lambda48_role", "")).lower()

    forbidden_artifacts = [
        p
        for p in list(OUT.rglob("*")) + list(REVIEW_PACKET.rglob("*"))
        if p.is_file()
        and re.search(r"expert_action_index_primary|promoted_primary|stage4a77.*primary_label|checkpoint|\\.pt$|\\.pth$", p.name, re.I)
    ]
    checks["no_forbidden_primary_or_checkpoint_artifacts"] = not forbidden_artifacts

    for key, ok in checks.items():
        if not ok:
            blockers.append(key)

    result = {
        "all_passed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "forbidden_artifacts": [str(p) for p in forbidden_artifacts],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
