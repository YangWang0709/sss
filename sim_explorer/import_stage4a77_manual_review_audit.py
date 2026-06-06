from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
REVIEW_PACKET = ROOT / "outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet"
DEFAULT_INPUT = REVIEW_PACKET / "stage4a72_manual_review_export_user_uploaded.json"
OUT = ROOT / "outputs/isaac_stage4a77_manual_review_import_audit"

REQUIRED_FIELDS = [
    "review_id",
    "start_id",
    "step_id",
    "sample_id",
    "human_review_status",
    "human_review_reason",
    "human_comment",
    "promote_candidate_yes_no",
]
ALLOWED_STATUS = {"unreviewed", "approve", "reject", "unsure", "needs_closer_inspection"}
ALLOWED_REASON = {
    "",
    "local_jitter_acceptable",
    "local_jitter_unacceptable",
    "unsafe_outside_stuck_revisit",
    "poor_uncertainty_choice",
    "poor_path_choice",
    "visual_mismatch",
    "good_exploration_choice",
    "other",
}
ALLOWED_PROMOTE = {"", "yes", "no"}
NEGATIVE_REASONS = {
    "local_jitter_unacceptable",
    "unsafe_outside_stuck_revisit",
    "poor_uncertainty_choice",
    "poor_path_choice",
    "visual_mismatch",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_row(row: dict) -> dict:
    normalized = {field: row.get(field, "") for field in REQUIRED_FIELDS}
    normalized["start_id"] = int(normalized["start_id"])
    normalized["step_id"] = int(normalized["step_id"])
    normalized["human_review_status"] = normalized["human_review_status"] or "unreviewed"
    normalized["human_review_reason"] = normalized["human_review_reason"] or ""
    normalized["human_comment"] = normalized["human_comment"] or ""
    normalized["promote_candidate_yes_no"] = normalized["promote_candidate_yes_no"] or ""
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    export_path = args.input
    export = load_json(export_path)
    sample_index = load_json(REVIEW_PACKET / "stage4a72_action_story_sample_index.json")
    expected = {
        item["sample_id"]: {
            "review_id": item["sample_id"],
            "start_id": int(item["start_id"]),
            "step_id": int(item["step_id"]),
            "sample_id": item["sample_id"],
        }
        for item in sample_index
    }

    rows = [normalize_row(row) for row in export.get("rows", [])]
    row_by_sample = {row["sample_id"]: row for row in rows}

    blockers: list[str] = []
    warnings: list[dict] = []
    if export.get("review_schema_version") != "stage4a76c_manual_review_export_v1":
        blockers.append("unexpected_review_schema_version")
    if export.get("source_stage") != "Stage 4A-7.2":
        blockers.append("unexpected_source_stage")
    if export.get("packet_stage") != "Stage 4A-7.6":
        blockers.append("unexpected_packet_stage")
    if len(rows) != 30:
        blockers.append("row_count_not_30")
    if len(row_by_sample) != len(rows):
        blockers.append("duplicate_sample_id")
    if set(row_by_sample) != set(expected):
        blockers.append("sample_id_set_mismatch")

    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            blockers.append(f"missing_fields:{row.get('sample_id', 'unknown')}:{','.join(missing)}")
        exp = expected.get(row["sample_id"])
        if exp:
            for key in ["review_id", "start_id", "step_id", "sample_id"]:
                if row[key] != exp[key]:
                    blockers.append(f"identity_mismatch:{row['sample_id']}:{key}")
        if row["human_review_status"] not in ALLOWED_STATUS:
            blockers.append(f"invalid_status:{row['sample_id']}")
        if row["human_review_reason"] not in ALLOWED_REASON:
            blockers.append(f"invalid_reason:{row['sample_id']}")
        if row["promote_candidate_yes_no"] not in ALLOWED_PROMOTE:
            blockers.append(f"invalid_promote:{row['sample_id']}")
        if row["promote_candidate_yes_no"] == "yes" and row["human_review_status"] != "approve":
            blockers.append(f"promote_yes_without_approve:{row['sample_id']}")
        if row["human_review_status"] == "approve" and row["human_review_reason"] in NEGATIVE_REASONS:
            warnings.append(
                {
                    "sample_id": row["sample_id"],
                    "type": "approve_with_negative_reason",
                    "human_review_reason": row["human_review_reason"],
                    "promote_candidate_yes_no": row["promote_candidate_yes_no"],
                    "note": "Do not auto-promote; future Stage 4A-7.7 should inspect this row.",
                }
            )

    status_counts = Counter(row["human_review_status"] for row in rows)
    reason_counts = Counter(row["human_review_reason"] for row in rows)
    promote_counts = Counter(row["promote_candidate_yes_no"] for row in rows)
    promote_yes_rows = [row for row in rows if row["promote_candidate_yes_no"] == "yes"]
    approve_rows = [row for row in rows if row["human_review_status"] == "approve"]
    reject_rows = [row for row in rows if row["human_review_status"] == "reject"]
    unsure_rows = [row for row in rows if row["human_review_status"] in {"unsure", "needs_closer_inspection"}]

    write_json(OUT / "stage4a77_imported_manual_review_rows.json", rows)
    write_csv(OUT / "stage4a77_imported_manual_review_rows.csv", rows, REQUIRED_FIELDS)
    write_json(OUT / "stage4a77_human_requested_promote_yes_rows.json", promote_yes_rows)
    write_csv(OUT / "stage4a77_human_requested_promote_yes_rows.csv", promote_yes_rows, REQUIRED_FIELDS)
    write_json(OUT / "stage4a77_rejected_or_uncertain_rows.json", reject_rows + unsure_rows)
    write_csv(OUT / "stage4a77_rejected_or_uncertain_rows.csv", reject_rows + unsure_rows, REQUIRED_FIELDS)
    write_json(OUT / "stage4a77_manual_review_import_warnings.json", warnings)

    summary = {
        "completed": not blockers,
        "blocked": bool(blockers),
        "blockers": blockers,
        "warnings": warnings,
        "warning_count": len(warnings),
        "input_review_export": str(export_path),
        "input_sha256": sha256(export_path),
        "review_schema_version": export.get("review_schema_version"),
        "source_stage": export.get("source_stage"),
        "packet_stage": export.get("packet_stage"),
        "exported_at_local_browser_time": export.get("exported_at_local_browser_time"),
        "reviewer_name": export.get("reviewer_name", ""),
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "promote_candidate_yes_no_counts": dict(promote_counts),
        "approve_count": len(approve_rows),
        "reject_count": len(reject_rows),
        "unsure_or_needs_closer_inspection_count": len(unsure_rows),
        "human_requested_promote_yes_count": len(promote_yes_rows),
        "stage4a77_promotion_performed": False,
        "expert_action_index_primary_created": False,
        "training": False,
        "checkpoint": False,
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
        "lambda48_role": "shadow/baseline only",
        "next_step": "Future Stage 4A-7.7 may inspect this import and make a separate promotion decision; this audit does not promote labels.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "stage4a77_manual_review_import_audit_summary.json", summary)
    md = ["# Stage 4A-7.7 Manual Review Import Audit", ""]
    for key, value in summary.items():
        if key in {"warnings", "blockers"}:
            continue
        md.append(f"- {key}: `{value}`" if isinstance(value, str) else f"- {key}: {value}")
    md.extend(["", "## Blockers", ""])
    md.extend([f"- {item}" for item in blockers] or ["- none"])
    md.extend(["", "## Warnings", ""])
    if warnings:
        for item in warnings:
            md.append(f"- `{item['sample_id']}`: {item['type']} ({item['human_review_reason']}, promote={item['promote_candidate_yes_no']})")
    else:
        md.append("- none")
    md.extend(
        [
            "",
            "## Safety",
            "",
            "- No label promotion was performed.",
            "- No expert_action_index_primary was created.",
            "- No training, checkpoint, Isaac startup, map_predict, rollout, or RL/GDPO/PPO was run.",
            "- Lambda48 remains shadow/baseline only.",
            "",
        ]
    )
    (OUT / "stage4a77_manual_review_import_audit_summary.md").write_text("\n".join(md), encoding="utf-8")

    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
