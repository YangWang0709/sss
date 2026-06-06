#!/usr/bin/env python3
"""Audit a Stage 4A-7.14 2D manual review export.

This is an import/readiness audit only. It does not promote labels, modify any
dataset, run training, create checkpoints, launch Isaac, run map_predict, run a
rollout, or run RL/GDPO/PPO.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = WORKSPACE / "outputs/stage4a714_2d_manual_review_import_audit/stage4a714_2d_manual_review_export.json"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/stage4a714_2d_manual_review_import_audit"

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
STATUS_VALUES = {"unreviewed", "approve", "reject", "unsure", "needs_closer_inspection"}
PROMOTE_VALUES = {"", "yes", "no"}


def load_export(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value)


def audit_export(export_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = load_export(export_path)
    rows = list(payload.get("rows") or [])
    status_counts = Counter(normalize_text(row.get("human_review_status")) for row in rows)
    promote_counts = Counter(normalize_text(row.get("promote_candidate_yes_no")) for row in rows)
    reason_counts_by_status: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        reason_counts_by_status[normalize_text(row.get("human_review_status"))][normalize_text(row.get("human_review_reason"))] += 1

    blockers: list[str] = []
    warnings: list[str] = []
    row_ids = [normalize_text(row.get("review_id")) for row in rows]
    duplicate_ids = sorted(review_id for review_id, count in Counter(row_ids).items() if count > 1)
    missing_required = [
        {"index": idx, "missing": [field for field in REQUIRED_FIELDS if field not in row]}
        for idx, row in enumerate(rows)
        if any(field not in row for field in REQUIRED_FIELDS)
    ]
    invalid_status = [
        {"index": idx, "review_id": row.get("review_id"), "status": row.get("human_review_status")}
        for idx, row in enumerate(rows)
        if normalize_text(row.get("human_review_status")) not in STATUS_VALUES
    ]
    invalid_promote = [
        {"index": idx, "review_id": row.get("review_id"), "promote_candidate_yes_no": row.get("promote_candidate_yes_no")}
        for idx, row in enumerate(rows)
        if normalize_text(row.get("promote_candidate_yes_no")) not in PROMOTE_VALUES
    ]
    illegal_promote_yes = [
        row
        for row in rows
        if normalize_text(row.get("promote_candidate_yes_no")) == "yes"
        and normalize_text(row.get("human_review_status")) != "approve"
    ]
    unreviewed_rows = [row for row in rows if normalize_text(row.get("human_review_status")) == "unreviewed"]
    unsure_rows = [row for row in rows if normalize_text(row.get("human_review_status")) == "unsure"]
    closer_rows = [row for row in rows if normalize_text(row.get("human_review_status")) == "needs_closer_inspection"]

    if len(rows) != 60:
        blockers.append(f"expected_60_rows_got_{len(rows)}")
    if duplicate_ids:
        blockers.append("duplicate_review_id")
    if missing_required:
        blockers.append("missing_required_fields")
    if invalid_status:
        blockers.append("invalid_human_review_status")
    if invalid_promote:
        blockers.append("invalid_promote_candidate_yes_no")
    if illegal_promote_yes:
        blockers.append("promote_yes_without_approve")
    if unreviewed_rows:
        blockers.append("manual_review_incomplete_unreviewed_rows")
    if unsure_rows:
        warnings.append("unsure_rows_require_future_policy_decision")
    if closer_rows:
        warnings.append("needs_closer_inspection_rows_require_future_policy_decision")

    approved_promote_yes_rows = [
        row
        for row in rows
        if normalize_text(row.get("human_review_status")) == "approve"
        and normalize_text(row.get("promote_candidate_yes_no")) == "yes"
    ]
    approved_no_rows = [
        row
        for row in rows
        if normalize_text(row.get("human_review_status")) == "approve"
        and normalize_text(row.get("promote_candidate_yes_no")) == "no"
    ]
    approved_empty_rows = [
        row
        for row in rows
        if normalize_text(row.get("human_review_status")) == "approve"
        and normalize_text(row.get("promote_candidate_yes_no")) == ""
    ]
    rejected_rows = [row for row in rows if normalize_text(row.get("human_review_status")) == "reject"]

    audit = {
        "stage": "Stage 4A-7.14 2D manual review export import audit",
        "audit_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(WORKSPACE),
        "export_json": str(export_path),
        "output_dir": str(output_dir),
        "review_schema_version": payload.get("review_schema_version"),
        "source_stage": payload.get("source_stage"),
        "packet_stage": payload.get("packet_stage"),
        "exported_at_local_browser_time": payload.get("exported_at_local_browser_time"),
        "reviewer_name": payload.get("reviewer_name", ""),
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "promote_candidate_counts": dict(sorted(promote_counts.items())),
        "reason_counts_by_status": {status: dict(counter) for status, counter in sorted(reason_counts_by_status.items())},
        "approved_promote_yes_count": len(approved_promote_yes_rows),
        "approved_no_count": len(approved_no_rows),
        "approved_empty_count": len(approved_empty_rows),
        "reject_count": len(rejected_rows),
        "unreviewed_count": len(unreviewed_rows),
        "unsure_count": len(unsure_rows),
        "needs_closer_inspection_count": len(closer_rows),
        "duplicate_review_ids": duplicate_ids,
        "missing_required": missing_required,
        "invalid_status": invalid_status,
        "invalid_promote": invalid_promote,
        "illegal_promote_yes": [
            {"review_id": row.get("review_id"), "human_review_status": row.get("human_review_status")}
            for row in illegal_promote_yes
        ],
        "unreviewed_rows": [
            {
                "review_id": row.get("review_id"),
                "start_id": row.get("start_id"),
                "step_id": row.get("step_id"),
                "sample_id": row.get("sample_id"),
            }
            for row in unreviewed_rows
        ],
        "promotion_ready": not blockers,
        "manual_review_complete": not unreviewed_rows and not unsure_rows and not closer_rows,
        "blockers": blockers,
        "warnings": warnings,
        "negative_scope": {
            "label_promotion": False,
            "dataset_modified": False,
            "expert_action_index_primary_created": False,
            "training": False,
            "checkpoint": False,
            "isaac_startup": False,
            "map_predict": False,
            "rollout": False,
            "rl_gdpo_ppo": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_decisions_csv(output_dir / "stage4a714_2d_manual_review_decisions.csv", rows)
    write_rows_csv(output_dir / "stage4a714_2d_manual_review_unreviewed_rows.csv", unreviewed_rows)
    (output_dir / "stage4a714_2d_manual_review_import_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "stage4a714_2d_manual_review_import_audit.md", audit)
    return audit


def write_decisions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: normalize_text(row.get(field)) for field in REQUIRED_FIELDS})


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["review_id", "start_id", "step_id", "sample_id", "human_review_status", "human_review_reason", "promote_candidate_yes_no"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: normalize_text(row.get(field)) for field in fields})


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Stage 4A-7.14 2D Manual Review Import Audit",
        "",
        f"- Export JSON: `{audit['export_json']}`",
        f"- Review schema: `{audit.get('review_schema_version')}`",
        f"- Source stage: `{audit.get('source_stage')}`",
        f"- Packet stage: `{audit.get('packet_stage')}`",
        f"- Exported at local browser time: `{audit.get('exported_at_local_browser_time')}`",
        f"- Rows: `{audit['row_count']}`",
        f"- Manual review complete: `{audit['manual_review_complete']}`",
        f"- Promotion/import ready: `{audit['promotion_ready']}`",
        f"- Blockers: `{audit['blockers']}`",
        "",
        "## Status Counts",
    ]
    for key, value in audit["status_counts"].items():
        lines.append(f"- {key or '<empty>'}: `{value}`")
    lines.extend(["", "## Promotion Candidate Counts"])
    for key, value in audit["promote_candidate_counts"].items():
        lines.append(f"- {key or '<empty>'}: `{value}`")
    lines.extend(
        [
            "",
            f"- Approved with promote_candidate_yes_no=yes: `{audit['approved_promote_yes_count']}`",
            f"- Approved with promote_candidate_yes_no=no: `{audit['approved_no_count']}`",
            f"- Approved with promote_candidate_yes_no empty: `{audit['approved_empty_count']}`",
            f"- Rejected: `{audit['reject_count']}`",
            f"- Unreviewed: `{audit['unreviewed_count']}`",
            "",
            "## Unreviewed Rows",
        ]
    )
    if audit["unreviewed_rows"]:
        for row in audit["unreviewed_rows"]:
            lines.append(f"- `{row['review_id']}` / `{row['sample_id']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Negative Scope"])
    for key, value in audit["negative_scope"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_json", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    audit = audit_export(args.export_json, args.output_dir)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if not audit["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
