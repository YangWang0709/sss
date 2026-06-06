#!/usr/bin/env python3
"""Generate the Stage 4A-7.7 gate for Stage 4A-7.14 manual review results.

This gate reads the completed human review export and the refreshed 2D review
records. It produces candidate decision lists only. It does not promote labels,
modify datasets, create expert_action_index_primary, train, checkpoint, launch
Isaac, run map_predict, run rollout, or run RL/GDPO/PPO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_EXPORT = (
    WORKSPACE
    / "outputs/stage4a714_2d_manual_review_import_audit/stage4a714_2d_manual_review_export.json"
)
DEFAULT_REVIEW_AUDIT = (
    WORKSPACE
    / "outputs/stage4a714_2d_manual_review_import_audit/stage4a714_2d_manual_review_import_audit.json"
)
DEFAULT_REVIEW_RECORDS = (
    WORKSPACE / "outputs/stage4a714_2d_review_packet/stage4a714_2d_rollout_review_records.json"
)
DEFAULT_RUNTIME_TRANSITIONS = (
    WORKSPACE / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime/transition_decisions.csv"
)
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/stage4a77_stage4a714_manual_review_promotion_gate"

REVIEW_FIELDS = [
    "review_id",
    "start_id",
    "step_id",
    "sample_id",
    "human_review_status",
    "human_review_reason",
    "human_comment",
    "promote_candidate_yes_no",
]
OUTPUT_FIELDS = REVIEW_FIELDS + [
    "source_to_action_distance_m",
    "source_to_action_distance_3d_m",
    "action_distance_flag",
    "newly_observed_xy_cells",
    "historical_observed_xy_cells",
    "observed_ratio_before",
    "observed_ratio_after_current_capture",
    "runtime_stage",
    "runtime_candidate_features",
    "runtime_rgb",
    "runtime_pose",
    "runtime_observed_state_reference",
    "gate_decision",
    "gate_bucket",
    "gate_notes",
]
NEGATIVE_REASONS = {
    "local_jitter_unacceptable",
    "unsafe_outside_stuck_revisit",
    "poor_uncertainty_choice",
    "poor_path_choice",
    "visual_mismatch",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = OUTPUT_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_sample_id(start_id: int, step_id: int) -> str:
    return f"start_{start_id:03d}_step_{step_id:03d}"


def normalize_review_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in REVIEW_FIELDS}
    out["start_id"] = int(out["start_id"])
    out["step_id"] = int(out["step_id"])
    out["sample_id"] = str(out["sample_id"])
    out["human_review_status"] = str(out["human_review_status"] or "unreviewed")
    out["human_review_reason"] = str(out["human_review_reason"] or "")
    out["human_comment"] = str(out["human_comment"] or "")
    out["promote_candidate_yes_no"] = str(out["promote_candidate_yes_no"] or "")
    return out


def build_record_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in records:
        sample_id = as_sample_id(int(row["start_variant_id"]), int(row["step_id"]))
        out[sample_id] = row
    return out


def build_runtime_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out = {}
    for row in rows:
        sample_id = as_sample_id(int(row["start_variant_id"]), int(row["step_id"]))
        out[sample_id] = row
    return out


def classify_row(review: dict[str, Any], record: dict[str, Any] | None, runtime: dict[str, str] | None) -> dict[str, Any]:
    notes: list[str] = []
    status = review["human_review_status"]
    reason = review["human_review_reason"]
    promote = review["promote_candidate_yes_no"]
    distance_flag = record.get("action_distance_flag", "missing_record") if record else "missing_record"

    if record is None:
        notes.append("missing_stage4a714_2d_review_record")
    if runtime is None:
        notes.append("missing_stage4a714_runtime_transition")
    if status == "unreviewed":
        notes.append("human_review_unreviewed")
    if status in {"unsure", "needs_closer_inspection"}:
        notes.append("human_review_requires_recheck")
    if reason in NEGATIVE_REASONS:
        notes.append(f"negative_review_reason:{reason}")
    if promote == "yes" and status != "approve":
        notes.append("promote_yes_without_approve")
    if distance_flag == "very_close":
        notes.append("very_close_source_to_action_distance_lt_0p25m")
    elif distance_flag == "close":
        notes.append("close_source_to_action_distance_lt_0p50m_warning")

    if status == "approve" and promote == "yes" and record and runtime and reason not in NEGATIVE_REASONS:
        if distance_flag == "very_close":
            gate_bucket = "distance_manual_recheck"
            gate_decision = "hold_manual_recheck"
        else:
            gate_bucket = "clean_promotion_candidate"
            gate_decision = "candidate_for_future_stage4a78_no_training_import"
    elif status == "approve" and promote != "yes":
        gate_bucket = "approved_no_promote"
        gate_decision = "do_not_promote_candidate"
    elif status == "reject":
        gate_bucket = "rejected"
        gate_decision = "do_not_promote_candidate"
    elif status in {"unsure", "needs_closer_inspection"}:
        gate_bucket = "manual_recheck"
        gate_decision = "hold_manual_recheck"
    else:
        gate_bucket = "blocked_or_invalid"
        gate_decision = "hold_manual_recheck"

    merged = {
        **review,
        "source_to_action_distance_m": record.get("source_to_action_distance_m", "") if record else "",
        "source_to_action_distance_3d_m": record.get("source_to_action_distance_3d_m", "") if record else "",
        "action_distance_flag": distance_flag,
        "newly_observed_xy_cells": record.get("newly_observed_xy_cells", "") if record else "",
        "historical_observed_xy_cells": record.get("historical_observed_xy_cells", "") if record else "",
        "observed_ratio_before": record.get("observed_ratio_before", "") if record else "",
        "observed_ratio_after_current_capture": record.get("observed_ratio_after_current_capture", "") if record else "",
        "runtime_stage": runtime.get("stage", "") if runtime else "",
        "runtime_candidate_features": runtime.get("candidate_features", "") if runtime else "",
        "runtime_rgb": runtime.get("rgb", "") if runtime else "",
        "runtime_pose": runtime.get("pose", "") if runtime else "",
        "runtime_observed_state_reference": runtime.get("observed_state_reference", "") if runtime else "",
        "gate_decision": gate_decision,
        "gate_bucket": gate_bucket,
        "gate_notes": ";".join(notes),
    }
    return merged


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 4A-7.7 Stage 4A-7.14 Manual Review Promotion Gate",
        "",
        "This packet is a gate/decision packet only. It does not promote labels or modify datasets.",
        "",
        "## Result",
        f"- Completed: `{summary['completed']}`",
        f"- Blocked: `{summary['blocked']}`",
        f"- Blockers: `{summary['blockers']}`",
        f"- Review rows: `{summary['row_count']}`",
        f"- Human approve: `{summary['status_counts'].get('approve', 0)}`",
        f"- Human reject: `{summary['status_counts'].get('reject', 0)}`",
        f"- Human promote yes: `{summary['promote_counts'].get('yes', 0)}`",
        f"- Clean promotion candidates: `{summary['clean_promotion_candidate_count']}`",
        f"- Distance manual recheck candidates: `{summary['distance_manual_recheck_count']}`",
        f"- Approved no-promote rows: `{summary['approved_no_promote_count']}`",
        f"- Rejected rows: `{summary['rejected_count']}`",
        "",
        "## Distance Gate",
        "- `very_close <0.25m` is held for manual recheck and not included in clean candidates.",
        "- `close <0.50m` is allowed as a clean candidate but keeps a warning note.",
        f"- Distance flag counts: `{summary['distance_flag_counts']}`",
        "",
        "## Outputs",
    ]
    for key, value in summary["output_files"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Label Lineage And Scope",
            "- Candidate actions come from the Stage 4A-7.14 medium bounded uncertainty-bonus expert rollout.",
            "- The gate does not recompute labels from lambda48.",
            "- Lambda48 remains shadow/baseline only.",
            "- No `expert_action_index_primary` is created.",
            "- No Stage 4A-7.0 primary BC dataset is modified.",
            "",
            "## Negative Scope",
        ]
    )
    for key, value in summary["negative_scope"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Recommended Next Step", summary["recommended_next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_gate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    review_export = read_json(args.review_export)
    review_audit = read_json(args.review_audit)
    records = read_json(args.review_records)
    runtime_rows = read_csv(args.runtime_transitions)
    record_by_sample = build_record_index(records)
    runtime_by_sample = build_runtime_index(runtime_rows)
    review_rows = [normalize_review_row(row) for row in review_export.get("rows", [])]

    blockers: list[str] = []
    if review_export.get("review_schema_version") != "stage4a714_2d_manual_review_v1":
        blockers.append("unexpected_review_schema_version")
    if review_export.get("source_stage") != "Stage 4A-7.14":
        blockers.append("unexpected_source_stage")
    if len(review_rows) != 60:
        blockers.append(f"row_count_expected_60_got_{len(review_rows)}")
    if review_audit.get("manual_review_complete") is not True:
        blockers.append("manual_review_not_complete")
    if review_audit.get("promotion_ready") is not True:
        blockers.append("review_audit_not_promotion_ready")
    if any(row["human_review_status"] == "unreviewed" for row in review_rows):
        blockers.append("unreviewed_rows_present")
    if any(row["promote_candidate_yes_no"] == "yes" and row["human_review_status"] != "approve" for row in review_rows):
        blockers.append("promote_yes_without_approve")

    merged_rows = []
    for row in review_rows:
        sample_id = row["sample_id"]
        expected_sample_id = as_sample_id(row["start_id"], row["step_id"])
        if sample_id != expected_sample_id:
            blockers.append(f"sample_identity_mismatch:{sample_id}:{expected_sample_id}")
        merged_rows.append(classify_row(row, record_by_sample.get(sample_id), runtime_by_sample.get(sample_id)))

    bucket_counts = Counter(row["gate_bucket"] for row in merged_rows)
    status_counts = Counter(row["human_review_status"] for row in merged_rows)
    promote_counts = Counter(row["promote_candidate_yes_no"] for row in merged_rows)
    distance_counts = Counter(row["action_distance_flag"] for row in merged_rows)

    clean = [row for row in merged_rows if row["gate_bucket"] == "clean_promotion_candidate"]
    distance_recheck = [row for row in merged_rows if row["gate_bucket"] == "distance_manual_recheck"]
    approved_no_promote = [row for row in merged_rows if row["gate_bucket"] == "approved_no_promote"]
    rejected = [row for row in merged_rows if row["gate_bucket"] == "rejected"]
    manual_recheck = [row for row in merged_rows if row["gate_decision"] == "hold_manual_recheck"]

    output_files = {
        "summary_json": str(output_dir / "stage4a77_stage4a714_promotion_gate_summary.json"),
        "summary_md": str(output_dir / "stage4a77_stage4a714_promotion_gate_summary.md"),
        "all_rows_csv": str(output_dir / "stage4a77_stage4a714_all_review_rows_with_gate.csv"),
        "clean_candidates_csv": str(output_dir / "stage4a77_stage4a714_clean_promotion_candidates.csv"),
        "distance_recheck_csv": str(output_dir / "stage4a77_stage4a714_distance_manual_recheck_rows.csv"),
        "rejected_csv": str(output_dir / "stage4a77_stage4a714_rejected_rows.csv"),
        "approved_no_promote_csv": str(output_dir / "stage4a77_stage4a714_approved_no_promote_rows.csv"),
        "manual_recheck_csv": str(output_dir / "stage4a77_stage4a714_manual_recheck_rows.csv"),
    }

    write_csv(Path(output_files["all_rows_csv"]), merged_rows)
    write_json(output_dir / "stage4a77_stage4a714_all_review_rows_with_gate.json", merged_rows)
    write_csv(Path(output_files["clean_candidates_csv"]), clean)
    write_json(output_dir / "stage4a77_stage4a714_clean_promotion_candidates.json", clean)
    write_csv(Path(output_files["distance_recheck_csv"]), distance_recheck)
    write_json(output_dir / "stage4a77_stage4a714_distance_manual_recheck_rows.json", distance_recheck)
    write_csv(Path(output_files["rejected_csv"]), rejected)
    write_json(output_dir / "stage4a77_stage4a714_rejected_rows.json", rejected)
    write_csv(Path(output_files["approved_no_promote_csv"]), approved_no_promote)
    write_json(output_dir / "stage4a77_stage4a714_approved_no_promote_rows.json", approved_no_promote)
    write_csv(Path(output_files["manual_recheck_csv"]), manual_recheck)
    write_json(output_dir / "stage4a77_stage4a714_manual_recheck_rows.json", manual_recheck)

    summary = {
        "stage": "Stage 4A-7.7 Stage 4A-7.14 manual review promotion gate",
        "completed": not blockers,
        "blocked": bool(blockers),
        "blockers": blockers,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(WORKSPACE),
        "review_export": str(args.review_export),
        "review_export_sha256": sha256_file(args.review_export),
        "review_audit": str(args.review_audit),
        "review_audit_sha256": sha256_file(args.review_audit),
        "review_records": str(args.review_records),
        "runtime_transitions": str(args.runtime_transitions),
        "output_dir": str(output_dir),
        "review_schema_version": review_export.get("review_schema_version"),
        "source_stage": review_export.get("source_stage"),
        "packet_stage": review_export.get("packet_stage"),
        "exported_at_local_browser_time": review_export.get("exported_at_local_browser_time"),
        "row_count": len(merged_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "promote_counts": dict(sorted(promote_counts.items())),
        "gate_bucket_counts": dict(sorted(bucket_counts.items())),
        "distance_flag_counts": dict(sorted(distance_counts.items())),
        "human_promote_yes_count": int(promote_counts.get("yes", 0)),
        "clean_promotion_candidate_count": len(clean),
        "distance_manual_recheck_count": len(distance_recheck),
        "approved_no_promote_count": len(approved_no_promote),
        "rejected_count": len(rejected),
        "manual_recheck_count": len(manual_recheck),
        "clean_candidate_sample_ids": [row["sample_id"] for row in clean],
        "distance_recheck_sample_ids": [row["sample_id"] for row in distance_recheck],
        "policy": {
            "clean_candidate_rule": "approve + promote_candidate_yes_no=yes + not very_close + no negative review reason + source records present",
            "distance_gate": "very_close source-to-action distance <0.25m is held for manual recheck; close <0.50m is warning-only",
            "lambda48_role": "shadow/baseline only; labels are not recomputed from lambda48",
            "primary_label_boundary": "no expert_action_index_primary is created in this gate",
        },
        "output_files": output_files,
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
        "recommended_next_step": (
            "Stage 4A-7.8 no-training import design can use the 25 clean candidates, "
            "while keeping the 3 very-close promote-yes rows in manual recheck. "
            "Actual dataset modification/training/checkpoint/RL remains a separate explicit gate."
        ),
    }
    write_json(Path(output_files["summary_json"]), summary)
    write_markdown(Path(output_files["summary_md"]), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_export", type=Path, default=DEFAULT_REVIEW_EXPORT)
    parser.add_argument("--review_audit", type=Path, default=DEFAULT_REVIEW_AUDIT)
    parser.add_argument("--review_records", type=Path, default=DEFAULT_REVIEW_RECORDS)
    parser.add_argument("--runtime_transitions", type=Path, default=DEFAULT_RUNTIME_TRANSITIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = generate_gate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
