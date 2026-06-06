from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a78_promotion_candidate_decision_packet"
STAGE77 = ROOT / "outputs/isaac_stage4a77_manual_review_import_audit"
STAGE76 = ROOT / "outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet"
STAGE75 = ROOT / "outputs/isaac_stage4a75_no_training_primary_label_policy_next_step_packet"
STAGE73 = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_no_training"
STAGE73_QA = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_qa_review"
STAGE72 = ROOT / "outputs/isaac_stage4a72_bounded_short_rollout_runtime"
STAGE72_QA = ROOT / "outputs/isaac_stage4a72_bounded_short_rollout_runtime_qa_review"

NEGATIVE_REASONS = {
    "unsafe_outside_stuck_revisit",
    "visual_mismatch",
    "local_jitter_unacceptable",
    "poor_uncertainty_choice",
    "poor_path_choice",
}

ROW_FIELDS = [
    "review_id",
    "sample_id",
    "start_id",
    "step_id",
    "human_review_status",
    "human_review_reason",
    "human_comment",
    "promote_candidate_yes_no",
    "adapter_sample_index",
    "candidate_action_index_uncertainty_bonus_executed",
    "expert_action_index_lambda48_shadow",
    "source_row_exists_in_stage4a73_adapter",
    "candidate_label_valid",
    "quality_keep",
    "quality_warning",
    "quality_blocker",
    "auto_quality_verdict",
    "clean_promotion_candidate",
    "decision_bucket",
    "decision_notes",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] = ROW_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True)


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_quality_rows() -> dict[str, dict]:
    quality = {}
    for path in sorted((STAGE76 / "samples").glob("start_*/step_*/review_row.json")):
        row = read_json(path)
        quality[row["sample_id"]] = row
    return quality


def hash_report(paths: dict[str, Path]) -> dict:
    items = {}
    for name, path in paths.items():
        items[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256_before": sha256_file(path),
            "sha256_after": None,
            "unchanged": None,
        }
    return items


def finalize_hash_report(report: dict) -> dict:
    for item in report.values():
        path = Path(item["path"])
        item["sha256_after"] = sha256_file(path)
        item["unchanged"] = item["sha256_before"] == item["sha256_after"]
    return report


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    source_paths = {
        "stage4a77_summary": STAGE77 / "stage4a77_manual_review_import_audit_summary.json",
        "stage4a77_rows": STAGE77 / "stage4a77_imported_manual_review_rows.csv",
        "stage4a77_promote_yes_rows": STAGE77 / "stage4a77_human_requested_promote_yes_rows.csv",
        "stage4a76_action_story_index": STAGE76 / "stage4a72_action_story_review_index.html",
        "stage4a73_adapter_sample_index": STAGE73 / "stage4a72_adapter_sample_index_table.csv",
        "stage4a73_adapter_candidate_features": STAGE73 / "stage4a72_adapter_candidate_model_feature_table.csv",
        "stage4a72_transition_decisions": STAGE72 / "transition_decisions.csv",
    }
    prior_paths = {
        "stage4a70_primary_dataset": ROOT / "outputs/isaac_stage4a70_bc_dataset_design_preparation/bc_dataset_primary_short_rollout.npz",
        "stage4a72_runtime_dataset": STAGE72 / "short_rollout_dataset_uncertainty_bonus.npz",
        "stage4a73_adapter_dataset": STAGE73 / "stage4a72_compact_v1_candidate_feature_adapter.npz",
        "stage4a73_adapter_sample_index": STAGE73 / "stage4a72_adapter_sample_index_table.csv",
        "stage4a77_review_import_rows": STAGE77 / "stage4a77_imported_manual_review_rows.csv",
    }
    source_hashes = hash_report(source_paths)
    prior_hashes = hash_report(prior_paths)

    stage77_summary = read_json(source_paths["stage4a77_summary"])
    review_rows = pd.read_csv(source_paths["stage4a77_rows"]).fillna("")
    adapter = pd.read_csv(source_paths["stage4a73_adapter_sample_index"]).fillna("")
    adapter_by_sample = {row["sample_id"]: row for _, row in adapter.iterrows()}
    quality_by_sample = load_quality_rows()

    rows: list[dict] = []
    conflicts: list[dict] = []
    clean: list[dict] = []
    approved: list[dict] = []
    promote_yes: list[dict] = []
    rejected: list[dict] = []
    unsure: list[dict] = []
    manual_recheck: list[dict] = []

    for _, review in review_rows.iterrows():
        sample_id = str(review["sample_id"])
        adapter_row = adapter_by_sample.get(sample_id)
        quality = quality_by_sample.get(sample_id, {})
        notes = []
        source_exists = adapter_row is not None
        if not source_exists:
            notes.append("missing_stage4a73_adapter_source_row")
        label_valid = source_exists and safe_bool(adapter_row["label_valid"])
        quality_keep = source_exists and safe_bool(adapter_row["quality_keep"])
        if source_exists and not label_valid:
            notes.append("adapter_label_invalid")
        if source_exists and not quality_keep:
            notes.append("adapter_quality_keep_false")
        quality_blocker = str(quality.get("quality_blocker", "") or "")
        quality_warning = str(quality.get("quality_warning", "") or "")
        if quality_blocker:
            notes.append("quality_blocker_present")
        status = str(review["human_review_status"])
        reason = str(review["human_review_reason"] or "")
        promote = str(review["promote_candidate_yes_no"] or "")
        if status in {"unsure", "needs_closer_inspection"}:
            notes.append("human_status_requires_recheck")
        if reason in NEGATIVE_REASONS:
            notes.append(f"negative_review_reason:{reason}")
        if promote == "yes" and status != "approve":
            notes.append("promote_yes_without_approve")
        clean_candidate = (
            status == "approve"
            and promote == "yes"
            and source_exists
            and label_valid
            and quality_keep
            and not quality_blocker
            and reason not in NEGATIVE_REASONS
        )
        if clean_candidate:
            bucket = "clean_promotion_candidate"
        elif status == "reject":
            bucket = "rejected"
        elif status in {"unsure", "needs_closer_inspection"}:
            bucket = "unsure_or_needs_closer_inspection"
        elif promote == "yes":
            bucket = "conflict_manual_recheck"
        elif status == "approve":
            bucket = "approved_no_promote"
        else:
            bucket = "manual_recheck"
        row = {
            "review_id": review["review_id"],
            "sample_id": sample_id,
            "start_id": int(review["start_id"]),
            "step_id": int(review["step_id"]),
            "human_review_status": status,
            "human_review_reason": reason,
            "human_comment": str(review["human_comment"] or ""),
            "promote_candidate_yes_no": promote,
            "adapter_sample_index": int(adapter_row["sample_index"]) if source_exists else "",
            "candidate_action_index_uncertainty_bonus_executed": int(adapter_row["candidate_action_index_uncertainty_bonus_executed"]) if source_exists else "",
            "expert_action_index_lambda48_shadow": int(adapter_row["expert_action_index_lambda48_shadow"]) if source_exists else "",
            "source_row_exists_in_stage4a73_adapter": source_exists,
            "candidate_label_valid": bool(label_valid),
            "quality_keep": bool(quality_keep),
            "quality_warning": quality_warning,
            "quality_blocker": quality_blocker,
            "auto_quality_verdict": quality.get("auto_quality_verdict", ""),
            "clean_promotion_candidate": clean_candidate,
            "decision_bucket": bucket,
            "decision_notes": ";".join(notes),
        }
        rows.append(row)
        if status == "approve":
            approved.append(row)
        if promote == "yes":
            promote_yes.append(row)
        if clean_candidate:
            clean.append(row)
        if bucket == "conflict_manual_recheck":
            conflicts.append(row)
        if status == "reject":
            rejected.append(row)
        if status in {"unsure", "needs_closer_inspection"}:
            unsure.append(row)
        if bucket in {"conflict_manual_recheck", "unsure_or_needs_closer_inspection", "manual_recheck"}:
            manual_recheck.append(row)

    write_csv(OUT / "all_human_approved_rows.csv", approved)
    write_json(OUT / "all_human_approved_rows.json", approved)
    write_csv(OUT / "human_promote_yes_rows.csv", promote_yes)
    write_json(OUT / "human_promote_yes_rows.json", promote_yes)
    write_csv(OUT / "clean_promotion_candidates.csv", clean)
    write_json(OUT / "clean_promotion_candidates.json", clean)
    write_csv(OUT / "conflict_manual_recheck_rows.csv", conflicts)
    write_json(OUT / "conflict_manual_recheck_rows.json", conflicts)
    write_csv(OUT / "rejected_rows.csv", rejected)
    write_json(OUT / "rejected_rows.json", rejected)
    write_csv(OUT / "unsure_rows.csv", unsure)
    write_json(OUT / "unsure_rows.json", unsure)
    write_csv(OUT / "manual_recheck_rows.csv", manual_recheck)
    write_json(OUT / "manual_recheck_rows.json", manual_recheck)

    policy = {
        "stage": "Stage 4A-7.8",
        "policy_name": "no_training_promotion_candidate_decision_only",
        "clean_promotion_candidate_requires": [
            "human_review_status=approve",
            "promote_candidate_yes_no=yes",
            "warnings/conflicts absent",
            "human_review_reason not unsafe/outside/stuck/revisit",
            "human_review_reason not visual_mismatch",
            "human_review_status not unsure",
            "human_review_status not needs_closer_inspection",
            "source row exists in Stage 4A-7.3 compact adapter",
            "candidate label is valid",
            "no quality blocker",
        ],
        "conflict_rows_must_not_be_promoted": True,
        "stage4a78_creates_expert_action_index_primary": False,
        "stage4a78_promotes_labels": False,
        "stage4a78_trains": False,
        "lambda48_role": "shadow/baseline only",
    }
    write_json(OUT / "promotion_candidate_policy.json", policy)
    (OUT / "promotion_candidate_policy.md").write_text(
        "# Stage 4A-7.8 Promotion Candidate Policy\n\n"
        + "\n".join(f"- {item}" for item in policy["clean_promotion_candidate_requires"])
        + "\n\nConflict rows must not be promoted. Stage 4A-7.8 does not create expert_action_index_primary, promote labels, train, checkpoint, run runtime, rollout, or RL/GDPO/PPO. Lambda48 remains shadow/baseline only.\n",
        encoding="utf-8",
    )

    no_promotion = {
        "label_promotion": False,
        "expert_action_index_primary_created": False,
        "reason": "Stage 4A-7.8 is a decision/audit packet only; future Stage 4A-7.9 must be explicitly approved to implement no-training promotion using clean candidates.",
    }
    no_training = {"training": False, "optimizer_step": False, "checkpoint": False}
    no_runtime = {"isaac_startup": False, "map_predict": False, "rollout": False, "rl_gdpo_ppo": False}
    write_json(OUT / "no_promotion_report.json", no_promotion)
    write_json(OUT / "no_training_report.json", no_training)
    write_json(OUT / "no_runtime_report.json", no_runtime)
    (OUT / "no_promotion_report.md").write_text("# No Promotion Report\n\nNo labels were promoted and no expert_action_index_primary was created.\n", encoding="utf-8")
    (OUT / "no_training_report.md").write_text("# No Training Report\n\nNo BC training, optimizer step, or checkpoint was run or created.\n", encoding="utf-8")
    (OUT / "no_runtime_report.md").write_text("# No Runtime Report\n\nNo Isaac startup, map_predict, rollout, or RL/GDPO/PPO was run.\n", encoding="utf-8")
    (OUT / "why_no_promotion_yet.md").write_text(
        "# Why No Promotion Yet\n\nStage 4A-7.8 only separates human-reviewed rows into decision buckets. Clean candidates are not promoted until a future Stage 4A-7.9 no-training promotion implementation is explicitly approved and validated.\n",
        encoding="utf-8",
    )
    write_json(OUT / "why_no_promotion_yet.json", no_promotion)
    (OUT / "future_stage4a79_no_training_promotion_implementation_sketch.md").write_text(
        "DO NOT RUN IN STAGE 4A-7.8.\n\n"
        "Future Stage 4A-7.9 may create a no-training promotion implementation that imports only `clean_promotion_candidates.csv/json`, verifies all source hashes again, creates a clearly versioned promoted-candidate artifact, and still avoids BC training/checkpoints/runtime unless separately approved.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_next_faithful_step.md").write_text(
        "# Recommended Next Faithful Step\n\nExplicitly approve future Stage 4A-7.9 no-training promotion implementation using clean candidates only, or keep hold/no-promotion/no-training.\n",
        encoding="utf-8",
    )

    source_hashes = finalize_hash_report(source_hashes)
    prior_hashes = finalize_hash_report(prior_hashes)
    write_json(OUT / "source_hash_report.json", source_hashes)
    write_json(OUT / "prior_dataset_hash_report.json", prior_hashes)
    (OUT / "source_hash_report.md").write_text(
        "# Source Hash Report\n\n" + "\n".join(f"- `{k}` unchanged: `{v['unchanged']}` path `{v['path']}`" for k, v in source_hashes.items()) + "\n",
        encoding="utf-8",
    )
    (OUT / "prior_dataset_hash_report.md").write_text(
        "# Prior Dataset Hash Report\n\n" + "\n".join(f"- `{k}` unchanged: `{v['unchanged']}` path `{v['path']}`" for k, v in prior_hashes.items()) + "\n",
        encoding="utf-8",
    )

    evidence = {
        "stage4a77_summary_path": str(source_paths["stage4a77_summary"]),
        "stage4a77_rows_path": str(source_paths["stage4a77_rows"]),
        "stage4a77_promote_yes_rows_path": str(source_paths["stage4a77_promote_yes_rows"]),
        "stage4a77_completed": stage77_summary.get("completed"),
        "stage4a77_blocked": stage77_summary.get("blocked"),
        "stage4a77_row_count": stage77_summary.get("row_count"),
        "stage4a77_status_counts": stage77_summary.get("status_counts"),
        "stage4a77_promote_counts": stage77_summary.get("promote_candidate_yes_no_counts"),
        "stage4a77_warning_count": stage77_summary.get("warning_count"),
        "stage4a77_input_sha256": stage77_summary.get("input_sha256"),
        "stage4a73_adapter_sample_count": int(len(adapter)),
        "stage4a73_label_valid_count": int(adapter["label_valid"].astype(bool).sum()),
        "stage4a72_review_packet": str(STAGE76),
        "stage4a75_policy_packet": str(STAGE75),
        "stage4a73_adapter_packet": str(STAGE73),
        "stage4a73_qa_packet": str(STAGE73_QA),
        "stage4a72_runtime_packet": str(STAGE72),
        "stage4a72_qa_packet": str(STAGE72_QA),
    }
    write_json(OUT / "loaded_stage4a77_review_evidence.json", evidence)
    (OUT / "loaded_stage4a77_review_evidence.md").write_text(
        "# Loaded Stage 4A-7.7 Review Evidence\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in evidence.items()) + "\n",
        encoding="utf-8",
    )

    summary = {
        "completed": True,
        "blocked": False,
        "stage": "Stage 4A-7.8",
        "output_dir": str(OUT),
        "stage4a77_review_import": str(source_paths["stage4a77_summary"]),
        "imported_rows": int(len(rows)),
        "approve": int(Counter(r["human_review_status"] for r in rows).get("approve", 0)),
        "reject": int(len(rejected)),
        "unsure": int(len(unsure)),
        "promote_yes": int(len(promote_yes)),
        "stage4a77_warnings": int(stage77_summary.get("warning_count", -1)),
        "clean_promotion_candidates": int(len(clean)),
        "conflict_rows": int(len(conflicts)),
        "rejected_rows": int(len(rejected)),
        "unsure_rows": int(len(unsure)),
        "manual_recheck_rows": int(len(manual_recheck)),
        "expert_action_index_primary_created": False,
        "label_promotion": False,
        "bc_training": False,
        "optimizer_step": False,
        "checkpoint": False,
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
        "lambda48_role": "shadow/baseline only",
        "prior_datasets_unchanged": all(item["unchanged"] for item in prior_hashes.values() if item["exists"]),
        "source_hashes_unchanged": all(item["unchanged"] for item in source_hashes.values() if item["exists"]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "stage4a78_promotion_candidate_decision_summary.json", summary)
    md = ["# Stage 4A-7.8 Promotion Candidate Decision Summary", ""]
    for key, value in summary.items():
        md.append(f"- {key}: `{value}`" if isinstance(value, str) else f"- {key}: {value}")
    (OUT / "stage4a78_promotion_candidate_decision_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
