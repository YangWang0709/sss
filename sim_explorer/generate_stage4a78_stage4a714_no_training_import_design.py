#!/usr/bin/env python3
"""Generate Stage 4A-7.8 no-training import design for Stage 4A-7.14 candidates.

This is a design/preparation packet only. It reads the Stage 4A-7.7 clean
candidate gate output and describes how those rows could be imported later.
It does not create or modify a dataset, does not create expert_action_index_primary,
does not promote labels, train, checkpoint, launch Isaac, run map_predict, run
rollout, or run RL/GDPO/PPO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_STAGE77 = WORKSPACE / "outputs/stage4a77_stage4a714_manual_review_promotion_gate"
DEFAULT_STAGE714 = WORKSPACE / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"
DEFAULT_STAGE70 = WORKSPACE / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/stage4a78_stage4a714_no_training_import_design"

OUTPUT_FIELDS = [
    "planned_import_id",
    "sample_id",
    "start_id",
    "step_id",
    "runtime_row_index",
    "action_index_primary_uncertainty_bonus",
    "action_index_lambda48_shadow",
    "selected_world_xyz_primary",
    "selected_yaw_primary",
    "source_to_action_distance_m",
    "action_distance_flag",
    "newly_observed_xy_cells",
    "human_review_status",
    "human_review_reason",
    "promote_candidate_yes_no",
    "candidate_features_csv",
    "runtime_observed_state_reference",
    "adapter_required",
    "direct_npz_concat_allowed",
    "planned_label_source",
    "lambda48_role",
    "import_status",
    "import_notes",
]


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


def hash_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
    }


def sample_id(start_id: int, step_id: int) -> str:
    return f"start_{start_id:03d}_step_{step_id:03d}"


def runtime_row_index(runtime_npz: np.lib.npyio.NpzFile) -> dict[str, int]:
    out = {}
    for idx, (start, step) in enumerate(zip(runtime_npz["start_variant_id"], runtime_npz["step_id"])):
        out[sample_id(int(start), int(step))] = idx
    return out


def csv_feature_columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def build_planned_rows(clean_rows: list[dict[str, str]], runtime_npz: np.lib.npyio.NpzFile, row_index: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for import_idx, clean in enumerate(clean_rows):
        sid = clean["sample_id"]
        idx = row_index[sid]
        action_idx = int(runtime_npz["action_index_primary_uncertainty_bonus"][idx])
        lambda48_idx = int(runtime_npz["action_index_lambda48_shadow"][idx])
        rows.append(
            {
                "planned_import_id": f"stage4a78_stage4a714_import_{import_idx:03d}",
                "sample_id": sid,
                "start_id": int(clean["start_id"]),
                "step_id": int(clean["step_id"]),
                "runtime_row_index": idx,
                "action_index_primary_uncertainty_bonus": action_idx,
                "action_index_lambda48_shadow": lambda48_idx,
                "selected_world_xyz_primary": json.dumps(runtime_npz["selected_world_xyz_primary"][idx].astype(float).tolist()),
                "selected_yaw_primary": float(runtime_npz["selected_yaw_primary"][idx]),
                "source_to_action_distance_m": float(clean["source_to_action_distance_m"]),
                "action_distance_flag": clean["action_distance_flag"],
                "newly_observed_xy_cells": int(float(clean["newly_observed_xy_cells"])),
                "human_review_status": clean["human_review_status"],
                "human_review_reason": clean["human_review_reason"],
                "promote_candidate_yes_no": clean["promote_candidate_yes_no"],
                "candidate_features_csv": clean["runtime_candidate_features"],
                "runtime_observed_state_reference": clean["runtime_observed_state_reference"],
                "adapter_required": True,
                "direct_npz_concat_allowed": False,
                "planned_label_source": "stage4a714_uncertainty_bonus_executed_clean_candidate_pending_future_import",
                "lambda48_role": "shadow/baseline only",
                "import_status": "planned_not_applied",
                "import_notes": "Clean Stage 4A-7.7 gate candidate; held for future no-training implementation because Stage 4A-7.8 is design-only.",
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 4A-7.8 Stage 4A-7.14 No-Training Import Design",
        "",
        "This packet prepares a future import path for the 25 clean candidates. It does not apply the import.",
        "",
        "## Result",
        f"- Completed: `{summary['completed']}`",
        f"- Blocked: `{summary['blocked']}`",
        f"- Blockers: `{summary['blockers']}`",
        f"- Planned clean imports: `{summary['planned_import_count']}`",
        f"- Held distance recheck rows: `{summary['held_distance_recheck_count']}`",
        f"- Direct NPZ concat allowed: `{summary['schema_compatibility']['direct_npz_concat_allowed']}`",
        f"- Adapter required: `{summary['schema_compatibility']['adapter_required']}`",
        "",
        "## Schema",
        f"- Existing primary BC model feature shape: `{summary['schema_compatibility']['existing_primary_bc_candidate_features_model_shape']}`",
        f"- Stage 4A-7.14 runtime candidate feature shape: `{summary['schema_compatibility']['stage4a714_runtime_candidate_features_shape']}`",
        f"- Existing compact_v1 feature count: `{summary['schema_compatibility']['existing_model_feature_count']}`",
        f"- Runtime compact source feature count: `{summary['schema_compatibility']['runtime_feature_count']}`",
        "",
        "## Outputs",
    ]
    for key, value in summary["output_files"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Label Policy",
            "- The 25 rows are planned future import candidates only.",
            "- Stage 4A-7.8 does not create `expert_action_index_primary`.",
            "- Stage 4A-7.8 does not recompute labels from lambda48.",
            "- Lambda48 remains shadow/baseline only.",
            "",
            "## Negative Scope",
        ]
    )
    for key, value in summary["negative_scope"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Recommended Next Step", summary["recommended_next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_design(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage77_summary = read_json(args.stage77 / "stage4a77_stage4a714_promotion_gate_summary.json")
    clean_rows = read_csv(args.stage77 / "stage4a77_stage4a714_clean_promotion_candidates.csv")
    distance_rows = read_csv(args.stage77 / "stage4a77_stage4a714_distance_manual_recheck_rows.csv")
    runtime_npz_path = args.stage714 / "short_rollout_dataset_uncertainty_bonus.npz"
    runtime_npz = np.load(runtime_npz_path, allow_pickle=False)
    primary_npz_path = args.stage70 / "bc_dataset_primary_short_rollout.npz"
    primary_npz = np.load(primary_npz_path, allow_pickle=False)
    schema_path = args.stage70 / "bc_candidate_feature_schema.json"
    schema = read_json(schema_path)
    row_index = runtime_row_index(runtime_npz)
    planned_rows = build_planned_rows(clean_rows, runtime_npz, row_index)

    candidate_feature_csv_columns = []
    if planned_rows:
        candidate_feature_csv_columns = csv_feature_columns(Path(planned_rows[0]["candidate_features_csv"]))

    blockers: list[str] = []
    if stage77_summary.get("completed") is not True or stage77_summary.get("blocked") is not False:
        blockers.append("stage4a77_gate_not_complete")
    if len(clean_rows) != 25:
        blockers.append(f"clean_candidate_count_expected_25_got_{len(clean_rows)}")
    if len(distance_rows) != 3:
        blockers.append(f"distance_recheck_count_expected_3_got_{len(distance_rows)}")
    missing_runtime_rows = [row["sample_id"] for row in clean_rows if row["sample_id"] not in row_index]
    if missing_runtime_rows:
        blockers.append("clean_candidates_missing_runtime_rows")

    runtime_feature_names = [str(x) for x in runtime_npz["candidate_feature_names"].tolist()]
    existing_model_features = [str(x) for x in schema["model_feature_names"]]
    adapter_mapping = {
        "gain_exp_norm_per_sample": "derive from runtime gain_exp per sample minmax",
        "inverse_path_cost": "derive from runtime path_cost as inverse cost",
        "source_occ_free_norm_per_sample": "derive from runtime source_occ_free per sample minmax",
        "candidate_confidence_mean": "runtime confidence_mean",
        "candidate_entropy_mean": "runtime entropy_mean",
        "candidate_margin_mean": "runtime margin_mean",
        "candidate_uncertain_fraction": "runtime uncertain_fraction",
        "uncertainty_composite": "runtime uncertainty_composite",
        "distance_xy": "derive from per-sample candidate_features.csv geometry columns",
        "yaw_delta": "derive from per-sample candidate_features.csv yaw_delta_rad",
        "astar_reachable": "derive from per-sample candidate_features.csv astar_reachable",
        "same_cell_target": "runtime same_cell_target / candidate CSV if available",
        "repeated_target": "runtime repeated_target / candidate CSV if available",
        "candidate_all_local": "runtime candidate_all_local / candidate CSV if available",
        "low_cost_artifact": "runtime low_cost_artifact",
        "historical_prior_basin": "runtime historical_prior_basin",
    }
    missing_mapping = [name for name in existing_model_features if name not in adapter_mapping]
    if missing_mapping:
        blockers.append("adapter_mapping_missing_model_features")

    output_files = {
        "summary_json": str(output_dir / "stage4a78_stage4a714_no_training_import_design_summary.json"),
        "summary_md": str(output_dir / "stage4a78_stage4a714_no_training_import_design_summary.md"),
        "planned_import_manifest_csv": str(output_dir / "stage4a78_stage4a714_planned_import_manifest.csv"),
        "planned_import_manifest_json": str(output_dir / "stage4a78_stage4a714_planned_import_manifest.json"),
        "planned_import_manifest_jsonl": str(output_dir / "stage4a78_stage4a714_planned_import_manifest.jsonl"),
        "schema_compatibility_json": str(output_dir / "stage4a78_stage4a714_schema_compatibility_report.json"),
        "schema_compatibility_md": str(output_dir / "stage4a78_stage4a714_schema_compatibility_report.md"),
        "label_policy_json": str(output_dir / "stage4a78_stage4a714_label_policy.json"),
        "no_promotion_md": str(output_dir / "stage4a78_stage4a714_no_promotion_report.md"),
    }

    schema_compatibility = {
        "direct_npz_concat_allowed": False,
        "adapter_required": True,
        "reason": "Stage 4A-7.14 runtime candidate_features are 13-wide, while existing primary BC candidate_features_model is 16-wide compact_v1.",
        "existing_primary_bc_candidate_features_model_shape": list(primary_npz["candidate_features_model"].shape),
        "stage4a714_runtime_candidate_features_shape": list(runtime_npz["candidate_features"].shape),
        "existing_model_feature_count": len(existing_model_features),
        "runtime_feature_count": len(runtime_feature_names),
        "existing_model_features": existing_model_features,
        "runtime_feature_names": runtime_feature_names,
        "candidate_features_csv_columns_available": candidate_feature_csv_columns,
        "adapter_mapping": adapter_mapping,
        "adapter_mapping_complete_for_model_features": not missing_mapping,
        "missing_adapter_mapping": missing_mapping,
    }
    label_policy = {
        "stage": "Stage 4A-7.8",
        "label_promotion": False,
        "expert_action_index_primary_created": False,
        "planned_future_label_source": "stage4a714_uncertainty_bonus_executed_clean_candidate_pending_future_import",
        "runtime_primary_index_key": "action_index_primary_uncertainty_bonus",
        "lambda48_role": "shadow/baseline only",
        "do_not_recompute_from_lambda48": True,
        "primary_bc_dataset_modified": False,
        "training": False,
    }
    source_hashes = hash_report(
        {
            "stage4a77_summary": args.stage77 / "stage4a77_stage4a714_promotion_gate_summary.json",
            "stage4a77_clean_candidates": args.stage77 / "stage4a77_stage4a714_clean_promotion_candidates.csv",
            "stage4a77_distance_recheck": args.stage77 / "stage4a77_stage4a714_distance_manual_recheck_rows.csv",
            "stage4a714_runtime_npz": runtime_npz_path,
            "stage4a714_runtime_transitions": args.stage714 / "transition_decisions.csv",
            "stage4a70_primary_npz": primary_npz_path,
            "stage4a70_feature_schema": schema_path,
        }
    )

    write_csv(Path(output_files["planned_import_manifest_csv"]), planned_rows)
    write_json(Path(output_files["planned_import_manifest_json"]), planned_rows)
    write_jsonl(Path(output_files["planned_import_manifest_jsonl"]), planned_rows)
    write_json(Path(output_files["schema_compatibility_json"]), schema_compatibility)
    write_json(Path(output_files["label_policy_json"]), label_policy)
    write_json(output_dir / "stage4a78_stage4a714_source_hash_report.json", source_hashes)
    (output_dir / "stage4a78_stage4a714_schema_compatibility_report.md").write_text(
        "# Stage 4A-7.14 Import Schema Compatibility\n\n"
        f"- Direct NPZ concat allowed: `{schema_compatibility['direct_npz_concat_allowed']}`\n"
        f"- Adapter required: `{schema_compatibility['adapter_required']}`\n"
        f"- Reason: {schema_compatibility['reason']}\n"
        f"- Existing model shape: `{schema_compatibility['existing_primary_bc_candidate_features_model_shape']}`\n"
        f"- Runtime shape: `{schema_compatibility['stage4a714_runtime_candidate_features_shape']}`\n"
        f"- Adapter mapping complete: `{schema_compatibility['adapter_mapping_complete_for_model_features']}`\n",
        encoding="utf-8",
    )
    (output_dir / "stage4a78_stage4a714_no_promotion_report.md").write_text(
        "# No Promotion Report\n\n"
        "Stage 4A-7.8 only creates a design/manifest packet. It does not promote labels, "
        "does not create expert_action_index_primary, and does not modify any dataset.\n",
        encoding="utf-8",
    )
    write_json(output_dir / "stage4a78_stage4a714_no_promotion_report.json", label_policy)
    no_training = {
        "training": False,
        "optimizer_step": False,
        "checkpoint": False,
        "bc_training": False,
    }
    no_runtime = {
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
    }
    write_json(output_dir / "stage4a78_stage4a714_no_training_report.json", no_training)
    write_json(output_dir / "stage4a78_stage4a714_no_runtime_report.json", no_runtime)
    (output_dir / "stage4a78_stage4a714_no_training_report.md").write_text(
        "# No Training Report\n\nNo BC training, optimizer step, checkpoint, or model save was run.\n",
        encoding="utf-8",
    )
    (output_dir / "stage4a78_stage4a714_no_runtime_report.md").write_text(
        "# No Runtime Report\n\nNo Isaac startup, map_predict, rollout, or RL/GDPO/PPO was run.\n",
        encoding="utf-8",
    )
    (output_dir / "future_stage4a79_stage4a714_no_training_import_implementation_sketch.md").write_text(
        "DO NOT RUN IN STAGE 4A-7.8.\n\n"
        "Future Stage 4A-7.9 may implement a no-training import artifact using only the 25 planned rows, "
        "after rebuilding compact_v1 candidate_features_model through the documented adapter mapping. "
        "Actual BC training/checkpoint/RL remains a separate explicit gate.\n",
        encoding="utf-8",
    )

    summary = {
        "stage": "Stage 4A-7.8 Stage 4A-7.14 no-training import design",
        "completed": not blockers,
        "blocked": bool(blockers),
        "blockers": blockers,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(WORKSPACE),
        "output_dir": str(output_dir),
        "stage4a77_gate_summary": str(args.stage77 / "stage4a77_stage4a714_promotion_gate_summary.json"),
        "planned_import_count": len(planned_rows),
        "held_distance_recheck_count": len(distance_rows),
        "rejected_count_from_gate": int(stage77_summary.get("rejected_count", 0)),
        "approved_no_promote_count_from_gate": int(stage77_summary.get("approved_no_promote_count", 0)),
        "schema_compatibility": schema_compatibility,
        "label_policy": label_policy,
        "output_files": output_files,
        "source_hashes": source_hashes,
        "negative_scope": {
            "label_promotion": False,
            "dataset_modified": False,
            "expanded_dataset_npz_created": False,
            "expert_action_index_primary_created": False,
            "training": False,
            "optimizer_step": False,
            "checkpoint": False,
            "isaac_startup": False,
            "map_predict": False,
            "rollout": False,
            "rl_gdpo_ppo": False,
        },
        "recommended_next_step": (
            "Future Stage 4A-7.9 can build a no-training import implementation artifact from the 25 planned rows, "
            "but must first materialize the documented compact_v1 adapter and rerun schema/lineage validation. "
            "Training/checkpoint/RL remains separate."
        ),
    }
    write_json(Path(output_files["summary_json"]), summary)
    write_markdown(Path(output_files["summary_md"]), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage77", type=Path, default=DEFAULT_STAGE77)
    parser.add_argument("--stage714", type=Path, default=DEFAULT_STAGE714)
    parser.add_argument("--stage70", type=Path, default=DEFAULT_STAGE70)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = generate_design(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
