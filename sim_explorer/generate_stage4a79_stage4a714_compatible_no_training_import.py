#!/usr/bin/env python3
"""Create a compatible no-training import artifact for Stage 4A-7.14 candidates.

This materializes the Stage 4A-7.8 design: the 25 clean Stage 4A-7.14 rows are
adapted into the existing compact_v1 BC feature schema. The script writes a
new versioned compatible artifact under outputs; it does not overwrite the
Stage 4A-7.0 primary dataset, train, checkpoint, launch Isaac, run map_predict,
run rollout, or run RL/GDPO/PPO.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bc_dataset_schema_utils import MODEL_FEATURE_NAMES, RAW_FEATURE_NAMES, as_bool, as_float, finite_minmax, rank_desc


WORKSPACE = Path(__file__).resolve().parents[1]
STAGE78 = WORKSPACE / "outputs/stage4a78_stage4a714_no_training_import_design"
STAGE77 = WORKSPACE / "outputs/stage4a77_stage4a714_manual_review_promotion_gate"
STAGE714 = WORKSPACE / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"
STAGE70 = WORKSPACE / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/stage4a79_stage4a714_compatible_no_training_import"

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
PRIMARY_POLICY_ORIGINAL = "stage4a613_uncertainty_bonus_executed_primary"
PRIMARY_POLICY_STAGE714_COMPAT = "stage4a714_uncertainty_bonus_executed_clean_candidate_compatible_import"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(jsonable(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_literal(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        try:
            return json.loads(str(value))
        except Exception:
            return default


def normalize_yaw_delta(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    return float(abs((a - b + math.pi) % (2 * math.pi) - math.pi))


def np_str(values: list[Any], width: int = 192) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=f"<U{width}")


def sample_id(start_id: int, step_id: int) -> str:
    return f"start_{start_id:03d}_step_{step_id:03d}"


def load_candidate_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(float(row["candidate_id"])): row for row in rows if row.get("candidate_id") not in {None, ""}}


def build_model_features(raw: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_idx = {name: pos for pos, name in enumerate(RAW_FEATURE_NAMES)}
    sample_count, candidate_count, _ = raw.shape
    out = np.full((sample_count, candidate_count, len(MODEL_FEATURE_NAMES)), np.nan, dtype=np.float32)
    for sample_index in range(sample_count):
        gain_norm = finite_minmax(raw[sample_index, :, raw_idx["gain_exp"]], valid_mask[sample_index])
        path = raw[sample_index, :, raw_idx["path_cost"]]
        inverse_path = np.where(np.isfinite(path) & (path >= 0), 1.0 / (1.0 + path), np.nan)
        columns = {
            "gain_exp_norm_per_sample": gain_norm,
            "inverse_path_cost": inverse_path,
            "source_occ_free_norm_per_sample": raw[sample_index, :, raw_idx["source_occ_free_norm_per_sample"]],
            "candidate_confidence_mean": raw[sample_index, :, raw_idx["candidate_confidence_mean"]],
            "candidate_entropy_mean": raw[sample_index, :, raw_idx["candidate_entropy_mean"]],
            "candidate_margin_mean": raw[sample_index, :, raw_idx["candidate_margin_mean"]],
            "candidate_uncertain_fraction": raw[sample_index, :, raw_idx["candidate_uncertain_fraction"]],
            "uncertainty_composite": raw[sample_index, :, raw_idx["uncertainty_composite"]],
            "distance_xy": raw[sample_index, :, raw_idx["distance_xy"]],
            "yaw_delta": raw[sample_index, :, raw_idx["yaw_delta"]],
            "astar_reachable": raw[sample_index, :, raw_idx["astar_reachable"]],
            "same_cell_target": raw[sample_index, :, raw_idx["same_cell_target"]],
            "repeated_target": raw[sample_index, :, raw_idx["repeated_target"]],
            "candidate_all_local": raw[sample_index, :, raw_idx["candidate_all_local"]],
            "low_cost_artifact": raw[sample_index, :, raw_idx["low_cost_artifact"]],
            "historical_prior_basin": raw[sample_index, :, raw_idx["historical_prior_basin"]],
        }
        for feature_index, name in enumerate(MODEL_FEATURE_NAMES):
            out[sample_index, :, feature_index] = np.asarray(columns[name], dtype=np.float32)
    missing = ~np.isfinite(out)
    imputed = np.where(missing, 0.0, out).astype(np.float32)
    feature_mask = (~missing) & valid_mask[:, :, None]
    return imputed, feature_mask, missing


def hash_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def build_runtime_index(runtime_npz: np.lib.npyio.NpzFile) -> dict[str, int]:
    return {
        sample_id(int(start), int(step)): idx
        for idx, (start, step) in enumerate(zip(runtime_npz["start_variant_id"], runtime_npz["step_id"]))
    }


def runtime_score_default(
    runtime_npz: np.lib.npyio.NpzFile,
    key: str,
    runtime_i: int,
    candidate_i: int,
    default: float = math.nan,
) -> float:
    arr = np.asarray(runtime_npz[key])
    if arr.ndim == 2:
        return float(arr[runtime_i, candidate_i])
    if arr.ndim == 1:
        return float(arr[runtime_i])
    return default


def build_stage714_raw(
    planned_rows: list[dict[str, str]],
    runtime_npz: np.lib.npyio.NpzFile,
    runtime_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    raw_idx = {name: pos for pos, name in enumerate(RAW_FEATURE_NAMES)}
    n = len(planned_rows)
    candidate_count = int(runtime_npz["candidate_features"].shape[1])
    raw = np.full((n, candidate_count, len(RAW_FEATURE_NAMES)), np.nan, dtype=np.float32)
    valid_mask = np.zeros((n, candidate_count), dtype=bool)
    candidate_mask = np.zeros((n, candidate_count), dtype=bool)
    row_reports: list[dict[str, Any]] = []

    for local_i, planned in enumerate(planned_rows):
        sid = planned["sample_id"]
        runtime_i = runtime_index[sid]
        csv_rows = load_candidate_csv(Path(planned["candidate_features_csv"]))
        pose = runtime_npz["pose"][runtime_i].astype(float)
        pose_xyz = pose[:3]
        pose_yaw = float(pose[3])
        sample_flags = {
            "same_cell_target": bool(runtime_npz["same_cell_target"][runtime_i]),
            "repeated_target": bool(runtime_npz["repeated_target"][runtime_i]),
            "outside_bounds_target": bool(runtime_npz["outside_bounds_target"][runtime_i]),
            "no_valid_candidate": bool(runtime_npz["no_valid_candidate"][runtime_i]),
            "low_cost_artifact": bool(runtime_npz["low_cost_artifact"][runtime_i]),
            "historical_prior_basin": bool(runtime_npz["historical_prior_basin"][runtime_i]),
            "candidate_all_local": bool(runtime_npz["candidate_all_local"][runtime_i]),
            "prediction_writeback": bool(runtime_npz["prediction_writeback"][runtime_i]),
            "uncertainty_writeback": bool(runtime_npz["uncertainty_writeback"][runtime_i]),
            "prediction_traversability_use": bool(runtime_npz["prediction_traversability_use"][runtime_i]),
            "uncertainty_traversability_use": bool(runtime_npz["uncertainty_traversability_use"][runtime_i]),
            "prediction_collision_use": bool(runtime_npz["prediction_collision_use"][runtime_i]),
            "uncertainty_collision_use": bool(runtime_npz["uncertainty_collision_use"][runtime_i]),
            "prediction_ray_blocking_use": bool(runtime_npz["prediction_ray_blocking_use"][runtime_i]),
            "uncertainty_ray_blocking_use": bool(runtime_npz["uncertainty_ray_blocking_use"][runtime_i]),
            "prediction_candidate_validity_use": bool(runtime_npz["prediction_candidate_validity_use"][runtime_i]),
            "uncertainty_candidate_validity_use": bool(runtime_npz["uncertainty_candidate_validity_use"][runtime_i]),
            "target_ground_truth_use": bool(runtime_npz["target_ground_truth_use"][runtime_i]),
            "future_observed_scoring_use": bool(runtime_npz["future_observed_scoring_use"][runtime_i]),
        }
        candidate_mask[local_i] = runtime_npz["candidate_mask"][runtime_i].astype(bool)
        valid_mask[local_i] = runtime_npz["valid_mask"][runtime_i].astype(bool)
        raw_rows_for_ranks: list[dict[str, float]] = []
        for candidate_i in range(candidate_count):
            row = csv_rows.get(candidate_i, {})
            grid = parse_literal(row.get("grid"), [math.nan, math.nan, math.nan])
            world = parse_literal(row.get("world"), [math.nan, math.nan, math.nan])
            yaw = as_float(row.get("yaw_rad"))
            delta = [
                float(world[axis]) - float(pose_xyz[axis]) if axis < len(world) and math.isfinite(float(world[axis])) else math.nan
                for axis in range(3)
            ]
            distance_xy = math.hypot(delta[0], delta[1]) if math.isfinite(delta[0]) and math.isfinite(delta[1]) else math.nan
            yaw_delta = as_float(row.get("yaw_delta_rad"), normalize_yaw_delta(yaw, pose_yaw))
            raw_row = {
                "candidate_id": float(candidate_i),
                "candidate_rank": float(candidate_i),
                "candidate_valid": float(bool(valid_mask[local_i, candidate_i])),
                "grid_x": as_float(grid[0] if len(grid) > 0 else math.nan),
                "grid_y": as_float(grid[1] if len(grid) > 1 else math.nan),
                "grid_z": as_float(grid[2] if len(grid) > 2 else math.nan),
                "world_x": as_float(world[0] if len(world) > 0 else math.nan),
                "world_y": as_float(world[1] if len(world) > 1 else math.nan),
                "world_z": as_float(world[2] if len(world) > 2 else math.nan),
                "yaw": yaw,
                "delta_x": delta[0],
                "delta_y": delta[1],
                "delta_z": delta[2],
                "distance_xy": distance_xy,
                "yaw_delta": yaw_delta,
                "step_id": float(runtime_npz["step_id"][runtime_i]),
                "start_variant_id": float(runtime_npz["start_variant_id"][runtime_i]),
                "gain_exp": as_float(row.get("gain_exp"), float(runtime_npz["gain_exp"][runtime_i, candidate_i])),
                "frontier_count": as_float(row.get("frontier_count_visible")),
                "frontier_adjacent_count": math.nan,
                "observed_ratio": float(runtime_npz["observed_ratio_before"][runtime_i]),
                "observed_count": float(runtime_npz["observed_count"][runtime_i]),
                "unknown_count": float(runtime_npz["unknown_count"][runtime_i]),
                "free_count": float(runtime_npz["free_count"][runtime_i]),
                "occupied_count": float(runtime_npz["occupied_count"][runtime_i]),
                "newly_observed_count_if_available": float(runtime_npz["newly_observed_count"][runtime_i]),
                "local_observed_density_if_available": math.nan,
                "path_cost": as_float(row.get("path_cost"), float(runtime_npz["path_cost"][runtime_i, candidate_i])),
                "astar_reachable": float(as_bool(row.get("astar_reachable"))),
                "astar_path_length_m": as_float(row.get("path_cost_m")),
                "astar_num_expanded": as_float(row.get("astar_num_expanded")),
                "reachable_component_count": math.nan,
                "reachable_frontier_adjacent_count": math.nan,
                "same_cell_target": float(sample_flags["same_cell_target"]),
                "repeated_target": float(sample_flags["repeated_target"]),
                "outside_bounds_target": float(sample_flags["outside_bounds_target"]),
                "source_occ_free": as_float(row.get("source_occ_free"), float(runtime_npz["source_occ_free"][runtime_i, candidate_i])),
                "gain_sc": as_float(row.get("gain_sc")),
                "gain_occ": as_float(row.get("raw_gain_sc"), as_float(row.get("source_occ_free"))),
                "gain_conf": as_float(row.get("source_confidence_gate_raw")),
                "prediction_valid_count": as_float(row.get("visible_prediction_voxel_count")),
                "predicted_unmeasured_count": as_float(row.get("visible_predicted_unmeasured_count")),
                "predicted_occupied_count": math.nan,
                "prediction_density": math.nan,
                "source_occ_free_norm_per_sample": as_float(row.get("source_occ_free_minmax_stage4a613"), as_float(row.get("source_occ_free_minmax"))),
                "candidate_confidence_mean": as_float(row.get("candidate_confidence_mean"), float(runtime_npz["confidence_mean"][runtime_i, candidate_i])),
                "candidate_confidence_min": as_float(row.get("candidate_confidence_min")),
                "candidate_confidence_p10": as_float(row.get("candidate_confidence_p10")),
                "candidate_confidence_p50": as_float(row.get("candidate_confidence_p50")),
                "candidate_confidence_p90": as_float(row.get("candidate_confidence_p90")),
                "candidate_entropy_mean": as_float(row.get("candidate_entropy_mean"), float(runtime_npz["entropy_mean"][runtime_i, candidate_i])),
                "candidate_entropy_max": as_float(row.get("candidate_entropy_max")),
                "candidate_entropy_p90": as_float(row.get("candidate_entropy_p90")),
                "candidate_margin_mean": as_float(row.get("candidate_margin_mean"), float(runtime_npz["margin_mean"][runtime_i, candidate_i])),
                "candidate_margin_min": as_float(row.get("candidate_margin_min")),
                "candidate_uncertain_fraction": as_float(row.get("candidate_uncertain_fraction"), float(runtime_npz["uncertain_fraction"][runtime_i, candidate_i])),
                "candidate_uncertain_voxel_count": as_float(row.get("uncertain_voxel_count"), float(runtime_npz["uncertain_voxel_count"][runtime_i, candidate_i])),
                "low_conf_count_0p7": as_float(row.get("low_conf_count_0p7")),
                "high_entropy_count_0p7": as_float(row.get("high_entropy_count_0p7")),
                "low_margin_count_0p2": as_float(row.get("low_margin_count_0p2")),
                "uncertainty_composite": as_float(row.get("uncertainty_composite"), float(runtime_npz["uncertainty_composite"][runtime_i, candidate_i])),
                "uncertainty_composite_norm_per_sample": math.nan,
                "score_measured": as_float(row.get("score_measured_only"), runtime_score_default(runtime_npz, "score_measured_shadow", runtime_i, candidate_i)),
                "score_lambda48": as_float(row.get("score_lambda48"), runtime_score_default(runtime_npz, "score_lambda48_shadow", runtime_i, candidate_i)),
                "score_confidence_gated": as_float(row.get("score_confidence_gated_6_11"), runtime_score_default(runtime_npz, "score_confidence_gated_shadow", runtime_i, candidate_i)),
                "score_uncertainty_bonus_composite_beta8": as_float(row.get("score_primary_uncertainty_bonus"), runtime_score_default(runtime_npz, "score_primary_uncertainty_bonus", runtime_i, candidate_i)),
                "final_score_primary": as_float(row.get("score_primary_uncertainty_bonus"), runtime_score_default(runtime_npz, "score_primary_uncertainty_bonus", runtime_i, candidate_i)),
                "score_rank_primary": math.nan,
                "score_rank_measured": math.nan,
                "score_rank_lambda48": math.nan,
                "score_rank_confidence_gated": math.nan,
                "no_valid_candidate": float(sample_flags["no_valid_candidate"]),
                "low_cost_artifact": float(sample_flags["low_cost_artifact"]),
                "historical_prior_basin": float(sample_flags["historical_prior_basin"]),
                "candidate_all_local": float(sample_flags["candidate_all_local"]),
                "high_uncertainty_selection": 0.0,
                "low_confidence_selection": 0.0,
                "low_margin_selection": 0.0,
                "formula_dominated_by_uncertainty": float(as_bool(row.get("formula_dominated_by_uncertainty"))),
                "prediction_writeback": float(sample_flags["prediction_writeback"]),
                "uncertainty_writeback": float(sample_flags["uncertainty_writeback"]),
                "prediction_traversability_use": float(sample_flags["prediction_traversability_use"]),
                "uncertainty_traversability_use": float(sample_flags["uncertainty_traversability_use"]),
                "prediction_collision_use": float(sample_flags["prediction_collision_use"]),
                "uncertainty_collision_use": float(sample_flags["uncertainty_collision_use"]),
                "prediction_ray_blocking_use": float(sample_flags["prediction_ray_blocking_use"]),
                "uncertainty_ray_blocking_use": float(sample_flags["uncertainty_ray_blocking_use"]),
                "prediction_candidate_validity_use": float(sample_flags["prediction_candidate_validity_use"]),
                "uncertainty_candidate_validity_use": float(sample_flags["uncertainty_candidate_validity_use"]),
                "target_ground_truth_use": float(sample_flags["target_ground_truth_use"]),
                "future_observed_scoring_use": float(sample_flags["future_observed_scoring_use"]),
            }
            raw_rows_for_ranks.append(raw_row)

        unc_values = np.asarray([r["uncertainty_composite"] for r in raw_rows_for_ranks], dtype=np.float64)
        unc_norm = finite_minmax(unc_values, valid_mask[local_i])
        rank_specs = {
            "score_rank_primary": "final_score_primary",
            "score_rank_measured": "score_measured",
            "score_rank_lambda48": "score_lambda48",
            "score_rank_confidence_gated": "score_confidence_gated",
        }
        ranks = {
            rank_name: rank_desc(
                np.asarray([r[score_name] for r in raw_rows_for_ranks], dtype=np.float64),
                valid_mask[local_i],
            )
            for rank_name, score_name in rank_specs.items()
        }
        for candidate_i, raw_row in enumerate(raw_rows_for_ranks):
            raw_row["uncertainty_composite_norm_per_sample"] = float(unc_norm[candidate_i])
            for rank_name, rank_values in ranks.items():
                raw_row[rank_name] = float(rank_values[candidate_i])
            for name, value in raw_row.items():
                if name in raw_idx and math.isfinite(float(value)):
                    raw[local_i, candidate_i, raw_idx[name]] = float(value)
        row_reports.append(
            {
                "sample_id": sid,
                "runtime_row_index": runtime_i,
                "candidate_csv_rows": len(csv_rows),
                "valid_candidate_count": int(np.sum(valid_mask[local_i])),
                "primary_action_index": int(runtime_npz["action_index_primary_uncertainty_bonus"][runtime_i]),
            }
        )
    return raw.astype(np.float32), candidate_mask, valid_mask, row_reports


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


def concat(a: np.ndarray, b: np.ndarray, dtype=None) -> np.ndarray:
    if dtype is not None:
        return np.concatenate([a.astype(dtype), b.astype(dtype)])
    return np.concatenate([a, b])


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 4A-7.9 Stage 4A-7.14 Compatible No-Training Import",
        "",
        "This stage creates a versioned compatible import artifact. It does not overwrite the existing primary dataset or train.",
        "",
        "## Result",
        f"- Completed: `{summary['completed']}`",
        f"- Blocked: `{summary['blocked']}`",
        f"- Blockers: `{summary['blockers']}`",
        f"- Original samples: `{summary['original_primary_sample_count']}`",
        f"- Imported compatible samples: `{summary['compatible_import_sample_count']}`",
        f"- Expanded compatible sample count: `{summary['expanded_compatible_sample_count']}`",
        f"- Compatible model feature shape: `{summary['compatible_dataset_shapes']['candidate_features_model']}`",
        f"- Existing dataset overwritten: `{summary['existing_primary_dataset_overwritten']}`",
        "",
        "## Outputs",
    ]
    for key, value in summary["output_files"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Label Lineage"])
    for key, value in summary["label_lineage"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Negative Scope"])
    for key, value in summary["negative_scope"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Recommended Next Step", summary["recommended_next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_import(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stage78_summary = read_json(args.stage78 / "stage4a78_stage4a714_no_training_import_design_summary.json")
    planned_rows = read_csv(args.stage78 / "stage4a78_stage4a714_planned_import_manifest.csv")
    stage77_summary = read_json(args.stage77 / "stage4a77_stage4a714_promotion_gate_summary.json")
    runtime_npz_path = args.stage714 / "short_rollout_dataset_uncertainty_bonus.npz"
    primary_npz_path = args.stage70 / "bc_dataset_primary_short_rollout.npz"
    runtime_npz = np.load(runtime_npz_path, allow_pickle=False)
    primary_npz = np.load(primary_npz_path, allow_pickle=False)
    runtime_index = build_runtime_index(runtime_npz)

    blockers: list[str] = []
    if stage78_summary.get("completed") is not True:
        blockers.append("stage4a78_design_not_complete")
    if len(planned_rows) != 25:
        blockers.append(f"planned_import_count_expected_25_got_{len(planned_rows)}")
    missing_runtime = [row["sample_id"] for row in planned_rows if row["sample_id"] not in runtime_index]
    if missing_runtime:
        blockers.append("planned_rows_missing_runtime_index")

    raw25, candidate_mask25, valid_mask25, adapter_row_reports = build_stage714_raw(planned_rows, runtime_npz, runtime_index)
    model25, feature_mask25, missing_model25 = build_model_features(raw25, valid_mask25)
    runtime_indices = np.asarray([runtime_index[row["sample_id"]] for row in planned_rows], dtype=np.int64)

    primary25 = runtime_npz["action_index_primary_uncertainty_bonus"][runtime_indices].astype(np.int64)
    measured25 = runtime_npz["action_index_measured_shadow"][runtime_indices].astype(np.int64)
    lambda25 = runtime_npz["action_index_lambda48_shadow"][runtime_indices].astype(np.int64)
    confidence25 = runtime_npz["action_index_confidence_gated_shadow"][runtime_indices].astype(np.int64)
    selected_world25 = runtime_npz["selected_world_xyz_primary"][runtime_indices].astype(np.float32)
    selected_yaw25 = runtime_npz["selected_yaw_primary"][runtime_indices].astype(np.float32)
    pose_world25 = runtime_npz["pose"][runtime_indices, :3].astype(np.float32)
    pose_yaw25 = runtime_npz["pose"][runtime_indices, 3].astype(np.float32)
    start25 = runtime_npz["start_variant_id"][runtime_indices].astype(np.int64)
    step25 = runtime_npz["step_id"][runtime_indices].astype(np.int64)
    sequence25 = start25 + (int(np.max(primary_npz["sequence_id"])) + 1)
    prev25, next25 = build_previous_next(sequence25, step25, list(range(int(primary_npz["sample_id"].shape[0]), int(primary_npz["sample_id"].shape[0]) + len(planned_rows))))
    split25 = np.where(start25 <= 6, 0, np.where(start25 <= 8, 1, 2)).astype(np.int64)
    quality25 = np.ones(len(planned_rows), dtype=bool)
    weight25 = np.ones(len(planned_rows), dtype=np.float32)
    sample_id25 = np_str([row["sample_id"] for row in planned_rows], width=64)
    source_stage25 = np.full(len(planned_rows), 714, dtype=np.int64)

    for local_i, label in enumerate(primary25):
        if not (0 <= int(label) < valid_mask25.shape[1] and bool(valid_mask25[local_i, int(label)])):
            blockers.append(f"invalid_primary_label:{planned_rows[local_i]['sample_id']}:{int(label)}")
    if model25.shape != (25, 64, 16):
        blockers.append(f"adapter_model_shape_expected_25_64_16_got_{model25.shape}")
    if raw25.shape != (25, 64, len(RAW_FEATURE_NAMES)):
        blockers.append(f"adapter_raw_shape_unexpected:{raw25.shape}")
    if set(planned_rows[i]["action_distance_flag"] for i in range(len(planned_rows))) == {"very_close"}:
        blockers.append("unexpected_all_planned_rows_very_close")

    raw_name_to_index = {name: idx for idx, name in enumerate(RAW_FEATURE_NAMES)}

    def score_matrix(raw_name: str) -> np.ndarray:
        scores = raw25[:, :, raw_name_to_index[raw_name]]
        return np.where(np.isfinite(scores), scores, 0.0).astype(np.float32)

    score_primary25 = score_matrix("final_score_primary")
    score_measured25 = score_matrix("score_measured")
    score_lambda4825 = score_matrix("score_lambda48")
    score_confidence25 = score_matrix("score_confidence_gated")

    original_n = int(primary_npz["sample_id"].shape[0])
    expanded_n = original_n + len(planned_rows)
    arrays25 = {
        "sample_id": sample_id25,
        "start_variant_id": start25,
        "step_id": step25,
        "source_stage_id": source_stage25,
        "candidate_features_raw": raw25.astype(np.float32),
        "candidate_features_model": model25.astype(np.float32),
        "candidate_feature_mask": feature_mask25.astype(bool),
        "candidate_valid_mask": valid_mask25.astype(bool),
        "expert_action_index_primary": primary25,
        "expert_action_index_measured_shadow": measured25,
        "expert_action_index_lambda48_shadow": lambda25,
        "expert_action_index_confidence_gated_shadow": confidence25,
        "score_primary": score_primary25,
        "score_measured": score_measured25,
        "score_lambda48": score_lambda4825,
        "score_confidence_gated": score_confidence25,
        "selected_world_xyz_primary": selected_world25,
        "selected_yaw_primary": selected_yaw25,
        "pose_world_xyz": pose_world25,
        "pose_yaw": pose_yaw25,
        "sequence_id": sequence25,
        "previous_sample_index": prev25,
        "next_sample_index": next25,
        "split_id": split25,
        "quality_keep_mask": quality25,
        "sample_weight": weight25,
        "missing_feature_mask": missing_model25.astype(bool),
    }
    expanded_arrays = {
        key: np.concatenate([primary_npz[key], arrays25[key]], axis=0)
        for key in arrays25
        if key in primary_npz.files
    }
    # Extra lineage arrays live only in the compatible artifact, leaving the source dataset untouched.
    expanded_arrays.update(
        {
            "source_stage": np_str(["stage4a70_original_primary"] * original_n + ["stage4a714_compatible_import"] * len(planned_rows)),
            "source_sample_id": np_str([str(x) for x in primary_npz["sample_id"]] + [row["sample_id"] for row in planned_rows], width=96),
            "promotion_source": np_str(["stage4a70_original_primary"] * original_n + ["stage4a77_stage4a714_clean_candidate"] * len(planned_rows)),
            "primary_label_policy": np_str([PRIMARY_POLICY_ORIGINAL] * original_n + [PRIMARY_POLICY_STAGE714_COMPAT] * len(planned_rows), width=128),
            "human_review_status": np_str(["not_applicable"] * original_n + [row["human_review_status"] for row in planned_rows]),
            "human_review_reason": np_str(["not_applicable"] * original_n + [row["human_review_reason"] for row in planned_rows]),
            "human_promote_candidate_yes_no": np_str([""] * original_n + [row["promote_candidate_yes_no"] for row in planned_rows]),
            "runtime_row_index": np.concatenate([np.full(original_n, -1, dtype=np.int64), runtime_indices]),
        }
    )

    forbidden_present = sorted(FORBIDDEN_KEYS.intersection(expanded_arrays))
    if forbidden_present:
        blockers.append("forbidden_keys_present")

    if blockers:
        summary = {
            "completed": False,
            "blocked": True,
            "blockers": blockers,
            "stage": "Stage 4A-7.9 Stage 4A-7.14 compatible no-training import",
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "stage4a79_stage4a714_compatible_import_summary.json", summary)
        return summary

    adapter_path = output_dir / "stage4a79_stage4a714_adapter_dataset_25.npz"
    expanded_path = output_dir / "stage4a79_stage4a714_compatible_expanded_dataset_55.npz"
    np.savez_compressed(adapter_path, **arrays25)
    np.savez_compressed(expanded_path, **expanded_arrays)

    manifest_rows: list[dict[str, Any]] = []
    for local_i, row in enumerate(planned_rows):
        expanded_idx = original_n + local_i
        manifest_rows.append(
            {
                "expanded_sample_index": expanded_idx,
                "adapter_sample_index": local_i,
                "sample_id": row["sample_id"],
                "start_id": int(row["start_id"]),
                "step_id": int(row["step_id"]),
                "runtime_row_index": int(runtime_indices[local_i]),
                "expert_action_index_primary": int(primary25[local_i]),
                "expert_action_index_lambda48_shadow": int(lambda25[local_i]),
                "primary_equals_lambda48_shadow": bool(primary25[local_i] == lambda25[local_i]),
                "selected_world_xyz_primary": selected_world25[local_i].astype(float).tolist(),
                "selected_yaw_primary": float(selected_yaw25[local_i]),
                "split_id": int(split25[local_i]),
                "sequence_id": int(sequence25[local_i]),
                "previous_sample_index": int(prev25[local_i]),
                "next_sample_index": int(next25[local_i]),
                "source_to_action_distance_m": float(row["source_to_action_distance_m"]),
                "action_distance_flag": row["action_distance_flag"],
                "human_review_status": row["human_review_status"],
                "human_review_reason": row["human_review_reason"],
                "promote_candidate_yes_no": row["promote_candidate_yes_no"],
                "primary_label_policy": PRIMARY_POLICY_STAGE714_COMPAT,
            }
        )
    write_csv(output_dir / "stage4a79_stage4a714_compatible_import_manifest.csv", manifest_rows)
    write_json(output_dir / "stage4a79_stage4a714_compatible_import_manifest.json", manifest_rows)
    write_jsonl(output_dir / "stage4a79_stage4a714_compatible_import_manifest.jsonl", manifest_rows)
    write_json(output_dir / "stage4a79_stage4a714_adapter_row_build_report.json", adapter_row_reports)

    source_hashes = hash_report(
        {
            "stage4a78_summary": args.stage78 / "stage4a78_stage4a714_no_training_import_design_summary.json",
            "stage4a78_manifest": args.stage78 / "stage4a78_stage4a714_planned_import_manifest.csv",
            "stage4a77_gate_summary": args.stage77 / "stage4a77_stage4a714_promotion_gate_summary.json",
            "stage4a714_runtime_npz": runtime_npz_path,
            "stage4a70_primary_npz": primary_npz_path,
        }
    )
    write_json(output_dir / "stage4a79_stage4a714_source_hash_report.json", source_hashes)

    integrity = {
        "adapter_dataset_path": str(adapter_path),
        "expanded_dataset_path": str(expanded_path),
        "adapter_sample_count": 25,
        "expanded_sample_count": expanded_n,
        "candidate_features_model_shape": list(expanded_arrays["candidate_features_model"].shape),
        "candidate_features_raw_shape": list(expanded_arrays["candidate_features_raw"].shape),
        "new_primary_indices_valid": True,
        "forbidden_keys_present": forbidden_present,
        "primary_differs_from_lambda48_count_new": int(np.sum(primary25 != lambda25)),
        "quality_keep_all_new": bool(np.all(quality25)),
        "missing_model_feature_count_new": int(np.sum(missing_model25)),
    }
    write_json(output_dir / "stage4a79_stage4a714_integrity_report.json", integrity)
    write_json(
        output_dir / "stage4a79_stage4a714_label_lineage_report.json",
        {
            "original_primary_policy": PRIMARY_POLICY_ORIGINAL,
            "compatible_import_policy": PRIMARY_POLICY_STAGE714_COMPAT,
            "new_primary_label_key": "action_index_primary_uncertainty_bonus",
            "lambda48_role": "shadow/baseline only",
            "labels_recomputed_from_lambda48": False,
            "existing_primary_dataset_overwritten": False,
        },
    )
    no_training = {
        "training": False,
        "optimizer_step": False,
        "checkpoint": False,
        "model_save": False,
        "bc_training": False,
    }
    no_runtime = {
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
    }
    write_json(output_dir / "stage4a79_stage4a714_no_training_report.json", no_training)
    write_json(output_dir / "stage4a79_stage4a714_no_runtime_report.json", no_runtime)
    (output_dir / "stage4a79_stage4a714_no_training_report.md").write_text(
        "# No Training Report\n\nNo BC training, optimizer step, checkpoint, or model save was run.\n",
        encoding="utf-8",
    )
    (output_dir / "stage4a79_stage4a714_no_runtime_report.md").write_text(
        "# No Runtime Report\n\nNo Isaac startup, map_predict, rollout, or RL/GDPO/PPO was run.\n",
        encoding="utf-8",
    )

    output_files = {
        "summary_json": str(output_dir / "stage4a79_stage4a714_compatible_import_summary.json"),
        "summary_md": str(output_dir / "stage4a79_stage4a714_compatible_import_summary.md"),
        "adapter_dataset_25": str(adapter_path),
        "compatible_expanded_dataset_55": str(expanded_path),
        "manifest_csv": str(output_dir / "stage4a79_stage4a714_compatible_import_manifest.csv"),
        "integrity_report": str(output_dir / "stage4a79_stage4a714_integrity_report.json"),
        "label_lineage_report": str(output_dir / "stage4a79_stage4a714_label_lineage_report.json"),
    }
    summary = {
        "stage": "Stage 4A-7.9 Stage 4A-7.14 compatible no-training import",
        "completed": True,
        "blocked": False,
        "blockers": [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(WORKSPACE),
        "output_dir": str(output_dir),
        "original_primary_sample_count": original_n,
        "compatible_import_sample_count": len(planned_rows),
        "expanded_compatible_sample_count": expanded_n,
        "adapter_dataset_created": True,
        "compatible_expanded_dataset_created": True,
        "existing_primary_dataset_overwritten": False,
        "compatible_dataset_shapes": {
            "candidate_features_model": list(expanded_arrays["candidate_features_model"].shape),
            "candidate_features_raw": list(expanded_arrays["candidate_features_raw"].shape),
            "expert_action_index_primary": list(expanded_arrays["expert_action_index_primary"].shape),
        },
        "label_lineage": {
            "new_primary_label_key": "action_index_primary_uncertainty_bonus",
            "new_primary_label_policy": PRIMARY_POLICY_STAGE714_COMPAT,
            "lambda48_role": "shadow/baseline only",
            "labels_recomputed_from_lambda48": False,
            "primary_differs_from_lambda48_count_new": int(np.sum(primary25 != lambda25)),
        },
        "source_hashes": source_hashes,
        "output_files": output_files,
        "negative_scope": {
            "existing_dataset_modified": False,
            "training": False,
            "optimizer_step": False,
            "checkpoint": False,
            "model_save": False,
            "isaac_startup": False,
            "map_predict": False,
            "rollout": False,
            "rl_gdpo_ppo": False,
        },
        "recommended_next_step": (
            "Run a no-training QA packet for this compatible artifact, then decide separately whether to use it for any BC dry-run. "
            "Do not train/checkpoint/RL without a separate explicit gate."
        ),
    }
    write_json(output_dir / "stage4a79_stage4a714_compatible_import_summary.json", summary)
    write_markdown_summary(output_dir / "stage4a79_stage4a714_compatible_import_summary.md", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage78", type=Path, default=STAGE78)
    parser.add_argument("--stage77", type=Path, default=STAGE77)
    parser.add_argument("--stage714", type=Path, default=STAGE714)
    parser.add_argument("--stage70", type=Path, default=STAGE70)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = generate_import(args)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary.get("completed") and not summary.get("blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
