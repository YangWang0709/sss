#!/usr/bin/env python3
"""Generate Stage 4A-7.10 no-training QA/readiness packet for Stage 4A-7.9."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/home/ubuntu22/sc_explorer_ws")
STAGE79_DIR = PROJECT_ROOT / "outputs/stage4a79_stage4a714_compatible_no_training_import"
STAGE714_REVIEW_DIR = PROJECT_ROOT / "outputs/stage4a714_2d_review_packet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/stage4a710_stage4a79_no_training_qa_readiness"

ADAPTER_NPZ = STAGE79_DIR / "stage4a79_stage4a714_adapter_dataset_25.npz"
EXPANDED_NPZ = STAGE79_DIR / "stage4a79_stage4a714_compatible_expanded_dataset_55.npz"
STAGE79_MANIFEST = STAGE79_DIR / "stage4a79_stage4a714_compatible_import_manifest.csv"
STAGE79_SUMMARY = STAGE79_DIR / "stage4a79_stage4a714_compatible_import_summary.json"
STAGE714_REVIEW_RECORDS = STAGE714_REVIEW_DIR / "stage4a714_2d_rollout_review_records.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def json_list(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    return []


def arrays_equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def rank_desc(scores: np.ndarray, valid_mask: np.ndarray, selected: int) -> int | None:
    if not (0 <= selected < scores.shape[0]) or not bool(valid_mask[selected]):
        return None
    valid_indices = np.flatnonzero(valid_mask)
    valid_scores = scores[valid_indices]
    selected_score = scores[selected]
    if not np.isfinite(selected_score):
        return None
    higher = np.sum(valid_scores > selected_score)
    return int(higher + 1)


def rel_from_output(path: str) -> str:
    return "../stage4a714_2d_review_packet/" + path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_html(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    blocker_items = "".join(f"<li>{esc(item)}</li>" for item in summary["blockers"]) or "<li>none</li>"
    cards = []
    for row in rows:
        status_class = "ok" if row["qa_status"] == "ok" else "warn" if row["qa_status"] == "warn" else "block"
        map_src = esc(row["map_image_for_html"])
        rgb_src = esc(row["rgb_for_html"])
        cards.append(
            f"""
      <article class="card {status_class}" data-review-card="1">
        <header>
          <div>
            <h2>{esc(row['sample_id'])}</h2>
            <p>expanded #{esc(row['expanded_sample_index'])} / adapter #{esc(row['adapter_sample_index'])}</p>
          </div>
          <span class="pill">{esc(row['qa_status'])}</span>
        </header>
        <div class="media">
          <img src="{map_src}" alt="2D map for {esc(row['sample_id'])}">
          <img src="{rgb_src}" alt="RGB camera for {esc(row['sample_id'])}">
        </div>
        <dl>
          <dt>human</dt><dd>{esc(row['human_review_status'])} / {esc(row['human_review_reason'])} / promote={esc(row['promote_candidate_yes_no'])}</dd>
          <dt>label</dt><dd>primary={esc(row['expert_action_index_primary'])}, lambda48_shadow={esc(row['expert_action_index_lambda48_shadow'])}, equal={esc(row['primary_equals_lambda48_shadow'])}</dd>
          <dt>score rank</dt><dd>primary rank among valid candidates = {esc(row['primary_score_rank'])}</dd>
          <dt>distance</dt><dd>{esc(row['source_to_action_distance_m'])} m, flag={esc(row['action_distance_flag'])}</dd>
          <dt>coverage</dt><dd>before={esc(row['observed_ratio_before'])}, after={esc(row['observed_ratio_after_current_capture'])}, new_xy={esc(row['newly_observed_xy_cells'])}</dd>
          <dt>validity</dt><dd>valid_candidates={esc(row['valid_candidate_count'])}, missing_model_features={esc(row['missing_model_feature_count'])}, quality_keep={esc(row['quality_keep_mask'])}</dd>
          <dt>coords</dt><dd>source=({esc(row['source_x'])}, {esc(row['source_y'])}, {esc(row['source_z'])}) -> action=({esc(row['action_x'])}, {esc(row['action_y'])}, {esc(row['action_z'])})</dd>
          <dt>note</dt><dd>{esc(row['qa_note'])}</dd>
        </dl>
      </article>
"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage 4A-7.10 QA Readiness</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5d6873;
      --line: #d8dee4;
      --ok: #0f766e;
      --warn: #b45309;
      --block: #b91c1c;
      --bg: #f6f8fa;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--ink); }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .summary {{ display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; font-size: 22px; margin-bottom: 4px; }}
    .boundary {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 12px; margin: 12px 0 18px; color: #7c2d12; }}
    .blockers {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }}
    .card {{ background: var(--panel); border: 2px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .card.ok {{ border-color: rgba(15, 118, 110, .45); }}
    .card.warn {{ border-color: rgba(180, 83, 9, .55); }}
    .card.block {{ border-color: rgba(185, 28, 28, .7); }}
    .card header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; padding: 12px 14px; border-bottom: 1px solid var(--line); }}
    .pill {{ white-space: nowrap; border: 1px solid currentColor; border-radius: 999px; padding: 4px 9px; font-size: 12px; text-transform: uppercase; }}
    .ok .pill {{ color: var(--ok); }}
    .warn .pill {{ color: var(--warn); }}
    .block .pill {{ color: var(--block); }}
    .media {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 8px; padding: 10px; background: #eef2f5; }}
    .media img {{ width: 100%; aspect-ratio: 16 / 10; object-fit: contain; background: #101820; border-radius: 6px; }}
    dl {{ display: grid; grid-template-columns: 140px 1fr; gap: 6px 12px; padding: 12px 14px 16px; margin: 0; font-size: 13px; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    @media (max-width: 760px) {{
      main {{ padding: 14px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      .media {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Stage 4A-7.10 Compatible Artifact QA Readiness</h1>
  <p>Generated at {esc(summary['generated_at_utc'])}. Source artifact: Stage 4A-7.9 compatible no-training import.</p>
  <section class="summary">
    <div class="metric"><b>{esc(summary['readiness_decision'])}</b><span>readiness</span></div>
    <div class="metric"><b>{esc(summary['import_rows'])}</b><span>new import rows</span></div>
    <div class="metric"><b>{esc(summary['expanded_rows'])}</b><span>expanded rows</span></div>
    <div class="metric"><b>{esc(summary['warning_count'])}</b><span>warnings</span></div>
    <div class="metric"><b>{esc(summary['blocker_count'])}</b><span>blockers</span></div>
  </section>
  <section class="boundary">
    No Isaac, no map_predict, no rollout, no training, no checkpoint/model save, no optimizer step, and no RL/GDPO/PPO were run. This page is an offline QA/readiness packet only.
  </section>
  <section class="blockers">
    <h2>Blockers</h2>
    <ul>{blocker_items}</ul>
  </section>
  <section class="grid">
    {''.join(cards)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    warnings: list[str] = []

    required = [ADAPTER_NPZ, EXPANDED_NPZ, STAGE79_MANIFEST, STAGE79_SUMMARY, STAGE714_REVIEW_RECORDS]
    for path in required:
        if not path.exists():
            blockers.append(f"missing_required_file:{path}")
    if blockers:
        summary = {
            "stage": "Stage 4A-7.10 Stage 4A-7.9 compatible artifact no-training QA/readiness",
            "completed": False,
            "readiness_decision": "blocked",
            "blockers": blockers,
            "warnings": warnings,
            "generated_at_utc": utc_now(),
            "negative_scope": negative_scope(),
        }
        write_json(OUTPUT_DIR / "stage4a710_stage4a79_no_training_qa_readiness_summary.json", summary)
        raise SystemExit(json.dumps(summary, indent=2))

    stage79_summary = read_json(STAGE79_SUMMARY)
    manifest_rows = read_csv(STAGE79_MANIFEST)
    review_rows = read_csv(STAGE714_REVIEW_RECORDS)
    review_by_sample = {
        f"start_{int(row['start_variant_id']):03d}_step_{int(row['step_id']):03d}": row
        for row in review_rows
    }

    adapter = np.load(ADAPTER_NPZ)
    expanded = np.load(EXPANDED_NPZ)
    primary_path = Path(stage79_summary["source_hashes"]["stage4a70_primary_npz"]["path"])
    primary = np.load(primary_path)

    original_n = int(primary["sample_id"].shape[0])
    import_n = int(adapter["sample_id"].shape[0])
    expanded_n = int(expanded["sample_id"].shape[0])

    if import_n != 25:
        blockers.append(f"expected_25_import_rows_got_{import_n}")
    if expanded_n != original_n + import_n:
        blockers.append(f"expanded_count_mismatch:{expanded_n}!={original_n}+{import_n}")
    if adapter["candidate_features_model"].shape != (25, 64, 16):
        blockers.append(f"adapter_model_shape_unexpected:{adapter['candidate_features_model'].shape}")
    if expanded["candidate_features_model"].shape != (55, 64, 16):
        blockers.append(f"expanded_model_shape_unexpected:{expanded['candidate_features_model'].shape}")
    if expanded["candidate_features_raw"].shape != (55, 64, 91):
        blockers.append(f"expanded_raw_shape_unexpected:{expanded['candidate_features_raw'].shape}")

    for key in primary.files:
        if key in expanded.files:
            if not arrays_equal_with_nan(primary[key], expanded[key][:original_n]):
                blockers.append(f"expanded_original_prefix_mismatch:{key}")

    for key in adapter.files:
        if key in expanded.files:
            if not arrays_equal_with_nan(adapter[key], expanded[key][original_n:]):
                blockers.append(f"expanded_import_suffix_mismatch:{key}")

    numeric_keys = [
        "candidate_features_model",
        "score_primary",
        "score_measured",
        "score_lambda48",
        "score_confidence_gated",
        "selected_world_xyz_primary",
        "pose_world_xyz",
        "pose_yaw",
        "selected_yaw_primary",
    ]
    numeric_health: dict[str, Any] = {}
    for key in numeric_keys:
        arr = adapter[key]
        nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
        inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
        numeric_health[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "stats": finite_stats(arr),
        }
        if nan_count or inf_count:
            blockers.append(f"adapter_numeric_nonfinite:{key}:nan={nan_count}:inf={inf_count}")

    labels = adapter["expert_action_index_primary"].astype(int)
    valid_mask = adapter["candidate_valid_mask"].astype(bool)
    if valid_mask.shape != (25, 64):
        blockers.append(f"adapter_valid_mask_shape_unexpected:{valid_mask.shape}")
    for i, label in enumerate(labels):
        if not (0 <= label < valid_mask.shape[1]):
            blockers.append(f"primary_label_out_of_range:{i}:{label}")
        elif not bool(valid_mask[i, label]):
            blockers.append(f"primary_label_invalid_candidate:{i}:{label}")

    if not bool(np.all(adapter["quality_keep_mask"])):
        blockers.append("adapter_quality_keep_mask_not_all_true")
    if not bool(np.all(adapter["source_stage_id"] == 714)):
        blockers.append("adapter_source_stage_id_not_all_714")

    manifest_by_sample = {row["sample_id"]: row for row in manifest_rows}
    if len(manifest_rows) != 25:
        blockers.append(f"manifest_row_count_expected_25_got_{len(manifest_rows)}")

    row_qa: list[dict[str, Any]] = []
    primary_ranks: list[int] = []
    distance_values: list[float] = []
    newly_observed_values: list[float] = []
    qa_counter: Counter[str] = Counter()

    for i in range(import_n):
        sid = str(adapter["sample_id"][i])
        manifest = manifest_by_sample.get(sid, {})
        review = review_by_sample.get(sid, {})
        label = int(adapter["expert_action_index_primary"][i])
        lambda_label = int(adapter["expert_action_index_lambda48_shadow"][i])
        rank = rank_desc(adapter["score_primary"][i], valid_mask[i], label)
        if rank is not None:
            primary_ranks.append(rank)
        distance = float_or_nan(manifest.get("source_to_action_distance_m"))
        if math.isfinite(distance):
            distance_values.append(distance)
        newly_observed = float_or_nan(review.get("newly_observed_xy_cells"))
        if math.isfinite(newly_observed):
            newly_observed_values.append(newly_observed)
        distance_flag = manifest.get("action_distance_flag", "")
        human_status = manifest.get("human_review_status", "")
        promote = manifest.get("promote_candidate_yes_no", "")
        row_blockers: list[str] = []
        row_warnings: list[str] = []
        if human_status != "approve":
            row_blockers.append("human_status_not_approve")
        if promote != "yes":
            row_blockers.append("promote_not_yes")
        if distance_flag == "very_close":
            row_blockers.append("very_close_row_should_be_held_out")
        if distance_flag == "close":
            row_warnings.append("close_distance_review_warning")
        if rank is not None and rank > 1:
            row_warnings.append(f"primary_score_rank_{rank}")
        if sid not in review_by_sample:
            row_blockers.append("missing_stage714_review_record")
        if not bool(adapter["quality_keep_mask"][i]):
            row_blockers.append("quality_keep_false")
        if row_blockers:
            qa_status = "block"
        elif row_warnings:
            qa_status = "warn"
        else:
            qa_status = "ok"
        qa_counter[qa_status] += 1
        row_qa.append(
            {
                "expanded_sample_index": int(manifest.get("expanded_sample_index", original_n + i)),
                "adapter_sample_index": i,
                "sample_id": sid,
                "start_id": manifest.get("start_id", ""),
                "step_id": manifest.get("step_id", ""),
                "runtime_row_index": manifest.get("runtime_row_index", ""),
                "human_review_status": human_status,
                "human_review_reason": manifest.get("human_review_reason", ""),
                "promote_candidate_yes_no": promote,
                "expert_action_index_primary": label,
                "expert_action_index_lambda48_shadow": lambda_label,
                "primary_equals_lambda48_shadow": label == lambda_label,
                "selected_primary_score": float(adapter["score_primary"][i, label]),
                "selected_lambda48_score": float(adapter["score_lambda48"][i, label]),
                "selected_measured_score": float(adapter["score_measured"][i, label]),
                "selected_confidence_gated_score": float(adapter["score_confidence_gated"][i, label]),
                "primary_score_rank": rank if rank is not None else "",
                "valid_candidate_count": int(valid_mask[i].sum()),
                "missing_model_feature_count": int(adapter["missing_feature_mask"][i].sum()),
                "quality_keep_mask": bool(adapter["quality_keep_mask"][i]),
                "source_to_action_distance_m": distance if math.isfinite(distance) else "",
                "action_distance_flag": distance_flag,
                "newly_observed_xy_cells": int(newly_observed) if math.isfinite(newly_observed) else "",
                "observed_ratio_before": review.get("observed_ratio_before", ""),
                "observed_ratio_after_current_capture": review.get("observed_ratio_after_current_capture", ""),
                "source_x": review.get("source_x", ""),
                "source_y": review.get("source_y", ""),
                "source_z": review.get("source_z", ""),
                "action_x": review.get("action_x", ""),
                "action_y": review.get("action_y", ""),
                "action_z": review.get("action_z", ""),
                "map_image": review.get("map_image", ""),
                "rgb": review.get("rgb", ""),
                "map_image_for_html": rel_from_output(review.get("map_image", "")),
                "rgb_for_html": rel_from_output(review.get("rgb", "")),
                "qa_status": qa_status,
                "qa_note": "; ".join(row_blockers + row_warnings) or "ok",
            }
        )

    for row in row_qa:
        if row["qa_status"] == "block":
            blockers.append(f"row_block:{row['sample_id']}:{row['qa_note']}")
        elif row["qa_status"] == "warn":
            warnings.append(f"row_warn:{row['sample_id']}:{row['qa_note']}")

    label_lineage = read_json(STAGE79_DIR / "stage4a79_stage4a714_label_lineage_report.json")
    if label_lineage.get("lambda48_role") != "shadow/baseline only":
        blockers.append("lambda48_not_shadow_only")
    if label_lineage.get("labels_recomputed_from_lambda48") is not False:
        blockers.append("labels_recomputed_from_lambda48_not_false")

    no_training = negative_scope()
    readiness_decision = "blocked" if blockers else "ready_for_tiny_bc_dry_run_consideration"

    distribution_report = {
        "action_distance_flag_counts": dict(Counter(str(row["action_distance_flag"]) for row in row_qa)),
        "human_review_status_counts": dict(Counter(str(row["human_review_status"]) for row in row_qa)),
        "human_review_reason_counts": dict(Counter(str(row["human_review_reason"]) for row in row_qa)),
        "qa_status_counts": dict(qa_counter),
        "primary_equals_lambda48_shadow_counts": dict(Counter(str(row["primary_equals_lambda48_shadow"]) for row in row_qa)),
        "primary_score_rank_counts": dict(Counter(str(row["primary_score_rank"]) for row in row_qa)),
        "source_to_action_distance_m": finite_stats(np.asarray(distance_values, dtype=np.float32)),
        "newly_observed_xy_cells": finite_stats(np.asarray(newly_observed_values, dtype=np.float32)),
        "valid_candidate_count": finite_stats(np.asarray([row["valid_candidate_count"] for row in row_qa], dtype=np.float32)),
        "missing_model_feature_count": finite_stats(np.asarray([row["missing_model_feature_count"] for row in row_qa], dtype=np.float32)),
    }

    summary = {
        "stage": "Stage 4A-7.10 Stage 4A-7.9 compatible artifact no-training QA/readiness",
        "completed": True,
        "blocked": bool(blockers),
        "readiness_decision": readiness_decision,
        "generated_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "source_stage": "Stage 4A-7.9 compatible no-training import",
        "input_files": {
            "adapter_npz": str(ADAPTER_NPZ),
            "expanded_npz": str(EXPANDED_NPZ),
            "manifest_csv": str(STAGE79_MANIFEST),
            "stage714_review_records": str(STAGE714_REVIEW_RECORDS),
        },
        "original_rows": original_n,
        "import_rows": import_n,
        "expanded_rows": expanded_n,
        "adapter_shapes": {
            "candidate_features_model": list(adapter["candidate_features_model"].shape),
            "candidate_features_raw": list(adapter["candidate_features_raw"].shape),
            "candidate_valid_mask": list(adapter["candidate_valid_mask"].shape),
            "expert_action_index_primary": list(adapter["expert_action_index_primary"].shape),
        },
        "expanded_shapes": {
            "candidate_features_model": list(expanded["candidate_features_model"].shape),
            "candidate_features_raw": list(expanded["candidate_features_raw"].shape),
            "candidate_valid_mask": list(expanded["candidate_valid_mask"].shape),
            "expert_action_index_primary": list(expanded["expert_action_index_primary"].shape),
        },
        "old_dataset_prefix_unchanged": not any(item.startswith("expanded_original_prefix_mismatch") for item in blockers),
        "adapter_matches_expanded_suffix": not any(item.startswith("expanded_import_suffix_mismatch") for item in blockers),
        "numeric_health": numeric_health,
        "distribution_report": distribution_report,
        "label_lineage": label_lineage,
        "warning_count": len(warnings),
        "blocker_count": len(blockers),
        "warnings": warnings,
        "blockers": blockers,
        "negative_scope": no_training,
        "output_files": {
            "main_html": str(OUTPUT_DIR / "stage4a710_stage4a79_readiness_index.html"),
            "row_qa_csv": str(OUTPUT_DIR / "stage4a710_stage4a79_row_qa.csv"),
            "row_qa_json": str(OUTPUT_DIR / "stage4a710_stage4a79_row_qa.json"),
            "summary_json": str(OUTPUT_DIR / "stage4a710_stage4a79_no_training_qa_readiness_summary.json"),
            "summary_md": str(OUTPUT_DIR / "stage4a710_stage4a79_no_training_qa_readiness_summary.md"),
            "distribution_report": str(OUTPUT_DIR / "stage4a710_stage4a79_distribution_report.json"),
        },
        "recommended_next_step": "If accepted, request a separate bounded tiny BC dry-run design/execution gate. Do not train/checkpoint/RL from this QA packet alone.",
    }

    row_fields = [
        "expanded_sample_index",
        "adapter_sample_index",
        "sample_id",
        "start_id",
        "step_id",
        "runtime_row_index",
        "human_review_status",
        "human_review_reason",
        "promote_candidate_yes_no",
        "expert_action_index_primary",
        "expert_action_index_lambda48_shadow",
        "primary_equals_lambda48_shadow",
        "selected_primary_score",
        "selected_lambda48_score",
        "selected_measured_score",
        "selected_confidence_gated_score",
        "primary_score_rank",
        "valid_candidate_count",
        "missing_model_feature_count",
        "quality_keep_mask",
        "source_to_action_distance_m",
        "action_distance_flag",
        "newly_observed_xy_cells",
        "observed_ratio_before",
        "observed_ratio_after_current_capture",
        "source_x",
        "source_y",
        "source_z",
        "action_x",
        "action_y",
        "action_z",
        "map_image",
        "rgb",
        "qa_status",
        "qa_note",
    ]
    write_csv(OUTPUT_DIR / "stage4a710_stage4a79_row_qa.csv", row_qa, row_fields)
    write_json(OUTPUT_DIR / "stage4a710_stage4a79_row_qa.json", row_qa)
    write_json(OUTPUT_DIR / "stage4a710_stage4a79_distribution_report.json", distribution_report)
    write_json(OUTPUT_DIR / "stage4a710_stage4a79_no_training_qa_readiness_summary.json", summary)
    write_json(OUTPUT_DIR / "stage4a710_stage4a79_readiness_decision.json", {
        "readiness_decision": readiness_decision,
        "blockers": blockers,
        "warnings": warnings,
        "ready_for_tiny_bc_dry_run_consideration": readiness_decision == "ready_for_tiny_bc_dry_run_consideration",
        "training_allowed_by_this_packet": False,
        "checkpoint_allowed_by_this_packet": False,
        "rl_gdpo_ppo_allowed_by_this_packet": False,
    })
    write_json(OUTPUT_DIR / "stage4a710_stage4a79_no_training_report.json", no_training)
    (OUTPUT_DIR / "stage4a710_stage4a79_no_training_report.md").write_text(
        "# Stage 4A-7.10 No-Training Report\n\n"
        "- training: false\n- optimizer_step: false\n- model_save: false\n- checkpoint: false\n- rl_gdpo_ppo: false\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage4a710_stage4a79_no_runtime_report.md").write_text(
        "# Stage 4A-7.10 No-Runtime Report\n\n"
        "- isaac_startup: false\n- map_predict: false\n- rollout: false\n- runtime_execution: false\n",
        encoding="utf-8",
    )
    write_json(OUTPUT_DIR / "stage4a710_stage4a79_no_runtime_report.json", {
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "runtime_execution": False,
    })
    (OUTPUT_DIR / "stage4a710_stage4a79_readiness_index.html").write_text(build_html(summary, row_qa), encoding="utf-8")

    summary_md = f"""# Stage 4A-7.10 Stage 4A-7.9 No-Training QA Readiness

- Readiness decision: `{readiness_decision}`
- Original Stage 4A-7.0 rows: `{original_n}`
- Stage 4A-7.14 imported rows: `{import_n}`
- Compatible expanded rows: `{expanded_n}`
- Adapter candidate_features_model shape: `{list(adapter['candidate_features_model'].shape)}`
- Expanded candidate_features_model shape: `{list(expanded['candidate_features_model'].shape)}`
- Old dataset prefix unchanged: `{summary['old_dataset_prefix_unchanged']}`
- Adapter matches expanded suffix: `{summary['adapter_matches_expanded_suffix']}`
- Warnings: `{len(warnings)}`
- Blockers: `{len(blockers)}`

Main HTML:
`{OUTPUT_DIR / 'stage4a710_stage4a79_readiness_index.html'}`

Row QA:
`{OUTPUT_DIR / 'stage4a710_stage4a79_row_qa.csv'}`

Distribution report:
`{OUTPUT_DIR / 'stage4a710_stage4a79_distribution_report.json'}`

## Interpretation

The Stage 4A-7.9 compatible artifact passed offline schema, label, numeric-health, old-prefix preservation, and adapter-to-expanded-suffix checks. This packet only supports a future decision about a separate bounded tiny BC dry-run. It does not authorize training, checkpoints, runtime, or RL.

## Negative Scope

- Isaac startup: `false`
- map_predict: `false`
- rollout: `false`
- training: `false`
- optimizer step: `false`
- model save/checkpoint: `false`
- RL/GDPO/PPO: `false`
- label promotion from lambda48: `false`
"""
    (OUTPUT_DIR / "stage4a710_stage4a79_no_training_qa_readiness_summary.md").write_text(summary_md, encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


def negative_scope() -> dict[str, bool]:
    return {
        "isaac_startup": False,
        "map_predict": False,
        "runtime_execution": False,
        "rollout": False,
        "long_rollout": False,
        "training": False,
        "optimizer_step": False,
        "model_save": False,
        "checkpoint": False,
        "label_promotion": False,
        "rl_gdpo_ppo": False,
    }


if __name__ == "__main__":
    main()
