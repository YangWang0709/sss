#!/usr/bin/env python3
"""Generate an offline 2D rollout review packet on top of the USD map.

The packet is for human review only.  It reads existing Stage 4A-7.14 rollout
artifacts, projects their 3D world coordinates directly to USD x/y, and
visualizes historical observed area, newly observed area, source poses, action
targets, and camera RGB frames.  It does not launch Isaac or run training.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from generate_usd_topdown_map import (
    DEFAULT_USD,
    extract_footprints,
    load_rollout_alignment_records,
    parse_usda_xforms,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_DIR = WORKSPACE / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"
DEFAULT_MANIFEST = DEFAULT_ROLLOUT_DIR / "short_rollout_manifest.jsonl"
DEFAULT_SCENE_METADATA = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix/scene_metadata.json"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/stage4a714_2d_review_packet"
VERY_CLOSE_ACTION_DISTANCE_M = 0.25
CLOSE_ACTION_DISTANCE_M = 0.50


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_to_output(path: str | Path, output_dir: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (WORKSPACE / p).resolve()
    return Path(os.path.relpath(p, output_dir)).as_posix()


def observed_xy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"observed_state must be 3D, got {arr.shape}: {path}")
    return np.any(arr != -1, axis=2)


def density_xy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"observed_state must be 3D, got {arr.shape}: {path}")
    return np.count_nonzero(arr != -1, axis=2).astype(np.float32) / float(arr.shape[2])


def draw_base(ax: Any, footprints: list[dict[str, Any]], *, labels: bool = False) -> None:
    from matplotlib.patches import Polygon

    styles = {
        "floor": {"face": "#f1efe3", "edge": "#c8c3aa", "alpha": 0.92, "lw": 0.35, "z": 1},
        "wall": {"face": "#111827", "edge": "#020617", "alpha": 0.98, "lw": 0.25, "z": 8},
        "furniture": {"face": "#d97706", "edge": "#7c2d12", "alpha": 0.42, "lw": 0.25, "z": 5},
        "door": {"face": "#991b1b", "edge": "#450a0a", "alpha": 0.55, "lw": 0.25, "z": 7},
        "structural_obstacle": {"face": "#6b7280", "edge": "#27272a", "alpha": 0.62, "lw": 0.25, "z": 4},
    }
    for category in ("floor", "structural_obstacle", "furniture", "door", "wall"):
        for fp in footprints:
            if fp["category"] != category:
                continue
            style = styles[category]
            ax.add_patch(
                Polygon(
                    fp["polygon_xy"],
                    closed=True,
                    facecolor=style["face"],
                    edgecolor=style["edge"],
                    alpha=style["alpha"],
                    linewidth=style["lw"],
                    zorder=style["z"],
                )
            )
            if labels and category in {"wall", "furniture"}:
                sx, sy, _ = fp["size_xyz"]
                if max(sx, sy) >= 1.5:
                    x, y, _ = fp["center_xyz"]
                    ax.text(x, y, fp["name"].replace("_", " ")[:18], ha="center", va="center", fontsize=4.0, zorder=9)


def draw_observed_layers(
    ax: Any,
    observed: np.ndarray | None,
    newly: np.ndarray | None,
    bounds: dict[str, list[float]],
) -> None:
    import matplotlib.colors as mcolors

    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    if observed is not None:
        obs = np.ma.masked_where(~observed, observed.astype(np.float32))
        cmap = mcolors.ListedColormap(["#60a5fa"])
        ax.imshow(obs.T, origin="lower", extent=extent, cmap=cmap, alpha=0.28, interpolation="nearest", zorder=2)
    if newly is not None:
        new = np.ma.masked_where(~newly, newly.astype(np.float32))
        cmap = mcolors.ListedColormap(["#22d3ee"])
        ax.imshow(new.T, origin="lower", extent=extent, cmap=cmap, alpha=0.60, interpolation="nearest", zorder=3)


def draw_camera_wedge(ax: Any, pose_xyz: list[float], yaw: float, *, fov_deg: float = 90.0, length_m: float = 4.0) -> None:
    from matplotlib.patches import Polygon

    x, y = pose_xyz[0], pose_xyz[1]
    half = math.radians(fov_deg) * 0.5
    pts = [(x, y)]
    for angle in (yaw - half, yaw + half):
        pts.append((x + length_m * math.cos(angle), y + length_m * math.sin(angle)))
    ax.add_patch(Polygon(pts, closed=True, facecolor="#fde047", edgecolor="#a16207", linewidth=0.5, alpha=0.18, zorder=6))


def draw_records(ax: Any, records: list[dict[str, Any]], current: dict[str, Any] | None = None) -> None:
    if not records:
        return
    xs = [row["source_pose_xyz"][0] for row in records]
    ys = [row["source_pose_xyz"][1] for row in records]
    ax.plot(xs, ys, color="#2563eb", linewidth=1.6, alpha=0.82, zorder=10)
    ax.scatter(xs, ys, s=26, color="#2563eb", edgecolor="white", linewidth=0.5, zorder=11)
    for row in records:
        sx, sy, _ = row["source_pose_xyz"]
        tx, ty, _ = row["action_world_xyz"]
        color = "#0f766e" if current and row["step_id"] == current["step_id"] else "#64748b"
        width = 1.4 if current and row["step_id"] == current["step_id"] else 0.75
        ax.annotate("", xy=(tx, ty), xytext=(sx, sy), arrowprops={"arrowstyle": "->", "color": color, "lw": width, "alpha": 0.78}, zorder=12)
        ax.scatter([tx], [ty], marker="x", s=36, color="#ec4899", linewidth=1.4, zorder=13)
        ax.text(tx + 0.05, ty + 0.05, f"{row['step_id']}", fontsize=5.5, color="#831843", zorder=14)
    if current:
        x, y, _ = current["source_pose_xyz"]
        yaw = current["source_yaw_rad"]
        ax.arrow(x, y, 0.55 * math.cos(yaw), 0.55 * math.sin(yaw), width=0.025, head_width=0.16, color="#1d4ed8", zorder=15)
        ax.scatter([x], [y], s=58, color="#1d4ed8", edgecolor="white", linewidth=0.8, zorder=15)


def setup_axes(ax: Any, bounds: dict[str, list[float]], title: str) -> None:
    from matplotlib.ticker import MultipleLocator

    ax.set_xlim(bounds["x"][0] - 0.7, bounds["x"][1] + 0.7)
    ax.set_ylim(bounds["y"][0] - 0.7, bounds["y"][1] + 0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.yaxis.set_major_locator(MultipleLocator(2.0))
    ax.grid(True, color="#d1d5db", linewidth=0.35, alpha=0.85)
    ax.axhline(0.0, color="#64748b", linewidth=0.7, alpha=0.55, zorder=2)
    ax.axvline(0.0, color="#64748b", linewidth=0.7, alpha=0.55, zorder=2)
    ax.scatter([0.0], [0.0], marker="+", s=70, c="#dc2626", linewidths=1.4, zorder=16)
    ax.text(0.15, 0.2, "world origin (0,0)", fontsize=6.2, color="#b91c1c", zorder=16)


def render_step_map(
    path: Path,
    footprints: list[dict[str, Any]],
    bounds: dict[str, list[float]],
    history: list[dict[str, Any]],
    current: dict[str, Any],
    observed: np.ndarray,
    newly: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10.8, 16.0), constrained_layout=True)
    draw_observed_layers(ax, observed, newly, bounds)
    draw_base(ax, footprints)
    draw_camera_wedge(ax, current["source_pose_xyz"], current["source_yaw_rad"])
    draw_records(ax, history, current=current)
    setup_axes(ax, bounds, f"start {current['start_variant_id']:03d} step {current['step_id']:03d}: history + current action")
    ax.legend(
        handles=[
            Patch(facecolor="#60a5fa", edgecolor="none", alpha=0.28, label="historically observed/swept area"),
            Patch(facecolor="#22d3ee", edgecolor="none", alpha=0.60, label="newly observed since previous saved step"),
            Patch(facecolor="#111827", edgecolor="#020617", label="USD wall"),
            Patch(facecolor="#d97706", edgecolor="#7c2d12", alpha=0.42, label="USD furniture/obstacle"),
            Line2D([0], [0], color="#2563eb", marker="o", label="source camera pose history"),
            Line2D([0], [0], color="#0f766e", label="current source -> action arrow"),
            Line2D([0], [0], color="#ec4899", marker="x", linestyle="None", label="action target"),
        ],
        loc="upper right",
        fontsize=6.6,
        frameon=True,
    )
    fig.savefig(path, dpi=190)
    plt.close(fig)


def render_start_overview(
    path: Path,
    footprints: list[dict[str, Any]],
    bounds: dict[str, list[float]],
    records: list[dict[str, Any]],
    final_observed: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 16.0), constrained_layout=True)
    draw_observed_layers(ax, final_observed, None, bounds)
    draw_base(ax, footprints)
    draw_records(ax, records, current=records[-1] if records else None)
    setup_axes(ax, bounds, f"start {records[0]['start_variant_id']:03d}: full rollout history and observed area")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def build_records(manifest: Path) -> list[dict[str, Any]]:
    records = load_rollout_alignment_records(manifest)
    for row in records:
        pose_path = Path(row["source_pose_file"])
        start = row["start_variant_id"]
        step = row["step_id"]
        sx, sy, sz = row["source_pose_xyz"]
        ax, ay, az = row["action_world_xyz"]
        distance_xy = math.hypot(float(ax) - float(sx), float(ay) - float(sy))
        distance_3d = math.sqrt((float(ax) - float(sx)) ** 2 + (float(ay) - float(sy)) ** 2 + (float(az) - float(sz)) ** 2)
        sample_dir = DEFAULT_ROLLOUT_DIR / "samples" / f"start_{start:03d}"
        row["observed_state_reference"] = str(sample_dir / f"step_{step:03d}_observed_state.npy")
        row["map_image"] = f"maps/start_{start:03d}_step_{step:03d}_2d_review.png"
        row["overview_image"] = f"maps/start_{start:03d}_overview_2d_review.png"
        row["relative_rgb"] = rel_to_output(row["rgb"], DEFAULT_OUTPUT_DIR)
        row["relative_pose"] = rel_to_output(pose_path, DEFAULT_OUTPUT_DIR)
        row["source_to_action_distance_m"] = float(distance_xy)
        row["source_to_action_distance_3d_m"] = float(distance_3d)
        if distance_xy < VERY_CLOSE_ACTION_DISTANCE_M:
            row["action_distance_flag"] = "very_close"
        elif distance_xy < CLOSE_ACTION_DISTANCE_M:
            row["action_distance_flag"] = "close"
        else:
            row["action_distance_flag"] = "normal"
    return records


def write_index_html(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    data = []
    for row in records:
        review_id = f"stage4a714_s{row['start_variant_id']:03d}_step{row['step_id']:03d}"
        sample_id = f"start_{row['start_variant_id']:03d}_step_{row['step_id']:03d}"
        data.append(
            {
                "review_id": review_id,
                "sample_id": sample_id,
                "start": row["start_variant_id"],
                "step": row["step_id"],
                "capture": row["capture_index"],
                "map": row["map_image"],
                "overview": row["overview_image"],
                "rgb": row["relative_rgb"],
                "source": [round(float(v), 4) for v in row["source_pose_xyz"]],
                "action": [round(float(v), 4) for v in row["action_world_xyz"]],
                "source_yaw_rad": round(float(row["source_yaw_rad"]), 6),
                "action_yaw_rad": round(float(row["action_yaw_rad"]), 6),
                "observed_before": round(float(row["observed_ratio_before"]), 6),
                "observed_after": round(float(row["observed_ratio_after_current_capture"]), 6),
                "newly_observed_xy_cells": int(row.get("newly_observed_xy_cells", 0)),
                "historical_observed_xy_cells": int(row.get("historical_observed_xy_cells", 0)),
                "source_to_action_distance_m": round(float(row["source_to_action_distance_m"]), 4),
                "source_to_action_distance_3d_m": round(float(row["source_to_action_distance_3d_m"]), 4),
                "action_distance_flag": row["action_distance_flag"],
            }
        )
    json_data = json.dumps(data, ensure_ascii=False)
    summary_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
        if not isinstance(v, (dict, list))
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Stage 4A-7.14 2D Rollout Review</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #f8fafc; color: #111827; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #d1d5db; background: white; position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 20px; }}
    .shell {{ display: grid; grid-template-columns: 280px minmax(0, 1fr); min-height: calc(100vh - 56px); }}
    aside {{ border-right: 1px solid #d1d5db; padding: 14px; background: #ffffff; overflow: auto; }}
    main {{ padding: 16px; }}
    .start-grid, .step-grid {{ display: grid; gap: 6px; }}
    .start-grid {{ grid-template-columns: repeat(5, 1fr); margin-bottom: 14px; }}
    .step-grid {{ grid-template-columns: repeat(3, 1fr); }}
    button {{ border: 1px solid #cbd5e1; background: #f8fafc; border-radius: 6px; padding: 7px 8px; cursor: pointer; font-size: 13px; }}
    button.active {{ background: #1d4ed8; color: white; border-color: #1d4ed8; }}
    button.primary {{ background: #0f766e; color: white; border-color: #0f766e; }}
    button.danger {{ background: #991b1b; color: white; border-color: #991b1b; }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
    .panel {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.55fr); gap: 14px; align-items: start; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #cbd5e1; background: white; }}
    .meta {{ background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
    .review-card {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; margin: 0 0 12px; }}
    .review-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .review-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .review-field label {{ display: block; font-size: 12px; color: #334155; font-weight: 650; margin-bottom: 4px; }}
    .review-field select, .review-field input, .review-field textarea {{ width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px 8px; font: inherit; font-size: 13px; background: white; }}
    .review-field textarea {{ min-height: 70px; resize: vertical; }}
    .review-actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }}
    .status-chip {{ display: inline-block; min-width: 78px; text-align: center; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 700; background: #e2e8f0; color: #334155; }}
    .status-approve {{ background: #dcfce7; color: #166534; }}
    .status-reject {{ background: #fee2e2; color: #991b1b; }}
    .status-unsure {{ background: #fef9c3; color: #854d0e; }}
    .status-needs_closer_inspection {{ background: #ffedd5; color: #9a3412; }}
    .distance-chip {{ display: inline-block; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 750; background: #e2e8f0; color: #334155; }}
    .distance-normal {{ background: #dcfce7; color: #166534; }}
    .distance-close {{ background: #fef9c3; color: #854d0e; }}
    .distance-very_close {{ background: #fee2e2; color: #991b1b; }}
    .export-box {{ width: 100%; min-height: 170px; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 7px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; width: 42%; }}
    .hint {{ font-size: 13px; color: #475569; margin: 6px 0 0; }}
    .warning {{ color: #991b1b; font-weight: 650; }}
    @media (max-width: 980px) {{ .shell, .panel {{ grid-template-columns: 1fr; }} aside {{ border-right: 0; border-bottom: 1px solid #d1d5db; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Stage 4A-7.14 2D Rollout Review</h1>
    <p class="hint">USD world x/y meters are preserved. Blue = historical camera poses, pink x = action targets, blue wash = historical observed/swept area, cyan = newly observed since previous saved step.</p>
    <p class="hint warning">Review-only packet. No label promotion, no training, no checkpoint, no Isaac/runtime, no RL/GDPO/PPO.</p>
    <p class="hint warning">Human approval does not automatically promote labels. Future Stage 4A-7.7 must import this review and make a separate promotion decision.</p>
  </header>
  <div class="shell">
    <aside>
      <div class="review-card">
        <h3>Review export</h3>
        <div class="review-field">
          <label for="reviewer">reviewer_name</label>
          <input id="reviewer" placeholder="optional" />
        </div>
        <div class="review-actions">
          <button id="markAllUnreviewed" class="danger" type="button">Mark all unreviewed</button>
          <button id="showSummary" type="button">Show review completion summary</button>
          <button id="exportJson" class="primary" type="button">Export review JSON</button>
          <button id="copyJson" type="button">Copy review JSON to clipboard</button>
          <button id="downloadJson" type="button">Download review JSON</button>
          <button id="downloadCsv" type="button">Download review CSV</button>
        </div>
        <p id="completionSummary" class="hint"></p>
        <textarea id="exportText" class="export-box" spellcheck="false" placeholder="Exported review JSON appears here."></textarea>
      </div>
      <strong>Start</strong>
      <div id="starts" class="start-grid"></div>
      <strong>Step</strong>
      <div id="steps" class="step-grid"></div>
      <table>{summary_rows}</table>
    </aside>
    <main>
      <div class="panel">
        <section>
          <img id="map" alt="2D rollout map" />
        </section>
        <section class="meta">
          <h2 id="title" style="font-size:17px;margin:0 0 8px;"></h2>
          <div class="review-card">
            <h3>Human decision <span id="statusChip" class="status-chip">unreviewed</span></h3>
            <p class="hint">Action distance <span id="distanceChip" class="distance-chip">--</span></p>
            <div class="review-grid">
              <div class="review-field">
                <label for="human_review_status">human_review_status</label>
                <select id="human_review_status">
                  <option value="unreviewed">unreviewed</option>
                  <option value="approve">approve</option>
                  <option value="reject">reject</option>
                  <option value="unsure">unsure</option>
                  <option value="needs_closer_inspection">needs_closer_inspection</option>
                </select>
              </div>
              <div class="review-field">
                <label for="human_review_reason">human_review_reason</label>
                <select id="human_review_reason">
                  <option value=""></option>
                  <option value="local_jitter_acceptable">local_jitter_acceptable</option>
                  <option value="local_jitter_unacceptable">local_jitter_unacceptable</option>
                  <option value="unsafe_outside_stuck_revisit">unsafe_outside_stuck_revisit</option>
                  <option value="poor_uncertainty_choice">poor_uncertainty_choice</option>
                  <option value="poor_path_choice">poor_path_choice</option>
                  <option value="visual_mismatch">visual_mismatch</option>
                  <option value="good_exploration_choice">good_exploration_choice</option>
                  <option value="other">other</option>
                </select>
              </div>
              <div class="review-field">
                <label for="promote_candidate_yes_no">promote_candidate_yes_no</label>
                <select id="promote_candidate_yes_no">
                  <option value=""></option>
                  <option value="yes">yes</option>
                  <option value="no">no</option>
                </select>
              </div>
              <div class="review-field">
                <label for="currentReviewId">review_id</label>
                <input id="currentReviewId" readonly />
              </div>
            </div>
            <div class="review-field" style="margin-top:10px;">
              <label for="human_comment">human_comment</label>
              <textarea id="human_comment" placeholder="optional note for future Stage 4A-7.7 import"></textarea>
            </div>
            <p id="validationHint" class="hint warning"></p>
          </div>
          <img id="rgb" alt="camera RGB" />
          <table id="details"></table>
          <p class="hint"><a id="overview" href="#">Open start overview</a></p>
        </section>
      </div>
    </main>
  </div>
  <script>
    const rows = {json_data};
    const REVIEW_KEY = 'stage4a714_2d_review_decisions_v1';
    const STATUS_VALUES = ['unreviewed', 'approve', 'reject', 'unsure', 'needs_closer_inspection'];
    const REASON_VALUES = ['', 'local_jitter_acceptable', 'local_jitter_unacceptable', 'unsafe_outside_stuck_revisit', 'poor_uncertainty_choice', 'poor_path_choice', 'visual_mismatch', 'good_exploration_choice', 'other'];
    const PROMOTE_VALUES = ['', 'yes', 'no'];
    function defaultDecision(row) {{
      return {{
        review_id: row.review_id,
        start_id: row.start,
        step_id: row.step,
        sample_id: row.sample_id,
        human_review_status: 'unreviewed',
        human_review_reason: '',
        human_comment: '',
        promote_candidate_yes_no: '',
      }};
    }}
    function loadDecisions() {{
      let saved = {{}};
      try {{
        saved = JSON.parse(localStorage.getItem(REVIEW_KEY) || '{{}}');
      }} catch (err) {{
        saved = {{}};
      }}
      const out = {{}};
      rows.forEach(row => {{
        out[row.review_id] = {{ ...defaultDecision(row), ...(saved[row.review_id] || {{}}) }};
        if (out[row.review_id].promote_candidate_yes_no === 'yes' && out[row.review_id].human_review_status !== 'approve') {{
          out[row.review_id].promote_candidate_yes_no = '';
        }}
      }});
      return out;
    }}
    let decisions = loadDecisions();
    let currentStart = rows[0]?.start ?? 0;
    let currentStep = rows[0]?.step ?? 0;
    const starts = [...new Set(rows.map(r => r.start))];
    function button(label, active, fn) {{
      const b = document.createElement('button');
      b.textContent = label;
      if (active) b.classList.add('active');
      b.onclick = fn;
      return b;
    }}
    function currentRow() {{
      return rows.find(r => r.start === currentStart && r.step === currentStep) || rows[0];
    }}
    function saveDecisions() {{
      localStorage.setItem(REVIEW_KEY, JSON.stringify(decisions));
    }}
    function setDecisionField(field, value) {{
      const row = currentRow();
      const d = decisions[row.review_id] || defaultDecision(row);
      d[field] = value;
      if (d.promote_candidate_yes_no === 'yes' && d.human_review_status !== 'approve') {{
        d.promote_candidate_yes_no = '';
      }}
      decisions[row.review_id] = d;
      saveDecisions();
      renderReviewControls(row);
      updateCompletionSummary(false);
    }}
    function renderNav() {{
      const startsEl = document.getElementById('starts');
      startsEl.innerHTML = '';
      starts.forEach(s => startsEl.appendChild(button('S' + s, s === currentStart, () => {{ currentStart = s; currentStep = rows.find(r => r.start === s).step; render(); }})));
      const stepsEl = document.getElementById('steps');
      stepsEl.innerHTML = '';
      rows.filter(r => r.start === currentStart).forEach(r => stepsEl.appendChild(button(String(r.step), r.step === currentStep, () => {{ currentStep = r.step; render(); }})));
    }}
    function renderDetails(row) {{
      const detailRows = [
        ['start / step', `S${{row.start}} / ${{row.step}}`],
        ['review id', row.review_id],
        ['sample id', row.sample_id],
        ['capture index', row.capture],
        ['source pose xyz', row.source.join(', ')],
        ['action target xyz', row.action.join(', ')],
        ['source -> action distance xy (m)', row.source_to_action_distance_m],
        ['source -> action distance 3d (m)', row.source_to_action_distance_3d_m],
        ['action distance flag', row.action_distance_flag],
        ['source yaw rad', row.source_yaw_rad],
        ['action yaw rad', row.action_yaw_rad],
        ['observed ratio before', row.observed_before],
        ['observed ratio after', row.observed_after],
        ['historical observed xy cells', row.historical_observed_xy_cells],
        ['newly observed xy cells', row.newly_observed_xy_cells],
      ];
      document.getElementById('details').innerHTML = detailRows.map(([k,v]) => `<tr><th>${{k}}</th><td>${{v}}</td></tr>`).join('');
    }}
    function renderReviewControls(row) {{
      const d = decisions[row.review_id] || defaultDecision(row);
      const statusEl = document.getElementById('human_review_status');
      const reasonEl = document.getElementById('human_review_reason');
      const promoteEl = document.getElementById('promote_candidate_yes_no');
      const commentEl = document.getElementById('human_comment');
      statusEl.value = STATUS_VALUES.includes(d.human_review_status) ? d.human_review_status : 'unreviewed';
      reasonEl.value = REASON_VALUES.includes(d.human_review_reason) ? d.human_review_reason : '';
      promoteEl.value = PROMOTE_VALUES.includes(d.promote_candidate_yes_no) ? d.promote_candidate_yes_no : '';
      commentEl.value = d.human_comment || '';
      document.getElementById('currentReviewId').value = row.review_id;
      [...promoteEl.options].forEach(opt => {{
        opt.disabled = opt.value === 'yes' && statusEl.value !== 'approve';
      }});
      const chip = document.getElementById('statusChip');
      chip.textContent = statusEl.value;
      chip.className = 'status-chip status-' + statusEl.value;
      const distanceChip = document.getElementById('distanceChip');
      distanceChip.textContent = `${{row.source_to_action_distance_m}} m / ${{row.action_distance_flag}}`;
      distanceChip.className = 'distance-chip distance-' + row.action_distance_flag;
      document.getElementById('validationHint').textContent =
        row.action_distance_flag === 'very_close'
          ? 'Very close action target (<0.25 m). Consider reject/poor_path_choice unless the camera change clearly adds useful view.'
          : (row.action_distance_flag === 'close'
            ? 'Close action target (<0.50 m). Check whether this is useful viewpoint refinement or local jitter.'
            : (statusEl.value === 'approve'
              ? 'Only approved samples may set promote_candidate_yes_no=yes. Promotion still requires a separate Stage 4A-7.7 import decision.'
              : 'promote_candidate_yes_no=yes is disabled unless human_review_status=approve.'));
    }}
    function exportPayload() {{
      return {{
        review_schema_version: 'stage4a714_2d_manual_review_v1',
        source_stage: 'Stage 4A-7.14',
        packet_stage: 'Stage 4A-7.14 2D rollout review packet',
        exported_at_local_browser_time: new Date().toISOString(),
        reviewer_name: document.getElementById('reviewer').value || '',
        rows: rows.map(row => decisions[row.review_id] || defaultDecision(row)),
        negative_scope: {{
          label_promotion: false,
          training: false,
          checkpoint: false,
          isaac_startup: false,
          map_predict: false,
          rollout: false,
          rl_gdpo_ppo: false,
        }},
      }};
    }}
    function csvEscape(value) {{
      const s = String(value ?? '');
      return /[",\\n\\r]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
    }}
    function exportCsvText() {{
      const fields = ['review_id', 'start_id', 'step_id', 'sample_id', 'human_review_status', 'human_review_reason', 'human_comment', 'promote_candidate_yes_no'];
      const lines = [fields.join(',')];
      exportPayload().rows.forEach(row => lines.push(fields.map(f => csvEscape(row[f])).join(',')));
      return lines.join('\\n') + '\\n';
    }}
    function downloadText(filename, mime, text) {{
      const blob = new Blob([text], {{ type: mime }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}
    function updateCompletionSummary(showAlert) {{
      const counts = Object.fromEntries(STATUS_VALUES.map(s => [s, 0]));
      rows.forEach(row => {{
        const status = (decisions[row.review_id] || defaultDecision(row)).human_review_status;
        counts[status] = (counts[status] || 0) + 1;
      }});
      const reviewed = rows.length - counts.unreviewed;
      const text = `Reviewed ${{reviewed}} / ${{rows.length}} | approve ${{counts.approve}} | reject ${{counts.reject}} | unsure ${{counts.unsure}} | needs closer inspection ${{counts.needs_closer_inspection}} | unreviewed ${{counts.unreviewed}}`;
      document.getElementById('completionSummary').textContent = text;
      if (showAlert) alert(text);
    }}
    function render() {{
      renderNav();
      const row = currentRow();
      document.getElementById('title').textContent = `Start ${{row.start}} Step ${{row.step}}`;
      document.getElementById('map').src = row.map;
      document.getElementById('rgb').src = row.rgb;
      document.getElementById('overview').href = row.overview;
      renderDetails(row);
      renderReviewControls(row);
    }}
    document.getElementById('human_review_status').onchange = e => setDecisionField('human_review_status', e.target.value);
    document.getElementById('human_review_reason').onchange = e => setDecisionField('human_review_reason', e.target.value);
    document.getElementById('promote_candidate_yes_no').onchange = e => setDecisionField('promote_candidate_yes_no', e.target.value);
    document.getElementById('human_comment').oninput = e => setDecisionField('human_comment', e.target.value);
    document.getElementById('reviewer').oninput = updateCompletionSummary.bind(null, false);
    document.getElementById('markAllUnreviewed').onclick = () => {{
      rows.forEach(row => decisions[row.review_id] = defaultDecision(row));
      saveDecisions();
      render();
      updateCompletionSummary(true);
    }};
    document.getElementById('showSummary').onclick = () => updateCompletionSummary(true);
    document.getElementById('exportJson').onclick = () => {{
      document.getElementById('exportText').value = JSON.stringify(exportPayload(), null, 2);
      updateCompletionSummary(false);
    }};
    document.getElementById('copyJson').onclick = async () => {{
      const text = JSON.stringify(exportPayload(), null, 2);
      const box = document.getElementById('exportText');
      box.value = text;
      try {{
        await navigator.clipboard.writeText(text);
      }} catch (err) {{
        box.focus();
        box.select();
        document.execCommand('copy');
      }}
    }};
    document.getElementById('downloadJson').onclick = () => downloadText('stage4a714_2d_manual_review_export.json', 'application/json', JSON.stringify(exportPayload(), null, 2) + '\\n');
    document.getElementById('downloadCsv').onclick = () => downloadText('stage4a714_2d_manual_review_export.csv', 'text/csv', exportCsvText());
    render();
    updateCompletionSummary(false);
  </script>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def write_how_to_save(path: Path) -> None:
    text = """# Stage 4A-7.14 2D Manual Review Save Instructions

Open `stage4a714_2d_rollout_review_index.html` in a browser.

1. Use the Start and Step buttons to inspect each of the 60 samples.
2. For each sample, set `human_review_status`.
3. Set `human_review_reason` and `human_comment` when useful.
4. Leave `promote_candidate_yes_no` empty unless the sample is approved and should be considered later.
5. Click `Export review JSON`, then `Download review JSON` or `Copy review JSON to clipboard`.
6. Click `Download review CSV` if a spreadsheet copy is easier to inspect.

Human approval does not automatically promote labels. Future Stage 4A-7.7 must import the exported review and make a separate promotion decision.

Negative scope: no label promotion, no training, no checkpoint, no Isaac startup, no map_predict, no rollout, no RL/GDPO/PPO.
"""
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "start_variant_id",
        "step_id",
        "capture_index",
        "source_x",
        "source_y",
        "source_z",
        "action_x",
        "action_y",
        "action_z",
        "observed_ratio_before",
        "observed_ratio_after_current_capture",
        "historical_observed_xy_cells",
        "newly_observed_xy_cells",
        "source_to_action_distance_m",
        "source_to_action_distance_3d_m",
        "action_distance_flag",
        "map_image",
        "rgb",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            sx, sy, sz = row["source_pose_xyz"]
            ax, ay, az = row["action_world_xyz"]
            writer.writerow(
                {
                    "start_variant_id": row["start_variant_id"],
                    "step_id": row["step_id"],
                    "capture_index": row["capture_index"],
                    "source_x": sx,
                    "source_y": sy,
                    "source_z": sz,
                    "action_x": ax,
                    "action_y": ay,
                    "action_z": az,
                    "observed_ratio_before": row["observed_ratio_before"],
                    "observed_ratio_after_current_capture": row["observed_ratio_after_current_capture"],
                    "historical_observed_xy_cells": row.get("historical_observed_xy_cells", 0),
                    "newly_observed_xy_cells": row.get("newly_observed_xy_cells", 0),
                    "source_to_action_distance_m": row["source_to_action_distance_m"],
                    "source_to_action_distance_3d_m": row["source_to_action_distance_3d_m"],
                    "action_distance_flag": row["action_distance_flag"],
                    "map_image": row["map_image"],
                    "rgb": row["relative_rgb"],
                }
            )


def write_distance_audit_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "review_id",
        "sample_id",
        "start_variant_id",
        "step_id",
        "source_to_action_distance_m",
        "source_to_action_distance_3d_m",
        "action_distance_flag",
        "newly_observed_xy_cells",
        "historical_observed_xy_cells",
    ]
    sorted_rows = sorted(records, key=lambda row: float(row["source_to_action_distance_m"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(
                {
                    "review_id": f"stage4a714_s{row['start_variant_id']:03d}_step{row['step_id']:03d}",
                    "sample_id": f"start_{row['start_variant_id']:03d}_step_{row['step_id']:03d}",
                    "start_variant_id": row["start_variant_id"],
                    "step_id": row["step_id"],
                    "source_to_action_distance_m": row["source_to_action_distance_m"],
                    "source_to_action_distance_3d_m": row["source_to_action_distance_3d_m"],
                    "action_distance_flag": row["action_distance_flag"],
                    "newly_observed_xy_cells": row.get("newly_observed_xy_cells", 0),
                    "historical_observed_xy_cells": row.get("historical_observed_xy_cells", 0),
                }
            )


def generate_packet(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    scene_metadata = load_json(args.scene_metadata)
    bounds = scene_metadata["map_bounds"]
    voxel_size = float(scene_metadata.get("voxel_size", args.voxel_size))
    footprints = extract_footprints(parse_usda_xforms(args.usd))
    records = build_records(args.manifest)
    by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_start[row["start_variant_id"]].append(row)

    for start, rows in sorted(by_start.items()):
        rows.sort(key=lambda item: item["step_id"])
        prev_obs: np.ndarray | None = None
        final_obs: np.ndarray | None = None
        for idx, row in enumerate(rows):
            obs_path = Path(row["observed_state_reference"])
            obs = observed_xy(obs_path)
            newly = obs if prev_obs is None else np.logical_and(obs, ~prev_obs)
            row["historical_observed_xy_cells"] = int(np.count_nonzero(obs))
            row["newly_observed_xy_cells"] = int(np.count_nonzero(newly))
            step_png = maps_dir / f"start_{start:03d}_step_{row['step_id']:03d}_2d_review.png"
            render_step_map(step_png, footprints, bounds, rows[: idx + 1], row, obs, newly)
            prev_obs = obs
            final_obs = obs
        if final_obs is not None:
            overview_png = maps_dir / f"start_{start:03d}_overview_2d_review.png"
            render_start_overview(overview_png, footprints, bounds, rows, final_obs)

    distances = np.array([float(row["source_to_action_distance_m"]) for row in records], dtype=np.float64)
    distance_counts = Counter(row["action_distance_flag"] for row in records)
    very_close_rows = [
        f"start_{row['start_variant_id']:03d}_step_{row['step_id']:03d}"
        for row in sorted(records, key=lambda item: float(item["source_to_action_distance_m"]))
        if row["action_distance_flag"] == "very_close"
    ]
    summary = {
        "stage": "Stage 4A-7.14 2D rollout review packet",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(WORKSPACE),
        "usd_path": str(args.usd),
        "rollout_manifest": str(args.manifest),
        "scene_metadata": str(args.scene_metadata),
        "output_dir": str(output_dir),
        "main_html": str(output_dir / "stage4a714_2d_rollout_review_index.html"),
        "save_instructions": str(output_dir / "stage4a714_2d_human_review_how_to_save.md"),
        "review_rows": len(records),
        "start_count": len(by_start),
        "step_map_count": len(records),
        "start_overview_count": len(by_start),
        "map_bounds": bounds,
        "voxel_size": voxel_size,
        "coordinate_contract": "3D world xyz projects to 2D as (world_x, world_y), with z ignored only for top-down display.",
        "history_layer": "historical observed/swept xy cells from observed_state != -1 across z",
        "new_layer": "newly observed xy cells compared with previous saved step within the same start; step0 uses its saved state as baseline",
        "action_distance_thresholds_m": {
            "very_close_lt": VERY_CLOSE_ACTION_DISTANCE_M,
            "close_lt": CLOSE_ACTION_DISTANCE_M,
        },
        "action_distance_counts": {
            "very_close": int(distance_counts.get("very_close", 0)),
            "close": int(distance_counts.get("close", 0)),
            "normal": int(distance_counts.get("normal", 0)),
        },
        "action_distance_stats_m": {
            "min": round(float(distances.min()), 6),
            "median": round(float(np.median(distances)), 6),
            "mean": round(float(distances.mean()), 6),
            "max": round(float(distances.max()), 6),
        },
        "very_close_action_rows": very_close_rows,
        "distance_review_note": "Very close source-to-action distances are review warnings only, not automatic rejection or label promotion.",
        "runtime_scope": "offline_existing_artifact_visualization_only_no_isaac_no_rollout_no_map_predict_no_training_no_checkpoint_no_rl",
        "negative_scope": {
            "isaac_startup": False,
            "runtime": False,
            "capture": False,
            "map_predict": False,
            "rollout": False,
            "training": False,
            "checkpoint": False,
            "label_promotion": False,
            "rl_gdpo_ppo": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage4a714_2d_rollout_review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "stage4a714_2d_rollout_review_records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "stage4a714_2d_rollout_review_records.csv", records)
    write_distance_audit_csv(output_dir / "stage4a714_2d_action_distance_audit.csv", records)
    write_index_html(output_dir / "stage4a714_2d_rollout_review_index.html", records, summary)
    write_how_to_save(output_dir / "stage4a714_2d_human_review_how_to_save.md")
    md = [
        "# Stage 4A-7.14 2D Rollout Review Packet",
        "",
        f"- Main HTML: `{summary['main_html']}`",
        f"- Save instructions: `{summary['save_instructions']}`",
        f"- Review rows: `{len(records)}`",
        f"- Starts: `{len(by_start)}`",
        f"- Step maps: `{len(records)}`",
        f"- Start overviews: `{len(by_start)}`",
        f"- Coordinate contract: `{summary['coordinate_contract']}`",
        f"- Runtime scope: `{summary['runtime_scope']}`",
        f"- Action distance thresholds: very_close `<{VERY_CLOSE_ACTION_DISTANCE_M}m`, close `<{CLOSE_ACTION_DISTANCE_M}m`.",
        f"- Action distance counts: `{summary['action_distance_counts']}`.",
        f"- Action distance stats meters: `{summary['action_distance_stats_m']}`.",
        "",
        "## Human Review Meaning",
        "- Blue wash: all historically observed/swept x/y cells through the selected step.",
        "- Cyan wash: newly observed x/y cells since the previous saved step in the same start.",
        "- Blue dots/line: historical source camera poses projected from 3D world x/y.",
        "- Pink x markers: selected action targets projected from 3D world x/y.",
        "- Green arrow: current source pose to current action target.",
        "- Human decision controls: `human_review_status`, `human_review_reason`, `promote_candidate_yes_no`, `human_comment`.",
        "- Export controls: `Export review JSON`, `Copy review JSON to clipboard`, `Download review JSON`, `Download review CSV`, `Show review completion summary`.",
        "- Action distance warnings are review cues only; a close move can still be valid when it creates a useful viewpoint change.",
        "- Human approval does not automatically promote labels; future Stage 4A-7.7 must import this review and make a separate promotion decision.",
        "",
        "## Negative Scope",
    ]
    for key, value in summary["negative_scope"].items():
        md.append(f"- {key}: `{value}`")
    (output_dir / "stage4a714_2d_rollout_review_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--rollout_dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene_metadata", type=Path, default=DEFAULT_SCENE_METADATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    args = parser.parse_args()
    summary = generate_packet(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
