#!/usr/bin/env python3
"""Inspect Phase 5 fallback rollout metadata for human review.

Reads rollout_steps.csv and candidate_summary.csv, prints per-start coverage and
summary statistics, and optionally writes coverage plots to review_plots/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_int(row, key, default=0):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def sparkline(values):
    if not values:
        return ""
    blocks = ".:-=+*#@"
    lo, hi = min(values), max(values)
    if hi == lo:
        return blocks[0] * len(values)
    return "".join(blocks[min(len(blocks) - 1, int((v - lo) / (hi - lo) * (len(blocks) - 1)))] for v in values)


def save_plots(curves, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc!r}"

    for start_id, rows in curves.items():
        xs = [as_int(r, "step_id") for r in rows]
        ys = [as_float(r, "known_ratio") for r in rows]
        plt.figure(figsize=(6, 3.5))
        plt.plot(xs, ys, marker="o", linewidth=1.5)
        plt.title(f"start_{start_id:03d} coverage")
        plt.xlabel("step_id")
        plt.ylabel("known_ratio")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"start_{start_id:03d}_coverage.png", dpi=140)
        plt.close()
    return True, "plots saved"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", default="runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    steps = read_csv(rollout_dir / "rollout_steps.csv")
    candidates = read_csv(rollout_dir / "candidate_summary.csv")
    failures = read_csv(rollout_dir / "failure_summary.csv")

    curves = {}
    for row in steps:
        sid = as_int(row, "start_id")
        curves.setdefault(sid, []).append(row)
    candidate_counts = {}
    for row in candidates:
        sid = as_int(row, "start_id")
        candidate_counts[sid] = candidate_counts.get(sid, 0) + 1

    per_start = []
    print("# Phase 5 Rollout Inspection")
    print(f"rollout_dir: {rollout_dir}")
    for sid in sorted(curves):
        rows = sorted(curves[sid], key=lambda r: as_int(r, "step_id"))
        known = [as_float(r, "known_ratio") for r in rows]
        failures_here = [r for r in rows if r.get("failure")]
        action_count = len([r for r in rows if not r.get("failure")])
        final_known = known[-1] if known else 0.0
        monotonic_non_decreasing = all(b >= a for a, b in zip(known, known[1:]))
        item = {
            "start_id": sid,
            "step_rows": len(rows),
            "action_count": action_count,
            "failure_count": len(failures_here),
            "final_known_ratio": round(final_known, 6),
            "candidate_rows": candidate_counts.get(sid, 0),
            "coverage_curve": [round(v, 6) for v in known],
            "coverage_sparkline": sparkline(known),
            "known_ratio_monotonic_non_decreasing": monotonic_non_decreasing,
        }
        per_start.append(item)
        print(json.dumps(item, sort_keys=True))

    summary = {
        "start_count": len(per_start),
        "total_action_count": sum(x["action_count"] for x in per_start),
        "total_step_rows": len(steps),
        "candidate_rows": len(candidates),
        "starts_with_failures": sum(1 for x in per_start if x["failure_count"] > 0),
        "average_final_known_ratio": round(mean(x["final_known_ratio"] for x in per_start), 6),
        "failure_summary_rows": len(failures),
        "training": False,
        "RL": False,
        "checkpoint": False,
    }
    print("# Summary")
    print(json.dumps(summary, indent=2, sort_keys=True))

    review_dir = rollout_dir / "review_plots"
    review_dir.mkdir(parents=True, exist_ok=True)
    with (review_dir / "coverage_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["start_id", "step_rows", "action_count", "failure_count", "final_known_ratio", "candidate_rows", "coverage_sparkline", "known_ratio_monotonic_non_decreasing"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_start:
            writer.writerow({k: item[k] for k in fieldnames})
    (review_dir / "inspection_summary.json").write_text(json.dumps({"summary": summary, "per_start": per_start}, indent=2, sort_keys=True), encoding="utf-8")

    if args.save_plots:
        ok, msg = save_plots(curves, review_dir)
        print(f"plots: {msg}")
        if not ok:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
