#!/usr/bin/env python3
"""Validate the Stage 4A-7.14 2D rollout review packet.

This validator is offline and review-only.  It does not launch Isaac, execute
actions, run map_predict, train, checkpoint, or promote labels.
"""

from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
OUT = WORKSPACE / "outputs/stage4a714_2d_review_packet"


def main() -> int:
    summary_path = OUT / "stage4a714_2d_rollout_review_summary.json"
    records_path = OUT / "stage4a714_2d_rollout_review_records.json"
    html_path = OUT / "stage4a714_2d_rollout_review_index.html"
    csv_path = OUT / "stage4a714_2d_rollout_review_records.csv"
    checks: dict[str, bool] = {
        "output_dir_exists": OUT.is_dir(),
        "summary_exists": summary_path.is_file(),
        "records_exists": records_path.is_file(),
        "html_exists": html_path.is_file(),
        "csv_exists": csv_path.is_file(),
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.is_file() else []
    html = html_path.read_text(encoding="utf-8", errors="replace") if html_path.is_file() else ""

    step_maps = sorted((OUT / "maps").glob("start_*_step_*_2d_review.png"))
    overview_maps = sorted((OUT / "maps").glob("start_*_overview_2d_review.png"))
    checks.update(
        {
            "review_rows_60": int(summary.get("review_rows", -1)) == 60 and len(records) == 60,
            "start_count_10": int(summary.get("start_count", -1)) == 10,
            "step_map_count_60": int(summary.get("step_map_count", -1)) == 60 and len(step_maps) == 60,
            "overview_count_10": int(summary.get("start_overview_count", -1)) == 10 and len(overview_maps) == 10,
            "coordinate_contract_present": "world_x, world_y" in str(summary.get("coordinate_contract", "")),
            "history_layer_present": "observed_state != -1" in str(summary.get("history_layer", "")),
            "html_has_start_controls": 'id="starts"' in html,
            "html_has_step_controls": 'id="steps"' in html,
            "html_has_map_image": 'id="map"' in html,
            "html_has_rgb_image": 'id="rgb"' in html,
            "html_mentions_observed_area": "historical observed/swept area" in html
            or "historically observed/swept area" in html,
            "html_mentions_action_targets": "action targets" in html,
            "negative_no_isaac": summary.get("negative_scope", {}).get("isaac_startup") is False,
            "negative_no_runtime": summary.get("negative_scope", {}).get("runtime") is False,
            "negative_no_capture": summary.get("negative_scope", {}).get("capture") is False,
            "negative_no_map_predict": summary.get("negative_scope", {}).get("map_predict") is False,
            "negative_no_rollout": summary.get("negative_scope", {}).get("rollout") is False,
            "negative_no_training": summary.get("negative_scope", {}).get("training") is False,
            "negative_no_checkpoint": summary.get("negative_scope", {}).get("checkpoint") is False,
            "negative_no_label_promotion": summary.get("negative_scope", {}).get("label_promotion") is False,
            "negative_no_rl_gdpo_ppo": summary.get("negative_scope", {}).get("rl_gdpo_ppo") is False,
        }
    )

    for idx in (0, 5, 12, 27, 59):
        if 0 <= idx < len(records):
            row = records[idx]
            checks[f"record_{idx}_has_map_image"] = (OUT / row["map_image"]).is_file()
            checks[f"record_{idx}_has_rgb_ref"] = bool(row.get("relative_rgb"))
            checks[f"record_{idx}_has_source_xyz"] = len(row.get("source_pose_xyz", [])) == 3
            checks[f"record_{idx}_has_action_xyz"] = len(row.get("action_world_xyz", [])) == 3
            checks[f"record_{idx}_has_observed_counts"] = int(row.get("historical_observed_xy_cells", -1)) >= 0 and int(
                row.get("newly_observed_xy_cells", -1)
            ) >= 0

    blockers = [name for name, ok in sorted(checks.items()) if not ok]
    result = {"all_passed": not blockers, "blockers": blockers, "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
