#!/usr/bin/env python3
"""Validate LR-7 bounded long rollout human review packet."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/autonomous_long_rollout_bridge/stage_lr7_long_rollout_review_packet"
RESULT = OUT / "test_long_rollout_review_packet_result.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    main_html = OUT / "long_rollout_2d_review_index.html"
    legacy_html = OUT / "stage4a714_2d_rollout_review_index.html"
    action_story = OUT / "long_rollout_action_story_index.html"
    records_path = OUT / "stage4a714_2d_rollout_review_records.json"
    instructions = OUT / "stage_lr7_long_rollout_human_review_how_to_save.md"
    mp4 = OUT / "long_rollout_flythrough.mp4"
    summary_path = OUT / "long_rollout_review_packet_summary.json"
    audit_path = OUT / "long_rollout_html_review_audit.json"

    html_text = main_html.read_text(encoding="utf-8", errors="replace") if main_html.is_file() else ""
    legacy_text = legacy_html.read_text(encoding="utf-8", errors="replace") if legacy_html.is_file() else ""
    story_text = action_story.read_text(encoding="utf-8", errors="replace") if action_story.is_file() else ""
    instructions_text = instructions.read_text(encoding="utf-8", errors="replace") if instructions.is_file() else ""
    combined_html = html_text + "\n" + legacy_text
    records = load(records_path)
    if not isinstance(records, list):
        records = []
    summary = load(summary_path)
    audit = load(audit_path)

    checks: dict[str, bool] = {
        "output_dir_exists": OUT.is_dir(),
        "main_html_exists": main_html.is_file(),
        "legacy_html_exists": legacy_html.is_file(),
        "action_story_exists": action_story.is_file(),
        "records_exists": records_path.is_file(),
        "summary_exists": summary_path.is_file(),
        "audit_exists": audit_path.is_file(),
        "instructions_exists": instructions.is_file(),
        "mp4_exists": mp4.is_file(),
        "review_rows_150": len(records) == 150,
        "map_pngs_150": len(list((OUT / "maps").glob("start_*_step_*_2d_review.png"))) == 150,
        "overview_pngs_10": len(list((OUT / "maps").glob("start_*_overview_2d_review.png"))) == 10,
        "export_review_json_text": "Export review JSON" in combined_html,
        "copy_review_json_text": "Copy review JSON to clipboard" in combined_html,
        "download_review_csv_text": "Download review CSV" in combined_html,
        "completion_summary_text": "Show review completion summary" in combined_html,
        "human_review_status_control": "human_review_status" in combined_html,
        "human_review_reason_control": "human_review_reason" in combined_html,
        "promote_candidate_control": "promote_candidate_yes_no" in combined_html,
        "human_comment_control": "human_comment" in combined_html,
        "no_promote_default_yes": 'promote_candidate_yes_no: "yes"' not in combined_html
        and "promote_candidate_yes_no: 'yes'" not in combined_html,
        "validation_reject_yes": "promote_candidate_yes_no" in combined_html and "approve" in combined_html,
        "action_story_cards_150": story_text.count('class="card"') == 150,
        "instructions_warn_no_auto_promotion": "does not automatically promote labels" in instructions_text,
        "summary_all_passed": summary.get("all_passed") is True,
        "audit_all_passed": audit.get("all_passed") is True,
        "no_isaac_startup": summary.get("negative_scope", {}).get("isaac_startup") is False,
        "no_map_predict": summary.get("negative_scope", {}).get("map_predict") is False,
        "no_runtime_rollout": summary.get("negative_scope", {}).get("runtime_rollout") is False,
        "no_training": summary.get("negative_scope", {}).get("training") is False,
        "no_checkpoint": summary.get("negative_scope", {}).get("checkpoint") is False,
        "no_label_promotion": summary.get("negative_scope", {}).get("label_promotion") is False,
        "no_rl_gdpo_ppo": summary.get("negative_scope", {}).get("RL_GDPO_PPO") is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    result = {
        "stage": "LR-7 bounded long rollout review packet validator",
        "checks": checks,
        "blockers": blockers,
        "all_passed": not blockers,
        "output_dir": str(OUT),
        "main_html": str(main_html),
        "action_story_html": str(action_story),
        "mp4": str(mp4),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
