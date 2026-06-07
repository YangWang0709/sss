#!/usr/bin/env python3
"""Generate LR-8 web/HTML review audit for the long rollout packet."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "outputs/autonomous_long_rollout_bridge"
PACKET = BRIDGE / "stage_lr7_long_rollout_review_packet"
OUT = BRIDGE / "stage_lr8_web_html_review_audit"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            lines.extend([f"## {key}", "", "```json", json.dumps(value, indent=2, sort_keys=True), "```", ""])
        else:
            lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    main_html = PACKET / "long_rollout_2d_review_index.html"
    legacy_html = PACKET / "stage4a714_2d_rollout_review_index.html"
    action_story = PACKET / "long_rollout_action_story_index.html"
    mp4 = PACKET / "long_rollout_flythrough.mp4"
    records_path = PACKET / "stage4a714_2d_rollout_review_records.json"
    summary_path = PACKET / "long_rollout_review_packet_summary.json"
    instructions = PACKET / "stage_lr7_long_rollout_human_review_how_to_save.md"

    html_text = text(main_html) + "\n" + text(legacy_html)
    story_text = text(action_story)
    instruction_text = text(instructions)
    records = load(records_path)
    if not isinstance(records, list):
        records = []
    summary = load(summary_path)

    map_files = sorted((PACKET / "maps").glob("start_*_step_*_2d_review.png"))
    overview_files = sorted((PACKET / "maps").glob("start_*_overview_2d_review.png"))
    very_close_rows = summary.get("source_packet_summary", {}).get("very_close_action_rows", [])
    distance_counts = summary.get("source_packet_summary", {}).get("action_distance_counts", {})

    checks = {
        "main_html_exists": main_html.is_file(),
        "legacy_html_exists": legacy_html.is_file(),
        "action_story_exists": action_story.is_file(),
        "mp4_exists": mp4.is_file() and mp4.stat().st_size > 0,
        "records_150": len(records) == 150,
        "step_maps_150": len(map_files) == 150,
        "overview_maps_10": len(overview_files) == 10,
        "action_story_cards_150": story_text.count('class="card"') == 150,
        "has_review_schema_version": "review_schema_version" in html_text,
        "exports_source_stage": "Stage 4A-7.2" in html_text or "source_stage" in html_text,
        "exports_packet_stage": "Stage 4A-7.6" in html_text or "packet_stage" in html_text,
        "has_export_json_button": "Export review JSON" in html_text,
        "has_copy_json_button": "Copy review JSON to clipboard" in html_text,
        "has_download_csv_button": "Download review CSV" in html_text,
        "has_completion_summary_button": "Show review completion summary" in html_text,
        "has_status_control": "human_review_status" in html_text,
        "has_reason_control": "human_review_reason" in html_text,
        "has_promote_control": "promote_candidate_yes_no" in html_text,
        "has_comment_control": "human_comment" in html_text,
        "promote_yes_not_default": "promote_candidate_yes_no: 'yes'" not in html_text and 'promote_candidate_yes_no: "yes"' not in html_text,
        "approval_no_auto_promotion_warning": "does not automatically promote labels" in html_text or "does not automatically promote labels" in instruction_text,
        "very_close_actions_visible_for_review": bool(very_close_rows) and int(distance_counts.get("very_close", 0)) == len(very_close_rows),
        "no_review_generation_training": summary.get("negative_scope", {}).get("training") is False,
        "no_review_generation_checkpoint": summary.get("negative_scope", {}).get("checkpoint") is False,
        "no_review_generation_label_promotion": summary.get("negative_scope", {}).get("label_promotion") is False,
        "no_review_generation_rl": summary.get("negative_scope", {}).get("RL_GDPO_PPO") is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    audit = {
        "stage": "LR-8 web/HTML review audit",
        "generated_at_utc": now(),
        "packet_dir": str(PACKET),
        "main_html": str(main_html),
        "action_story_html": str(action_story),
        "mp4": str(mp4),
        "records": str(records_path),
        "review_rows": len(records),
        "map_count": len(map_files),
        "overview_count": len(overview_files),
        "very_close_action_count": int(distance_counts.get("very_close", 0)),
        "close_action_count": int(distance_counts.get("close", 0)),
        "normal_action_count": int(distance_counts.get("normal", 0)),
        "checks": checks,
        "blockers": blockers,
        "external_gpt_chrome_review": {
            "attempted": True,
            "submitted": False,
            "reason": "GPT Chrome window was minimized/active-user-input detected; no prompt was sent to avoid interrupting the user.",
        },
        "critic_decision": "approve" if not blockers else "block",
        "all_passed": not blockers,
    }
    critic = {
        "stage": "LR-8 local critic review",
        "generated_at_utc": now(),
        "decision": audit["critic_decision"],
        "basis": "Static HTML/export/media audit over LR-7 packet; no runtime or training performed.",
        "must_not_start_rl": True,
        "must_not_promote_labels": True,
        "lambda48_role": "shadow/baseline only",
        "all_passed": audit["all_passed"],
        "blockers": blockers,
    }
    write_json(OUT / "web_html_review_audit.json", audit)
    write_md(OUT / "web_html_review_audit.md", "LR-8 Web HTML Review Audit", audit)
    write_json(OUT / "critic_html_review_audit.json", critic)
    write_md(OUT / "critic_html_review_audit.md", "LR-8 Critic HTML Review Audit", critic)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
