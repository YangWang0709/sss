from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet"
HTML = OUT / "stage4a72_action_story_review_index.html"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["main_html_exists"] = HTML.is_file()
    html = HTML.read_text(encoding="utf-8") if HTML.is_file() else ""

    required_text = [
        "human_review_status",
        "human_review_reason",
        "promote_candidate_yes_no",
        "human_comment",
        "Mark all unreviewed",
        "Export review JSON",
        "Copy review JSON to clipboard",
        "Download review CSV",
        "Show review completion summary",
        "Human approval does not automatically promote labels",
        "review_schema_version",
        "source_stage: \"Stage 4A-7.2\"",
        "packet_stage: \"Stage 4A-7.6\"",
        "exported_at_local_browser_time",
        "reviewer_name",
        "stage4a72_manual_review_export.json",
        "stage4a72_manual_review_export.csv",
    ]
    for text in required_text:
        checks[f"html_contains_{text[:40]}"] = text in html

    checks["review_cards_30"] = len(re.findall(r"<article\b[^>]*data-review-card=\"true\"", html)) == 30
    checks["status_selects_30"] = html.count('data-field="human_review_status"') == 30
    checks["reason_selects_30"] = html.count('data-field="human_review_reason"') == 30
    checks["promote_selects_30"] = html.count('data-field="promote_candidate_yes_no"') == 30
    checks["comment_boxes_30"] = html.count('data-field="human_comment"') == 30
    checks["default_status_unreviewed_30"] = html.count('<option value="unreviewed" selected>unreviewed</option>') == 30
    checks["no_promote_yes_default"] = 'value="yes" selected' not in html and "value='yes' selected" not in html
    checks["browser_validation_present"] = "promote_candidate_yes_no=yes is only allowed when human_review_status=approve" in html
    checks["reject_unsure_needs_cannot_promote_yes"] = all(x in html for x in ["reject", "unsure", "needs_closer_inspection"]) and "status.value !== 'approve'" in html

    checks["instruction_file_exists"] = (OUT / "stage4a72_human_review_how_to_save.md").is_file()
    instruction = (OUT / "stage4a72_human_review_how_to_save.md").read_text(encoding="utf-8") if checks["instruction_file_exists"] else ""
    checks["instruction_mentions_export_json_csv"] = "Export review JSON" in instruction and "Download review CSV" in instruction
    checks["report_exists"] = (OUT / "stage4a76c_review_export_controls_report.json").is_file()
    report = load_json(OUT / "stage4a76c_review_export_controls_report.json") if checks["report_exists"] else {}
    checks["report_completed"] = report.get("completed") is True and report.get("blocked") is False
    checks["report_30_cards"] = report.get("review_cards") == 30
    checks["no_expert_action_index_primary_created_from_stage4a72"] = report.get("expert_action_index_primary_created_from_stage4a72") is False
    checks["no_label_promotion"] = report.get("label_promotion") is False
    checks["no_training"] = report.get("training") is False
    checks["no_checkpoint"] = report.get("checkpoint") is False
    checks["no_isaac"] = report.get("isaac_startup") is False
    checks["no_map_predict"] = report.get("map_predict") is False
    checks["no_rollout"] = report.get("rollout") is False
    checks["no_rl_gdpo_ppo"] = report.get("rl_gdpo_ppo") is False
    checks["lambda48_shadow_only"] = "shadow" in str(report.get("lambda48_role", "")).lower()

    # Guard against accidental Stage 4A-7.2 primary-label creation artifacts.
    forbidden_names = [
        p
        for p in OUT.rglob("*")
        if p.is_file() and re.search(r"expert_action_index_primary|promoted_primary|stage4a72.*primary_label", p.name, re.I)
    ]
    checks["no_forbidden_primary_artifact_names"] = not forbidden_names

    for key, ok in checks.items():
        if not ok:
            blockers.append(key)

    result = {
        "all_passed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "forbidden_primary_artifact_names": [str(p) for p in forbidden_names],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
