#!/usr/bin/env python3
"""Validate autonomous bounded long expert rollout goal closure."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "outputs/autonomous_long_rollout_bridge"
OUT = BRIDGE / "stage_lr9_goal_closure"
RESULT = OUT / "autonomous_long_rollout_goal_closure_validator.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def main() -> int:
    lr6 = load(BRIDGE / "stage_lr6_postrun_safety_audit/test_long_rollout_postrun_safety_audit_result.json")
    lr7 = load(BRIDGE / "stage_lr7_long_rollout_review_packet/test_long_rollout_review_packet_result.json")
    lr8 = load(BRIDGE / "stage_lr8_web_html_review_audit/web_html_review_audit.json")
    no_train = load(BRIDGE / "stage_lr6_postrun_safety_audit/no_training_checkpoint_rl_report.json")
    current = text(ROOT / ".project_context/CURRENT_STATE.md")
    todo = text(ROOT / ".project_context/TODO.md")
    log = text(ROOT / ".project_context/CODEX_LOG.md")

    checks: dict[str, bool] = {
        "lr0_current_state_report_exists": (BRIDGE / "stage_lr0_current_state_verification/current_state_verification_report.json").is_file(),
        "lr4_critic_gate_exists": (BRIDGE / "stage_lr4_web_design_preflight_review/critic_design_preflight_review.json").is_file(),
        "lr6_postrun_validator_passed": lr6.get("all_passed") is True,
        "lr7_review_packet_validator_passed": lr7.get("all_passed") is True,
        "lr8_web_html_audit_passed": lr8.get("all_passed") is True,
        "current_state_mentions_lr5": "LR-5 bounded long expert rollout" in current,
        "current_state_mentions_lr6": "LR-6 postrun safety audit" in current,
        "current_state_mentions_lr7": "LR-7 long rollout review packet" in current,
        "todo_mentions_human_review": "long_rollout_2d_review_index.html" in todo,
        "codex_log_mentions_goal": "autonomous bounded long expert rollout" in log.lower(),
        "no_training": no_train.get("training") is False,
        "no_checkpoint": no_train.get("checkpoint") is False and not no_train.get("checkpoint_like_artifacts_under_runtime"),
        "no_label_promotion": no_train.get("label_promotion") is False,
        "no_rl_gdpo_ppo": no_train.get("RL_GDPO_PPO") is False,
        "lambda48_shadow_only": no_train.get("lambda48_role") == "shadow/baseline only",
    }
    blockers = [name for name, passed in checks.items() if not passed]
    result = {
        "stage": "LR-9 autonomous bounded long expert rollout goal closure validator",
        "checks": checks,
        "blockers": blockers,
        "all_passed": not blockers,
        "bridge_dir": str(BRIDGE),
        "main_review_html": str(BRIDGE / "stage_lr7_long_rollout_review_packet/long_rollout_2d_review_index.html"),
        "action_story_html": str(BRIDGE / "stage_lr7_long_rollout_review_packet/long_rollout_action_story_index.html"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
