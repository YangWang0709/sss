#!/usr/bin/env python3
"""Validate Stage 4A-7.10 no-training QA/readiness packet."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUTPUT_DIR = PROJECT_ROOT / "outputs/stage4a710_stage4a79_no_training_qa_readiness"
STAGE79_DIR = PROJECT_ROOT / "outputs/stage4a79_stage4a714_compatible_no_training_import"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    summary_path = OUTPUT_DIR / "stage4a710_stage4a79_no_training_qa_readiness_summary.json"
    html_path = OUTPUT_DIR / "stage4a710_stage4a79_readiness_index.html"
    row_csv = OUTPUT_DIR / "stage4a710_stage4a79_row_qa.csv"
    row_json = OUTPUT_DIR / "stage4a710_stage4a79_row_qa.json"
    distribution_path = OUTPUT_DIR / "stage4a710_stage4a79_distribution_report.json"
    decision_path = OUTPUT_DIR / "stage4a710_stage4a79_readiness_decision.json"
    no_training_path = OUTPUT_DIR / "stage4a710_stage4a79_no_training_report.json"
    no_runtime_path = OUTPUT_DIR / "stage4a710_stage4a79_no_runtime_report.json"

    paths = {
        "output_dir_exists": OUTPUT_DIR,
        "summary_json_exists": summary_path,
        "main_html_exists": html_path,
        "row_csv_exists": row_csv,
        "row_json_exists": row_json,
        "distribution_report_exists": distribution_path,
        "decision_json_exists": decision_path,
        "no_training_report_exists": no_training_path,
        "no_runtime_report_exists": no_runtime_path,
    }
    checks: dict[str, bool] = {name: path.exists() for name, path in paths.items()}
    blockers: list[str] = [name for name, ok in checks.items() if not ok]

    if blockers:
        print(json.dumps({"all_passed": False, "checks": checks, "blockers": blockers}, indent=2, sort_keys=True))
        raise SystemExit(1)

    summary = load_json(summary_path)
    decision = load_json(decision_path)
    no_training = load_json(no_training_path)
    no_runtime = load_json(no_runtime_path)
    rows = csv_rows(row_csv)
    row_json_rows = load_json(row_json)
    html = html_path.read_text(encoding="utf-8")
    adapter = np.load(STAGE79_DIR / "stage4a79_stage4a714_adapter_dataset_25.npz")
    expanded = np.load(STAGE79_DIR / "stage4a79_stage4a714_compatible_expanded_dataset_55.npz")

    checks.update(
        {
            "completed_true": summary.get("completed") is True,
            "blocked_false": summary.get("blocked") is False,
            "readiness_ready": summary.get("readiness_decision") == "ready_for_tiny_bc_dry_run_consideration",
            "decision_ready_true": decision.get("ready_for_tiny_bc_dry_run_consideration") is True,
            "training_not_allowed_by_packet": decision.get("training_allowed_by_this_packet") is False,
            "checkpoint_not_allowed_by_packet": decision.get("checkpoint_allowed_by_this_packet") is False,
            "rl_not_allowed_by_packet": decision.get("rl_gdpo_ppo_allowed_by_this_packet") is False,
            "blocker_count_zero": summary.get("blocker_count") == 0,
            "import_rows_25": summary.get("import_rows") == 25,
            "expanded_rows_55": summary.get("expanded_rows") == 55,
            "adapter_model_shape": summary.get("adapter_shapes", {}).get("candidate_features_model") == [25, 64, 16],
            "expanded_model_shape": summary.get("expanded_shapes", {}).get("candidate_features_model") == [55, 64, 16],
            "expanded_raw_shape": summary.get("expanded_shapes", {}).get("candidate_features_raw") == [55, 64, 91],
            "old_dataset_prefix_unchanged": summary.get("old_dataset_prefix_unchanged") is True,
            "adapter_matches_expanded_suffix": summary.get("adapter_matches_expanded_suffix") is True,
            "row_csv_25": len(rows) == 25,
            "row_json_25": len(row_json_rows) == 25,
            "html_25_cards": html.count('data-review-card="1"') == 25,
            "html_mentions_stage_4a710": "Stage 4A-7.10" in html,
            "html_mentions_no_training": "no training" in html.lower(),
            "all_rows_not_block": all(row.get("qa_status") != "block" for row in rows),
            "no_very_close_import_rows": all(row.get("action_distance_flag") != "very_close" for row in rows),
            "all_human_approved": all(row.get("human_review_status") == "approve" for row in rows),
            "all_promote_yes": all(row.get("promote_candidate_yes_no") == "yes" for row in rows),
            "lambda48_shadow_only": summary.get("label_lineage", {}).get("lambda48_role") == "shadow/baseline only",
            "not_recomputed_from_lambda48": summary.get("label_lineage", {}).get("labels_recomputed_from_lambda48") is False,
            "no_training": no_training.get("training") is False,
            "no_optimizer_step": no_training.get("optimizer_step") is False,
            "no_model_save": no_training.get("model_save") is False,
            "no_checkpoint": no_training.get("checkpoint") is False,
            "no_isaac": no_runtime.get("isaac_startup") is False,
            "no_map_predict": no_runtime.get("map_predict") is False,
            "no_rollout": no_runtime.get("rollout") is False,
            "adapter_npz_shape_runtime": adapter["candidate_features_model"].shape == (25, 64, 16),
            "expanded_npz_shape_runtime": expanded["candidate_features_model"].shape == (55, 64, 16),
            "all_primary_labels_valid_runtime": all(
                0 <= int(label) < adapter["candidate_valid_mask"].shape[1]
                and bool(adapter["candidate_valid_mask"][idx, int(label)])
                for idx, label in enumerate(adapter["expert_action_index_primary"])
            ),
        }
    )

    forbidden_names = ["checkpoint", "ppo", "gdpo", "rl_model", "optimizer_state"]
    output_names = [p.name.lower() for p in OUTPUT_DIR.iterdir()]
    forbidden_present = [name for name in output_names for forbidden in forbidden_names if forbidden in name]
    checks["no_forbidden_output_names"] = not forbidden_present

    blockers = [name for name, ok in checks.items() if not ok]
    result = {
        "all_passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "forbidden_present": forbidden_present,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
