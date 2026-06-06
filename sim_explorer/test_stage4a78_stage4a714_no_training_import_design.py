#!/usr/bin/env python3
"""Validate Stage 4A-7.8 no-training import design for Stage 4A-7.14 candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
OUT = WORKSPACE / "outputs/stage4a78_stage4a714_no_training_import_design"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    summary_path = OUT / "stage4a78_stage4a714_no_training_import_design_summary.json"
    summary_md = OUT / "stage4a78_stage4a714_no_training_import_design_summary.md"
    manifest_csv = OUT / "stage4a78_stage4a714_planned_import_manifest.csv"
    manifest_json = OUT / "stage4a78_stage4a714_planned_import_manifest.json"
    manifest_jsonl = OUT / "stage4a78_stage4a714_planned_import_manifest.jsonl"
    schema_json = OUT / "stage4a78_stage4a714_schema_compatibility_report.json"
    label_policy = OUT / "stage4a78_stage4a714_label_policy.json"
    no_promotion = OUT / "stage4a78_stage4a714_no_promotion_report.json"
    no_training = OUT / "stage4a78_stage4a714_no_training_report.json"
    no_runtime = OUT / "stage4a78_stage4a714_no_runtime_report.json"
    future_sketch = OUT / "future_stage4a79_stage4a714_no_training_import_implementation_sketch.md"
    checks: dict[str, bool] = {
        "output_dir_exists": OUT.is_dir(),
        "summary_json_exists": summary_path.is_file(),
        "summary_md_exists": summary_md.is_file(),
        "manifest_csv_exists": manifest_csv.is_file(),
        "manifest_json_exists": manifest_json.is_file(),
        "manifest_jsonl_exists": manifest_jsonl.is_file(),
        "schema_json_exists": schema_json.is_file(),
        "label_policy_exists": label_policy.is_file(),
        "no_promotion_exists": no_promotion.is_file(),
        "no_training_exists": no_training.is_file(),
        "no_runtime_exists": no_runtime.is_file(),
        "future_sketch_exists": future_sketch.is_file(),
    }
    summary = read_json(summary_path) or {}
    schema = read_json(schema_json) or {}
    label = read_json(label_policy) or {}
    no_promo = read_json(no_promotion) or {}
    no_train = read_json(no_training) or {}
    no_run = read_json(no_runtime) or {}
    manifest = read_csv(manifest_csv)
    negative = summary.get("negative_scope", {})
    npz_outputs = sorted(OUT.glob("*.npz"))

    checks.update(
        {
            "completed_true": summary.get("completed") is True,
            "blocked_false": summary.get("blocked") is False,
            "no_blockers": summary.get("blockers") == [],
            "planned_import_count_25": summary.get("planned_import_count") == 25 and len(manifest) == 25,
            "held_recheck_count_3": summary.get("held_distance_recheck_count") == 3,
            "schema_adapter_required": schema.get("adapter_required") is True
            and summary.get("schema_compatibility", {}).get("adapter_required") is True,
            "direct_npz_concat_disallowed": schema.get("direct_npz_concat_allowed") is False
            and summary.get("schema_compatibility", {}).get("direct_npz_concat_allowed") is False,
            "shape_mismatch_documented": schema.get("existing_primary_bc_candidate_features_model_shape") == [30, 64, 16]
            and schema.get("stage4a714_runtime_candidate_features_shape") == [60, 64, 13],
            "adapter_mapping_complete": schema.get("adapter_mapping_complete_for_model_features") is True,
            "manifest_all_planned_not_applied": all(row.get("import_status") == "planned_not_applied" for row in manifest),
            "manifest_no_direct_concat": all(row.get("direct_npz_concat_allowed") == "False" for row in manifest),
            "manifest_adapter_required": all(row.get("adapter_required") == "True" for row in manifest),
            "no_npz_created": not npz_outputs,
            "no_label_promotion": negative.get("label_promotion") is False and no_promo.get("label_promotion") is False,
            "no_dataset_modified": negative.get("dataset_modified") is False,
            "no_expanded_dataset_npz_created": negative.get("expanded_dataset_npz_created") is False,
            "no_expert_action_index_primary_created": negative.get("expert_action_index_primary_created") is False
            and label.get("expert_action_index_primary_created") is False,
            "lambda48_shadow_only": "shadow" in str(label.get("lambda48_role", "")).lower(),
            "do_not_recompute_from_lambda48": label.get("do_not_recompute_from_lambda48") is True,
            "no_training": negative.get("training") is False and no_train.get("training") is False,
            "no_optimizer_step": negative.get("optimizer_step") is False and no_train.get("optimizer_step") is False,
            "no_checkpoint": negative.get("checkpoint") is False and no_train.get("checkpoint") is False,
            "no_isaac": negative.get("isaac_startup") is False and no_run.get("isaac_startup") is False,
            "no_map_predict": negative.get("map_predict") is False and no_run.get("map_predict") is False,
            "no_rollout": negative.get("rollout") is False and no_run.get("rollout") is False,
            "no_rl_gdpo_ppo": negative.get("rl_gdpo_ppo") is False and no_run.get("rl_gdpo_ppo") is False,
            "future_sketch_guard": future_sketch.read_text(encoding="utf-8").splitlines()[0] == "DO NOT RUN IN STAGE 4A-7.8."
            if future_sketch.is_file()
            else False,
        }
    )
    blockers = [name for name, ok in sorted(checks.items()) if not ok]
    result = {"all_passed": not blockers, "blockers": blockers, "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
