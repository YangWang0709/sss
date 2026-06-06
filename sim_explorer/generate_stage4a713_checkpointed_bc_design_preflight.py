#!/usr/bin/env python3
"""Generate Stage 4A-7.13 checkpointed BC design/preflight packet.

This stage is design/preflight only. It reads Stage 4A-7.12 decision evidence
and Stage 4A-7.9 dataset metadata, then writes a bounded experiment design for a
future checkpointed BC run. It does not train, step an optimizer, save weights,
create checkpoints, run runtime, or run RL.
"""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/stage4a713_checkpointed_bc_design_preflight"
STAGE712 = ROOT / "outputs/stage4a712_stage4a79_review_readiness_decision_packet"
STAGE79 = ROOT / "outputs/stage4a79_stage4a714_compatible_no_training_import"

STAGE712_SUMMARY = STAGE712 / "stage4a712_review_readiness_decision_summary.md"
STAGE712_DECISION = STAGE712 / "selected_next_step_decision.json"
STAGE712_OPTIONS = STAGE712 / "decision_option_matrix.json"
STAGE712_RISKS = STAGE712 / "risk_register.json"
STAGE712_HANDOFF = STAGE712 / "handoff_to_stage4a713_design_preflight.md"

DATASET = STAGE79 / "stage4a79_stage4a714_compatible_expanded_dataset_55.npz"
STAGE79_SUMMARY = STAGE79 / "stage4a79_stage4a714_compatible_import_summary.md"
STAGE79_INTEGRITY = STAGE79 / "stage4a79_stage4a714_integrity_report.json"
STAGE79_LINEAGE = STAGE79 / "stage4a79_stage4a714_label_lineage_report.json"

FUTURE_CHECKPOINT_DIR = ROOT / "checkpoints/stage4a714_checkpointed_bc_experiment"
FUTURE_CHECKPOINT_PATH = FUTURE_CHECKPOINT_DIR / "stage4a79_compatible_candidate_mlp_policy.pt"
FUTURE_CHECKPOINT_METADATA = FUTURE_CHECKPOINT_DIR / "stage4a79_compatible_candidate_mlp_policy_metadata.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def git_status() -> str:
    return run(["git", "status", "--short"]).stdout


def git_head() -> str:
    result = run(["git", "rev-parse", "HEAD"])
    return result.stdout.strip()


def git_check_ignored(path: Path) -> dict[str, Any]:
    result = run(["git", "check-ignore", "-v", str(path.relative_to(ROOT))])
    return {
        "path": str(path),
        "is_ignored": result.returncode == 0,
        "evidence": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|")


def write_md_table(path: Path, title: str, rows: list[tuple[str, Any]], extra: str = "") -> None:
    lines = [f"# {title}", "", "| field | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| `{key}` | {format_value(value)} |")
    if extra:
        lines.extend(["", extra.rstrip()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_md_list(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def counter_from_array(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(map(str, values.tolist())).items()}


def load_dataset_preflight() -> dict[str, Any]:
    forbidden_tokens = [
        "target",
        "ground_truth",
        "future_observed",
        "prediction_writeback",
        "uncertainty_writeback",
    ]
    with np.load(DATASET, allow_pickle=False) as data:
        keys = sorted(data.files)
        features = np.asarray(data["candidate_features_model"])
        valid = np.asarray(data["candidate_valid_mask"], dtype=bool)
        labels = np.asarray(data["expert_action_index_primary"], dtype=np.int64)
        quality = np.asarray(data["quality_keep_mask"], dtype=bool)
        source_stage = np.asarray(data["source_stage"]).astype(str)
        source_stage_id = np.asarray(data["source_stage_id"], dtype=np.int64)
        primary_policy = np.asarray(data["primary_label_policy"]).astype(str)
        lambda_shadow = np.asarray(data["expert_action_index_lambda48_shadow"], dtype=np.int64)
        split_id = np.asarray(data["split_id"], dtype=np.int64)
        review_status = np.asarray(data["human_review_status"]).astype(str)

        row_count, candidate_count, d_model = list(features.shape)
        label_in_range = (labels >= 0) & (labels < candidate_count)
        valid_label_mask = np.zeros(row_count, dtype=bool)
        valid_label_mask[label_in_range] = valid[np.arange(row_count)[label_in_range], labels[label_in_range]]
        rejected_tokens = {"reject", "rejected", "held", "recheck", "needs_recheck"}
        report = {
            "dataset_path": str(DATASET),
            "dataset_exists": DATASET.is_file(),
            "dataset_sha256": sha256_file(DATASET),
            "dataset_size_bytes": DATASET.stat().st_size if DATASET.is_file() else None,
            "sample_count": int(row_count),
            "candidate_count": int(candidate_count),
            "d_model": int(d_model),
            "candidate_features_model_shape": [int(x) for x in features.shape],
            "candidate_valid_mask_shape": [int(x) for x in valid.shape],
            "finite_features": bool(np.isfinite(features).all()),
            "valid_primary_labels": bool(valid_label_mask.all()),
            "invalid_primary_label_count": int((~valid_label_mask).sum()),
            "quality_keep_all": bool(quality.all()),
            "source_stage_counts": counter_from_array(source_stage),
            "source_stage_id_counts": counter_from_array(source_stage_id),
            "primary_label_policy_counts": counter_from_array(primary_policy),
            "split_id_counts": counter_from_array(split_id),
            "review_status_counts": counter_from_array(review_status),
            "expected_sample_count": 55,
            "expected_candidate_count": 64,
            "expected_d_model": 16,
            "provenance": {
                "stage4a70_original_primary_rows": int((source_stage == "stage4a70_original_primary").sum()),
                "stage4a714_compatible_imported_rows": int((source_stage == "stage4a714_compatible_import").sum()),
                "stage4a613_source_stage_id_rows": int((source_stage_id == 613).sum()),
                "stage4a714_source_stage_id_rows": int((source_stage_id == 714).sum()),
            },
            "lambda48_role": "shadow/baseline only",
            "lambda48_labels_present_as_shadow": bool(lambda_shadow.shape == labels.shape),
            "labels_recomputed_from_lambda48": False,
            "no_rejected_held_recheck_rows_included": not any(str(v).lower() in rejected_tokens for v in review_status.tolist()),
            "forbidden_key_matches": [
                key for key in keys for token in forbidden_tokens if token in key.lower()
            ],
            "dataset_keys": keys,
        }

    report["shape_matches_expected"] = (
        report["sample_count"] == report["expected_sample_count"]
        and report["candidate_count"] == report["expected_candidate_count"]
        and report["d_model"] == report["expected_d_model"]
    )
    report["provenance_matches_expected"] = (
        report["provenance"]["stage4a70_original_primary_rows"] == 30
        and report["provenance"]["stage4a714_compatible_imported_rows"] == 25
    )
    report["forbidden_fields_absent"] = report["forbidden_key_matches"] == []
    return report


def source_hash_report() -> dict[str, dict[str, Any]]:
    paths = {
        "stage4a712_summary_md": STAGE712_SUMMARY,
        "stage4a712_decision_json": STAGE712_DECISION,
        "stage4a712_option_matrix_json": STAGE712_OPTIONS,
        "stage4a712_risk_register_json": STAGE712_RISKS,
        "stage4a712_handoff_md": STAGE712_HANDOFF,
        "stage4a79_dataset_npz": DATASET,
        "stage4a79_summary_md": STAGE79_SUMMARY,
        "stage4a79_integrity_json": STAGE79_INTEGRITY,
        "stage4a79_lineage_json": STAGE79_LINEAGE,
    }
    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for key, path in paths.items()
    }


def checkpoint_like_outputs() -> list[str]:
    hits: list[str] = []
    if not OUT.exists():
        return hits
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(OUT))
        lower = rel.lower()
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".tar"}:
            hits.append(rel)
        elif any(token in lower for token in ["model_weights", "optimizer_state", "state_dict", "replay_buffer"]):
            hits.append(rel)
    return sorted(hits)


def build_html(summary: dict[str, Any], reports: dict[str, dict[str, Any]]) -> str:
    esc = lambda value: html.escape("" if value is None else str(value))
    cards = "".join(
        f"<div class='metric'><b>{esc(value)}</b><span>{esc(label)}</span></div>"
        for label, value in [
            ("decision", summary["decision"]),
            ("dataset rows", summary["sample_count"]),
            ("candidate shape", summary["candidate_shape"]),
            ("optimizer steps", summary["optimizer_step_count"]),
            ("checkpoint created", summary["checkpoint_created"]),
            ("lambda48 role", summary["lambda48_role"]),
        ]
    )
    report_rows = "".join(
        "<tr>"
        f"<td>{esc(name)}</td>"
        f"<td>{esc(report.get('decision', report.get('status', report.get('completed', 'ok'))))}</td>"
        f"<td>{esc(report.get('path', ''))}</td>"
        "</tr>"
        for name, report in reports.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage 4A-7.13 Checkpointed BC Design Preflight</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; color: #182027; background: #f7f8fa; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin-top: 24px; font-size: 20px; letter-spacing: 0; }}
    p {{ color: #5c6670; line-height: 1.5; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: #fff; border: 1px solid #d9dee5; border-radius: 8px; padding: 12px; min-height: 72px; }}
    .metric b {{ display: block; font-size: 20px; margin-bottom: 6px; overflow-wrap: anywhere; }}
    .boundary {{ background: #fff7ed; border: 1px solid #fdc991; border-radius: 8px; padding: 12px; color: #7a2e0e; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9dee5; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e6eaf0; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #eef2f6; }}
    code {{ background: #eef2f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Stage 4A-7.13 Checkpointed BC Design / Preflight</h1>
  <p>Generated at {esc(summary['generated_at_utc'])}. This packet defines a future bounded checkpointed BC experiment and does not execute it.</p>
  <section class="metrics">{cards}</section>
  <section class="boundary">
    No BC training, optimizer step, model save, checkpoint creation, Isaac startup, map_predict, rollout, runtime, label promotion, or RL/GDPO/PPO occurred in Stage 4A-7.13.
  </section>
  <h2>Future Approval Gate</h2>
  <p>The future execution phrase is stored in <code>exact_approval_phrase_for_stage4a714_checkpointed_bc_execution.md</code>. It must be supplied before any checkpointed BC execution.</p>
  <h2>Report Inventory</h2>
  <table><thead><tr><th>Report</th><th>Status</th><th>Path</th></tr></thead><tbody>{report_rows}</tbody></table>
</main>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    blockers: list[str] = []
    required_inputs = {
        "stage4a712_summary_md": STAGE712_SUMMARY,
        "stage4a712_decision_json": STAGE712_DECISION,
        "stage4a712_option_matrix_json": STAGE712_OPTIONS,
        "stage4a712_risk_register_json": STAGE712_RISKS,
        "stage4a712_handoff_md": STAGE712_HANDOFF,
        "stage4a79_dataset_npz": DATASET,
        "stage4a79_summary_md": STAGE79_SUMMARY,
        "stage4a79_integrity_json": STAGE79_INTEGRITY,
        "stage4a79_lineage_json": STAGE79_LINEAGE,
    }
    for key, path in required_inputs.items():
        if not path.exists():
            blockers.append(f"missing_input:{key}:{path}")

    decision = read_json(STAGE712_DECISION) if STAGE712_DECISION.exists() else {}
    option_matrix = read_json(STAGE712_OPTIONS) if STAGE712_OPTIONS.exists() else {}
    risk_register = read_json(STAGE712_RISKS) if STAGE712_RISKS.exists() else {"risks": []}
    integrity = read_json(STAGE79_INTEGRITY) if STAGE79_INTEGRITY.exists() else {}
    lineage = read_json(STAGE79_LINEAGE) if STAGE79_LINEAGE.exists() else {}
    dataset = load_dataset_preflight() if DATASET.exists() else {}

    expected_decision_checks = {
        "selected_option_b": decision.get("selected_option") == "B",
        "selected_next_stage_713": decision.get("selected_next_stage")
        == "Stage 4A-7.13 bounded checkpointed BC experiment design/preflight only",
        "approved_to_create_design_preflight": decision.get("approved_to_create_design_preflight") is True,
        "checkpointed_bc_not_approved_now": decision.get("approved_to_run_checkpointed_bc_now") is False,
        "checkpoint_save_not_approved_now": decision.get("approved_to_save_checkpoint_now") is False,
        "runtime_not_approved_now": decision.get("approved_to_run_runtime_now") is False,
        "rollout_not_approved_now": decision.get("approved_to_run_rollout_now") is False,
        "rl_not_approved_now": decision.get("approved_to_run_rl_gdpo_ppo_now") is False,
    }
    blockers.extend([f"stage4a712_decision_mismatch:{key}" for key, ok in expected_decision_checks.items() if not ok])

    label_lineage_ok = (
        lineage.get("lambda48_role") == "shadow/baseline only"
        and lineage.get("labels_recomputed_from_lambda48") is False
        and lineage.get("original_primary_policy") == "stage4a613_uncertainty_bonus_executed_primary"
        and lineage.get("compatible_import_policy")
        == "stage4a714_uncertainty_bonus_executed_clean_candidate_compatible_import"
    )
    if not label_lineage_ok:
        blockers.append("label_lineage_not_safe")

    if dataset:
        if not dataset.get("shape_matches_expected"):
            blockers.append("dataset_shape_mismatch")
        if not dataset.get("provenance_matches_expected"):
            blockers.append("dataset_provenance_mismatch")
        if not dataset.get("valid_primary_labels"):
            blockers.append("invalid_primary_labels")
        if not dataset.get("forbidden_fields_absent"):
            blockers.append("forbidden_fields_present")
        if not dataset.get("no_rejected_held_recheck_rows_included"):
            blockers.append("rejected_held_recheck_rows_present")

    loaded_decision_evidence = {
        "source_stage": "Stage 4A-7.12",
        "stage4a712_summary_path": str(STAGE712_SUMMARY),
        "selected_next_step_decision_path": str(STAGE712_DECISION),
        "decision_option_matrix_path": str(STAGE712_OPTIONS),
        "risk_register_path": str(STAGE712_RISKS),
        "handoff_path": str(STAGE712_HANDOFF),
        "selected_next_step_decision": decision,
        "selected_option_from_matrix": option_matrix.get("selected_option"),
        "risk_count": len(risk_register.get("risks", [])),
        "expected_decision_checks": expected_decision_checks,
        "all_expected_decision_checks_passed": all(expected_decision_checks.values()),
        "approved_to_create_design_preflight": decision.get("approved_to_create_design_preflight"),
        "approved_to_run_checkpointed_bc_now": decision.get("approved_to_run_checkpointed_bc_now"),
        "approved_to_save_checkpoint_now": decision.get("approved_to_save_checkpoint_now"),
        "approved_to_run_runtime_now": decision.get("approved_to_run_runtime_now"),
        "approved_to_run_rollout_now": decision.get("approved_to_run_rollout_now"),
        "approved_to_run_rl_gdpo_ppo_now": decision.get("approved_to_run_rl_gdpo_ppo_now"),
    }

    feature_names = [f"candidate_feature_{idx:02d}" for idx in range(int(dataset.get("d_model", 16) or 16))]
    model_design = {
        "status": "design_only_not_executed",
        "model_family": "CandidateMLPPolicy",
        "allowed_variants": [
            {
                "name": "CandidateMLPPolicy_small",
                "input_dim": 16,
                "hidden_dim": 64,
                "dropout": 0.0,
                "use_invalid_candidate_mask": True,
                "selection_reason": "matches Stage 4A-7.11 tiny dry-run and minimizes overfit risk",
            },
            {
                "name": "CandidateMLPPolicy_medium",
                "input_dim": 16,
                "hidden_dim": 128,
                "dropout": 0.05,
                "use_invalid_candidate_mask": True,
                "selection_reason": "only for a capped comparison if small model underfits",
            },
        ],
        "recommended_default": "CandidateMLPPolicy_small",
        "input_dim": 16,
        "candidate_count": 64,
        "invalid_candidate_mask_required": True,
        "score_space": "64-way masked candidate logits",
        "feature_names": feature_names,
    }

    split_policy = {
        "status": "design_only_not_executed",
        "primary_split": "use existing split_id from Stage 4A-7.9 artifact",
        "split_id_mapping": {"0": "train", "1": "val", "2": "test"},
        "observed_split_counts": dataset.get("split_id_counts", {}),
        "expected_counts": {"train": 39, "val": 10, "test": 6},
        "subgroup_balance_required": True,
        "fallback_diagnostic_split": "leave-one-start/source diagnostic only if primary split metrics are unstable",
        "no_dataset_modification": True,
    }

    training_plan = {
        "status": "design_only_not_executed",
        "checkpointed_bc_experiment_is_experimental_only": True,
        "dataset_path": str(DATASET),
        "loss": "masked 64-way cross entropy",
        "optimizer": "Adam or AdamW",
        "learning_rate_candidates": [0.001, 0.0003],
        "batch_size_candidates": [8, 16],
        "max_epochs_cap": 20,
        "max_optimizer_steps_cap": 200,
        "early_stopping": {
            "monitor": "validation CE loss with top3 tie-break",
            "patience_epochs": 3,
            "min_delta": 0.0001,
            "restore_best_validation_checkpoint": True,
        },
        "overfit_monitoring": [
            "train vs validation CE loss gap",
            "train vs validation top1/top3 gap",
            "original 30 vs imported 25 subgroup gap",
        ],
        "runtime_during_training": False,
        "rl_during_training": False,
        "this_stage_optimizer_steps": 0,
        "this_stage_training": False,
    }

    metric_plan = {
        "status": "design_only_not_executed",
        "primary_metrics": ["CE loss", "top1", "top3", "top5", "MRR"],
        "report_per_split": ["train", "val", "test", "all"],
        "acceptance_is_not_runtime_authorization": True,
        "must_report_invalid_mask_usage": True,
        "must_report_confusion_examples": True,
    }

    subgroup_metric_plan = {
        "status": "design_only_not_executed",
        "required_subgroups": {
            "original_stage4a70_primary_30": "source_stage == stage4a70_original_primary",
            "imported_stage4a714_clean_candidate_25": "source_stage == stage4a714_compatible_import",
        },
        "required_metrics": ["CE loss", "top1", "top3", "top5", "MRR", "sample_count"],
        "watch_items": [
            "imported subgroup should not collapse while original subgroup improves",
            "original subgroup should not be overwritten by compatible import rows",
        ],
    }

    checkpoint_policy = {
        "status": "policy_only_no_checkpoint_created",
        "future_checkpoint_dir": str(FUTURE_CHECKPOINT_DIR),
        "future_checkpoint_path": str(FUTURE_CHECKPOINT_PATH),
        "future_checkpoint_metadata_path": str(FUTURE_CHECKPOINT_METADATA),
        "checkpoint_dir_gitignore_check": git_check_ignored(FUTURE_CHECKPOINT_PATH),
        "checkpoint_output_under_ignored_directory": str(FUTURE_CHECKPOINT_PATH).startswith(str(ROOT / "checkpoints/")),
        "checkpoint_must_not_be_committed": True,
        "future_approval_required": True,
        "stage4a713_created_checkpoint": False,
        "stage4a713_created_checkpoint_dir": FUTURE_CHECKPOINT_DIR.exists(),
        "metadata_required_fields": {
            "dataset_path": str(DATASET),
            "dataset_sha256": dataset.get("dataset_sha256"),
            "git_commit": git_head(),
            "model_config": model_design["recommended_default"],
            "feature_names": feature_names,
            "label_policy": lineage,
            "lambda48_shadow_only_declaration": "lambda48 remains shadow/baseline only",
            "no_rl_declaration": "no RL/GDPO/PPO in checkpointed BC execution",
        },
        "rollback_policy": [
            "delete checkpoint file and metadata if validation fails",
            "do not promote labels from checkpoint metrics",
            "do not run runtime/rollout from a checkpoint unless a later gate approves it",
        ],
    }

    risk_control = {
        "status": "design_only_not_executed",
        "loaded_risks_from_stage4a712": risk_register.get("risks", []),
        "additional_controls": [
            "hard cap epochs and optimizer steps",
            "keep checkpoint directory ignored",
            "validate no outputs/logs/checkpoints are tracked",
            "require subgroup metrics for original vs imported rows",
            "require future explicit approval phrase before execution",
        ],
        "safety_policy": {
            "no_isaac": True,
            "no_map_predict": True,
            "no_rollout": True,
            "no_runtime": True,
            "no_rl_gdpo_ppo": True,
            "no_replay_buffer_learning": True,
            "no_dataset_modification": True,
            "no_label_promotion": True,
            "no_target_ground_truth_future_observed_fields": True,
            "no_prediction_uncertainty_writeback": True,
        },
    }

    no_training = {
        "training": False,
        "bc_training": False,
        "checkpointed_bc_execution": False,
        "optimizer_step_count": 0,
        "backward_count": 0,
        "replay_buffer_training": False,
        "dataset_modified": False,
        "label_promotion": False,
    }
    no_checkpoint = {
        "checkpoint_created": False,
        "model_saved": False,
        "torch_save_calls": 0,
        "checkpoint_dir_created_by_stage4a713": FUTURE_CHECKPOINT_DIR.exists(),
        "checkpoint_like_outputs": checkpoint_like_outputs(),
    }
    no_runtime = {
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "long_rollout": False,
        "runtime": False,
        "action_execution": False,
        "capture": False,
    }
    no_rl = {
        "rl": False,
        "gdpo": False,
        "ppo": False,
        "replay_buffer_learning": False,
    }
    forbidden_field = {
        "dataset_path": str(DATASET),
        "forbidden_tokens": [
            "target",
            "ground_truth",
            "future_observed",
            "prediction_writeback",
            "uncertainty_writeback",
        ],
        "forbidden_key_matches": dataset.get("forbidden_key_matches", []),
        "forbidden_fields_absent": dataset.get("forbidden_fields_absent", False),
        "prediction_uncertainty_writeback": False,
        "dataset_modified": False,
    }

    approval_phrase = (
        "Approve Stage 4A-7.14 bounded checkpointed BC execution: run bounded checkpointed BC "
        "experiment using the Stage 4A-7.9 compatible expanded dataset "
        f"{DATASET}; save checkpoint only to the ignored checkpoint dir {FUTURE_CHECKPOINT_DIR}; "
        "do not run runtime, rollout, or RL/GDPO/PPO; keep lambda48 shadow-only and do not "
        "recompute labels from lambda48."
    )
    future_sketch = [
        "DO NOT RUN IN STAGE 4A-7.13.",
        "",
        "Future Stage 4A-7.14 checkpointed BC execution sketch, gated by the exact approval phrase:",
        "",
        "1. Re-run dataset hash and label-lineage checks on the Stage 4A-7.9 compatible expanded dataset.",
        "2. Instantiate CandidateMLPPolicy_small by default with the invalid candidate mask enabled.",
        "3. Train only within the capped epoch/optimizer-step budget and record all split/subgroup metrics.",
        "4. Save exactly one checkpoint and metadata JSON under the ignored checkpoint directory if validation passes.",
        "5. Do not run Isaac, map_predict, rollout, runtime, label promotion, or RL/GDPO/PPO.",
        "6. If validation fails, delete any checkpoint artifact and keep the run as a failed experiment report.",
    ]
    recommended_next = [
        "Recommended next faithful step:",
        "",
        "- Human/web review this Stage 4A-7.13 design/preflight packet.",
        "- If accepted, provide the exact approval phrase before any Stage 4A-7.14 checkpointed BC execution.",
        "- Keep runtime, rollout, and RL behind later separate gates even if a checkpoint is produced.",
    ]

    completed = not blockers
    summary = {
        "stage": "Stage 4A-7.13",
        "mode": "bounded_checkpointed_bc_design_preflight_only",
        "generated_at_utc": utc_now(),
        "completed": completed,
        "blocked": bool(blockers),
        "blockers": blockers,
        "decision": "design_preflight_complete_ready_for_separate_stage4a714_checkpointed_bc_execution_review"
        if completed
        else "blocked",
        "selected_dataset": str(DATASET),
        "sample_count": dataset.get("sample_count"),
        "candidate_shape": dataset.get("candidate_features_model_shape"),
        "d_model": dataset.get("d_model"),
        "model": model_design["recommended_default"],
        "training_plan": "bounded checkpointed BC experiment design only; not executed",
        "split_policy": split_policy["primary_split"],
        "metric_plan": metric_plan["primary_metrics"],
        "checkpoint_policy": checkpoint_policy["status"],
        "lambda48_role": "shadow/baseline only",
        "labels_recomputed_from_lambda48": False,
        "bc_training": False,
        "optimizer_step_count": 0,
        "backward_count": 0,
        "checkpoint_created": False,
        "model_saved": False,
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "runtime": False,
        "rl_gdpo_ppo": False,
        "label_promotion": False,
        "exact_approval_phrase_path": str(OUT / "exact_approval_phrase_for_stage4a714_checkpointed_bc_execution.md"),
    }

    reports = {
        "loaded_stage4a712_decision_evidence": loaded_decision_evidence,
        "dataset_preflight": dataset,
        "model_design": model_design,
        "training_plan": training_plan,
        "checkpoint_policy": checkpoint_policy,
        "split_policy": split_policy,
        "metric_plan": metric_plan,
        "subgroup_metric_plan": subgroup_metric_plan,
        "risk_control": risk_control,
        "no_training": no_training,
        "no_checkpoint": no_checkpoint,
        "no_runtime": no_runtime,
        "no_rl_gdpo_ppo": no_rl,
        "forbidden_field_preflight": forbidden_field,
        "source_hash_report": source_hash_report(),
    }

    write_json(OUT / "loaded_stage4a712_decision_evidence.json", loaded_decision_evidence)
    write_md_table(
        OUT / "loaded_stage4a712_decision_evidence.md",
        "Loaded Stage 4A-7.12 Decision Evidence",
        [
            ("selected option", decision.get("selected_option")),
            ("selected next stage", decision.get("selected_next_stage")),
            ("approved to create design/preflight", decision.get("approved_to_create_design_preflight")),
            ("approved to run checkpointed BC now", decision.get("approved_to_run_checkpointed_bc_now")),
            ("approved to save checkpoint now", decision.get("approved_to_save_checkpoint_now")),
            ("approved to run runtime now", decision.get("approved_to_run_runtime_now")),
            ("approved to run rollout now", decision.get("approved_to_run_rollout_now")),
            ("approved to run RL/GDPO/PPO now", decision.get("approved_to_run_rl_gdpo_ppo_now")),
            ("all expected checks passed", loaded_decision_evidence["all_expected_decision_checks_passed"]),
        ],
    )

    for name, report in reports.items():
        if name == "loaded_stage4a712_decision_evidence":
            continue
        json_name = f"{name}_report.json" if not name.endswith("_report") else f"{name}.json"
        md_name = f"{name}_report.md" if not name.endswith("_report") else f"{name}.md"
        write_json(OUT / json_name, report)
        write_md_table(
            OUT / md_name,
            " ".join(part.capitalize() for part in name.replace("_report", "").split("_")) + " Report",
            [(key, value) for key, value in report.items() if key not in {"dataset_keys", "feature_names"}],
        )

    write_json(OUT / "stage4a713_checkpointed_bc_design_preflight_summary.json", summary)
    write_md_table(
        OUT / "stage4a713_checkpointed_bc_design_preflight_summary.md",
        "Stage 4A-7.13 Checkpointed BC Design/Preflight Summary",
        [
            ("completed", summary["completed"]),
            ("blocked", summary["blocked"]),
            ("decision", summary["decision"]),
            ("selected dataset", summary["selected_dataset"]),
            ("sample count", summary["sample_count"]),
            ("candidate shape", summary["candidate_shape"]),
            ("model", summary["model"]),
            ("checkpoint policy", summary["checkpoint_policy"]),
            ("lambda48 role", summary["lambda48_role"]),
            ("optimizer steps", summary["optimizer_step_count"]),
            ("checkpoint created", summary["checkpoint_created"]),
            ("runtime", summary["runtime"]),
            ("RL/GDPO/PPO", summary["rl_gdpo_ppo"]),
            ("blockers", summary["blockers"] or "none"),
        ],
        "This is a design/preflight packet only. It does not approve or execute checkpointed BC training.",
    )

    write_md_list(OUT / "exact_approval_phrase_for_stage4a714_checkpointed_bc_execution.md", "Exact Approval Phrase For Stage 4A-7.14 Checkpointed BC Execution", [approval_phrase])
    write_md_list(OUT / "future_stage4a714_checkpointed_bc_execution_sketch.md", "Future Stage 4A-7.14 Checkpointed BC Execution Sketch", future_sketch)
    write_md_list(OUT / "recommended_next_faithful_step.md", "Recommended Next Faithful Step", recommended_next)
    (OUT / "stage4a713_checkpointed_bc_design_preflight_index.html").write_text(
        build_html(summary, reports), encoding="utf-8"
    )

    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
