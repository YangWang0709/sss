from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a711_expanded_tiny_no_checkpoint_eval"
SUBAGENT_OUT = OUT / "subagent_reports"
STAGE79 = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation"
STAGE70 = ROOT / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
STAGE73 = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_no_training"

REQUIRED_REPORTS = [
    "stage4a711_expanded_tiny_eval_summary.json",
    "stage4a711_expanded_tiny_eval_summary.md",
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a710_gate_report.json",
    "loaded_stage4a710_gate_report.md",
    "expanded_dataset_load_report.json",
    "expanded_dataset_load_report.md",
    "expanded_eval_config.json",
    "expanded_eval_config.md",
    "forward_only_smoke_report.json",
    "forward_only_smoke_report.md",
    "split_plan_report.json",
    "split_plan_report.md",
    "tiny_split_eval_metrics.csv",
    "tiny_split_eval_metrics.json",
    "tiny_split_eval_metrics.md",
    "looso_fold_metrics.csv",
    "looso_fold_metrics.json",
    "looso_fold_metrics.md",
    "looso_aggregate_report.json",
    "looso_aggregate_report.md",
    "overfit_sanity_report.json",
    "overfit_sanity_report.md",
    "baseline_comparison_report.json",
    "baseline_comparison_report.md",
    "subgroup_original_vs_promoted_report.json",
    "subgroup_original_vs_promoted_report.md",
    "no_checkpoint_report.json",
    "no_checkpoint_report.md",
    "no_model_save_report.json",
    "no_model_save_report.md",
    "no_runtime_report.json",
    "no_runtime_report.md",
    "no_rl_gdpo_ppo_report.json",
    "no_rl_gdpo_ppo_report.md",
    "forbidden_field_training_audit.json",
    "forbidden_field_training_audit.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "prior_dataset_hash_report.json",
    "prior_dataset_hash_report.md",
    "git_status_before.txt",
    "git_status_after.txt",
    "future_stage4a712_controlled_bc_checkpoint_or_medium_rollout_decision_sketch.md",
    "recommended_next_faithful_step.md",
]
SUBAGENT_REPORTS = [
    "context_gate_agent_report.md",
    "dataset_split_agent_report.md",
    "training_smoke_agent_report.md",
    "looso_fold_eval_agent_report.md",
    "overfit_sanity_agent_report.md",
    "safety_no_checkpoint_agent_report.md",
    "qa_validator_agent_report.md",
]
PLOTS = [
    "expanded_tiny_loss_curve.png",
    "expanded_eval_topk_by_fold.png",
    "expanded_eval_loss_by_fold.png",
    "baseline_vs_expanded_metrics.png",
    "overfit_loss_curve.png",
    "original_vs_promoted_subgroup_metrics.png",
    "expanded_tiny_eval_index.html",
]
FORBIDDEN_FIELDS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "future_observed",
    "reward",
    "policy_logits",
    "replay_buffer",
    "optimizer",
    "training_state",
    "class_prob",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def scan_output_checkpoint_like() -> list[str]:
    hits = []
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(OUT))
        lower = rel.lower()
        if path.suffix.lower() in {".pth", ".pt", ".ckpt", ".tar"}:
            hits.append(rel)
        elif any(token in lower for token in ["state_dict", "model_weights", "optimizer_state", "replay_buffer"]):
            hits.append(rel)
    return sorted(hits)


def write_validator_report(result: dict) -> None:
    lines = [
        "# QA Validator Agent Report",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| all_passed | {result['all_passed']} |",
        f"| blockers | {', '.join(result['blockers']) or 'none'} |",
        f"| expanded samples | {result.get('expanded_samples')} |",
        f"| fold count | {result.get('fold_count')} |",
    ]
    SUBAGENT_OUT.mkdir(parents=True, exist_ok=True)
    (SUBAGENT_OUT / "qa_validator_agent_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["output_dir_exists"] = OUT.is_dir()
    checks["subagent_dir_exists"] = SUBAGENT_OUT.is_dir()
    for name in REQUIRED_REPORTS:
        checks[f"required:{name}"] = (OUT / name).is_file()
    for name in SUBAGENT_REPORTS:
        checks[f"subagent:{name}"] = (SUBAGENT_OUT / name).is_file()
    for name in PLOTS:
        checks[f"plot_or_index:{name}"] = (OUT / name).is_file()

    if not checks["output_dir_exists"] or not all(v for k, v in checks.items() if k.startswith(("required:", "subagent:"))):
        for key, ok in checks.items():
            if not ok:
                blockers.append(key)
        result = {"all_passed": False, "blockers": blockers, "checks": checks}
        print(json.dumps(result, indent=2, sort_keys=True))
        write_validator_report(result)
        return 1

    summary = load_json(OUT / "stage4a711_expanded_tiny_eval_summary.json")
    gate = load_json(OUT / "loaded_stage4a710_gate_report.json")
    dataset_load = load_json(OUT / "expanded_dataset_load_report.json")
    config = load_json(OUT / "expanded_eval_config.json")
    forward = load_json(OUT / "forward_only_smoke_report.json")
    split_plan = load_json(OUT / "split_plan_report.json")
    tiny = load_json(OUT / "tiny_split_eval_metrics.json")
    fold_rows = load_json(OUT / "looso_fold_metrics.json")
    looso = load_json(OUT / "looso_aggregate_report.json")
    overfit = load_json(OUT / "overfit_sanity_report.json")
    baseline = load_json(OUT / "baseline_comparison_report.json")
    subgroup = load_json(OUT / "subgroup_original_vs_promoted_report.json")
    no_checkpoint = load_json(OUT / "no_checkpoint_report.json")
    no_model_save = load_json(OUT / "no_model_save_report.json")
    no_runtime = load_json(OUT / "no_runtime_report.json")
    no_rl = load_json(OUT / "no_rl_gdpo_ppo_report.json")
    forbidden = load_json(OUT / "forbidden_field_training_audit.json")
    source_hash = load_json(OUT / "source_hash_report.json")
    prior_hash = load_json(OUT / "prior_dataset_hash_report.json")

    expanded_path = STAGE79 / "expanded_primary_bc_dataset.npz"
    expanded = np.load(expanded_path, allow_pickle=False)
    original = np.load(STAGE70 / "bc_dataset_primary_short_rollout.npz", allow_pickle=False)
    adapter = np.load(STAGE73 / "stage4a72_compact_v1_candidate_feature_adapter.npz", allow_pickle=False)

    checks["summary_completed"] = summary.get("completed") is True and summary.get("blocked") is False
    checks["gate_passed"] = gate.get("passed") is True
    checks["dataset_load_passed"] = dataset_load.get("passed") is True
    checks["sample_count_47"] = expanded["sample_id"].shape[0] == 47 and summary.get("samples") == 47
    checks["candidate_count_64"] = expanded["candidate_features_model"].shape[1] == 64 and summary.get("candidate_count") == 64
    checks["D_model_16"] = expanded["candidate_features_model"].shape[2] == 16 and summary.get("D_model") == 16
    checks["valid_labels"] = dataset_load.get("valid_labels") is True
    labels = expanded["expert_action_index_primary"].astype(np.int64)
    valid = expanded["candidate_valid_mask"].astype(bool)
    checks["candidate_valid_mask_at_label"] = bool(np.all(valid[np.arange(labels.shape[0]), labels]))
    checks["labels_not_lambda48_primary"] = (
        summary.get("lambda48_primary_use") is False
        and forbidden.get("labels_not_recomputed_from_lambda48") is True
        and forbidden.get("lambda48_shadow_baseline_only") is True
    )
    checks["first_30_original_retained"] = (
        np.array_equal(expanded["sample_id"][:30], original["sample_id"])
        and np.array_equal(expanded["expert_action_index_primary"][:30], original["expert_action_index_primary"])
    )
    checks["promoted_17_from_adapter"] = expanded["sample_id"][30:].shape[0] == 17
    checks["forward_ce_finite"] = is_finite_number(forward.get("ce_loss")) and forward.get("passed") is True
    checks["forward_only_has_no_backward_optimizer"] = forward.get("backward") is False and forward.get("optimizer") is False
    checks["tiny_metrics_exist"] = len(read_csv(OUT / "tiny_split_eval_metrics.csv")) >= 4 and tiny.get("passed") is True
    checks["looso_ran"] = len(fold_rows) == split_plan.get("fold_count") == 10 and looso.get("passed") is True
    metric_values = []
    for row in fold_rows:
        metric_values.extend([row["eval_loss"], row["eval_top1"], row["eval_top3"], row["eval_top5"], row["eval_mrr"]])
    metric_values.extend(
        [
            looso.get("eval_loss_mean"),
            looso.get("eval_loss_stdev"),
            looso.get("eval_top1"),
            looso.get("eval_top3"),
            looso.get("eval_top5"),
            looso.get("eval_mrr"),
            tiny.get("initial_train_loss"),
            tiny.get("final_train_loss"),
            overfit.get("initial_loss"),
            overfit.get("final_loss"),
            overfit.get("final_top1"),
        ]
    )
    checks["metrics_finite"] = all(is_finite_number(v) for v in metric_values)
    checks["overfit_report_passed"] = overfit.get("passed") is True
    checks["baseline_comparison_exists"] = baseline.get("no_generalization_claim") is True
    checks["subgroup_report_exists"] = subgroup.get("passed") is True
    checks["backward_optimizer_counts_tiny_only"] = (
        int(summary.get("num_backward_calls", -1)) > 0
        and int(summary.get("num_optimizer_steps", -1)) > 0
        and int(summary.get("num_backward_calls")) == int(summary.get("num_optimizer_steps"))
        and int(summary.get("num_optimizer_steps")) <= 200
    )
    checks["model_saved_false"] = summary.get("model_saved") is False and no_model_save.get("model_saved") is False
    checks["checkpoint_created_false"] = summary.get("checkpoint_created") is False and no_checkpoint.get("checkpoint_created") is False
    checks["no_checkpoint_like_files_in_output"] = scan_output_checkpoint_like() == []
    checks["no_runtime"] = (
        no_runtime.get("isaac_startup") is False
        and no_runtime.get("capture") is False
        and no_runtime.get("map_predict") is False
        and no_runtime.get("sscnet_inference") is False
        and no_runtime.get("action_execution") is False
        and no_runtime.get("rollout") is False
        and no_runtime.get("long_rollout") is False
    )
    checks["no_rl_gdpo_ppo"] = no_rl.get("rl") is False and no_rl.get("gdpo") is False and no_rl.get("ppo") is False
    checks["forbidden_field_audit_passed"] = (
        forbidden.get("passed") is True and forbidden.get("target_ground_truth_future_fields_used") is False
    )
    checks["future_sketch_first_line"] = (
        (OUT / "future_stage4a712_controlled_bc_checkpoint_or_medium_rollout_decision_sketch.md").read_text(encoding="utf-8").splitlines()[0]
        == "DO NOT RUN IN STAGE 4A-7.11."
    )
    checks["prior_hashes_unchanged"] = all(item.get("unchanged") is True for item in prior_hash.values() if item.get("exists"))
    checks["source_hashes_unchanged"] = all(item.get("unchanged") is True for item in source_hash.values() if item.get("exists"))
    checks["expanded_dataset_unchanged"] = prior_hash["stage4a79_expanded_dataset"].get("unchanged") is True
    checks["config_no_save"] = (
        config.get("save_checkpoint") is False
        and config.get("save_model") is False
        and config.get("save_state_dict") is False
        and config.get("full_training") is False
    )

    tracked = git_files()
    forbidden_tracked = [
        path
        for path in tracked
        if path.startswith(("outputs/", "logs/", "checkpoints/", "assets/"))
        or path.lower().endswith((".npz", ".npy", ".png", ".mp4", ".usd", ".usda", ".usdc", ".pt", ".pth", ".ckpt", ".tar"))
    ]
    large_tracked = [
        path
        for path in tracked
        if (ROOT / path).is_file() and (ROOT / path).stat().st_size > 50 * 1024 * 1024
    ]
    checks["git_large_artifact_policy_preserved"] = not forbidden_tracked and not large_tracked

    for key, ok in checks.items():
        if not ok:
            blockers.append(key)

    result = {
        "all_passed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "expanded_samples": int(expanded["sample_id"].shape[0]),
        "fold_count": len(fold_rows),
        "forbidden_tracked": forbidden_tracked,
        "large_tracked": large_tracked,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    write_validator_report(result)
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
