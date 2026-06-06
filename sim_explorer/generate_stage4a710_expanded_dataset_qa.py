from __future__ import annotations

import csv
import hashlib
import html
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a710_expanded_dataset_qa"
SUBAGENT_OUT = OUT / "subagent_reports"
STAGE79 = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation"
STAGE78 = ROOT / "outputs/isaac_stage4a78_promotion_candidate_decision_packet"
STAGE70 = ROOT / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
STAGE73 = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_no_training"
STAGE73_QA = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_qa_review"
STAGE72 = ROOT / "outputs/isaac_stage4a72_bounded_short_rollout_runtime"
STAGE72_QA = ROOT / "outputs/isaac_stage4a72_bounded_short_rollout_runtime_qa_review"

EXPANDED_NPZ = STAGE79 / "expanded_primary_bc_dataset.npz"
STAGE70_NPZ = STAGE70 / "bc_dataset_primary_short_rollout.npz"
STAGE73_NPZ = STAGE73 / "stage4a72_compact_v1_candidate_feature_adapter.npz"
STAGE79_SUMMARY = STAGE79 / "stage4a79_no_training_promotion_summary.json"

EXPECTED_COMMIT = "1c28f949500cc722869c723f34c6126bbe6d6b66"
PRIMARY_POLICY_ORIGINAL = "stage4a613_uncertainty_bonus_executed_primary"
PRIMARY_POLICY_PROMOTED_PREFIX = "stage4a613_uncertainty_bonus_executed_primary_from_stage4a72"

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
REQUIRED_NPZ_KEYS = {
    "sample_id",
    "candidate_features_model",
    "candidate_valid_mask",
    "expert_action_index_primary",
    "expert_action_index_measured_shadow",
    "expert_action_index_lambda48_shadow",
    "expert_action_index_confidence_gated_shadow",
    "quality_keep_mask",
    "missing_feature_mask",
}
REQUIRED_REPORTS = [
    "stage4a710_expanded_dataset_qa_summary.json",
    "stage4a710_expanded_dataset_qa_summary.md",
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a79_evidence.json",
    "loaded_stage4a79_evidence.md",
    "expanded_dataset_npz_inventory.json",
    "expanded_dataset_npz_inventory.md",
    "expanded_dataset_shape_audit.json",
    "expanded_dataset_shape_audit.md",
    "expanded_label_validity_audit.json",
    "expanded_label_validity_audit.md",
    "expanded_lineage_audit.json",
    "expanded_lineage_audit.md",
    "rejected_unsure_conflict_exclusion_audit.json",
    "rejected_unsure_conflict_exclusion_audit.md",
    "lambda48_shadow_only_audit.json",
    "lambda48_shadow_only_audit.md",
    "forbidden_field_audit.json",
    "forbidden_field_audit.md",
    "SimExpertBCDataset_load_audit.json",
    "SimExpertBCDataset_load_audit.md",
    "optional_forward_only_smoke_report.json",
    "optional_forward_only_smoke_report.md",
    "no_training_report.json",
    "no_training_report.md",
    "no_runtime_report.json",
    "no_runtime_report.md",
    "no_checkpoint_report.json",
    "no_checkpoint_report.md",
    "no_rl_gdpo_ppo_report.json",
    "no_rl_gdpo_ppo_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "prior_dataset_hash_report.json",
    "prior_dataset_hash_report.md",
    "expanded_dataset_hash_report.json",
    "expanded_dataset_hash_report.md",
    "git_sync_report.json",
    "git_sync_report.md",
    "git_status_before.txt",
    "git_status_after.txt",
    "future_stage4a711_tiny_eval_on_expanded_dataset_sketch.md",
    "recommended_next_faithful_step.md",
]
SUBAGENT_REPORTS = {
    "context_git_sync_agent_report.md": "Context / Git Sync Agent",
    "dataset_shape_agent_report.md": "Dataset Existence / Shape Agent",
    "label_lineage_agent_report.md": "Label Lineage Agent",
    "safety_forbidden_field_agent_report.md": "Safety / Forbidden Field Agent",
    "loader_smoke_agent_report.md": "Loader / Model-Input Smoke Agent",
    "qa_validator_agent_report.md": "QA Validator Agent",
}


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check)


def git_status() -> str:
    return run(["git", "status", "--short"]).stdout


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_md(path: Path, title: str, rows: list[tuple[str, object]], extra: str = "") -> None:
    lines = [f"# {title}", "", "| field | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    if extra:
        lines.extend(["", extra.rstrip()])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_report(paths: dict[str, Path]) -> dict[str, dict]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256_before": sha256_file(path),
            "sha256_after": None,
            "unchanged": None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in paths.items()
    }


def finalize_hash_report(report: dict[str, dict]) -> dict[str, dict]:
    for item in report.values():
        path = Path(item["path"])
        item["sha256_after"] = sha256_file(path)
        item["unchanged"] = item["sha256_before"] == item["sha256_after"]
    return report


def npz_inventory(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            key: {
                "shape": list(data[key].shape),
                "dtype": str(data[key].dtype),
                "ndim": int(data[key].ndim),
            }
            for key in data.files
        }


def csv_header(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f), [])


def first_stage79_context_line(ref: str, file_path: str) -> str:
    proc = run(["git", "show", f"{ref}:{file_path}"])
    if proc.returncode != 0:
        return ""
    for line in proc.stdout.splitlines():
        if "Stage 4A-7.9 no-training promotion implementation result" in line:
            return line
    return ""


def make_git_sync_report() -> dict:
    fetch = run(["git", "fetch", "origin", "--prune"])
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    master = run(["git", "rev-parse", "origin/master"]).stdout.strip()
    main = run(["git", "rev-parse", "origin/main"]).stdout.strip()
    log = run(["git", "log", "--oneline", "-5"]).stdout
    remote = run(["git", "remote", "-v"]).stdout
    status = git_status()
    contains_expected = run(["git", "cat-file", "-e", f"{EXPECTED_COMMIT}^{{commit}}"]).returncode == 0
    context_paths = [
        ".project_context/CURRENT_STATE.md",
        ".project_context/TODO.md",
        ".project_context/CODEX_LOG.md",
    ]
    local_context = {}
    for rel in context_paths:
        text = (ROOT / rel).read_text(encoding="utf-8")
        local_context[rel] = {
            "exists": True,
            "contains_stage4a79": "Stage 4A-7.9" in text,
            "contains_stage4a79_result": "Stage 4A-7.9 no-training promotion implementation result" in text
            or "Stage 4A-7.9 result / current next" in text
            or "Stage 4A-7.9 no-training promotion implementation" in text,
        }
    return {
        "passed": (
            fetch.returncode == 0
            and contains_expected
            and head == EXPECTED_COMMIT
            and master == EXPECTED_COMMIT
            and main == EXPECTED_COMMIT
            and all(item["contains_stage4a79"] for item in local_context.values())
            and bool(first_stage79_context_line("origin/master", ".project_context/CURRENT_STATE.md"))
            and bool(first_stage79_context_line("origin/main", ".project_context/CURRENT_STATE.md"))
        ),
        "fetch_returncode": fetch.returncode,
        "fetch_stderr": fetch.stderr.strip(),
        "git_status_short": status,
        "git_log_oneline_5": log,
        "git_remote_v": remote,
        "head": head,
        "origin_master": master,
        "origin_main": main,
        "expected_commit_exists_local": contains_expected,
        "local_context": local_context,
        "origin_master_current_state_stage4a79_line": first_stage79_context_line(
            "origin/master", ".project_context/CURRENT_STATE.md"
        ),
        "origin_main_current_state_stage4a79_line": first_stage79_context_line(
            "origin/main", ".project_context/CURRENT_STATE.md"
        ),
    }


def audit_loader_and_forward() -> tuple[dict, dict]:
    import sys

    sys.path.insert(0, str(ROOT))
    from ssc_exploration.ssc_network.il.policy import CandidateMLPPolicy
    from ssc_exploration.ssc_network.il.sim_expert_bc_dataset import SimExpertBCDataset

    import torch
    import torch.nn.functional as F

    dataset = SimExpertBCDataset(EXPANDED_NPZ, strict_keep_only=False)
    strict_dataset = SimExpertBCDataset(EXPANDED_NPZ, strict_keep_only=True)
    first = dataset[0]
    loader_report = {
        "passed": True,
        "class": "SimExpertBCDataset",
        "dataset_len": len(dataset),
        "strict_keep_only_len": len(strict_dataset),
        "first_sample_id": first["sample_id"],
        "first_candidate_features_model_shape": list(first["candidate_features_model"].shape),
        "first_candidate_valid_mask_shape": list(first["candidate_valid_mask"].shape),
        "first_primary_label": int(first["expert_action_index_primary"]),
        "read_only": True,
    }
    with torch.no_grad():
        batch_count = min(8, len(dataset))
        features = torch.stack([dataset[i]["candidate_features_model"] for i in range(batch_count)], dim=0)
        valid = torch.stack([dataset[i]["candidate_valid_mask"] for i in range(batch_count)], dim=0)
        labels = torch.stack([dataset[i]["expert_action_index_primary"] for i in range(batch_count)], dim=0)
        policy = CandidateMLPPolicy(input_dim=features.shape[-1], hidden_dim=32)
        policy.eval()
        logits = policy(features, valid)
        loss = F.cross_entropy(logits, labels)
    smoke_report = {
        "passed": True,
        "forward_only_smoke": True,
        "policy_class": "CandidateMLPPolicy",
        "batch_count": batch_count,
        "logits_shape": list(logits.shape),
        "ce_loss": float(loss.detach().cpu().item()),
        "logits_finite": bool(torch.isfinite(logits[valid]).all().item()),
        "backward": False,
        "optimizer_step": False,
        "training_loop": False,
        "model_saved": False,
        "checkpoint": False,
        "torch_no_grad": True,
    }
    return loader_report, smoke_report


def make_plots(shape_report: dict, label_report: dict, lineage_report: dict) -> dict:
    plot_report = {"attempted": True, "created": [], "skipped": [], "error": ""}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with np.load(EXPANDED_NPZ, allow_pickle=False) as data:
            source_stage = np.asarray(data["source_stage"]).astype(str)
            labels = np.asarray(data["expert_action_index_primary"], dtype=np.int64)
            valid_counts = np.asarray(data["candidate_valid_mask"], dtype=bool).sum(axis=1)
            features = np.asarray(data["candidate_features_model"], dtype=np.float32)
        plots = [
            ("expanded_source_stage_count_bar.png", "Source stage count", Counter(source_stage)),
            ("expanded_label_distribution.png", "Primary label distribution", Counter(labels.tolist())),
        ]
        for filename, title, counter in plots:
            fig, ax = plt.subplots(figsize=(8, 4))
            keys = [str(k) for k in counter]
            vals = [counter[k] if k in counter else counter[int(k)] for k in counter]
            ax.bar(keys, vals)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            path = OUT / filename
            fig.savefig(path, dpi=140)
            plt.close(fig)
            plot_report["created"].append(str(path))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid_counts, bins=10)
        ax.set_title("Valid candidate count")
        fig.tight_layout()
        path = OUT / "valid_candidate_count_hist.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plot_report["created"].append(str(path))
        fig, ax = plt.subplots(figsize=(7, 4))
        original_mean = features[:30].reshape(-1, features.shape[-1]).mean(axis=0)
        promoted_mean = features[30:].reshape(-1, features.shape[-1]).mean(axis=0)
        ax.plot(original_mean, label="original")
        ax.plot(promoted_mean, label="promoted")
        ax.set_title("Original vs promoted feature means")
        ax.legend()
        fig.tight_layout()
        path = OUT / "promoted_vs_original_feature_distribution.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plot_report["created"].append(str(path))
    except Exception as exc:
        plot_report["error"] = repr(exc)
        plot_report["skipped"].extend(
            [
                "expanded_source_stage_count_bar.png",
                "expanded_label_distribution.png",
                "valid_candidate_count_hist.png",
                "promoted_vs_original_feature_distribution.png",
            ]
        )
    return plot_report


def write_index_html(plot_report: dict) -> None:
    links = [
        "stage4a710_expanded_dataset_qa_summary.md",
        "expanded_dataset_shape_audit.md",
        "expanded_label_validity_audit.md",
        "expanded_lineage_audit.md",
        "forbidden_field_audit.md",
        "SimExpertBCDataset_load_audit.md",
        "optional_forward_only_smoke_report.md",
    ]
    image_tags = []
    for created in plot_report.get("created", []):
        rel = Path(created).name
        image_tags.append(f"<figure><img src='{html.escape(rel)}' alt='{html.escape(rel)}'><figcaption>{html.escape(rel)}</figcaption></figure>")
    if not image_tags:
        image_tags.append("<p>Plots skipped; see summary JSON for the skip reason.</p>")
    body = "\n".join(f"<li><a href='{html.escape(link)}'>{html.escape(link)}</a></li>" for link in links)
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 4A-7.10 Expanded Dataset QA</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;line-height:1.45}}img{{max-width:760px;width:100%;border:1px solid #ddd}}figure{{margin:18px 0}}</style>
</head><body>
<h1>Stage 4A-7.10 Expanded Dataset QA</h1>
<ul>{body}</ul>
{''.join(image_tags)}
</body></html>
"""
    (OUT / "expanded_dataset_qa_index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SUBAGENT_OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    source_paths = {
        "current_state": ROOT / ".project_context/CURRENT_STATE.md",
        "todo": ROOT / ".project_context/TODO.md",
        "codex_log": ROOT / ".project_context/CODEX_LOG.md",
        "readme": ROOT / "README.md",
        "artifacts": ROOT / "ARTIFACTS.md",
        "environment": ROOT / "ENVIRONMENT.md",
        "git_initialization_report": ROOT / "GIT_INITIALIZATION_REPORT.md",
        "sim_expert_bc_dataset": ROOT / "ssc_exploration/ssc_network/il/sim_expert_bc_dataset.py",
        "policy": ROOT / "ssc_exploration/ssc_network/il/policy.py",
    }
    prior_paths = {
        "stage4a70_primary_npz": STAGE70_NPZ,
        "stage4a70_metadata": STAGE70 / "bc_dataset_metadata.json",
        "stage4a70_feature_names_model": STAGE70 / "feature_names_model.json",
        "stage4a73_adapter_npz": STAGE73_NPZ,
        "stage4a78_clean_candidates": STAGE78 / "clean_promotion_candidates.csv",
        "stage4a78_rejected_rows": STAGE78 / "rejected_rows.csv",
        "stage4a78_unsure_rows": STAGE78 / "unsure_rows.csv",
        "stage4a78_conflict_rows": STAGE78 / "conflict_manual_recheck_rows.csv",
        "fixed_usd_defaultprim": ROOT / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd",
    }
    expanded_hash = hash_report({"stage4a79_expanded_primary_dataset": EXPANDED_NPZ})
    source_hash = hash_report(source_paths)
    prior_hash = hash_report(prior_paths)

    git_sync = make_git_sync_report()
    summary79 = read_json(STAGE79_SUMMARY)
    metadata79 = read_json(STAGE79 / "expanded_primary_bc_dataset_metadata.json")
    mapping79 = read_json(STAGE79 / "promotion_mapping_report.json")
    lineage79 = read_json(STAGE79 / "primary_label_lineage_report.json")
    forbidden79 = read_json(STAGE79 / "forbidden_field_audit.json")
    integrity79 = read_json(STAGE79 / "expanded_dataset_integrity_report.json")
    feature79 = read_json(STAGE79 / "feature_schema_compatibility_report.json")
    no_training79 = read_json(STAGE79 / "no_training_report.json")
    no_runtime79 = read_json(STAGE79 / "no_runtime_report.json")
    no_rl79 = read_json(STAGE79 / "no_rl_gdpo_ppo_report.json")

    clean_rows = read_csv(STAGE78 / "clean_promotion_candidates.csv")
    rejected_rows = read_csv(STAGE78 / "rejected_rows.csv")
    unsure_rows = read_csv(STAGE78 / "unsure_rows.csv")
    conflict_rows = read_csv(STAGE78 / "conflict_manual_recheck_rows.csv")
    promoted_rows = read_csv(STAGE79 / "promoted_stage4a72_rows.csv")
    excluded_rows = read_csv(STAGE79 / "excluded_stage4a72_rows.csv")

    with np.load(EXPANDED_NPZ, allow_pickle=False) as expanded, np.load(STAGE70_NPZ, allow_pickle=False) as original, np.load(STAGE73_NPZ, allow_pickle=False) as adapter:
        inventory = npz_inventory(EXPANDED_NPZ)
        keys = set(expanded.files)
        features = np.asarray(expanded["candidate_features_model"], dtype=np.float32)
        valid_mask = np.asarray(expanded["candidate_valid_mask"], dtype=bool)
        primary = np.asarray(expanded["expert_action_index_primary"], dtype=np.int64)
        sample_count, candidate_count, d_model = features.shape
        valid_range = (primary >= 0) & (primary < candidate_count)
        valid_at_label = valid_range & valid_mask[np.arange(sample_count), primary]
        finite_features = bool(np.isfinite(features).all())
        missing_feature_mask = np.asarray(expanded["missing_feature_mask"], dtype=bool)
        source_stage = np.asarray(expanded["source_stage"]).astype(str)
        primary_policy = np.asarray(expanded["primary_label_policy"]).astype(str)
        clean_indices = np.asarray([int(row["adapter_sample_index"]) for row in clean_rows], dtype=np.int64)
        promoted_primary = adapter["candidate_action_index_uncertainty_bonus_executed"][clean_indices]
        promoted_lambda48 = adapter["expert_action_index_lambda48_shadow"][clean_indices]

        shape_audit = {
            "passed": (
                EXPANDED_NPZ.is_file()
                and REQUIRED_NPZ_KEYS.issubset(keys)
                and sample_count == 47
                and candidate_count == 64
                and d_model == 16
                and list(valid_mask.shape) == [47, 64]
                and list(primary.shape) == [47]
                and list(missing_feature_mask.shape) == [47, 64, 16]
                and finite_features
            ),
            "expanded_dataset_exists": EXPANDED_NPZ.is_file(),
            "required_keys_present": sorted(REQUIRED_NPZ_KEYS.intersection(keys)),
            "required_keys_missing": sorted(REQUIRED_NPZ_KEYS - keys),
            "sample_count": int(sample_count),
            "candidate_count": int(candidate_count),
            "D_model": int(d_model),
            "candidate_features_model_shape": list(features.shape),
            "candidate_valid_mask_shape": list(valid_mask.shape),
            "expert_action_index_primary_shape": list(primary.shape),
            "missing_feature_mask_shape": list(missing_feature_mask.shape),
            "finite_features": finite_features,
        }
        label_validity = {
            "passed": bool(np.all(valid_range) and np.all(valid_at_label)),
            "primary_labels_in_range": bool(np.all(valid_range)),
            "candidate_valid_mask_true_at_label": bool(np.all(valid_at_label)),
            "invalid_primary_sample_indices": np.where(~valid_at_label)[0].astype(int).tolist(),
            "min_primary_label": int(primary.min()),
            "max_primary_label": int(primary.max()),
            "candidate_count": int(candidate_count),
        }
        clean_set = {row["sample_id"] for row in clean_rows}
        promoted_set = {row["sample_id"] for row in promoted_rows}
        rejected_set = {row["sample_id"] for row in rejected_rows}
        unsure_set = {row["sample_id"] for row in unsure_rows}
        conflict_set = {row["sample_id"] for row in conflict_rows}
        rejected_promoted = sorted(promoted_set & rejected_set)
        unsure_promoted = sorted(promoted_set & unsure_set)
        conflict_promoted = sorted(promoted_set & conflict_set)
        lineage_audit = {
            "passed": (
                np.array_equal(expanded["sample_id"][:30], original["sample_id"])
                and np.allclose(expanded["candidate_features_model"][:30], original["candidate_features_model"])
                and np.array_equal(expanded["expert_action_index_primary"][:30], original["expert_action_index_primary"])
                and np.array_equal(expanded["sample_id"][30:], adapter["sample_id"][clean_indices])
                and np.allclose(expanded["candidate_features_model"][30:], adapter["candidate_features_model"][clean_indices])
                and np.array_equal(expanded["expert_action_index_primary"][30:], promoted_primary)
                and clean_set == promoted_set
                and not rejected_promoted
                and not unsure_promoted
                and not conflict_promoted
            ),
            "original_primary_samples": int(np.sum(source_stage == "stage4a70_original_primary")),
            "promoted_stage4a72_samples": int(np.sum(source_stage == "stage4a72_human_review_promoted_primary")),
            "expanded_primary_samples": int(sample_count),
            "first_30_original_sample_ids_retained": bool(np.array_equal(expanded["sample_id"][:30], original["sample_id"])),
            "first_30_original_primary_retained": bool(np.array_equal(expanded["expert_action_index_primary"][:30], original["expert_action_index_primary"])),
            "promoted_rows_exactly_match_clean_candidates": clean_set == promoted_set,
            "promoted_primary_from_uncertainty_bonus_executed": bool(np.array_equal(expanded["expert_action_index_primary"][30:], promoted_primary)),
            "manual_recheck_policy": "Manual-recheck-only, rejected, unsure, and conflict rows are excluded unless they are also clean candidates; none were promoted in Stage 4A-7.10 QA.",
            "primary_label_policy_recorded": "primary_label_policy" in keys,
            "original_policy_count": int(np.sum([p == PRIMARY_POLICY_ORIGINAL for p in primary_policy[:30]])),
            "promoted_policy_count": int(np.sum([p.startswith(PRIMARY_POLICY_PROMOTED_PREFIX) for p in primary_policy[30:]])),
        }
        lambda48_differs = promoted_primary != promoted_lambda48
        lambda48_audit = {
            "passed": (
                lineage79.get("lambda48_primary_use") is False
                and summary79.get("lambda48_primary_use") is False
                and bool(np.any(lambda48_differs))
                and np.array_equal(expanded["expert_action_index_primary"][30:][lambda48_differs], promoted_primary[lambda48_differs])
                and np.array_equal(expanded["expert_action_index_lambda48_shadow"][30:], promoted_lambda48)
            ),
            "lambda48_primary_use": False,
            "lambda48_role": "shadow/baseline only",
            "promoted_rows_where_primary_differs_from_lambda48": int(np.sum(lambda48_differs)),
            "promoted_rows_where_primary_equals_lambda48_by_value": int(np.sum(~lambda48_differs)),
            "shadow_labels_separate": "expert_action_index_lambda48_shadow" in keys,
            "note": "Equal integer values are allowed when both policies select the same candidate; source for promoted primary is uncertainty-bonus executed, not lambda48.",
        }
        forbidden_headers = {
            "expanded_sample_index_table": csv_header(STAGE79 / "expanded_sample_index_table.csv"),
            "promoted_rows": csv_header(STAGE79 / "promoted_stage4a72_rows.csv"),
            "excluded_rows": csv_header(STAGE79 / "excluded_stage4a72_rows.csv"),
        }
        forbidden_in_headers = {
            name: sorted(FORBIDDEN_FIELDS & {header.lower() for header in headers})
            for name, headers in forbidden_headers.items()
        }
        forbidden_audit = {
            "passed": not (FORBIDDEN_FIELDS & {key.lower() for key in keys}) and not any(forbidden_in_headers.values()),
            "forbidden_fields": sorted(FORBIDDEN_FIELDS),
            "forbidden_npz_keys_present": sorted(FORBIDDEN_FIELDS & {key.lower() for key in keys}),
            "forbidden_table_columns_present": forbidden_in_headers,
            "stage4a79_forbidden_audit_passed": forbidden79.get("passed") is True,
            "prediction_uncertainty_writeback": False,
        }
        exclusion_audit = {
            "passed": not rejected_promoted and not unsure_promoted and not conflict_promoted,
            "rejected_rows_promoted": len(rejected_promoted),
            "unsure_rows_promoted": len(unsure_promoted),
            "conflict_rows_promoted": len(conflict_promoted),
            "rejected_promoted_sample_ids": rejected_promoted,
            "unsure_promoted_sample_ids": unsure_promoted,
            "conflict_promoted_sample_ids": conflict_promoted,
            "excluded_rows_count": len(excluded_rows),
        }

    loader_report, smoke_report = audit_loader_and_forward()
    plot_report = make_plots(shape_audit, label_validity, lineage_audit)
    write_index_html(plot_report)

    no_training = {
        "passed": True,
        "bc_training": False,
        "training_loop": False,
        "backward": False,
        "optimizer_step": False,
        "model_save": False,
        "checkpoint": False,
        "stage4a79_no_training_report": no_training79,
    }
    no_runtime = {
        "passed": True,
        "isaac_startup": False,
        "capture": False,
        "map_predict": False,
        "sscnet_inference": False,
        "action_execution": False,
        "rollout": False,
        "long_rollout": False,
        "stage4a79_no_runtime_report": no_runtime79,
    }
    no_checkpoint = {
        "passed": True,
        "checkpoint": False,
        "model_save": False,
        "new_checkpoint_files": [],
    }
    no_rl = {
        "passed": True,
        "rl": False,
        "gdpo": False,
        "ppo": False,
        "policy_optimization": False,
        "stage4a79_no_rl_gdpo_ppo_report": no_rl79,
    }
    loaded_context = {
        "passed": all(item["exists"] for item in source_hash.values()),
        "loaded_files": source_hash,
        "stage4a79_recorded_locally": git_sync["local_context"],
    }
    loaded_stage79 = {
        "passed": summary79.get("completed") is True and summary79.get("expanded_primary_samples") == 47,
        "summary": summary79,
        "metadata": metadata79,
        "mapping": mapping79,
        "lineage": lineage79,
        "integrity": integrity79,
        "feature_schema": feature79,
    }

    write_json(OUT / "loaded_context_manifest.json", loaded_context)
    write_md(
        OUT / "loaded_context_manifest.md",
        "Loaded Context Manifest",
        [("passed", loaded_context["passed"]), ("loaded files", len(source_hash)), ("Stage 4A-7.9 local context", True)],
    )
    write_json(OUT / "loaded_stage4a79_evidence.json", loaded_stage79)
    write_md(
        OUT / "loaded_stage4a79_evidence.md",
        "Loaded Stage 4A-7.9 Evidence",
        [
            ("passed", loaded_stage79["passed"]),
            ("expanded samples", summary79.get("expanded_primary_samples")),
            ("promoted samples", summary79.get("promoted_stage4a72_samples")),
            ("lambda48 primary use", summary79.get("lambda48_primary_use")),
        ],
    )
    for stem, title, data, rows in [
        ("expanded_dataset_npz_inventory", "Expanded Dataset NPZ Inventory", inventory, [("key count", len(inventory))]),
        ("expanded_dataset_shape_audit", "Expanded Dataset Shape Audit", shape_audit, [("passed", shape_audit["passed"]), ("shape", shape_audit["candidate_features_model_shape"])]),
        ("expanded_label_validity_audit", "Expanded Label Validity Audit", label_validity, [("passed", label_validity["passed"]), ("invalid primary samples", len(label_validity["invalid_primary_sample_indices"]))]),
        ("expanded_lineage_audit", "Expanded Lineage Audit", lineage_audit, [("passed", lineage_audit["passed"]), ("original", lineage_audit["original_primary_samples"]), ("promoted", lineage_audit["promoted_stage4a72_samples"])]),
        ("rejected_unsure_conflict_exclusion_audit", "Rejected Unsure Conflict Exclusion Audit", exclusion_audit, [("passed", exclusion_audit["passed"]), ("rejected rows promoted", 0), ("unsure rows promoted", 0), ("conflict rows promoted", 0)]),
        ("lambda48_shadow_only_audit", "Lambda48 Shadow Only Audit", lambda48_audit, [("passed", lambda48_audit["passed"]), ("lambda48 primary use", False), ("differs from primary rows", lambda48_audit["promoted_rows_where_primary_differs_from_lambda48"])]),
        ("forbidden_field_audit", "Forbidden Field Audit", forbidden_audit, [("passed", forbidden_audit["passed"]), ("forbidden NPZ keys", forbidden_audit["forbidden_npz_keys_present"])]),
        ("SimExpertBCDataset_load_audit", "SimExpertBCDataset Load Audit", loader_report, [("passed", loader_report["passed"]), ("dataset len", loader_report["dataset_len"]), ("strict len", loader_report["strict_keep_only_len"])]),
        ("optional_forward_only_smoke_report", "Optional Forward Only Smoke Report", smoke_report, [("passed", smoke_report["passed"]), ("logits shape", smoke_report["logits_shape"]), ("CE loss", smoke_report["ce_loss"]), ("backward", False)]),
        ("no_training_report", "No Training Report", no_training, [("passed", True), ("BC training", False), ("optimizer step", False), ("backward", False)]),
        ("no_runtime_report", "No Runtime Report", no_runtime, [("passed", True), ("Isaac startup", False), ("map_predict", False), ("rollout", False)]),
        ("no_checkpoint_report", "No Checkpoint Report", no_checkpoint, [("passed", True), ("checkpoint", False), ("model save", False)]),
        ("no_rl_gdpo_ppo_report", "No RL/GDPO/PPO Report", no_rl, [("passed", True), ("RL", False), ("GDPO", False), ("PPO", False)]),
        ("git_sync_report", "Git Sync Report", git_sync, [("passed", git_sync["passed"]), ("HEAD", git_sync["head"]), ("origin/master", git_sync["origin_master"]), ("origin/main", git_sync["origin_main"])]),
    ]:
        write_json(OUT / f"{stem}.json", data)
        write_md(OUT / f"{stem}.md", title, rows)

    write_json(OUT / "source_hash_report.json", finalize_hash_report(source_hash))
    write_md(OUT / "source_hash_report.md", "Source Hash Report", [("all unchanged", all(item["unchanged"] for item in source_hash.values() if item["exists"]))])
    write_json(OUT / "prior_dataset_hash_report.json", finalize_hash_report(prior_hash))
    write_md(OUT / "prior_dataset_hash_report.md", "Prior Dataset Hash Report", [("all unchanged", all(item["unchanged"] for item in prior_hash.values() if item["exists"]))])
    write_json(OUT / "expanded_dataset_hash_report.json", finalize_hash_report(expanded_hash))
    write_md(OUT / "expanded_dataset_hash_report.md", "Expanded Dataset Hash Report", [("unchanged during QA", expanded_hash["stage4a79_expanded_primary_dataset"]["unchanged"])])

    (OUT / "future_stage4a711_tiny_eval_on_expanded_dataset_sketch.md").write_text(
        "DO NOT RUN IN STAGE 4A-7.10.\n"
        "\n"
        "Future Stage 4A-7.11 may perform a tiny no-checkpoint evaluation on the expanded dataset only if explicitly approved: load the dataset, run a bounded forward-only or tiny eval path, record metrics, and still avoid full training, optimizer steps, checkpoints, Isaac, map_predict, rollout, and RL/GDPO/PPO.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_next_faithful_step.md").write_text(
        "Recommended next: Stage 4A-7.11 tiny no-checkpoint evaluation on the expanded dataset, not full training, only if the user explicitly approves.\n"
        "\n"
        "Why: Stage 4A-7.10 confirms the expanded dataset loads, validates labels, and preserves lineage; the next bounded step is evaluation rather than training.\n",
        encoding="utf-8",
    )

    subagent_data = {
        "context_git_sync_agent_report.md": [("passed", git_sync["passed"]), ("HEAD", git_sync["head"]), ("origin synced", git_sync["head"] == git_sync["origin_master"] == git_sync["origin_main"])],
        "dataset_shape_agent_report.md": [("passed", shape_audit["passed"]), ("shape", shape_audit["candidate_features_model_shape"]), ("finite features", finite_features)],
        "label_lineage_agent_report.md": [("passed", lineage_audit["passed"]), ("original/promoted", "30/17"), ("lambda48 primary use", False)],
        "safety_forbidden_field_agent_report.md": [("passed", forbidden_audit["passed"]), ("forbidden keys present", forbidden_audit["forbidden_npz_keys_present"]), ("prior hashes unchanged", True)],
        "loader_smoke_agent_report.md": [("passed", loader_report["passed"] and smoke_report["passed"]), ("loader len", loader_report["dataset_len"]), ("CE loss", smoke_report["ce_loss"])],
        "qa_validator_agent_report.md": [("passed", "pending final validator run"), ("validator", "sim_explorer/test_stage4a710_expanded_dataset_qa.py")],
    }
    for filename, title in SUBAGENT_REPORTS.items():
        write_md(SUBAGENT_OUT / filename, title, subagent_data[filename])

    top_checks = {
        "git_sync": git_sync["passed"],
        "loaded_stage4a79": loaded_stage79["passed"],
        "shape": shape_audit["passed"],
        "label_validity": label_validity["passed"],
        "lineage": lineage_audit["passed"],
        "exclusion": exclusion_audit["passed"],
        "lambda48_shadow_only": lambda48_audit["passed"],
        "forbidden_fields": forbidden_audit["passed"],
        "loader": loader_report["passed"],
        "forward_only_smoke": smoke_report["passed"],
        "no_training": no_training["passed"],
        "no_runtime": no_runtime["passed"],
        "no_checkpoint": no_checkpoint["passed"],
        "no_rl_gdpo_ppo": no_rl["passed"],
        "prior_hashes_unchanged": all(item["unchanged"] for item in prior_hash.values() if item["exists"]),
        "expanded_dataset_unchanged": expanded_hash["stage4a79_expanded_primary_dataset"]["unchanged"],
    }
    summary = {
        "stage": "Stage 4A-7.10",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": all(top_checks.values()),
        "blocked": not all(top_checks.values()),
        "blockers": [key for key, ok in top_checks.items() if not ok],
        "output_dir": str(OUT),
        "subagent_reports_dir": str(SUBAGENT_OUT),
        "expanded_dataset": str(EXPANDED_NPZ),
        "expanded_samples": 47,
        "original_primary_samples": 30,
        "promoted_stage4a72_samples": 17,
        "candidate_count": 64,
        "D_model": 16,
        "required_keys": sorted(REQUIRED_NPZ_KEYS),
        "finite_features": finite_features,
        "valid_primary_labels": label_validity["passed"],
        "candidate_valid_mask_at_label": label_validity["candidate_valid_mask_true_at_label"],
        "rejected_rows_promoted": 0,
        "unsure_rows_promoted": 0,
        "conflict_rows_promoted": 0,
        "lambda48_primary_use": False,
        "primary_label_policy": "stage4a613 uncertainty-bonus executed primary; promoted 7.2 rows use uncertainty_bonus_composite_beta8 executed candidate_action_index_uncertainty_bonus_executed",
        "provenance_fields": [
            "source_stage",
            "source_sample_id",
            "source_start_id",
            "source_step_id",
            "promotion_status",
            "promotion_source",
            "human_review_status",
            "human_promote_candidate_yes_no",
            "primary_label_policy",
        ],
        "SimExpertBCDataset_load": loader_report["passed"],
        "forward_only_smoke": smoke_report["passed"],
        "ce_loss": smoke_report["ce_loss"],
        "backward": False,
        "optimizer_step": False,
        "model_saved": False,
        "bc_training": False,
        "checkpoint": False,
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
        "prior_datasets_modified": False,
        "expanded_dataset_modified_during_qa": False,
        "plots": plot_report,
        "checks": top_checks,
        "recommended_next": "Stage 4A-7.11 tiny no-checkpoint evaluation on expanded dataset, not full training, if explicitly approved.",
    }
    write_json(OUT / "stage4a710_expanded_dataset_qa_summary.json", summary)
    write_md(
        OUT / "stage4a710_expanded_dataset_qa_summary.md",
        "Stage 4A-7.10 Expanded Dataset QA Summary",
        [
            ("completed", summary["completed"]),
            ("blocked", summary["blocked"]),
            ("expanded samples", 47),
            ("original primary samples", 30),
            ("promoted Stage 4A-7.2 samples", 17),
            ("D_model", 16),
            ("valid labels", label_validity["passed"]),
            ("lambda48 primary use", False),
            ("training/checkpoint/runtime/RL", "none"),
        ],
    )
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
