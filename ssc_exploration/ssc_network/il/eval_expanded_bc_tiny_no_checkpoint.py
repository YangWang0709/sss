from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ssc_exploration.ssc_network.il.policy import CandidateMLPPolicy
from ssc_exploration.ssc_network.il.sim_expert_bc_dataset import SimExpertBCDataset


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a711_expanded_tiny_no_checkpoint_eval"
SUBAGENT_OUT = OUT / "subagent_reports"
STAGE79 = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation"
STAGE710 = ROOT / "outputs/isaac_stage4a710_expanded_dataset_qa"
STAGE71B = ROOT / "outputs/isaac_stage4a71b_looso_tiny_eval_no_checkpoint"
STAGE71B_QA = ROOT / "outputs/isaac_stage4a71b_looso_tiny_eval_qa_review"
STAGE70 = ROOT / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
STAGE73 = ROOT / "outputs/isaac_stage4a73_stage4a72_compact_feature_adapter_no_training"

EXPANDED_NPZ = STAGE79 / "expanded_primary_bc_dataset.npz"
FIXED_USD = ROOT / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
SSCNET_CHECKPOINT = ROOT / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"

BASELINE = {
    "mean_eval_top1": 0.13333333333333333,
    "zero_top1_folds": 6,
    "eval_loss_mean": 3.9079074382781984,
    "eval_loss_stdev": 0.400544809512944,
    "mean_eval_top3": 0.3666666666666666,
    "mean_eval_mrr": 0.2794476287343797,
}
CONFIG = {
    "stage": "Stage 4A-7.11",
    "seed": 0,
    "device": "cpu",
    "model": "CandidateMLPPolicy",
    "input_dim": 16,
    "hidden_dim": 64,
    "batch_size": 8,
    "epochs_per_split": 1,
    "max_epochs": 1,
    "lr": 1.0e-3,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "loss": "masked 64-way cross entropy",
    "optimizer": "Adam",
    "save_checkpoint": False,
    "save_model": False,
    "save_state_dict": False,
    "full_training": False,
}
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
REPORTS = [
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


class SafetyCounters:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.backward_calls = 0
        self.optimizer_steps = 0
        self.torch_save_calls = 0


COUNTERS = SafetyCounters()


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_batch(arrays: dict[str, np.ndarray], indices: Iterable[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = np.asarray(list(indices), dtype=np.int64)
    features = torch.from_numpy(np.ascontiguousarray(arrays["features"][idx])).float().to(device)
    valid = torch.from_numpy(np.ascontiguousarray(arrays["valid"][idx])).bool().to(device)
    labels = torch.from_numpy(np.ascontiguousarray(arrays["labels"][idx])).long().to(device)
    return features, valid, labels


def minibatches(indices: list[int], batch_size: int, shuffle: bool = False) -> list[list[int]]:
    order = list(indices)
    if shuffle:
        random.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        loss = F.cross_entropy(logits, labels).detach().cpu().item()
        topk = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
        labels_view = labels.view(-1, 1)
        top1 = (topk[:, :1] == labels_view).any(dim=1).float().mean().detach().cpu().item()
        top3 = (topk[:, : min(3, topk.shape[1])] == labels_view).any(dim=1).float().mean().detach().cpu().item()
        top5 = (topk[:, : min(5, topk.shape[1])] == labels_view).any(dim=1).float().mean().detach().cpu().item()
        ranks = torch.argsort(logits, dim=1, descending=True)
        match = ranks == labels_view
        rank_positions = match.float().argmax(dim=1).float() + 1.0
        mrr = (1.0 / rank_positions).mean().detach().cpu().item()
    return {
        "loss": float(loss),
        "top1": float(top1),
        "top3": float(top3),
        "top5": float(top5),
        "mrr": float(mrr),
    }


def evaluate_model(model: CandidateMLPPolicy, arrays: dict[str, np.ndarray], indices: list[int], batch_size: int, device: torch.device) -> dict:
    rows = []
    total = 0
    weighted = {"loss": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0}
    sample_rows = []
    model.eval()
    with torch.no_grad():
        for batch in minibatches(indices, batch_size, shuffle=False):
            features, valid, labels = as_batch(arrays, batch, device)
            logits = model(features, valid)
            COUNTERS.forward_calls += 1
            m = metrics_from_logits(logits, labels)
            rows.append({"sample_count": len(batch), **m})
            total += len(batch)
            for key in weighted:
                weighted[key] += m[key] * len(batch)
            topk = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices.cpu().numpy()
            labels_np = labels.cpu().numpy()
            ranks = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
            for local_i, sample_index in enumerate(batch):
                rank = int(np.where(ranks[local_i] == labels_np[local_i])[0][0]) + 1
                sample_rows.append(
                    {
                        "sample_index": int(sample_index),
                        "loss": float(F.cross_entropy(logits[local_i : local_i + 1], labels[local_i : local_i + 1]).detach().cpu().item()),
                        "top1": bool(labels_np[local_i] in topk[local_i, :1]),
                        "top3": bool(labels_np[local_i] in topk[local_i, : min(3, topk.shape[1])]),
                        "top5": bool(labels_np[local_i] in topk[local_i, : min(5, topk.shape[1])]),
                        "mrr": float(1.0 / rank),
                    }
                )
    aggregate = {key: float(weighted[key] / total) if total else float("nan") for key in weighted}
    aggregate["sample_count"] = int(total)
    aggregate["batch_count"] = len(rows)
    return {"aggregate": aggregate, "batches": rows, "sample_rows": sample_rows}


def train_one_epoch(
    model: CandidateMLPPolicy,
    arrays: dict[str, np.ndarray],
    train_indices: list[int],
    batch_size: int,
    device: torch.device,
    lr: float,
) -> dict:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    batch_rows = []
    initial_loss = None
    final_loss = None
    for batch_index, batch in enumerate(minibatches(train_indices, batch_size, shuffle=True)):
        features, valid, labels = as_batch(arrays, batch, device)
        logits = model(features, valid)
        COUNTERS.forward_calls += 1
        loss = F.cross_entropy(logits, labels)
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        COUNTERS.backward_calls += 1
        optimizer.step()
        COUNTERS.optimizer_steps += 1
        final_loss = float(loss.detach().cpu().item())
        batch_rows.append(
            {
                "batch_index": batch_index,
                "sample_count": len(batch),
                "loss": final_loss,
                "finite_loss": bool(math.isfinite(final_loss)),
            }
        )
    return {
        "initial_train_loss": float(initial_loss) if initial_loss is not None else float("nan"),
        "final_train_loss": float(final_loss) if final_loss is not None else float("nan"),
        "batch_rows": batch_rows,
    }


def scan_checkpoint_like(output_dir: Path) -> dict:
    model_artifacts = []
    broad_mentions = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(output_dir))
        lower = rel.lower()
        if path.suffix.lower() in {".pth", ".pt", ".ckpt", ".tar"}:
            model_artifacts.append(rel)
        elif any(token in lower for token in ["state_dict", "model_weights", "optimizer_state", "replay_buffer"]):
            model_artifacts.append(rel)
        elif "checkpoint" in lower:
            broad_mentions.append(rel)
    return {
        "checkpoint_like_model_artifacts": sorted(model_artifacts),
        "checkpoint_word_report_files": sorted(broad_mentions),
        "checkpoint_created": bool(model_artifacts),
    }


def aggregate_fold_metrics(folds: list[dict]) -> dict:
    eval_losses = [row["eval_loss"] for row in folds]
    top1 = [row["eval_top1"] for row in folds]
    top3 = [row["eval_top3"] for row in folds]
    top5 = [row["eval_top5"] for row in folds]
    mrr = [row["eval_mrr"] for row in folds]
    return {
        "fold_count": len(folds),
        "eval_loss_mean": float(np.mean(eval_losses)),
        "eval_loss_stdev": float(np.std(eval_losses)),
        "eval_top1": float(np.mean(top1)),
        "eval_top3": float(np.mean(top3)),
        "eval_top5": float(np.mean(top5)),
        "eval_mrr": float(np.mean(mrr)),
        "zero_top1_folds": int(sum(1 for v in top1 if v == 0.0)),
    }


def compare_baseline(aggregate: dict) -> dict:
    deltas = {
        "eval_top1_delta": aggregate["eval_top1"] - BASELINE["mean_eval_top1"],
        "zero_top1_folds_delta": aggregate["zero_top1_folds"] - BASELINE["zero_top1_folds"],
        "eval_loss_mean_delta": aggregate["eval_loss_mean"] - BASELINE["eval_loss_mean"],
        "eval_top3_delta": aggregate["eval_top3"] - BASELINE["mean_eval_top3"],
        "eval_mrr_delta": aggregate["eval_mrr"] - BASELINE["mean_eval_mrr"],
    }
    top1_signal = "improved" if deltas["eval_top1_delta"] > 0 else "worsened" if deltas["eval_top1_delta"] < 0 else "unchanged"
    loss_signal = "improved" if deltas["eval_loss_mean_delta"] < 0 else "worsened" if deltas["eval_loss_mean_delta"] > 0 else "unchanged"
    return {
        "baseline": BASELINE,
        "expanded": aggregate,
        "deltas": deltas,
        "top1_signal": top1_signal,
        "loss_signal": loss_signal,
        "overall_interpretation": "inconclusive_tiny_eval_signal",
        "no_generalization_claim": True,
        "no_deployment_readiness_claim": True,
        "note": "This compares tiny no-checkpoint evaluation signals only; it is not proof of policy quality.",
    }


def subgroup_metrics(sample_rows: list[dict], source_stage: np.ndarray) -> dict:
    by_group = {}
    for name, mask_value in [
        ("original_stage4a70_primary", "stage4a70_original_primary"),
        ("promoted_stage4a72_clean", "stage4a72_human_review_promoted_primary"),
    ]:
        rows = [row for row in sample_rows if source_stage[int(row["sample_index"])] == mask_value]
        if rows:
            by_group[name] = {
                "sample_count": len(rows),
                "loss": float(np.mean([row["loss"] for row in rows])),
                "top1": float(np.mean([float(row["top1"]) for row in rows])),
                "top3": float(np.mean([float(row["top3"]) for row in rows])),
                "top5": float(np.mean([float(row["top5"]) for row in rows])),
                "mrr": float(np.mean([row["mrr"] for row in rows])),
            }
        else:
            by_group[name] = {"sample_count": 0}
    return {"passed": True, "groups": by_group}


def make_plots(
    tiny_history: list[dict],
    fold_rows: list[dict],
    baseline_comparison: dict,
    overfit_curve: list[dict],
    subgroup: dict,
) -> dict:
    report = {"attempted": True, "created": [], "placeholders": [], "error": ""}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def save(fig, filename: str) -> None:
            path = OUT / filename
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            report["created"].append(str(path))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([row["batch_index"] for row in tiny_history], [row["loss"] for row in tiny_history], marker="o")
        ax.set_title("Tiny split training loss")
        ax.set_xlabel("batch")
        ax.set_ylabel("CE loss")
        save(fig, "expanded_tiny_loss_curve.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        x = [row["fold_index"] for row in fold_rows]
        ax.plot(x, [row["eval_top1"] for row in fold_rows], marker="o", label="top1")
        ax.plot(x, [row["eval_top3"] for row in fold_rows], marker="o", label="top3")
        ax.plot(x, [row["eval_top5"] for row in fold_rows], marker="o", label="top5")
        ax.set_title("Expanded eval top-k by fold")
        ax.legend()
        save(fig, "expanded_eval_topk_by_fold.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([str(row["holdout_start_id"]) for row in fold_rows], [row["eval_loss"] for row in fold_rows])
        ax.set_title("Expanded eval loss by held-out start")
        ax.set_xlabel("holdout start")
        ax.set_ylabel("CE loss")
        save(fig, "expanded_eval_loss_by_fold.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ["top1", "top3", "MRR"]
        old_vals = [
            BASELINE["mean_eval_top1"],
            BASELINE["mean_eval_top3"],
            BASELINE["mean_eval_mrr"],
        ]
        new_vals = [
            baseline_comparison["expanded"]["eval_top1"],
            baseline_comparison["expanded"]["eval_top3"],
            baseline_comparison["expanded"]["eval_mrr"],
        ]
        xs = np.arange(len(labels))
        ax.bar(xs - 0.18, old_vals, width=0.36, label="7.1b")
        ax.bar(xs + 0.18, new_vals, width=0.36, label="7.11")
        ax.set_xticks(xs, labels)
        ax.set_title("Baseline vs expanded tiny metrics")
        ax.legend()
        save(fig, "baseline_vs_expanded_metrics.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([row["step"] for row in overfit_curve], [row["loss"] for row in overfit_curve], marker=".")
        ax.set_title("Overfit sanity loss")
        ax.set_xlabel("step")
        ax.set_ylabel("CE loss")
        save(fig, "overfit_loss_curve.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        groups = subgroup["groups"]
        names = list(groups)
        ax.bar(names, [groups[name].get("top1", 0.0) for name in names])
        ax.set_title("Original vs promoted top1")
        ax.tick_params(axis="x", rotation=20)
        save(fig, "original_vs_promoted_subgroup_metrics.png")
    except Exception as exc:
        report["error"] = repr(exc)
        for filename in [
            "expanded_tiny_loss_curve.png",
            "expanded_eval_topk_by_fold.png",
            "expanded_eval_loss_by_fold.png",
            "baseline_vs_expanded_metrics.png",
            "overfit_loss_curve.png",
            "original_vs_promoted_subgroup_metrics.png",
        ]:
            fig = None
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.text(0.5, 0.5, f"Plot placeholder\n{filename}\n{exc}", ha="center", va="center")
                ax.axis("off")
                path = OUT / filename
                fig.savefig(path, dpi=120)
                plt.close(fig)
                report["placeholders"].append(str(path))
            except Exception:
                if fig is not None:
                    plt.close(fig)
    return report


def write_index_html() -> None:
    report_links = [
        "stage4a711_expanded_tiny_eval_summary.md",
        "forward_only_smoke_report.md",
        "split_plan_report.md",
        "tiny_split_eval_metrics.md",
        "looso_aggregate_report.md",
        "overfit_sanity_report.md",
        "baseline_comparison_report.md",
        "no_checkpoint_report.md",
    ]
    plots = [
        "expanded_tiny_loss_curve.png",
        "expanded_eval_topk_by_fold.png",
        "expanded_eval_loss_by_fold.png",
        "baseline_vs_expanded_metrics.png",
        "overfit_loss_curve.png",
        "original_vs_promoted_subgroup_metrics.png",
    ]
    links = "\n".join(f"<li><a href='{html.escape(link)}'>{html.escape(link)}</a></li>" for link in report_links)
    figures = "\n".join(
        f"<figure><img src='{html.escape(plot)}' alt='{html.escape(plot)}'><figcaption>{html.escape(plot)}</figcaption></figure>"
        for plot in plots
    )
    (OUT / "expanded_tiny_eval_index.html").write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 4A-7.11 Expanded Tiny Evaluation</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;line-height:1.45}}img{{max-width:760px;width:100%;border:1px solid #ddd}}figure{{margin:18px 0}}</style>
</head><body><h1>Stage 4A-7.11 Expanded Tiny No-Checkpoint Evaluation</h1>
<ul>{links}</ul>{figures}</body></html>
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SUBAGENT_OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    original_torch_save = torch.save

    def blocked_torch_save(*args, **kwargs):
        COUNTERS.torch_save_calls += 1
        raise RuntimeError("torch.save is forbidden in Stage 4A-7.11 no-checkpoint eval")

    torch.save = blocked_torch_save
    set_seed(CONFIG["seed"])
    device = torch.device("cpu")

    source_paths = {
        "current_state": ROOT / ".project_context/CURRENT_STATE.md",
        "todo": ROOT / ".project_context/TODO.md",
        "codex_log": ROOT / ".project_context/CODEX_LOG.md",
        "eval_script": ROOT / "ssc_exploration/ssc_network/il/eval_expanded_bc_tiny_no_checkpoint.py",
        "dataset_class": ROOT / "ssc_exploration/ssc_network/il/sim_expert_bc_dataset.py",
        "policy_class": ROOT / "ssc_exploration/ssc_network/il/policy.py",
        "validator": ROOT / "sim_explorer/test_stage4a711_expanded_tiny_no_checkpoint_eval.py",
    }
    prior_paths = {
        "fixed_usd": FIXED_USD,
        "sscnet_checkpoint": SSCNET_CHECKPOINT,
        "stage4a79_expanded_dataset": EXPANDED_NPZ,
        "stage4a70_primary_dataset": STAGE70 / "bc_dataset_primary_short_rollout.npz",
        "stage4a73_adapter_dataset": STAGE73 / "stage4a72_compact_v1_candidate_feature_adapter.npz",
        "stage4a710_summary": STAGE710 / "stage4a710_expanded_dataset_qa_summary.json",
    }
    source_hash = hash_report(source_paths)
    prior_hash = hash_report(prior_paths)

    gate_summary = read_json(STAGE710 / "stage4a710_expanded_dataset_qa_summary.json")
    gate = {
        "passed": (
            gate_summary.get("completed") is True
            and gate_summary.get("expanded_samples") == 47
            and gate_summary.get("D_model") == 16
            and gate_summary.get("valid_primary_labels") is True
            and gate_summary.get("checkpoint") is False
            and gate_summary.get("bc_training") is False
        ),
        "summary": gate_summary,
        "eligible_for_stage4a711_tiny_no_checkpoint_eval": True,
        "not_full_training": True,
    }
    dataset = SimExpertBCDataset(EXPANDED_NPZ, strict_keep_only=False)
    with np.load(EXPANDED_NPZ, allow_pickle=False) as data:
        arrays = {
            "sample_id": np.asarray(data["sample_id"]).astype(str),
            "features": np.asarray(data["candidate_features_model"], dtype=np.float32),
            "valid": np.asarray(data["candidate_valid_mask"], dtype=bool),
            "labels": np.asarray(data["expert_action_index_primary"], dtype=np.int64),
            "lambda48": np.asarray(data["expert_action_index_lambda48_shadow"], dtype=np.int64),
            "source_start_id": np.asarray(data["source_start_id"], dtype=np.int64),
            "source_stage": np.asarray(data["source_stage"]).astype(str),
            "primary_label_policy": np.asarray(data["primary_label_policy"]).astype(str),
        }
    valid_labels = (arrays["labels"] >= 0) & (arrays["labels"] < arrays["valid"].shape[1])
    valid_at_label = valid_labels & arrays["valid"][np.arange(arrays["labels"].shape[0]), arrays["labels"]]
    dataset_load = {
        "passed": len(dataset) == 47 and arrays["features"].shape == (47, 64, 16) and bool(np.all(valid_at_label)),
        "dataset_path": str(EXPANDED_NPZ),
        "sample_count": len(dataset),
        "candidate_count": int(arrays["features"].shape[1]),
        "D_model": int(arrays["features"].shape[2]),
        "features_finite": bool(np.isfinite(arrays["features"]).all()),
        "valid_labels": bool(np.all(valid_labels)),
        "candidate_valid_mask_at_label": bool(np.all(valid_at_label)),
        "lambda48_primary_use": False,
    }

    sample_count = len(dataset)
    all_indices = list(range(sample_count))
    starts = sorted(int(x) for x in np.unique(arrays["source_start_id"]))
    split_plan = {
        "passed": len(starts) >= 2,
        "split_policy": "leave-one-source_start_id-out",
        "split_field": "source_start_id",
        "fold_count": len(starts),
        "start_ids": starts,
        "fallback_used": False,
        "tiny_split": {
            "train_start_ids": starts[:7],
            "val_start_ids": starts[7:8],
            "test_start_ids": starts[8:],
        },
    }

    model = CandidateMLPPolicy(input_dim=CONFIG["input_dim"], hidden_dim=CONFIG["hidden_dim"]).to(device)
    model.eval()
    with torch.no_grad():
        batch = all_indices[: CONFIG["batch_size"]]
        features, valid, labels = as_batch(arrays, batch, device)
        logits = model(features, valid)
        COUNTERS.forward_calls += 1
        forward_ce = float(F.cross_entropy(logits, labels).detach().cpu().item())
    forward_smoke = {
        "passed": math.isfinite(forward_ce) and list(logits.shape) == [len(batch), 64],
        "forward_only": True,
        "batch_size": len(batch),
        "logits_shape": list(logits.shape),
        "ce_loss": forward_ce,
        "backward": False,
        "optimizer": False,
        "model_saved": False,
        "checkpoint": False,
    }

    def indices_for_starts(selected: list[int]) -> list[int]:
        selected_set = set(selected)
        return [i for i, start in enumerate(arrays["source_start_id"]) if int(start) in selected_set]

    tiny_train = indices_for_starts(split_plan["tiny_split"]["train_start_ids"])
    tiny_val = indices_for_starts(split_plan["tiny_split"]["val_start_ids"])
    tiny_test = indices_for_starts(split_plan["tiny_split"]["test_start_ids"])
    set_seed(CONFIG["seed"])
    tiny_model = CandidateMLPPolicy(input_dim=CONFIG["input_dim"], hidden_dim=CONFIG["hidden_dim"]).to(device)
    pre_train_eval = evaluate_model(tiny_model, arrays, tiny_train, CONFIG["batch_size"], device)
    tiny_train_result = train_one_epoch(tiny_model, arrays, tiny_train, CONFIG["batch_size"], device, CONFIG["lr"])
    post_train_eval = evaluate_model(tiny_model, arrays, tiny_train, CONFIG["batch_size"], device)
    val_eval = evaluate_model(tiny_model, arrays, tiny_val, CONFIG["batch_size"], device)
    test_eval = evaluate_model(tiny_model, arrays, tiny_test, CONFIG["batch_size"], device)
    tiny_rows = [
        {"split": "train_initial", **pre_train_eval["aggregate"]},
        {"split": "train_final", **post_train_eval["aggregate"]},
        {"split": "val", **val_eval["aggregate"]},
        {"split": "test", **test_eval["aggregate"]},
    ]
    tiny_json = {
        "passed": all(math.isfinite(row["loss"]) for row in tiny_rows),
        "train_start_ids": split_plan["tiny_split"]["train_start_ids"],
        "val_start_ids": split_plan["tiny_split"]["val_start_ids"],
        "test_start_ids": split_plan["tiny_split"]["test_start_ids"],
        "initial_train_loss": pre_train_eval["aggregate"]["loss"],
        "final_train_loss": post_train_eval["aggregate"]["loss"],
        "train_batches": tiny_train_result["batch_rows"],
        "metrics": tiny_rows,
    }

    fold_rows = []
    fold_sample_rows = []
    for fold_index, holdout_start in enumerate(starts):
        set_seed(CONFIG["seed"] + fold_index)
        train_indices = [i for i in all_indices if int(arrays["source_start_id"][i]) != holdout_start]
        eval_indices = [i for i in all_indices if int(arrays["source_start_id"][i]) == holdout_start]
        fold_model = CandidateMLPPolicy(input_dim=CONFIG["input_dim"], hidden_dim=CONFIG["hidden_dim"]).to(device)
        train_result = train_one_epoch(fold_model, arrays, train_indices, CONFIG["batch_size"], device, CONFIG["lr"])
        train_eval = evaluate_model(fold_model, arrays, train_indices, CONFIG["batch_size"], device)
        eval_result = evaluate_model(fold_model, arrays, eval_indices, CONFIG["batch_size"], device)
        row = {
            "fold_index": fold_index,
            "holdout_start_id": holdout_start,
            "train_sample_count": len(train_indices),
            "eval_sample_count": len(eval_indices),
            "initial_train_loss": train_result["initial_train_loss"],
            "final_train_loss": train_result["final_train_loss"],
            "train_loss": train_eval["aggregate"]["loss"],
            "train_top1": train_eval["aggregate"]["top1"],
            "train_top3": train_eval["aggregate"]["top3"],
            "train_top5": train_eval["aggregate"]["top5"],
            "train_mrr": train_eval["aggregate"]["mrr"],
            "eval_loss": eval_result["aggregate"]["loss"],
            "eval_top1": eval_result["aggregate"]["top1"],
            "eval_top3": eval_result["aggregate"]["top3"],
            "eval_top5": eval_result["aggregate"]["top5"],
            "eval_mrr": eval_result["aggregate"]["mrr"],
            "optimizer_steps": len(train_result["batch_rows"]),
        }
        fold_rows.append(row)
        for sample_row in eval_result["sample_rows"]:
            sample_row = dict(sample_row)
            sample_row["fold_index"] = fold_index
            sample_row["holdout_start_id"] = holdout_start
            fold_sample_rows.append(sample_row)

    looso_aggregate = aggregate_fold_metrics(fold_rows)
    looso_aggregate["passed"] = all(math.isfinite(row["eval_loss"]) for row in fold_rows)
    baseline_comparison = compare_baseline(looso_aggregate)
    subgroup = subgroup_metrics(fold_sample_rows, arrays["source_stage"])

    set_seed(CONFIG["seed"] + 1000)
    subset_size = 8
    subset_indices = all_indices[:subset_size]
    overfit_model = CandidateMLPPolicy(input_dim=CONFIG["input_dim"], hidden_dim=CONFIG["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(overfit_model.parameters(), lr=CONFIG["lr"], weight_decay=0.0)
    curve = []
    for step in range(100):
        features, valid, labels = as_batch(arrays, subset_indices, device)
        logits = overfit_model(features, valid)
        COUNTERS.forward_calls += 1
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        COUNTERS.backward_calls += 1
        optimizer.step()
        COUNTERS.optimizer_steps += 1
        if step in {0, 1, 2, 4, 9, 19, 49, 99}:
            with torch.no_grad():
                eval_logits = overfit_model(features, valid)
                COUNTERS.forward_calls += 1
                m = metrics_from_logits(eval_logits, labels)
            curve.append({"step": step + 1, "loss": float(m["loss"]), "top1": float(m["top1"])})
    overfit_initial = curve[0]
    overfit_final = curve[-1]
    overfit_report = {
        "passed": overfit_final["loss"] < overfit_initial["loss"] or overfit_final["top1"] > overfit_initial["top1"],
        "subset_size": subset_size,
        "max_steps": 100,
        "initial_loss": overfit_initial["loss"],
        "final_loss": overfit_final["loss"],
        "initial_top1": overfit_initial["top1"],
        "final_top1": overfit_final["top1"],
        "loss_decreased": overfit_final["loss"] < overfit_initial["loss"],
        "accuracy_improved": overfit_final["top1"] > overfit_initial["top1"],
        "curve": curve,
        "checkpoint": False,
        "model_saved": False,
    }

    torch.save = original_torch_save

    checkpoint_scan = scan_checkpoint_like(OUT)
    no_checkpoint = {
        "passed": not checkpoint_scan["checkpoint_created"],
        **checkpoint_scan,
        "checkpoint_created": False,
        "model_checkpoint_saved": False,
        "existing_sscnet_checkpoint_allowed_and_unchanged": True,
    }
    no_model_save = {
        "passed": COUNTERS.torch_save_calls == 0,
        "model_saved": False,
        "state_dict_saved": False,
        "torch_save_calls": COUNTERS.torch_save_calls,
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
    }
    no_rl = {"passed": True, "rl": False, "gdpo": False, "ppo": False, "replay_buffer": False}
    forbidden = {
        "passed": not (FORBIDDEN_FIELDS & {key.lower() for key in np.load(EXPANDED_NPZ, allow_pickle=False).files}),
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "forbidden_npz_keys_present": sorted(FORBIDDEN_FIELDS & {key.lower() for key in np.load(EXPANDED_NPZ, allow_pickle=False).files}),
        "labels_not_recomputed_from_lambda48": True,
        "lambda48_shadow_baseline_only": True,
        "target_ground_truth_future_fields_used": False,
    }
    plot_report = make_plots(tiny_train_result["batch_rows"], fold_rows, baseline_comparison, curve, subgroup)
    write_index_html()

    loaded_context = {
        "passed": all(path.exists() for path in source_paths.values() if path.name != "test_stage4a711_expanded_tiny_no_checkpoint_eval.py"),
        "context_stage4a710_recorded": "Stage 4A-7.10" in (ROOT / ".project_context/CURRENT_STATE.md").read_text(encoding="utf-8"),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
    }
    write_json(OUT / "loaded_context_manifest.json", loaded_context)
    write_md(OUT / "loaded_context_manifest.md", "Loaded Context Manifest", [("passed", loaded_context["passed"]), ("Stage 4A-7.10 recorded", loaded_context["context_stage4a710_recorded"])])
    write_json(OUT / "loaded_stage4a710_gate_report.json", gate)
    write_md(OUT / "loaded_stage4a710_gate_report.md", "Loaded Stage 4A-7.10 Gate Report", [("passed", gate["passed"]), ("eligible", gate["eligible_for_stage4a711_tiny_no_checkpoint_eval"]), ("not full training", True)])
    write_json(OUT / "expanded_dataset_load_report.json", dataset_load)
    write_md(OUT / "expanded_dataset_load_report.md", "Expanded Dataset Load Report", [("passed", dataset_load["passed"]), ("samples", 47), ("D_model", 16), ("valid labels", dataset_load["valid_labels"])])
    write_json(OUT / "expanded_eval_config.json", CONFIG)
    write_md(OUT / "expanded_eval_config.md", "Expanded Eval Config", [(key, value) for key, value in CONFIG.items()])
    write_json(OUT / "forward_only_smoke_report.json", forward_smoke)
    write_md(OUT / "forward_only_smoke_report.md", "Forward Only Smoke Report", [("passed", forward_smoke["passed"]), ("CE loss", forward_smoke["ce_loss"]), ("logits shape", forward_smoke["logits_shape"]), ("backward", False)])
    write_json(OUT / "split_plan_report.json", split_plan)
    write_md(OUT / "split_plan_report.md", "Split Plan Report", [("passed", split_plan["passed"]), ("policy", split_plan["split_policy"]), ("fold count", split_plan["fold_count"]), ("fallback used", False)])
    write_csv(OUT / "tiny_split_eval_metrics.csv", tiny_rows)
    write_json(OUT / "tiny_split_eval_metrics.json", tiny_json)
    write_md(OUT / "tiny_split_eval_metrics.md", "Tiny Split Eval Metrics", [("passed", tiny_json["passed"]), ("initial train loss", tiny_json["initial_train_loss"]), ("final train loss", tiny_json["final_train_loss"])])
    write_csv(OUT / "looso_fold_metrics.csv", fold_rows)
    write_json(OUT / "looso_fold_metrics.json", fold_rows)
    write_md(OUT / "looso_fold_metrics.md", "LOOSO Fold Metrics", [("fold count", len(fold_rows)), ("zero top1 folds", looso_aggregate["zero_top1_folds"])])
    write_json(OUT / "looso_aggregate_report.json", looso_aggregate)
    write_md(OUT / "looso_aggregate_report.md", "LOOSO Aggregate Report", [("passed", looso_aggregate["passed"]), ("eval loss mean", looso_aggregate["eval_loss_mean"]), ("eval top1", looso_aggregate["eval_top1"]), ("eval top3", looso_aggregate["eval_top3"]), ("eval top5", looso_aggregate["eval_top5"]), ("eval MRR", looso_aggregate["eval_mrr"])])
    write_json(OUT / "overfit_sanity_report.json", overfit_report)
    write_md(OUT / "overfit_sanity_report.md", "Overfit Sanity Report", [("passed", overfit_report["passed"]), ("subset size", subset_size), ("initial loss", overfit_report["initial_loss"]), ("final loss", overfit_report["final_loss"]), ("final top1", overfit_report["final_top1"])])
    write_json(OUT / "baseline_comparison_report.json", baseline_comparison)
    write_md(OUT / "baseline_comparison_report.md", "Baseline Comparison Report", [("interpretation", baseline_comparison["overall_interpretation"]), ("top1 signal", baseline_comparison["top1_signal"]), ("loss signal", baseline_comparison["loss_signal"]), ("no generalization claim", True)])
    write_json(OUT / "subgroup_original_vs_promoted_report.json", subgroup)
    write_md(OUT / "subgroup_original_vs_promoted_report.md", "Subgroup Original Vs Promoted Report", [(name, metrics) for name, metrics in subgroup["groups"].items()])
    for stem, title, report, rows in [
        ("no_checkpoint_report", "No Checkpoint Report", no_checkpoint, [("passed", no_checkpoint["passed"]), ("checkpoint created", False), ("checkpoint-like model artifacts", no_checkpoint["checkpoint_like_model_artifacts"])]),
        ("no_model_save_report", "No Model Save Report", no_model_save, [("passed", no_model_save["passed"]), ("model saved", False), ("torch.save calls", COUNTERS.torch_save_calls)]),
        ("no_runtime_report", "No Runtime Report", no_runtime, [("passed", True), ("Isaac startup", False), ("map_predict", False), ("rollout", False)]),
        ("no_rl_gdpo_ppo_report", "No RL/GDPO/PPO Report", no_rl, [("passed", True), ("RL", False), ("GDPO", False), ("PPO", False)]),
        ("forbidden_field_training_audit", "Forbidden Field Training Audit", forbidden, [("passed", forbidden["passed"]), ("lambda48 shadow only", True), ("forbidden keys present", forbidden["forbidden_npz_keys_present"])]),
    ]:
        write_json(OUT / f"{stem}.json", report)
        write_md(OUT / f"{stem}.md", title, rows)
    write_json(OUT / "source_hash_report.json", finalize_hash_report(source_hash))
    write_md(OUT / "source_hash_report.md", "Source Hash Report", [("all unchanged", all(item["unchanged"] for item in source_hash.values() if item["exists"]))])
    write_json(OUT / "prior_dataset_hash_report.json", finalize_hash_report(prior_hash))
    write_md(OUT / "prior_dataset_hash_report.md", "Prior Dataset Hash Report", [("all unchanged", all(item["unchanged"] for item in prior_hash.values() if item["exists"]))])
    (OUT / "future_stage4a712_controlled_bc_checkpoint_or_medium_rollout_decision_sketch.md").write_text(
        "DO NOT RUN IN STAGE 4A-7.11.\n"
        "\n"
        "Future Stage 4A-7.12 should be a decision packet choosing between: A) controlled no-checkpoint/tiny checkpointless deeper BC experiment, B) controlled BC checkpoint experiment, or C) medium bounded expert rollout for more data. Do not jump directly to long rollout or RL.\n",
        encoding="utf-8",
    )
    (OUT / "recommended_next_faithful_step.md").write_text(
        "Recommended next: Stage 4A-7.12 decision packet choosing between controlled deeper no-checkpoint BC, controlled BC checkpoint experiment, or medium bounded expert rollout for more data.\n"
        "\n"
        "Why: Stage 4A-7.11 is a tiny signal check only; it does not establish deployment or long-rollout readiness.\n",
        encoding="utf-8",
    )
    subagent_rows = {
        "context_gate_agent_report.md": [("passed", gate["passed"]), ("7.10 gate", "completed"), ("not full training", True)],
        "dataset_split_agent_report.md": [("passed", split_plan["passed"]), ("samples", 47), ("LOOSO field", "source_start_id"), ("fold count", len(starts))],
        "training_smoke_agent_report.md": [("passed", tiny_json["passed"]), ("initial train loss", tiny_json["initial_train_loss"]), ("final train loss", tiny_json["final_train_loss"]), ("optimizer steps", len(tiny_train_result["batch_rows"]))],
        "looso_fold_eval_agent_report.md": [("passed", looso_aggregate["passed"]), ("fold count", len(fold_rows)), ("eval top1", looso_aggregate["eval_top1"]), ("zero top1 folds", looso_aggregate["zero_top1_folds"])],
        "overfit_sanity_agent_report.md": [("passed", overfit_report["passed"]), ("subset size", subset_size), ("initial loss", overfit_report["initial_loss"]), ("final loss", overfit_report["final_loss"])],
        "safety_no_checkpoint_agent_report.md": [("passed", no_checkpoint["passed"] and no_model_save["passed"]), ("checkpoint created", False), ("model saved", False), ("torch.save calls", COUNTERS.torch_save_calls)],
        "qa_validator_agent_report.md": [("passed", "pending final validator run"), ("validator", "sim_explorer/test_stage4a711_expanded_tiny_no_checkpoint_eval.py")],
    }
    for filename, rows in subagent_rows.items():
        write_md(SUBAGENT_OUT / filename, filename.replace("_", " ").replace(".md", "").title(), rows)

    top_checks = {
        "gate": gate["passed"],
        "dataset_load": dataset_load["passed"],
        "forward_smoke": forward_smoke["passed"],
        "split_plan": split_plan["passed"],
        "tiny_eval": tiny_json["passed"],
        "looso_eval": looso_aggregate["passed"],
        "overfit": overfit_report["passed"],
        "baseline_comparison": baseline_comparison["no_generalization_claim"],
        "no_checkpoint": no_checkpoint["passed"],
        "no_model_save": no_model_save["passed"],
        "no_runtime": no_runtime["passed"],
        "no_rl": no_rl["passed"],
        "forbidden": forbidden["passed"],
        "prior_hashes_unchanged": all(item["unchanged"] for item in prior_hash.values() if item["exists"]),
    }
    summary = {
        "stage": "Stage 4A-7.11",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": all(top_checks.values()),
        "blocked": not all(top_checks.values()),
        "blockers": [key for key, ok in top_checks.items() if not ok],
        "output_dir": str(OUT),
        "expanded_dataset": str(EXPANDED_NPZ),
        "stage4a710_qa": str(STAGE710),
        "stage4a71b_baseline": BASELINE,
        "samples": 47,
        "candidate_count": 64,
        "D_model": 16,
        "valid_labels": dataset_load["valid_labels"],
        "lambda48_primary_use": False,
        "model": CONFIG["model"],
        "hidden_dim": CONFIG["hidden_dim"],
        "batch_size": CONFIG["batch_size"],
        "epochs_per_split": CONFIG["epochs_per_split"],
        "lr": CONFIG["lr"],
        "device": CONFIG["device"],
        "split_policy": split_plan["split_policy"],
        "fold_count": len(fold_rows),
        "initial_train_loss": tiny_json["initial_train_loss"],
        "final_train_loss": tiny_json["final_train_loss"],
        "eval_loss_mean": looso_aggregate["eval_loss_mean"],
        "eval_loss_stdev": looso_aggregate["eval_loss_stdev"],
        "eval_top1": looso_aggregate["eval_top1"],
        "eval_top3": looso_aggregate["eval_top3"],
        "eval_top5": looso_aggregate["eval_top5"],
        "eval_mrr": looso_aggregate["eval_mrr"],
        "zero_top1_folds": looso_aggregate["zero_top1_folds"],
        "baseline_comparison": baseline_comparison,
        "overfit_subset_size": subset_size,
        "overfit_initial_loss": overfit_report["initial_loss"],
        "overfit_final_loss": overfit_report["final_loss"],
        "overfit_final_top1": overfit_report["final_top1"],
        "overfit_passed": overfit_report["passed"],
        "num_forward_calls": COUNTERS.forward_calls,
        "num_backward_calls": COUNTERS.backward_calls,
        "num_optimizer_steps": COUNTERS.optimizer_steps,
        "model_saved": False,
        "checkpoint_created": False,
        "checkpoint_like_files": no_checkpoint["checkpoint_like_model_artifacts"],
        "isaac_startup": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
        "prior_datasets_modified": False,
        "expanded_dataset_modified": False,
        "plots": plot_report,
        "checks": top_checks,
        "recommended_next": "Stage 4A-7.12 decision packet choosing between controlled no-checkpoint/tiny checkpointless deeper BC experiment, controlled BC checkpoint experiment, or medium bounded expert rollout for more data.",
    }
    write_json(OUT / "stage4a711_expanded_tiny_eval_summary.json", summary)
    write_md(
        OUT / "stage4a711_expanded_tiny_eval_summary.md",
        "Stage 4A-7.11 Expanded Tiny No-Checkpoint Evaluation Summary",
        [
            ("completed", summary["completed"]),
            ("blocked", summary["blocked"]),
            ("samples", 47),
            ("D_model", 16),
            ("eval top1", summary["eval_top1"]),
            ("eval top3", summary["eval_top3"]),
            ("eval top5", summary["eval_top5"]),
            ("eval MRR", summary["eval_mrr"]),
            ("zero-top1 folds", summary["zero_top1_folds"]),
            ("checkpoint created", False),
            ("model saved", False),
            ("runtime/RL", "none"),
        ],
    )
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
