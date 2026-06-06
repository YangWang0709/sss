#!/usr/bin/env python3
"""Stage 4A-7.11 bounded tiny BC dry-run on the Stage 4A-7.9 artifact.

This is intentionally a tiny in-memory training smoke, not full BC training.
It never writes model weights or checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path("/home/ubuntu22/sc_explorer_ws")
STAGE79_DIR = PROJECT_ROOT / "outputs/stage4a79_stage4a714_compatible_no_training_import"
STAGE710_DIR = PROJECT_ROOT / "outputs/stage4a710_stage4a79_no_training_qa_readiness"
OUTPUT_DIR = PROJECT_ROOT / "outputs/stage4a711_stage4a79_bounded_tiny_bc_dry_run"
EXPANDED_NPZ = STAGE79_DIR / "stage4a79_stage4a714_compatible_expanded_dataset_55.npz"
ADAPTER_NPZ = STAGE79_DIR / "stage4a79_stage4a714_adapter_dataset_25.npz"
READINESS_JSON = STAGE710_DIR / "stage4a710_stage4a79_no_training_qa_readiness_summary.json"

CONFIG = {
    "stage": "Stage 4A-7.11",
    "mode": "bounded_tiny_bc_dry_run_no_checkpoint",
    "seed": 11,
    "device": "cpu",
    "policy_class": "CandidateMLPPolicy",
    "hidden_dim": 64,
    "batch_size": 8,
    "max_optimizer_steps": 8,
    "max_epochs": 2,
    "lr": 1.0e-3,
    "weight_decay": 0.0,
    "loss": "masked 64-way cross entropy",
    "optimizer": "Adam",
    "save_model": False,
    "save_checkpoint": False,
    "save_state": False,
    "full_training": False,
    "runtime": False,
    "rl_gdpo_ppo": False,
}


class Counters:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.backward_calls = 0
        self.optimizer_steps = 0


COUNTERS = Counters()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, title: str, rows: list[tuple[str, Any]], extra: str = "") -> None:
    lines = [f"# {title}", "", "| field | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    if extra:
        lines.extend(["", extra.rstrip()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
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


def finalize_hash_report(report: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for item in report.values():
        after = sha256_file(Path(item["path"]))
        item["sha256_after"] = after
        item["unchanged"] = item["sha256_before"] == after
    return report


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {
            "sample_id": np.asarray(data["sample_id"]).astype(str),
            "features": np.asarray(data["candidate_features_model"], dtype=np.float32),
            "valid": np.asarray(data["candidate_valid_mask"], dtype=bool),
            "labels": np.asarray(data["expert_action_index_primary"], dtype=np.int64),
            "split_id": np.asarray(data["split_id"], dtype=np.int64),
            "source_stage_id": np.asarray(data["source_stage_id"], dtype=np.int64),
            "quality_keep": np.asarray(data["quality_keep_mask"], dtype=bool),
            "lambda48": np.asarray(data["expert_action_index_lambda48_shadow"], dtype=np.int64),
        }
    return arrays


def make_batch(arrays: dict[str, np.ndarray], indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = np.asarray(indices, dtype=np.int64)
    features = torch.from_numpy(np.ascontiguousarray(arrays["features"][idx])).float().to(device)
    valid = torch.from_numpy(np.ascontiguousarray(arrays["valid"][idx])).bool().to(device)
    labels = torch.from_numpy(np.ascontiguousarray(arrays["labels"][idx])).long().to(device)
    return features, valid, labels


def minibatches(indices: list[int], batch_size: int, shuffle: bool) -> list[list[int]]:
    order = list(indices)
    if shuffle:
        random.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def metric_dict(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        loss = float(F.cross_entropy(logits, labels).detach().cpu().item())
        topk = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
        labels_view = labels.view(-1, 1)
        top1 = float((topk[:, :1] == labels_view).any(dim=1).float().mean().detach().cpu().item())
        top3 = float((topk[:, : min(3, topk.shape[1])] == labels_view).any(dim=1).float().mean().detach().cpu().item())
        top5 = float((topk[:, : min(5, topk.shape[1])] == labels_view).any(dim=1).float().mean().detach().cpu().item())
        ranks = torch.argsort(logits, dim=1, descending=True)
        rank_positions = (ranks == labels_view).float().argmax(dim=1).float() + 1.0
        mrr = float((1.0 / rank_positions).mean().detach().cpu().item())
    return {"loss": loss, "top1": top1, "top3": top3, "top5": top5, "mrr": mrr}


def evaluate(model: torch.nn.Module, arrays: dict[str, np.ndarray], indices: list[int], batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    weighted = {"loss": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0}
    sample_count = 0
    batch_rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(minibatches(indices, batch_size=batch_size, shuffle=False)):
            features, valid, labels = make_batch(arrays, batch, device)
            logits = model(features, valid)
            COUNTERS.forward_calls += 1
            metrics = metric_dict(logits, labels)
            for key, value in metrics.items():
                weighted[key] += value * len(batch)
            sample_count += len(batch)
            batch_rows.append({"batch_index": batch_index, "sample_count": len(batch), **metrics})
    aggregate = {key: float(value / sample_count) if sample_count else float("nan") for key, value in weighted.items()}
    aggregate["sample_count"] = int(sample_count)
    aggregate["batch_count"] = len(batch_rows)
    aggregate["metrics_finite"] = all(math.isfinite(float(v)) for k, v in aggregate.items() if k not in {"sample_count", "batch_count"})
    return {"aggregate": aggregate, "batches": batch_rows}


def train_tiny(model: torch.nn.Module, arrays: dict[str, np.ndarray], train_indices: list[int], device: torch.device) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    model.train()
    step_rows: list[dict[str, Any]] = []
    step = 0
    for epoch in range(CONFIG["max_epochs"]):
        for batch in minibatches(train_indices, batch_size=CONFIG["batch_size"], shuffle=True):
            if step >= CONFIG["max_optimizer_steps"]:
                break
            features, valid, labels = make_batch(arrays, batch, device)
            logits = model(features, valid)
            COUNTERS.forward_calls += 1
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            COUNTERS.backward_calls += 1
            optimizer.step()
            COUNTERS.optimizer_steps += 1
            loss_value = float(loss.detach().cpu().item())
            step_rows.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "sample_count": len(batch),
                    "loss": loss_value,
                    "finite_loss": math.isfinite(loss_value),
                }
            )
            step += 1
        if step >= CONFIG["max_optimizer_steps"]:
            break
    return {
        "ran": True,
        "optimizer": CONFIG["optimizer"],
        "optimizer_step_count": int(step),
        "max_optimizer_steps": CONFIG["max_optimizer_steps"],
        "initial_loss": step_rows[0]["loss"] if step_rows else None,
        "final_loss": step_rows[-1]["loss"] if step_rows else None,
        "loss_delta": (step_rows[-1]["loss"] - step_rows[0]["loss"]) if len(step_rows) >= 2 else None,
        "all_losses_finite": all(row["finite_loss"] for row in step_rows),
        "steps": step_rows,
    }


def scan_checkpoint_like(output_dir: Path) -> list[str]:
    hits: list[str] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(output_dir))
        lower = rel.lower()
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".tar"}:
            hits.append(rel)
        elif any(token in lower for token in ["model_weights", "optimizer_state", "state_dict", "replay_buffer"]):
            hits.append(rel)
    return sorted(hits)


def build_html(summary: dict[str, Any], train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> str:
    esc = lambda value: html.escape("" if value is None else str(value))
    train_table = "".join(
        f"<tr><td>{row['step']}</td><td>{row['epoch']}</td><td>{row['sample_count']}</td><td>{row['loss']:.6f}</td></tr>"
        for row in train_rows
    )
    eval_table = "".join(
        f"<tr><td>{esc(row['split'])}</td><td>{row['sample_count']}</td><td>{row['loss']:.6f}</td><td>{row['top1']:.3f}</td><td>{row['top3']:.3f}</td><td>{row['top5']:.3f}</td><td>{row['mrr']:.3f}</td></tr>"
        for row in eval_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage 4A-7.11 Tiny BC Dry-Run</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: #f6f8fa; color: #172026; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin-top: 24px; font-size: 20px; letter-spacing: 0; }}
    p {{ color: #5d6873; line-height: 1.5; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: #fff; border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; font-size: 24px; margin-bottom: 4px; }}
    .boundary {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 12px; color: #7c2d12; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d8dee4; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e9ef; text-align: left; font-size: 14px; }}
    th {{ background: #eef2f5; }}
  </style>
</head>
<body>
<main>
  <h1>Stage 4A-7.11 Bounded Tiny BC Dry-Run</h1>
  <p>Generated at {esc(summary['generated_at_utc'])}. This is a tiny in-memory BC smoke on the Stage 4A-7.9 compatible artifact.</p>
  <section class="metrics">
    <div class="metric"><b>{esc(summary['decision'])}</b><span>decision</span></div>
    <div class="metric"><b>{esc(summary['sample_count'])}</b><span>samples</span></div>
    <div class="metric"><b>{esc(summary['optimizer_step_count'])}</b><span>optimizer steps</span></div>
    <div class="metric"><b>{esc(summary['blocker_count'])}</b><span>blockers</span></div>
  </section>
  <section class="boundary">
    No checkpoint or model file was written. Isaac, map_predict, rollout, long rollout, RL/GDPO/PPO, and runtime execution were not run.
  </section>
  <h2>Training Steps</h2>
  <table><thead><tr><th>step</th><th>epoch</th><th>samples</th><th>loss</th></tr></thead><tbody>{train_table}</tbody></table>
  <h2>Evaluation</h2>
  <table><thead><tr><th>split</th><th>samples</th><th>loss</th><th>top1</th><th>top3</th><th>top5</th><th>MRR</th></tr></thead><tbody>{eval_table}</tbody></table>
</main>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    source_hash = hash_report(
        {
            "stage4a79_expanded_npz": EXPANDED_NPZ,
            "stage4a79_adapter_npz": ADAPTER_NPZ,
            "stage4a710_readiness_json": READINESS_JSON,
        }
    )

    blockers: list[str] = []
    warnings: list[str] = []
    for path in [EXPANDED_NPZ, ADAPTER_NPZ, READINESS_JSON]:
        if not path.exists():
            blockers.append(f"missing_input:{path}")
    if blockers:
        summary = {
            "completed": False,
            "blocked": True,
            "decision": "blocked",
            "blockers": blockers,
            "warnings": warnings,
            "generated_at_utc": utc_now(),
        }
        write_json(OUTPUT_DIR / "stage4a711_stage4a79_tiny_bc_dry_run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(1)

    readiness = read_json(READINESS_JSON)
    if readiness.get("readiness_decision") != "ready_for_tiny_bc_dry_run_consideration":
        blockers.append("stage4a710_not_ready_for_tiny_bc_dry_run_consideration")

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from ssc_exploration.ssc_network.il.policy import CandidateMLPPolicy
    from ssc_exploration.ssc_network.il.sim_expert_bc_dataset import SimExpertBCDataset

    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    arrays = load_arrays(EXPANDED_NPZ)
    dataset = SimExpertBCDataset(EXPANDED_NPZ, strict_keep_only=False)
    strict_dataset = SimExpertBCDataset(EXPANDED_NPZ, strict_keep_only=True)

    sample_count, candidate_count, input_dim = arrays["features"].shape
    labels = arrays["labels"]
    valid = arrays["valid"]
    valid_labels = bool(np.all(valid[np.arange(sample_count), labels]))
    if not valid_labels:
        blockers.append("primary_labels_not_valid")
    if sample_count != 55 or candidate_count != 64 or input_dim != 16:
        blockers.append(f"unexpected_shape:{arrays['features'].shape}")
    if len(strict_dataset) != sample_count:
        blockers.append(f"strict_keep_only_len_mismatch:{len(strict_dataset)}!={sample_count}")

    train_idx = np.flatnonzero((arrays["split_id"] == 0) & arrays["quality_keep"]).astype(int).tolist()
    val_idx = np.flatnonzero((arrays["split_id"] == 1) & arrays["quality_keep"]).astype(int).tolist()
    test_idx = np.flatnonzero((arrays["split_id"] == 2) & arrays["quality_keep"]).astype(int).tolist()
    all_idx = np.flatnonzero(arrays["quality_keep"]).astype(int).tolist()
    if not train_idx or not val_idx or not test_idx:
        blockers.append("missing_train_val_test_split")

    policy = CandidateMLPPolicy(input_dim=input_dim, hidden_dim=CONFIG["hidden_dim"]).to(device)

    forward_before = evaluate(policy, arrays, all_idx[: min(16, len(all_idx))], CONFIG["batch_size"], device)
    tiny_train = train_tiny(policy, arrays, train_idx, device)
    eval_reports = {
        "train": evaluate(policy, arrays, train_idx, CONFIG["batch_size"], device),
        "val": evaluate(policy, arrays, val_idx, CONFIG["batch_size"], device),
        "test": evaluate(policy, arrays, test_idx, CONFIG["batch_size"], device),
        "all": evaluate(policy, arrays, all_idx, CONFIG["batch_size"], device),
        "imported_stage714": evaluate(
            policy,
            arrays,
            np.flatnonzero((arrays["source_stage_id"] == 714) & arrays["quality_keep"]).astype(int).tolist(),
            CONFIG["batch_size"],
            device,
        ),
        "original_stage613": evaluate(
            policy,
            arrays,
            np.flatnonzero((arrays["source_stage_id"] == 613) & arrays["quality_keep"]).astype(int).tolist(),
            CONFIG["batch_size"],
            device,
        ),
    }

    if tiny_train["optimizer_step_count"] <= 0:
        blockers.append("optimizer_step_count_zero")
    if tiny_train["optimizer_step_count"] > CONFIG["max_optimizer_steps"]:
        blockers.append("optimizer_step_count_exceeded_cap")
    if not tiny_train["all_losses_finite"]:
        blockers.append("nonfinite_tiny_train_loss")

    eval_rows: list[dict[str, Any]] = []
    for split_name, report in eval_reports.items():
        row = {"split": split_name, **report["aggregate"]}
        eval_rows.append(row)
        if not report["aggregate"]["metrics_finite"]:
            blockers.append(f"nonfinite_eval_metrics:{split_name}")

    checkpoint_like = scan_checkpoint_like(OUTPUT_DIR)
    if checkpoint_like:
        blockers.append(f"checkpoint_like_outputs_present:{checkpoint_like}")

    label_lineage = {
        "lambda48_role": "shadow/baseline only",
        "labels_recomputed_from_lambda48": False,
        "primary_label_source": "stage4a613_uncertainty_bonus_executed_primary plus stage4a714_uncertainty_bonus_executed_clean_candidate_compatible_import",
        "lambda48_primary_use": False,
        "primary_differs_from_lambda48_count": int(np.sum(arrays["labels"] != arrays["lambda48"])),
    }
    split_report = {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "all_count": len(all_idx),
    }
    dataset_load_report = {
        "passed": not blockers,
        "dataset_class": "SimExpertBCDataset",
        "dataset_len": len(dataset),
        "strict_keep_only_len": len(strict_dataset),
        "sample_count": sample_count,
        "candidate_count": candidate_count,
        "input_dim": input_dim,
        "valid_primary_labels": valid_labels,
        "finite_features": bool(np.isfinite(arrays["features"]).all()),
    }
    safety_report = {
        "tiny_bc_dry_run_training": True,
        "full_training": False,
        "optimizer_step_count": COUNTERS.optimizer_steps,
        "backward_call_count": COUNTERS.backward_calls,
        "max_optimizer_steps": CONFIG["max_optimizer_steps"],
        "model_saved": False,
        "checkpoint_created": False,
        "checkpoint_like_outputs": checkpoint_like,
        "torch_save_called": False,
        "state_written": False,
    }
    no_runtime_report = {
        "isaac_startup": False,
        "capture": False,
        "map_predict": False,
        "runtime_execution": False,
        "action_execution": False,
        "rollout": False,
        "long_rollout": False,
    }
    no_rl_report = {"rl": False, "gdpo": False, "ppo": False, "policy_optimization_rl": False}

    summary = {
        "stage": "Stage 4A-7.11 Stage 4A-7.9 bounded tiny BC dry-run gate",
        "completed": not blockers,
        "blocked": bool(blockers),
        "decision": "tiny_bc_dry_run_passed_no_checkpoint" if not blockers else "blocked",
        "generated_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "input_artifact": str(EXPANDED_NPZ),
        "sample_count": sample_count,
        "candidate_count": candidate_count,
        "input_dim": input_dim,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "optimizer_step_count": COUNTERS.optimizer_steps,
        "backward_call_count": COUNTERS.backward_calls,
        "forward_call_count": COUNTERS.forward_calls,
        "max_optimizer_steps": CONFIG["max_optimizer_steps"],
        "full_training": False,
        "checkpoint_created": False,
        "model_saved": False,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "blockers": blockers,
        "warnings": warnings,
        "config": CONFIG,
        "stage4a710_readiness_decision": readiness.get("readiness_decision"),
        "forward_before": forward_before["aggregate"],
        "tiny_train": {key: value for key, value in tiny_train.items() if key != "steps"},
        "eval": {name: report["aggregate"] for name, report in eval_reports.items()},
        "label_lineage": label_lineage,
        "safety": safety_report,
        "no_runtime": no_runtime_report,
        "no_rl_gdpo_ppo": no_rl_report,
        "recommended_next_step": "Stage 4A-7.12 should be a review/readiness decision packet before any checkpointed BC or runtime use.",
    }

    source_hash = finalize_hash_report(source_hash)
    write_json(OUTPUT_DIR / "stage4a711_stage4a79_tiny_bc_dry_run_summary.json", summary)
    write_json(OUTPUT_DIR / "stage4a711_tiny_bc_config.json", CONFIG)
    write_json(OUTPUT_DIR / "dataset_load_report.json", dataset_load_report)
    write_json(OUTPUT_DIR / "split_plan_report.json", split_report)
    write_json(OUTPUT_DIR / "forward_only_smoke_report.json", forward_before)
    write_json(OUTPUT_DIR / "tiny_train_metrics.json", tiny_train)
    write_csv(OUTPUT_DIR / "tiny_train_metrics.csv", tiny_train["steps"])
    write_json(OUTPUT_DIR / "eval_metrics.json", eval_reports)
    write_csv(OUTPUT_DIR / "eval_metrics.csv", eval_rows)
    write_json(OUTPUT_DIR / "label_lineage_report.json", label_lineage)
    write_json(OUTPUT_DIR / "safety_no_checkpoint_report.json", safety_report)
    write_json(OUTPUT_DIR / "no_runtime_report.json", no_runtime_report)
    write_json(OUTPUT_DIR / "no_rl_gdpo_ppo_report.json", no_rl_report)
    write_json(OUTPUT_DIR / "source_hash_report.json", source_hash)
    write_json(OUTPUT_DIR / "stage4a711_readiness_decision.json", {
        "decision": summary["decision"],
        "tiny_bc_dry_run_passed": not blockers,
        "ready_for_checkpointed_training": False,
        "ready_for_runtime": False,
        "ready_for_rl_gdpo_ppo": False,
        "requires_separate_stage4a712_decision": True,
    })

    write_md_table(
        OUTPUT_DIR / "stage4a711_stage4a79_tiny_bc_dry_run_summary.md",
        "Stage 4A-7.11 Bounded Tiny BC Dry-Run Summary",
        [
            ("decision", summary["decision"]),
            ("samples", sample_count),
            ("shape", list(arrays["features"].shape)),
            ("train/val/test", f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}"),
            ("optimizer steps", COUNTERS.optimizer_steps),
            ("max optimizer steps", CONFIG["max_optimizer_steps"]),
            ("initial train loss", tiny_train["initial_loss"]),
            ("final train loss", tiny_train["final_loss"]),
            ("eval all top1/top3/top5", f"{summary['eval']['all']['top1']:.3f}/{summary['eval']['all']['top3']:.3f}/{summary['eval']['all']['top5']:.3f}"),
            ("blockers", blockers or "none"),
            ("checkpoint created", False),
            ("model saved", False),
            ("Isaac/map_predict/rollout/RL", "not run"),
        ],
        "This was a bounded in-memory dry-run only. It does not authorize checkpointed BC training, runtime use, or RL/GDPO/PPO.",
    )
    write_md_table(OUTPUT_DIR / "dataset_load_report.md", "Dataset Load Report", list(dataset_load_report.items()))
    write_md_table(OUTPUT_DIR / "safety_no_checkpoint_report.md", "Safety No-Checkpoint Report", list(safety_report.items()))
    write_md_table(OUTPUT_DIR / "no_runtime_report.md", "No Runtime Report", list(no_runtime_report.items()))
    write_md_table(OUTPUT_DIR / "no_rl_gdpo_ppo_report.md", "No RL/GDPO/PPO Report", list(no_rl_report.items()))
    (OUTPUT_DIR / "recommended_next_step.md").write_text(
        "Recommended next: Stage 4A-7.12 review/readiness decision packet. "
        "Do not create checkpoints, run runtime, or start RL/GDPO/PPO without a separate explicit gate.\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage4a711_stage4a79_tiny_bc_dry_run_index.html").write_text(
        build_html(summary, tiny_train["steps"], eval_rows),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "git_status_after.txt").write_text(git_status(), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
