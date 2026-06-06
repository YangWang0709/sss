#!/usr/bin/env python3
"""Generate Stage 4A-7.12 review/readiness decision packet.

This stage is decision-only. It reads Stage 4A-7.9/7.10/7.11 evidence and
does not train, save checkpoints, run runtime, or start RL.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/stage4a712_stage4a79_review_readiness_decision_packet"
STAGE79 = ROOT / "outputs/stage4a79_stage4a714_compatible_no_training_import"
STAGE710 = ROOT / "outputs/stage4a710_stage4a79_no_training_qa_readiness"
STAGE711 = ROOT / "outputs/stage4a711_stage4a79_bounded_tiny_bc_dry_run"

STAGE79_SUMMARY = STAGE79 / "stage4a79_stage4a714_compatible_import_summary.json"
STAGE79_EXPANDED = STAGE79 / "stage4a79_stage4a714_compatible_expanded_dataset_55.npz"
STAGE710_SUMMARY = STAGE710 / "stage4a710_stage4a79_no_training_qa_readiness_summary.json"
STAGE711_SUMMARY = STAGE711 / "stage4a711_stage4a79_tiny_bc_dry_run_summary.json"
STAGE711_DECISION = STAGE711 / "stage4a711_readiness_decision.json"
STAGE711_SAFETY = STAGE711 / "safety_no_checkpoint_report.json"
STAGE711_NO_RUNTIME = STAGE711 / "no_runtime_report.json"
STAGE711_NO_RL = STAGE711 / "no_rl_gdpo_ppo_report.json"
STAGE711_EVAL = STAGE711 / "eval_metrics.json"
STAGE711_TRAIN = STAGE711 / "tiny_train_metrics.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def git_status() -> str:
    return run(["git", "status", "--short"]).stdout


def git_head() -> str:
    return run(["git", "rev-parse", "HEAD"]).stdout.strip()


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


def write_md_table(path: Path, title: str, rows: list[tuple[str, Any]], extra: str = "") -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value).replace("|", "\\|")

    lines = [f"# {title}", "", "| field | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| `{key}` | {fmt(value)} |")
    if extra:
        lines.extend(["", extra.rstrip()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def negative_scope() -> dict[str, Any]:
    return {
        "stage4a712_bc_training": False,
        "stage4a712_optimizer_step_count": 0,
        "stage4a712_backward_count": 0,
        "stage4a712_model_saved": False,
        "stage4a712_checkpoint_created": False,
        "stage4a712_torch_save_calls": 0,
        "stage4a712_isaac_startup": False,
        "stage4a712_map_predict": False,
        "stage4a712_runtime_execution": False,
        "stage4a712_rollout": False,
        "stage4a712_long_rollout": False,
        "stage4a712_label_promotion": False,
        "stage4a712_rl": False,
        "stage4a712_gdpo": False,
        "stage4a712_ppo": False,
    }


def source_hash_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for key, path in paths.items()
    }


def scan_checkpoint_like() -> list[str]:
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


def build_html(summary: dict[str, Any], options: list[dict[str, Any]], risks: list[dict[str, Any]]) -> str:
    esc = lambda value: html.escape("" if value is None else str(value))
    option_rows = "".join(
        "<tr>"
        f"<td>{esc(row['id'])}</td>"
        f"<td>{esc(row['name'])}</td>"
        f"<td>{esc(row['decision'])}</td>"
        f"<td>{esc(row['reason'])}</td>"
        "</tr>"
        for row in options
    )
    risk_rows = "".join(
        "<tr>"
        f"<td>{esc(row['risk'])}</td>"
        f"<td>{esc(row['level'])}</td>"
        f"<td>{esc(row['mitigation'])}</td>"
        "</tr>"
        for row in risks
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage 4A-7.12 Review Readiness Decision</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; color: #172026; background: #f6f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin-top: 24px; font-size: 20px; letter-spacing: 0; }}
    p {{ color: #5d6873; line-height: 1.5; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: #fff; border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; font-size: 22px; margin-bottom: 4px; }}
    .boundary {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 12px; color: #7c2d12; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d8dee4; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e5e9ef; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #eef2f5; }}
  </style>
</head>
<body>
<main>
  <h1>Stage 4A-7.12 Review / Readiness Decision Packet</h1>
  <p>Generated at {esc(summary['generated_at_utc'])}. This packet reviews Stage 4A-7.9, 7.10, and 7.11 evidence only.</p>
  <section class="metrics">
    <div class="metric"><b>{esc(summary['decision'])}</b><span>decision</span></div>
    <div class="metric"><b>{esc(summary['selected_next_stage'])}</b><span>selected next stage</span></div>
    <div class="metric"><b>{esc(summary['stage711_optimizer_steps'])}</b><span>prior tiny steps</span></div>
    <div class="metric"><b>{esc(summary['blocker_count'])}</b><span>blockers</span></div>
  </section>
  <section class="boundary">
    Stage 4A-7.12 did not train, did not perform optimizer steps, did not save a model/checkpoint, did not start Isaac, did not run map_predict/rollout/runtime, and did not run RL/GDPO/PPO.
  </section>
  <h2>Option Matrix</h2>
  <table><thead><tr><th>ID</th><th>Option</th><th>Decision</th><th>Reason</th></tr></thead><tbody>{option_rows}</tbody></table>
  <h2>Risk Register</h2>
  <table><thead><tr><th>Risk</th><th>Level</th><th>Mitigation</th></tr></thead><tbody>{risk_rows}</tbody></table>
</main>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    blockers: list[str] = []
    required = {
        "stage4a79_summary": STAGE79_SUMMARY,
        "stage4a79_expanded": STAGE79_EXPANDED,
        "stage4a710_summary": STAGE710_SUMMARY,
        "stage4a711_summary": STAGE711_SUMMARY,
        "stage4a711_decision": STAGE711_DECISION,
        "stage4a711_safety": STAGE711_SAFETY,
        "stage4a711_no_runtime": STAGE711_NO_RUNTIME,
        "stage4a711_no_rl": STAGE711_NO_RL,
        "stage4a711_eval": STAGE711_EVAL,
        "stage4a711_train": STAGE711_TRAIN,
    }
    for key, path in required.items():
        if not path.exists():
            blockers.append(f"missing_input:{key}:{path}")

    if blockers:
        summary = {
            "stage": "Stage 4A-7.12 review/readiness decision packet",
            "completed": False,
            "blocked": True,
            "decision": "blocked",
            "blockers": blockers,
            "blocker_count": len(blockers),
            "generated_at_utc": utc_now(),
            "output_dir": str(OUT),
        }
        write_json(OUT / "stage4a712_review_readiness_decision_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    s79 = read_json(STAGE79_SUMMARY)
    s710 = read_json(STAGE710_SUMMARY)
    s711 = read_json(STAGE711_SUMMARY)
    d711 = read_json(STAGE711_DECISION)
    safety711 = read_json(STAGE711_SAFETY)
    no_runtime711 = read_json(STAGE711_NO_RUNTIME)
    no_rl711 = read_json(STAGE711_NO_RL)
    eval711 = read_json(STAGE711_EVAL)
    train711 = read_json(STAGE711_TRAIN)

    with np.load(STAGE79_EXPANDED, allow_pickle=False) as expanded:
        expanded_shape = list(expanded["candidate_features_model"].shape)
        labels = expanded["expert_action_index_primary"].astype(np.int64)
        valid = expanded["candidate_valid_mask"].astype(bool)
        valid_labels = bool(np.all(valid[np.arange(labels.shape[0]), labels]))
        source_stage_counts = {
            str(int(stage)): int(np.sum(expanded["source_stage_id"] == stage))
            for stage in sorted(set(expanded["source_stage_id"].astype(int).tolist()))
        }

    evidence = {
        "git_head": git_head(),
        "stage4a79_completed": s79.get("completed") is True,
        "stage4a79_expanded_rows": s79.get("expanded_compatible_sample_count"),
        "stage4a79_shape": expanded_shape,
        "stage4a79_valid_labels": valid_labels,
        "stage4a79_source_stage_counts": source_stage_counts,
        "stage4a710_readiness_decision": s710.get("readiness_decision"),
        "stage4a710_blocker_count": s710.get("blocker_count"),
        "stage4a711_decision": s711.get("decision"),
        "stage4a711_sample_count": s711.get("sample_count"),
        "stage4a711_optimizer_steps": s711.get("optimizer_step_count"),
        "stage4a711_backward_calls": s711.get("backward_call_count"),
        "stage4a711_checkpoint_created": s711.get("checkpoint_created"),
        "stage4a711_model_saved": s711.get("model_saved"),
        "stage4a711_full_training": s711.get("full_training"),
        "stage4a711_eval_all_top1": s711.get("eval", {}).get("all", {}).get("top1"),
        "stage4a711_eval_all_top3": s711.get("eval", {}).get("all", {}).get("top3"),
        "stage4a711_eval_all_top5": s711.get("eval", {}).get("all", {}).get("top5"),
        "stage4a711_imported_top1": s711.get("eval", {}).get("imported_stage714", {}).get("top1"),
        "stage4a711_original_top1": s711.get("eval", {}).get("original_stage613", {}).get("top1"),
        "stage4a711_initial_loss": train711.get("initial_loss"),
        "stage4a711_final_loss": train711.get("final_loss"),
        "stage4a711_checkpoint_like_outputs": safety711.get("checkpoint_like_outputs"),
        "stage4a711_no_runtime": no_runtime711,
        "stage4a711_no_rl_gdpo_ppo": no_rl711,
    }

    hard_checks = {
        "stage4a79_completed": evidence["stage4a79_completed"],
        "stage4a79_shape_55_64_16": expanded_shape == [55, 64, 16],
        "stage4a79_valid_labels": valid_labels,
        "stage4a710_ready": evidence["stage4a710_readiness_decision"] == "ready_for_tiny_bc_dry_run_consideration",
        "stage4a710_no_blockers": evidence["stage4a710_blocker_count"] == 0,
        "stage4a711_passed": evidence["stage4a711_decision"] == "tiny_bc_dry_run_passed_no_checkpoint",
        "stage4a711_steps_bounded": evidence["stage4a711_optimizer_steps"] == 8,
        "stage4a711_no_checkpoint": evidence["stage4a711_checkpoint_created"] is False and safety711.get("checkpoint_created") is False,
        "stage4a711_no_model_save": evidence["stage4a711_model_saved"] is False and safety711.get("model_saved") is False,
        "stage4a711_not_full_training": evidence["stage4a711_full_training"] is False and safety711.get("full_training") is False,
        "stage4a711_no_runtime": all(value is False for value in no_runtime711.values()),
        "stage4a711_no_rl": all(value is False for value in no_rl711.values()),
        "stage4a711_metrics_finite": all(
            finite(value)
            for report in eval711.values()
            for key, value in report.get("aggregate", {}).items()
            if key not in {"sample_count", "batch_count", "metrics_finite"}
        ),
        "lambda48_shadow_only": s711.get("label_lineage", {}).get("lambda48_role") == "shadow/baseline only",
        "labels_not_recomputed_from_lambda48": s711.get("label_lineage", {}).get("labels_recomputed_from_lambda48") is False,
    }
    for key, ok in hard_checks.items():
        if not ok:
            blockers.append(key)

    option_rows = [
        {
            "id": "A",
            "name": "More no-checkpoint diagnostics",
            "decision": "allowed_not_selected",
            "reason": "Useful if the user wants more confidence, but 7.11 already proved the loader/model/loss path works on the 55-row artifact.",
        },
        {
            "id": "B",
            "name": "Bounded checkpointed BC experiment design/preflight",
            "decision": "selected_next_design_preflight_only",
            "reason": "7.9 compatibility, 7.10 QA, and 7.11 tiny dry-run all passed. The next faithful step is design/preflight, not immediate checkpoint writing.",
        },
        {
            "id": "C",
            "name": "Runtime or medium rollout",
            "decision": "not_approved",
            "reason": "A learned checkpoint does not exist yet, and runtime/rollout should not be coupled to this decision packet.",
        },
        {
            "id": "D",
            "name": "RL/GDPO/PPO",
            "decision": "not_approved",
            "reason": "The project is still in BC evidence gathering; RL remains out of scope.",
        },
    ]
    risk_rows = [
        {
            "risk": "small dataset overfit",
            "level": "medium",
            "mitigation": "Stage 4A-7.13 should cap epochs/steps, keep no-runtime boundaries, and report train/val/test plus original/imported subgroup metrics.",
        },
        {
            "risk": "checkpoint appears more authoritative than evidence supports",
            "level": "medium",
            "mitigation": "Any checkpointed experiment must be explicitly labeled experimental and cannot authorize runtime/RL by itself.",
        },
        {
            "risk": "lambda48 accidentally treated as primary",
            "level": "high",
            "mitigation": "Keep validator checks that lambda48 is shadow/baseline only and labels were not recomputed from lambda48.",
        },
        {
            "risk": "runtime jump before policy QA",
            "level": "high",
            "mitigation": "Stage 4A-7.12 approves design/preflight only; runtime, rollout, and RL remain false.",
        },
    ]

    selected_next = {
        "selected_option": "B",
        "selected_next_stage": "Stage 4A-7.13 bounded checkpointed BC experiment design/preflight only",
        "approved_to_create_design_preflight": not blockers,
        "approved_to_run_checkpointed_bc_now": False,
        "approved_to_save_checkpoint_now": False,
        "approved_to_run_runtime_now": False,
        "approved_to_run_rollout_now": False,
        "approved_to_run_rl_gdpo_ppo_now": False,
        "requires_separate_user_approval_for_execution": True,
        "reason": "Proceed only to design/preflight because the tiny dry-run passed but evidence volume is still modest.",
    }

    stage712_negative = negative_scope()
    checkpoint_like = scan_checkpoint_like()
    if checkpoint_like:
        blockers.append(f"stage4a712_checkpoint_like_outputs:{checkpoint_like}")

    summary = {
        "stage": "Stage 4A-7.12 Stage 4A-7.9 review/readiness decision packet",
        "completed": not blockers,
        "blocked": bool(blockers),
        "decision": "ready_for_stage4a713_checkpointed_bc_design_preflight_only" if not blockers else "blocked",
        "selected_option": "B",
        "selected_next_stage": selected_next["selected_next_stage"],
        "generated_at_utc": utc_now(),
        "project_root": str(ROOT),
        "output_dir": str(OUT),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "hard_checks": hard_checks,
        "evidence": evidence,
        "stage711_optimizer_steps": evidence["stage4a711_optimizer_steps"],
        "stage712_optimizer_step_count": 0,
        "stage712_training": False,
        "stage712_checkpoint_created": False,
        "stage712_model_saved": False,
        "stage712_runtime": False,
        "stage712_rl_gdpo_ppo": False,
        "label_lineage": {
            "lambda48_role": "shadow/baseline only",
            "labels_recomputed_from_lambda48": False,
            "primary_label_source": "Stage 4A-6.13 primary plus Stage 4A-7.14 compatible uncertainty-bonus import",
        },
        "recommended_next_step": selected_next["selected_next_stage"],
    }

    source_hashes = source_hash_report(required)
    packet_manifest = {
        "stage": summary["stage"],
        "output_dir": str(OUT),
        "files": {
            "summary_json": str(OUT / "stage4a712_review_readiness_decision_summary.json"),
            "summary_md": str(OUT / "stage4a712_review_readiness_decision_summary.md"),
            "index_html": str(OUT / "stage4a712_review_readiness_decision_index.html"),
            "selected_next_step": str(OUT / "selected_next_step_decision.json"),
            "option_matrix": str(OUT / "decision_option_matrix.json"),
            "risk_register": str(OUT / "risk_register.json"),
        },
    }

    write_json(OUT / "loaded_evidence_report.json", evidence)
    write_md_table(OUT / "loaded_evidence_report.md", "Loaded Evidence Report", list(evidence.items()))
    write_json(OUT / "decision_option_matrix.json", {"selected_option": "B", "options": option_rows})
    write_csv(OUT / "decision_option_matrix.csv", option_rows)
    write_md_table(OUT / "decision_option_matrix.md", "Decision Option Matrix", [(row["id"], f"{row['decision']}: {row['name']}") for row in option_rows])
    write_json(OUT / "selected_next_step_decision.json", selected_next)
    write_md_table(OUT / "selected_next_step_decision.md", "Selected Next Step Decision", list(selected_next.items()))
    write_json(OUT / "risk_register.json", {"risks": risk_rows})
    write_csv(OUT / "risk_register.csv", risk_rows)
    write_md_table(OUT / "risk_register.md", "Risk Register", [(row["risk"], f"{row['level']}: {row['mitigation']}") for row in risk_rows])
    write_json(OUT / "negative_scope_report.json", stage712_negative)
    write_md_table(OUT / "negative_scope_report.md", "Stage 4A-7.12 Negative Scope", list(stage712_negative.items()))
    write_json(OUT / "no_training_report.json", {"training": False, "optimizer_step_count": 0, "backward_count": 0})
    write_json(OUT / "no_checkpoint_report.json", {"checkpoint_created": False, "model_saved": False, "torch_save_calls": 0, "checkpoint_like_outputs": checkpoint_like})
    write_json(OUT / "no_runtime_report.json", {"isaac_startup": False, "map_predict": False, "runtime_execution": False, "rollout": False, "long_rollout": False})
    write_json(OUT / "no_rl_gdpo_ppo_report.json", {"rl": False, "gdpo": False, "ppo": False})
    write_json(OUT / "lambda48_shadow_only_audit.json", summary["label_lineage"])
    write_json(OUT / "source_hash_report.json", source_hashes)
    write_json(OUT / "packet_manifest.json", packet_manifest)
    write_json(OUT / "stage4a712_review_readiness_decision_summary.json", summary)
    write_md_table(
        OUT / "stage4a712_review_readiness_decision_summary.md",
        "Stage 4A-7.12 Review Readiness Decision Summary",
        [
            ("decision", summary["decision"]),
            ("selected option", "B"),
            ("selected next stage", summary["selected_next_stage"]),
            ("7.11 optimizer steps", evidence["stage4a711_optimizer_steps"]),
            ("7.12 optimizer steps", 0),
            ("7.12 checkpoint created", False),
            ("7.12 runtime/RL", False),
            ("blockers", blockers or "none"),
        ],
        "This packet approves only the next design/preflight. It does not approve checkpoint creation, runtime, rollout, or RL.",
    )
    (OUT / "stage4a712_web_review_request.md").write_text(
        "# Stage 4A-7.12 Web Review Request\n\n"
        "Please review the Stage 4A-7.12 decision packet. The selected next step is "
        "Stage 4A-7.13 bounded checkpointed BC experiment design/preflight only. "
        "This packet does not approve checkpoint creation, runtime, rollout, or RL/GDPO/PPO.\n",
        encoding="utf-8",
    )
    (OUT / "handoff_to_stage4a713_design_preflight.md").write_text(
        "# Handoff To Stage 4A-7.13 Design/Preflight\n\n"
        "- Use the Stage 4A-7.9 compatible expanded artifact.\n"
        "- Keep checkpointed BC execution behind a separate explicit approval.\n"
        "- Require no runtime, no rollout, and no RL/GDPO/PPO in the preflight.\n"
        "- Preserve lambda48 as shadow/baseline only.\n",
        encoding="utf-8",
    )
    (OUT / "stage4a712_review_readiness_decision_index.html").write_text(
        build_html(summary, option_rows, risk_rows),
        encoding="utf-8",
    )
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
