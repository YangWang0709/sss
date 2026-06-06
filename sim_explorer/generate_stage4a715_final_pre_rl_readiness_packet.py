from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_pre_rl_bridge"
OUT = BRIDGE / "stage4a715_final_pre_rl_readiness_packet"

STAGE79 = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation/stage4a79_no_training_promotion_summary.json"
STAGE710 = ROOT / "outputs/isaac_stage4a710_expanded_dataset_qa/stage4a710_expanded_dataset_qa_summary.json"
STAGE711 = ROOT / "outputs/isaac_stage4a711_expanded_tiny_no_checkpoint_eval/stage4a711_expanded_tiny_eval_summary.json"
STAGE712 = BRIDGE / "stage4a712_pre_rl_direction_decision_packet/stage4a712_pre_rl_direction_decision_summary.json"
STAGE713 = BRIDGE / "stage4a713_medium_expert_rollout_design_preflight/stage4a713_medium_expert_rollout_design_preflight_summary.json"
STAGE714B = BRIDGE / "stage4a714b_medium_postrun_safety_audit/stage4a714b_medium_postrun_safety_audit_summary.json"

RUNTIME714 = ROOT / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"

EVIDENCE = {
    "stage4a79_no_training_promotion": STAGE79,
    "stage4a710_expanded_dataset_qa": STAGE710,
    "stage4a711_tiny_no_checkpoint_eval": STAGE711,
    "stage4a712_pre_rl_direction_decision": STAGE712,
    "stage4a713_medium_preflight": STAGE713,
    "stage4a714b_medium_postrun_safety": STAGE714B,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def md_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return text.replace("|", "\\|")


def write_md_table(path: Path, title: str, rows: list[tuple[str, Any]], extra: str = "") -> None:
    lines = [f"# {title}", "", "| field | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| `{key}` | {md_value(value)} |")
    if extra:
        lines.extend(["", extra.rstrip()])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_pair(stem: str, data: dict[str, Any], title: str, rows: list[tuple[str, Any]] | None = None, extra: str = "") -> None:
    write_json(OUT / f"{stem}.json", data)
    write_md_table(OUT / f"{stem}.md", title, rows or list(data.items()), extra)


def git_status() -> str:
    proc = subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout


def git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.strip()


def git_files() -> list[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.splitlines()


def process_hits() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,pgid,stat,etime,cmd"], text=True, capture_output=True)
    hits = []
    for line in proc.stdout.splitlines():
        if "run_stage4a714_medium_uncertainty_bonus_rollout.py" in line:
            hits.append(line.strip())
        elif "run_with_isaac_close_guard.py" in line and "Stage4A-7.14" in line:
            hits.append(line.strip())
    return hits


def stage_gate_matrix(stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s79 = stages["stage4a79_no_training_promotion"]
    s710 = stages["stage4a710_expanded_dataset_qa"]
    s711 = stages["stage4a711_tiny_no_checkpoint_eval"]
    s712 = stages["stage4a712_pre_rl_direction_decision"]
    s713 = stages["stage4a713_medium_preflight"]
    s714b = stages["stage4a714b_medium_postrun_safety"]

    return {
        "stage4a79_promotion_implementation_no_training": {
            "passed": (
                s79.get("completed") is True
                and s79.get("blocked") is False
                and s79.get("expanded_primary_samples") == 47
                and s79.get("promoted_stage4a72_samples") == 17
                and s79.get("lambda48_primary_use") is False
                and s79.get("bc_training") is False
                and s79.get("checkpoint") is False
                and s79.get("rl_gdpo_ppo") is False
            ),
            "evidence": str(STAGE79),
        },
        "stage4a710_dataset_qa": {
            "passed": (
                s710.get("completed") is True
                and s710.get("blocked") is False
                and s710.get("expanded_samples") == 47
                and s710.get("valid_primary_labels") is True
                and s710.get("finite_features") is True
                and s710.get("candidate_valid_mask_at_label") is True
                and s710.get("lambda48_primary_use") is False
                and s710.get("checkpoint") is False
                and s710.get("rl_gdpo_ppo") is False
            ),
            "evidence": str(STAGE710),
        },
        "stage4a711_tiny_no_checkpoint_eval": {
            "passed": (
                s711.get("completed") is True
                and s711.get("blocked") is False
                and s711.get("checkpoint_created") is False
                and s711.get("model_saved") is False
                and s711.get("lambda48_primary_use") is False
                and s711.get("rl_gdpo_ppo") is False
                and s711.get("rollout") is False
                and s711.get("overfit_passed") is True
                and s711.get("samples") == 47
            ),
            "evidence": str(STAGE711),
            "metric_note": {
                "eval_top1": s711.get("eval_top1"),
                "eval_top3": s711.get("eval_top3"),
                "zero_top1_folds": s711.get("zero_top1_folds"),
                "interpretation": "Weak/high-variance tiny no-checkpoint metrics are not treated as policy-quality proof; they motivated the medium expert rollout branch.",
            },
        },
        "stage4a712_direction_decision": {
            "passed": (
                s712.get("completed") is True
                and s712.get("blocked") is False
                and s712.get("selected_option") == "C"
                and s712.get("training") is False
                and s712.get("checkpoint_created") is False
                and s712.get("runtime_started") is False
                and s712.get("lambda48_primary_use") is False
                and s712.get("rl_gdpo_ppo") is False
            ),
            "evidence": str(STAGE712),
        },
        "stage4a713_medium_preflight": {
            "passed": (
                s713.get("completed") is True
                and s713.get("blocked") is False
                and s713.get("preflight_passed") is True
                and s713.get("runtime_allowed") is True
                and s713.get("primary_formula") == "uncertainty_bonus_composite_beta8"
                and s713.get("lambda48_primary_use") is False
                and s713.get("training") is False
                and s713.get("checkpoint_created") is False
                and s713.get("runtime_started") is False
                and s713.get("rl_gdpo_ppo") is False
            ),
            "evidence": str(STAGE713),
        },
        "stage4a714b_medium_postrun_safety": {
            "passed": (
                s714b.get("completed") is True
                and s714b.get("passed") is True
                and s714b.get("blocked") is False
                and s714b.get("actual_rl_training_allowed") is False
                and s714b.get("negative_scope", {}).get("training") is False
                and s714b.get("negative_scope", {}).get("checkpoint") is False
                and s714b.get("negative_scope", {}).get("rl") is False
                and s714b.get("negative_scope", {}).get("lambda48_primary") is False
            ),
            "evidence": str(STAGE714B),
        },
    }


def build_summary() -> dict[str, Any]:
    stages = {name: read_json(path) for name, path in EVIDENCE.items()}
    evidence_exists = {name: path.is_file() for name, path in EVIDENCE.items()}
    gates = stage_gate_matrix(stages)
    gate_passed = {name: data["passed"] for name, data in gates.items()}
    negative_scope = {
        "actual_rl_training_allowed": False,
        "rl": False,
        "gdpo": False,
        "ppo": False,
        "checkpoint_training": False,
        "bc_training": False,
        "optimizer_step": False,
        "model_save": False,
        "label_promotion": False,
        "long_rollout": False,
        "isaac_startup_in_stage715": False,
        "map_predict_in_stage715": False,
        "lambda48_primary": False,
    }
    tracked = git_files()
    repo_safety_passed = (
        not any(path.startswith(("outputs/", "logs/")) for path in tracked)
        and not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked)
    )
    live_processes = process_hits()
    all_gates_passed = all(evidence_exists.values()) and all(gate_passed.values()) and repo_safety_passed and live_processes == []
    blockers = []
    if not all(evidence_exists.values()):
        blockers.extend(f"missing_evidence:{name}" for name, exists in evidence_exists.items() if not exists)
    blockers.extend(f"gate_failed:{name}" for name, passed in gate_passed.items() if not passed)
    if not repo_safety_passed:
        blockers.append("repo_safety_failed")
    if live_processes:
        blockers.append("stage4a714_process_still_running")

    s711 = stages["stage4a711_tiny_no_checkpoint_eval"]
    s714b = stages["stage4a714b_medium_postrun_safety"]
    warnings = []
    if s711.get("zero_top1_folds", 0):
        warnings.append("stage4a711_tiny_eval_weak_high_variance_metrics")
    warnings.extend(s714b.get("warnings", []))

    return {
        "stage": "Stage 4A-7.15",
        "packet": "final_pre_rl_readiness_packet",
        "created_at_utc": utc_now(),
        "output_dir": str(OUT),
        "completed": True,
        "blocked": bool(blockers),
        "pre_rl_readiness_packet_complete": True,
        "pre_rl_readiness_passed": all_gates_passed,
        "pre_rl_readiness_decision": "ready_for_future_explicit_pre_rl_design_review" if all_gates_passed else "blocked_before_pre_rl_design_review",
        "main_blocker": blockers[0] if blockers else "",
        "blockers": blockers,
        "warnings": warnings,
        "evidence_exists": evidence_exists,
        "gate_passed": gate_passed,
        "gate_matrix": gates,
        "repo_safety_passed": repo_safety_passed,
        "live_stage4a714_processes": live_processes,
        "latest_commit_at_generation": git_head(),
        "dataset_readiness": {
            "expanded_primary_samples": stages["stage4a79_no_training_promotion"].get("expanded_primary_samples"),
            "stage4a710_valid_primary_labels": stages["stage4a710_expanded_dataset_qa"].get("valid_primary_labels"),
            "stage4a714_medium_actions": s714b.get("medium_runtime_counts", {}).get("executed_actions"),
            "stage4a714_medium_captures": s714b.get("medium_runtime_counts", {}).get("captures"),
        },
        "label_lineage": {
            "primary_label_source": "stage4a613_uncertainty_bonus_executed_primary",
            "primary_formula": "uncertainty_bonus_composite_beta8",
            "lambda48_role": "shadow/baseline only",
            "lambda48_primary_use": False,
            "no_recompute_from_lambda48": True,
        },
        "negative_scope": negative_scope,
        "actual_rl_training_allowed": False,
        "next_faithful_step": (
            "Human/web review of this final readiness packet, then a separate explicit Stage 4A-7.16 design-only pre-RL/RL plan if approved. "
            "Do not start actual RL/GDPO/PPO, checkpoint training, or model save from this packet."
        ),
        "review_files": {
            "this_packet_index": str(OUT / "stage4a715_final_pre_rl_readiness_index.html"),
            "this_packet_summary": str(OUT / "stage4a715_final_pre_rl_readiness_summary.md"),
            "stage4a714_runtime_html": str(RUNTIME714 / "short_rollout_uncertainty_bonus_index.html"),
            "stage4a714_runtime_mp4": str(RUNTIME714 / "short_rollout_flythrough.mp4"),
            "stage4a714b_medium_postrun_summary": str(STAGE714B.with_suffix(".md")),
        },
    }


def write_index(summary: dict[str, Any]) -> None:
    rows = [
        ("Completed", summary["completed"]),
        ("Pre-RL Readiness Passed", summary["pre_rl_readiness_passed"]),
        ("Blocked", summary["blocked"]),
        ("Decision", summary["pre_rl_readiness_decision"]),
        ("Main Blocker", summary["main_blocker"]),
        ("Actual RL Training Allowed", summary["actual_rl_training_allowed"]),
        ("Next Faithful Step", summary["next_faithful_step"]),
    ]
    table = "\n".join(f"<tr><th>{key}</th><td><code>{value}</code></td></tr>" for key, value in rows)
    gate_rows = "\n".join(
        f"<tr><td><code>{name}</code></td><td>{passed}</td><td><code>{summary['gate_matrix'][name]['evidence']}</code></td></tr>"
        for name, passed in summary["gate_passed"].items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 4A-7.15 Final Pre-RL Readiness Packet</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #1d242d; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1300px; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #c8d0da; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ width: 260px; background: #eef3f8; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
    .warn {{ padding: 12px; background: #fff2cc; border: 1px solid #e2bd55; max-width: 1300px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-7.15 Final Pre-RL Readiness Packet</h1>
  <p class="warn">This packet stops before actual RL training. It does not authorize PPO, GDPO, checkpoint training, or model save.</p>
  <table>{table}</table>
  <h2>Gate Matrix</h2>
  <table><tr><th>Gate</th><th>Passed</th><th>Evidence</th></tr>{gate_rows}</table>
  <h2>Files</h2>
  <ul>
    <li><a href="stage4a715_final_pre_rl_readiness_summary.md">Summary</a></li>
    <li><a href="stage4a715_final_pre_rl_gate_matrix.md">Gate matrix</a></li>
    <li><a href="stage4a715_upload_to_web_review_file_list.md">Upload file list</a></li>
  </ul>
</body>
</html>
"""
    (OUT / "stage4a715_final_pre_rl_readiness_index.html").write_text(html, encoding="utf-8")


def write_upload_list(summary: dict[str, Any]) -> None:
    files = [
        summary["review_files"]["this_packet_index"],
        summary["review_files"]["this_packet_summary"],
        str(OUT / "stage4a715_final_pre_rl_gate_matrix.md"),
        str(OUT / "stage4a715_final_pre_rl_evidence_manifest.md"),
        str(OUT / "stage4a715_final_pre_rl_negative_scope.md"),
        summary["review_files"]["stage4a714_runtime_html"],
        summary["review_files"]["stage4a714_runtime_mp4"],
        summary["review_files"]["stage4a714b_medium_postrun_summary"],
    ]
    lines = ["# Stage 4A-7.15 Upload To Web Review File List", ""]
    lines.extend(f"- `{item}`" for item in files)
    lines.extend([
        "",
        "Do not upload private keys, credentials, checkpoints, or unrelated logs.",
        "Actual RL/GDPO/PPO training remains unauthorized.",
    ])
    (OUT / "stage4a715_upload_to_web_review_file_list.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")
    summary = build_summary()
    report_pair("stage4a715_final_pre_rl_readiness_summary", summary, "Stage 4A-7.15 Final Pre-RL Readiness Summary")
    report_pair(
        "stage4a715_final_pre_rl_gate_matrix",
        {"stage": summary["stage"], "passed": summary["pre_rl_readiness_passed"], "gate_matrix": summary["gate_matrix"], "blockers": summary["blockers"]},
        "Stage 4A-7.15 Final Pre-RL Gate Matrix",
    )
    report_pair(
        "stage4a715_final_pre_rl_evidence_manifest",
        {"stage": summary["stage"], "evidence_exists": summary["evidence_exists"], "review_files": summary["review_files"]},
        "Stage 4A-7.15 Evidence Manifest",
    )
    report_pair("stage4a715_final_pre_rl_negative_scope", summary["negative_scope"], "Stage 4A-7.15 Negative Scope")
    write_upload_list(summary)
    write_index(summary)
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pre_rl_readiness_packet_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
