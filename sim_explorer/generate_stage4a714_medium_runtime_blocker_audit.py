from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_pre_rl_bridge"
OUT = BRIDGE / "stage4a714_medium_runtime_blocker_audit"
RUNTIME = ROOT / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"

EXPECTED = {
    "starts": 10,
    "steps_per_start": 6,
    "executed_actions": 60,
    "decision_frames": 60,
    "captures": 70,
    "primary_formula": "uncertainty_bonus_composite_beta8",
    "primary_label_source": "stage4a613_uncertainty_bonus_executed_primary",
    "lambda48_role": "shadow/baseline only",
}

RUNTIME_JSON = {
    "runtime_summary": RUNTIME / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json",
    "expert_quality": RUNTIME / "expert_data_quality_audit.json",
    "uncertainty_bonus_quality": RUNTIME / "uncertainty_bonus_runtime_quality_audit.json",
    "prediction_safety": RUNTIME / "prediction_safety_audit.json",
    "uncertainty_safety": RUNTIME / "uncertainty_safety_audit.json",
    "rollout_safety": RUNTIME / "rollout_safety_audit.json",
    "dataset_integrity": RUNTIME / "dataset_integrity_report.json",
    "finalization_sentinel": RUNTIME / "stage_finalized_before_isaac_close.json",
    "isaac_shutdown": RUNTIME / "isaac_shutdown_report.json",
    "manual_process_termination": RUNTIME / "manual_process_termination_report.json",
    "no_training_rl_bc": RUNTIME / "no_training_rl_bc_report.json",
    "checkpoint_hash": RUNTIME / "checkpoint_hash_report.json",
    "source_hash": RUNTIME / "source_hash_report.json",
    "no_long_rollout": RUNTIME / "no_long_rollout_report.json",
}

RUNTIME_REQUIRED_OUTPUTS = {
    "manifest": RUNTIME / "short_rollout_manifest.jsonl",
    "primary_decisions_jsonl": RUNTIME / "primary_uncertainty_bonus_decisions.jsonl",
    "primary_decisions_csv": RUNTIME / "primary_uncertainty_bonus_decisions.csv",
    "dataset_npz": RUNTIME / "short_rollout_dataset_uncertainty_bonus.npz",
    "html_index": RUNTIME / "short_rollout_uncertainty_bonus_index.html",
    "mp4_flythrough": RUNTIME / "short_rollout_flythrough.mp4",
    "expert_quality_audit": RUNTIME / "expert_data_quality_audit.json",
    "runtime_quality_audit": RUNTIME / "uncertainty_bonus_runtime_quality_audit.json",
    "prediction_safety_audit": RUNTIME / "prediction_safety_audit.json",
    "uncertainty_safety_audit": RUNTIME / "uncertainty_safety_audit.json",
    "rollout_safety_audit": RUNTIME / "rollout_safety_audit.json",
    "dataset_integrity_report": RUNTIME / "dataset_integrity_report.json",
    "finalization_sentinel": RUNTIME / "stage_finalized_before_isaac_close.json",
    "manual_process_termination": RUNTIME / "manual_process_termination_report.json",
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
    write_md_table(OUT / f"{stem}.md", title, rows or list(data.items()), extra=extra)


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


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
    hits: list[str] = []
    for line in proc.stdout.splitlines():
        if "run_stage4a714_medium_uncertainty_bonus_rollout.py" in line:
            hits.append(line.strip())
        elif "run_with_isaac_close_guard.py" in line and "Stage4A-7.14" in line:
            hits.append(line.strip())
    return hits


def artifact_manifest() -> dict[str, Any]:
    required = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in RUNTIME_REQUIRED_OUTPUTS.items()
    }
    return {
        "stage": "Stage 4A-7.14",
        "runtime_output_dir": str(RUNTIME),
        "audit_packet_dir": str(OUT),
        "created_at_utc": utc_now(),
        "expected": EXPECTED,
        "required_outputs": required,
        "manifest_rows": line_count(RUNTIME / "short_rollout_manifest.jsonl"),
        "primary_decision_rows": line_count(RUNTIME / "primary_uncertainty_bonus_decisions.jsonl"),
        "action_rgb_png_count": len(list(RUNTIME.glob("action_rgb_*.png"))),
        "action_pose_json_count": len(list(RUNTIME.glob("action_pose_*.json"))),
        "topdown_png_count": len(list(RUNTIME.glob("*topdown*.png"))),
        "runtime_json_files_present": {name: path.is_file() for name, path in RUNTIME_JSON.items()},
    }


def build_summary(artifacts: dict[str, Any]) -> dict[str, Any]:
    runtime_summary = read_json(RUNTIME_JSON["runtime_summary"])
    expert_quality = read_json(RUNTIME_JSON["expert_quality"])
    uncertainty_bonus_quality = read_json(RUNTIME_JSON["uncertainty_bonus_quality"])
    prediction_safety = read_json(RUNTIME_JSON["prediction_safety"])
    uncertainty_safety = read_json(RUNTIME_JSON["uncertainty_safety"])
    rollout_safety = read_json(RUNTIME_JSON["rollout_safety"])
    dataset_integrity = read_json(RUNTIME_JSON["dataset_integrity"])
    sentinel = read_json(RUNTIME_JSON["finalization_sentinel"])
    shutdown = read_json(RUNTIME_JSON["isaac_shutdown"])
    manual_term = read_json(RUNTIME_JSON["manual_process_termination"])
    no_training = read_json(RUNTIME_JSON["no_training_rl_bc"])
    checkpoint = read_json(RUNTIME_JSON["checkpoint_hash"])
    source_hash = read_json(RUNTIME_JSON["source_hash"])
    no_long = read_json(RUNTIME_JSON["no_long_rollout"])

    required_outputs_available = all(item["exists"] for item in artifacts["required_outputs"].values())
    counts_match_medium = (
        artifacts["manifest_rows"] == EXPECTED["executed_actions"]
        and artifacts["primary_decision_rows"] == EXPECTED["executed_actions"]
        and runtime_summary.get("executed_action_count") == EXPECTED["executed_actions"]
        and runtime_summary.get("decision_frame_count") == EXPECTED["decision_frames"]
        and runtime_summary.get("capture_count") == EXPECTED["captures"]
    )
    quality_core_passed = (
        expert_quality.get("passed") is True
        and uncertainty_bonus_quality.get("passed") is True
        and prediction_safety.get("passed") is True
        and uncertainty_safety.get("passed") is True
    )
    legacy_short_audit_failed = (
        rollout_safety.get("passed") is False
        and rollout_safety.get("action_execution_exceeds_3_per_start") is True
        and rollout_safety.get("total_action_limit_respected") is False
        and rollout_safety.get("max_decision_steps_per_start") == 3
    )
    dataset_failed_only_due_rollout = (
        dataset_integrity.get("passed") is False
        and dataset_integrity.get("rollout_safety_audit_passed") is False
        and dataset_integrity.get("expert_data_quality_audit_passed") is True
        and dataset_integrity.get("prediction_safety_audit_passed") is True
        and dataset_integrity.get("uncertainty_safety_audit_passed") is True
        and dataset_integrity.get("missing_required_outputs") == []
    )
    sentinel_unsafe_due_audit = (
        sentinel.get("required_output_checks_passed") is True
        and sentinel.get("audit_checks_passed") is False
        and sentinel.get("safe_to_terminate_after_close_timeout") is False
    )
    close_hung_documented = shutdown.get("close_hung_after_finalization") is True and shutdown.get("close_returned") is False
    manual_termination_clean = (
        manual_term.get("still_running_after_final_check") is False
        and manual_term.get("scope", {}).get("matched_only_stage4a714_runner") is True
        and manual_term.get("scope", {}).get("unrelated_process_killed") is False
    )
    no_training_checkpoint_rl = (
        no_training.get("training") is False
        and no_training.get("BC") is False
        and no_training.get("RL") is False
        and no_training.get("GDPO") is False
        and no_training.get("PPO") is False
        and checkpoint.get("checkpoint_sha256_before") == checkpoint.get("checkpoint_sha256_after")
        and source_hash.get("source_usd_sha256_before") == source_hash.get("source_usd_sha256_after")
        and source_hash.get("fixed_usd_sha256_before") == source_hash.get("fixed_usd_sha256_after")
        and no_long.get("long_rollout_executed") is False
    )
    lambda48_not_primary = (
        runtime_summary.get("beta") == 8
        and runtime_summary.get("lambda") == 48
        and artifacts["required_outputs"]["primary_decisions_jsonl"]["exists"] is True
        and (RUNTIME / "lambda48_shadow_decisions.jsonl").is_file()
        and "lambda48" not in str(runtime_summary.get("primary_label_source", "")).lower()
    )

    process_snapshot = process_hits()
    tracked = git_files()
    no_outputs_tracked = not any(path.startswith(("outputs/", "logs/")) for path in tracked)
    no_checkpoints_tracked = not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked)

    blocked = True
    main_blocker = "legacy_short_rollout_safety_audit_hardcoded_for_3_step_30_action_envelope"
    pre_rl_ready = False
    return {
        "stage": "Stage 4A-7.14",
        "packet": "stage4a714_medium_runtime_blocker_audit",
        "created_at_utc": utc_now(),
        "runtime_output_dir": str(RUNTIME),
        "audit_packet_dir": str(OUT),
        "completed": True,
        "blocked": blocked,
        "main_blocker": main_blocker,
        "pre_rl_ready": pre_rl_ready,
        "pre_rl_readiness_decision": "not_ready_blocked",
        "required_outputs_available": required_outputs_available,
        "counts_match_medium_bounds": counts_match_medium,
        "quality_core_passed": quality_core_passed,
        "legacy_short_audit_failed": legacy_short_audit_failed,
        "dataset_failed_only_due_rollout_safety": dataset_failed_only_due_rollout,
        "sentinel_unsafe_due_audit_failure": sentinel_unsafe_due_audit,
        "close_hung_after_finalization": close_hung_documented,
        "manual_termination_clean": manual_termination_clean,
        "no_training_checkpoint_rl": no_training_checkpoint_rl,
        "lambda48_primary_use": False,
        "lambda48_not_primary": lambda48_not_primary,
        "runtime_processes_still_running": process_snapshot,
        "no_outputs_or_logs_tracked": no_outputs_tracked,
        "no_checkpoint_files_tracked": no_checkpoints_tracked,
        "latest_commit": git_head(),
        "runtime_counts": {
            "starts": runtime_summary.get("start_count"),
            "executed_actions": runtime_summary.get("executed_action_count"),
            "decision_frames": runtime_summary.get("decision_frame_count"),
            "captures": runtime_summary.get("capture_count"),
            "map_predict_calls": runtime_summary.get("map_predict_calls"),
            "manifest_rows": artifacts["manifest_rows"],
            "primary_decision_rows": artifacts["primary_decision_rows"],
            "action_rgb_png_count": artifacts["action_rgb_png_count"],
            "action_pose_json_count": artifacts["action_pose_json_count"],
            "topdown_png_count": artifacts["topdown_png_count"],
        },
        "negative_scope": {
            "training": False,
            "checkpoint": False,
            "rl": False,
            "gdpo": False,
            "ppo": False,
            "label_promotion": False,
            "prediction_uncertainty_writeback": False,
            "lambda48_primary": False,
            "long_rollout": False,
        },
        "runtime_artifacts_for_review": {
            "html": str(RUNTIME / "short_rollout_uncertainty_bonus_index.html"),
            "mp4": str(RUNTIME / "short_rollout_flythrough.mp4"),
            "summary": str(RUNTIME / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.md"),
            "expert_quality": str(RUNTIME / "expert_data_quality_audit.md"),
            "runtime_quality": str(RUNTIME / "uncertainty_bonus_runtime_quality_audit.md"),
            "rollout_safety": str(RUNTIME / "rollout_safety_audit.md"),
            "dataset_integrity": str(RUNTIME / "dataset_integrity_report.md"),
            "manual_termination": str(RUNTIME / "manual_process_termination_report.json"),
        },
    }


def write_legacy_mismatch_report(summary: dict[str, Any]) -> dict[str, Any]:
    rollout = read_json(RUNTIME_JSON["rollout_safety"])
    dataset = read_json(RUNTIME_JSON["dataset_integrity"])
    sentinel = read_json(RUNTIME_JSON["finalization_sentinel"])
    data = {
        "stage": "Stage 4A-7.14",
        "passed": False,
        "blocker": summary["main_blocker"],
        "expected_medium_bounds": EXPECTED,
        "legacy_rollout_safety_audit": {
            "passed": rollout.get("passed"),
            "max_decision_steps_per_start": rollout.get("max_decision_steps_per_start"),
            "action_execution_exceeds_3_per_start": rollout.get("action_execution_exceeds_3_per_start"),
            "total_action_limit_respected": rollout.get("total_action_limit_respected"),
            "executed_action_count": rollout.get("executed_action_count"),
            "decision_frame_count": rollout.get("decision_frame_count"),
            "capture_count": rollout.get("capture_count"),
        },
        "dataset_integrity_dependency": {
            "passed": dataset.get("passed"),
            "rollout_safety_audit_passed": dataset.get("rollout_safety_audit_passed"),
            "missing_required_outputs": dataset.get("missing_required_outputs"),
        },
        "sentinel_effect": {
            "required_output_checks_passed": sentinel.get("required_output_checks_passed"),
            "audit_checks_passed": sentinel.get("audit_checks_passed"),
            "safe_to_terminate_after_close_timeout": sentinel.get("safe_to_terminate_after_close_timeout"),
        },
        "interpretation": "The medium runtime produced the expected 60-action packet, but legacy safety checks still encode the prior short 3-step/30-action envelope. This blocks Pre-RL readiness until a medium-specific postrun safety validator replaces or layers over that audit.",
    }
    report_pair(
        "stage4a714_legacy_short_bound_audit_mismatch_report",
        data,
        "Stage 4A-7.14 Legacy Short-Bound Audit Mismatch",
    )
    return data


def write_no_training_report(summary: dict[str, Any]) -> dict[str, Any]:
    data = {
        "stage": "Stage 4A-7.14",
        "passed": summary["no_training_checkpoint_rl"],
        "training": False,
        "bc": False,
        "optimizer_step": False,
        "checkpoint": False,
        "model_save": False,
        "rl": False,
        "gdpo": False,
        "ppo": False,
        "label_promotion": False,
        "lambda48_primary": False,
        "long_rollout": False,
        "source_usd_modified": False,
        "fixed_usd_modified": False,
    }
    report_pair("stage4a714_no_training_no_checkpoint_no_rl_report", data, "Stage 4A-7.14 Negative Scope Report")
    return data


def write_recommendation(summary: dict[str, Any]) -> None:
    text = f"""# Stage 4A-7.14 Recommended Next Faithful Step

Status: blocked, not Pre-RL ready.

Main blocker:
`{summary['main_blocker']}`.

Do not start PPO/GDPO/RL or any checkpoint training from this packet. The medium runtime produced the intended bounded artifacts, but the postrun safety gate is still wired to the previous short rollout envelope. The next faithful step is a source-only Stage 4A-7.14b medium postrun safety validator/audit update that explicitly accepts the approved 10-start, 6-step, 60-action, 70-capture envelope while preserving all negative-scope checks.

After that validator passes, regenerate a Pre-RL readiness packet. Lambda48 remains shadow/baseline only; primary expert lineage remains `stage4a613_uncertainty_bonus_executed_primary` from `uncertainty_bonus_composite_beta8`.
"""
    (OUT / "stage4a714_recommended_next_faithful_step.md").write_text(text, encoding="utf-8")


def write_index(summary: dict[str, Any]) -> None:
    rows = [
        ("Completed", summary["completed"]),
        ("Blocked", summary["blocked"]),
        ("Pre-RL Ready", summary["pre_rl_ready"]),
        ("Main Blocker", summary["main_blocker"]),
        ("Runtime Output", summary["runtime_output_dir"]),
        ("HTML Review", summary["runtime_artifacts_for_review"]["html"]),
        ("MP4 Review", summary["runtime_artifacts_for_review"]["mp4"]),
        ("No Training/Checkpoint/RL", summary["no_training_checkpoint_rl"]),
    ]
    table = "\n".join(f"<tr><th>{key}</th><td><code>{value}</code></td></tr>" for key, value in rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 4A-7.14 Medium Runtime Blocker Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #1d242d; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1200px; }}
    th, td {{ border: 1px solid #c8d0da; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ width: 240px; background: #eef3f8; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
    .warn {{ padding: 12px; background: #fff2cc; border: 1px solid #e2bd55; max-width: 1200px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-7.14 Medium Runtime Blocker Audit</h1>
  <p class="warn">This packet is not a Pre-RL green light. It documents a blocked gate after bounded medium runtime output generation.</p>
  <table>{table}</table>
  <h2>Review Files</h2>
  <ul>
    <li><a href="stage4a714_medium_runtime_blocker_audit_summary.md">Summary</a></li>
    <li><a href="stage4a714_medium_runtime_artifact_manifest.md">Artifact manifest</a></li>
    <li><a href="stage4a714_legacy_short_bound_audit_mismatch_report.md">Legacy short-bound audit mismatch</a></li>
    <li><a href="stage4a714_no_training_no_checkpoint_no_rl_report.md">Negative scope report</a></li>
    <li><a href="stage4a714_recommended_next_faithful_step.md">Recommended next faithful step</a></li>
  </ul>
</body>
</html>
"""
    (OUT / "stage4a714_medium_runtime_blocker_audit_index.html").write_text(html, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")
    artifacts = artifact_manifest()
    report_pair("stage4a714_medium_runtime_artifact_manifest", artifacts, "Stage 4A-7.14 Medium Runtime Artifact Manifest")

    summary = build_summary(artifacts)
    report_pair("stage4a714_medium_runtime_blocker_audit_summary", summary, "Stage 4A-7.14 Medium Runtime Blocker Audit")
    write_legacy_mismatch_report(summary)
    write_no_training_report(summary)
    write_recommendation(summary)
    write_index(summary)
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
