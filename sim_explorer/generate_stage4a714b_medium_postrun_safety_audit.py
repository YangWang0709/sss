from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_pre_rl_bridge"
OUT = BRIDGE / "stage4a714b_medium_postrun_safety_audit"
RUNTIME = ROOT / "outputs/isaac_stage4a714_medium_bounded_expert_rollout_runtime"

EXPECTED = {
    "stage": "Stage 4A-7.14-medium-bounded-uncertainty-bonus-expert-rollout",
    "starts": 10,
    "steps_per_start": 6,
    "terminal_frames_per_start": 1,
    "captures_per_start": 7,
    "total_actions": 60,
    "total_decision_frames": 60,
    "total_captures": 70,
    "primary_formula": "uncertainty_bonus_composite_beta8",
    "primary_label_source": "stage4a613_uncertainty_bonus_executed_primary",
    "lambda48_role": "shadow/baseline only",
}

REQUIRED_RUNTIME_FILES = {
    "manifest": RUNTIME / "short_rollout_manifest.jsonl",
    "primary_decisions_jsonl": RUNTIME / "primary_uncertainty_bonus_decisions.jsonl",
    "primary_decisions_csv": RUNTIME / "primary_uncertainty_bonus_decisions.csv",
    "lambda48_shadow_jsonl": RUNTIME / "lambda48_shadow_decisions.jsonl",
    "dataset": RUNTIME / "short_rollout_dataset_uncertainty_bonus.npz",
    "html": RUNTIME / "short_rollout_uncertainty_bonus_index.html",
    "mp4": RUNTIME / "short_rollout_flythrough.mp4",
    "per_start_csv": RUNTIME / "per_start_summary.csv",
    "per_step_csv": RUNTIME / "per_step_summary.csv",
    "summary": RUNTIME / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json",
    "expert_quality": RUNTIME / "expert_data_quality_audit.json",
    "uncertainty_bonus_quality": RUNTIME / "uncertainty_bonus_runtime_quality_audit.json",
    "prediction_safety": RUNTIME / "prediction_safety_audit.json",
    "uncertainty_safety": RUNTIME / "uncertainty_safety_audit.json",
    "legacy_rollout_safety": RUNTIME / "rollout_safety_audit.json",
    "legacy_dataset_integrity": RUNTIME / "dataset_integrity_report.json",
    "finalization_sentinel": RUNTIME / "stage_finalized_before_isaac_close.json",
    "isaac_shutdown": RUNTIME / "isaac_shutdown_report.json",
    "manual_process_termination": RUNTIME / "manual_process_termination_report.json",
    "no_training_rl_bc": RUNTIME / "no_training_rl_bc_report.json",
    "no_long_rollout": RUNTIME / "no_long_rollout_report.json",
    "checkpoint_hash": RUNTIME / "checkpoint_hash_report.json",
    "source_hash": RUNTIME / "source_hash_report.json",
}

FORBIDDEN_MANIFEST_FLAGS = [
    "future_observed_scoring_use",
    "target_ground_truth_use",
    "prediction_candidate_validity_use",
    "prediction_collision_use",
    "prediction_ray_blocking_use",
    "prediction_traversability_use",
    "prediction_writeback",
    "uncertainty_candidate_validity_use",
    "uncertainty_collision_use",
    "uncertainty_ray_blocking_use",
    "uncertainty_traversability_use",
    "uncertainty_writeback",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def process_hits() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,pgid,stat,etime,cmd"], text=True, capture_output=True)
    hits: list[str] = []
    for line in proc.stdout.splitlines():
        if "run_stage4a714_medium_uncertainty_bonus_rollout.py" in line:
            hits.append(line.strip())
        elif "run_with_isaac_close_guard.py" in line and "Stage4A-7.14" in line:
            hits.append(line.strip())
    return hits


def git_status() -> str:
    proc = subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout


def git_files() -> list[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.splitlines()


def file_report() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in REQUIRED_RUNTIME_FILES.items()
    }


def analyze_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
    forbidden_counts: Counter[str] = Counter()
    missing_paths: list[str] = []
    nonfinite_action_rows: list[dict[str, Any]] = []
    observed_ratio_regressions: list[dict[str, Any]] = []
    previous_after_by_start: dict[int, float] = {}
    no_valid_candidate_count = 0
    out_of_bounds_count = 0
    repeated_target_count = 0
    same_cell_target_count = 0
    total_newly_observed = 0
    stage_mismatch_count = 0

    for index, row in enumerate(rows):
        start = int(row.get("start_variant_id", -1))
        by_start[start].append(row)
        if row.get("stage") != EXPECTED["stage"]:
            stage_mismatch_count += 1
        for flag in FORBIDDEN_MANIFEST_FLAGS:
            if to_bool(row.get(flag)):
                forbidden_counts[flag] += 1
        if to_bool(row.get("no_valid_candidate")):
            no_valid_candidate_count += 1
        if to_bool(row.get("outside_bounds_target")):
            out_of_bounds_count += 1
        if to_bool(row.get("repeated_target")):
            repeated_target_count += 1
        if to_bool(row.get("same_cell_target")):
            same_cell_target_count += 1
        total_newly_observed += int(row.get("newly_observed_count") or 0)
        coords = row.get("action_world_xyz") or []
        if len(coords) != 3 or not all(finite_number(item) for item in coords) or not finite_number(row.get("action_yaw")):
            nonfinite_action_rows.append({"index": index, "start": start, "step": row.get("step_id")})
        for field in ["candidate_features", "dense_prediction_uncertainty", "depth", "observed_state_reference", "pose", "rgb"]:
            path_text = row.get(field)
            if path_text and not Path(path_text).is_file():
                missing_paths.append(path_text)
        before = row.get("observed_ratio_before")
        after = row.get("observed_ratio_after_current_capture")
        if finite_number(before) and finite_number(after):
            before_f = float(before)
            after_f = float(after)
            if after_f + 1e-12 < before_f:
                observed_ratio_regressions.append({"index": index, "start": start, "step": row.get("step_id"), "before": before_f, "after": after_f})
            prev = previous_after_by_start.get(start)
            if prev is not None and before_f + 1e-12 < prev:
                observed_ratio_regressions.append({"index": index, "start": start, "step": row.get("step_id"), "previous_after": prev, "before": before_f})
            previous_after_by_start[start] = after_f

    per_start_steps = {str(start): len(items) for start, items in sorted(by_start.items())}
    step_sets = {str(start): sorted(int(item.get("step_id", -999)) for item in items) for start, items in sorted(by_start.items())}
    return {
        "row_count": len(rows),
        "start_ids": sorted(by_start.keys()),
        "start_count": len(by_start),
        "per_start_steps": per_start_steps,
        "step_sets": step_sets,
        "all_starts_have_6_steps": all(count == EXPECTED["steps_per_start"] for count in per_start_steps.values()),
        "all_step_sets_expected": all(steps == list(range(EXPECTED["steps_per_start"])) for steps in step_sets.values()),
        "stage_mismatch_count": stage_mismatch_count,
        "forbidden_flag_counts": dict(forbidden_counts),
        "missing_referenced_path_count": len(missing_paths),
        "missing_referenced_paths_sample": missing_paths[:20],
        "nonfinite_action_rows": nonfinite_action_rows,
        "observed_ratio_regressions": observed_ratio_regressions,
        "no_valid_candidate_count": no_valid_candidate_count,
        "out_of_bounds_target_count": out_of_bounds_count,
        "repeated_target_count": repeated_target_count,
        "same_cell_target_count": same_cell_target_count,
        "total_newly_observed": total_newly_observed,
    }


def analyze_per_start(rows: list[dict[str, str]]) -> dict[str, Any]:
    start_ids = sorted(to_int(row.get("start_variant_id")) for row in rows if to_int(row.get("start_variant_id")) is not None)
    return {
        "row_count": len(rows),
        "start_ids": start_ids,
        "all_start_rows_expected": start_ids == list(range(EXPECTED["starts"])),
        "all_executed_action_count_6": all(to_int(row.get("executed_action_count")) == EXPECTED["steps_per_start"] for row in rows),
        "all_decision_frame_count_6": all(to_int(row.get("decision_frame_count")) == EXPECTED["steps_per_start"] for row in rows),
        "all_terminal_frame_count_1": all(to_int(row.get("terminal_frame_count")) == EXPECTED["terminal_frames_per_start"] for row in rows),
        "all_capture_count_7": all(to_int(row.get("capture_count")) == EXPECTED["captures_per_start"] for row in rows),
        "all_done_reason_max_steps": all(row.get("done_reason") == "max_steps_reached" for row in rows),
        "any_early_stop": any(to_bool(row.get("early_stop")) for row in rows),
        "terminal_observed_state_missing": [
            row.get("terminal_observed_state", "")
            for row in rows
            if row.get("terminal_observed_state") and not Path(row["terminal_observed_state"]).is_file()
        ],
    }


def analyze_primary_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    formulas = Counter(row.get("formula", "") for row in rows)
    quality_flags = Counter(row.get("quality_flags", "") for row in rows)
    return {
        "row_count": len(rows),
        "formula_counts": dict(formulas),
        "all_formula_beta8": list(formulas.keys()) == [EXPECTED["primary_formula"]] if formulas else False,
        "selected_candidate_finite": all(finite_number(row.get("selected_candidate_id")) for row in rows),
        "final_score_finite": all(finite_number(row.get("final_score")) for row in rows),
        "source_occ_free_kept_separate": all(to_bool(row.get("source_occ_free_kept_separate_from_uncertainty")) for row in rows),
        "no_invalid_candidates": all(not to_bool(row.get("no_valid_candidate")) for row in rows),
        "no_low_cost_artifacts": all(not to_bool(row.get("low_cost_artifact")) for row in rows),
        "no_historical_prior_basin": all(not to_bool(row.get("historical_prior_basin")) for row in rows),
        "no_candidate_all_local": all(not to_bool(row.get("candidate_all_local")) for row in rows),
        "quality_flags_counts": dict(quality_flags),
    }


def build_summary() -> dict[str, Any]:
    files = file_report()
    manifest = read_jsonl(REQUIRED_RUNTIME_FILES["manifest"])
    primary_jsonl = read_jsonl(REQUIRED_RUNTIME_FILES["primary_decisions_jsonl"])
    primary_csv = read_csv(REQUIRED_RUNTIME_FILES["primary_decisions_csv"])
    per_start = read_csv(REQUIRED_RUNTIME_FILES["per_start_csv"])

    runtime_summary = read_json(REQUIRED_RUNTIME_FILES["summary"])
    expert_quality = read_json(REQUIRED_RUNTIME_FILES["expert_quality"])
    runtime_quality = read_json(REQUIRED_RUNTIME_FILES["uncertainty_bonus_quality"])
    prediction_safety = read_json(REQUIRED_RUNTIME_FILES["prediction_safety"])
    uncertainty_safety = read_json(REQUIRED_RUNTIME_FILES["uncertainty_safety"])
    legacy_rollout = read_json(REQUIRED_RUNTIME_FILES["legacy_rollout_safety"])
    legacy_dataset = read_json(REQUIRED_RUNTIME_FILES["legacy_dataset_integrity"])
    sentinel = read_json(REQUIRED_RUNTIME_FILES["finalization_sentinel"])
    shutdown = read_json(REQUIRED_RUNTIME_FILES["isaac_shutdown"])
    manual_term = read_json(REQUIRED_RUNTIME_FILES["manual_process_termination"])
    no_training = read_json(REQUIRED_RUNTIME_FILES["no_training_rl_bc"])
    no_long = read_json(REQUIRED_RUNTIME_FILES["no_long_rollout"])
    checkpoint = read_json(REQUIRED_RUNTIME_FILES["checkpoint_hash"])
    source_hash = read_json(REQUIRED_RUNTIME_FILES["source_hash"])

    manifest_audit = analyze_manifest(manifest)
    per_start_audit = analyze_per_start(per_start)
    primary_audit = analyze_primary_rows(primary_csv)
    live_processes = process_hits()
    tracked = git_files()

    medium_envelope_passed = (
        runtime_summary.get("executed_action_count") == EXPECTED["total_actions"]
        and runtime_summary.get("decision_frame_count") == EXPECTED["total_decision_frames"]
        and runtime_summary.get("capture_count") == EXPECTED["total_captures"]
        and runtime_summary.get("map_predict_calls") == EXPECTED["total_actions"]
        and manifest_audit["row_count"] == EXPECTED["total_actions"]
        and manifest_audit["start_ids"] == list(range(EXPECTED["starts"]))
        and manifest_audit["all_starts_have_6_steps"] is True
        and manifest_audit["all_step_sets_expected"] is True
        and per_start_audit["all_start_rows_expected"] is True
        and per_start_audit["all_executed_action_count_6"] is True
        and per_start_audit["all_decision_frame_count_6"] is True
        and per_start_audit["all_terminal_frame_count_1"] is True
        and per_start_audit["all_capture_count_7"] is True
        and len(primary_jsonl) == EXPECTED["total_actions"]
        and primary_audit["row_count"] == EXPECTED["total_actions"]
    )
    required_outputs_passed = all(item["exists"] and (item["size_bytes"] or 0) > 0 for item in files.values())
    manifest_safety_passed = (
        manifest_audit["stage_mismatch_count"] == 0
        and manifest_audit["forbidden_flag_counts"] == {}
        and manifest_audit["missing_referenced_path_count"] == 0
        and manifest_audit["nonfinite_action_rows"] == []
        and manifest_audit["observed_ratio_regressions"] == []
        and manifest_audit["no_valid_candidate_count"] == 0
        and manifest_audit["out_of_bounds_target_count"] == 0
        and manifest_audit["total_newly_observed"] > 0
    )
    primary_policy_passed = (
        primary_audit["all_formula_beta8"] is True
        and primary_audit["selected_candidate_finite"] is True
        and primary_audit["final_score_finite"] is True
        and primary_audit["source_occ_free_kept_separate"] is True
        and primary_audit["no_invalid_candidates"] is True
        and primary_audit["no_low_cost_artifacts"] is True
        and primary_audit["no_historical_prior_basin"] is True
        and primary_audit["no_candidate_all_local"] is True
        and REQUIRED_RUNTIME_FILES["lambda48_shadow_jsonl"].is_file()
    )
    quality_audits_passed = (
        expert_quality.get("passed") is True
        and runtime_quality.get("passed") is True
        and prediction_safety.get("passed") is True
        and uncertainty_safety.get("passed") is True
    )
    legacy_mismatch_expected = (
        legacy_rollout.get("passed") is False
        and legacy_rollout.get("max_decision_steps_per_start") == 3
        and legacy_rollout.get("action_execution_exceeds_3_per_start") is True
        and legacy_rollout.get("total_action_limit_respected") is False
        and legacy_dataset.get("passed") is False
        and legacy_dataset.get("rollout_safety_audit_passed") is False
        and legacy_dataset.get("missing_required_outputs") == []
        and legacy_dataset.get("expert_data_quality_audit_passed") is True
        and legacy_dataset.get("prediction_safety_audit_passed") is True
        and legacy_dataset.get("uncertainty_safety_audit_passed") is True
    )
    finalization_passed = (
        sentinel.get("finalization_complete") is True
        and sentinel.get("required_output_checks_passed") is True
        and sentinel.get("missing_required_outputs") == []
        and sentinel.get("empty_required_outputs") == []
        and shutdown.get("close_called") is True
        and shutdown.get("close_hung_after_finalization") is True
        and manual_term.get("still_running_after_final_check") is False
        and manual_term.get("scope", {}).get("matched_only_stage4a714_runner") is True
        and manual_term.get("scope", {}).get("unrelated_process_killed") is False
        and live_processes == []
    )
    negative_scope_passed = (
        no_training.get("passed") is True
        and no_training.get("training") is False
        and no_training.get("BC") is False
        and no_training.get("RL") is False
        and no_training.get("GDPO") is False
        and no_training.get("PPO") is False
        and no_long.get("long_rollout_executed") is False
        and checkpoint.get("checkpoint_sha256_before") == checkpoint.get("checkpoint_sha256_after")
        and source_hash.get("source_usd_sha256_before") == source_hash.get("source_usd_sha256_after")
        and source_hash.get("fixed_usd_sha256_before") == source_hash.get("fixed_usd_sha256_after")
    )
    repo_safety_passed = (
        not any(path.startswith(("outputs/", "logs/")) for path in tracked)
        and not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked)
    )

    checks = {
        "required_outputs_passed": required_outputs_passed,
        "medium_envelope_passed": medium_envelope_passed,
        "manifest_safety_passed": manifest_safety_passed,
        "primary_policy_passed": primary_policy_passed,
        "quality_audits_passed": quality_audits_passed,
        "legacy_mismatch_expected": legacy_mismatch_expected,
        "finalization_passed": finalization_passed,
        "negative_scope_passed": negative_scope_passed,
        "repo_safety_passed": repo_safety_passed,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    warnings = []
    if manifest_audit["same_cell_target_count"]:
        warnings.append("same_cell_targets_present_but_expert_quality_audit_passed")
    if manifest_audit["repeated_target_count"]:
        warnings.append("repeated_targets_present_but_expert_quality_audit_passed")
    if expert_quality.get("warnings"):
        warnings.extend(f"expert_quality:{item}" for item in expert_quality.get("warnings", []))

    return {
        "stage": "Stage 4A-7.14b",
        "packet": "stage4a714b_medium_postrun_safety_audit",
        "created_at_utc": utc_now(),
        "runtime_output_dir": str(RUNTIME),
        "output_dir": str(OUT),
        "completed": True,
        "passed": not blockers,
        "blocked": bool(blockers),
        "blockers": blockers,
        "warnings": warnings,
        "main_blocker": blockers[0] if blockers else "",
        "approved_medium_envelope": EXPECTED,
        "checks": checks,
        "medium_runtime_counts": {
            "starts": runtime_summary.get("start_count"),
            "executed_actions": runtime_summary.get("executed_action_count"),
            "decision_frames": runtime_summary.get("decision_frame_count"),
            "captures": runtime_summary.get("capture_count"),
            "map_predict_calls": runtime_summary.get("map_predict_calls"),
            "manifest_rows": manifest_audit["row_count"],
            "primary_rows": primary_audit["row_count"],
            "action_rgb_png_count": len(list(RUNTIME.glob("action_rgb_*.png"))),
            "action_pose_json_count": len(list(RUNTIME.glob("action_pose_*.json"))),
        },
        "file_report": files,
        "manifest_audit": manifest_audit,
        "per_start_audit": per_start_audit,
        "primary_policy_audit": primary_audit,
        "quality_audit_inputs": {
            "expert_quality_passed": expert_quality.get("passed"),
            "uncertainty_bonus_quality_passed": runtime_quality.get("passed"),
            "prediction_safety_passed": prediction_safety.get("passed"),
            "uncertainty_safety_passed": uncertainty_safety.get("passed"),
        },
        "legacy_short_bound_audit_handling": {
            "legacy_rollout_safety_passed": legacy_rollout.get("passed"),
            "legacy_dataset_integrity_passed": legacy_dataset.get("passed"),
            "legacy_mismatch_expected": legacy_mismatch_expected,
            "interpretation": "The old rollout safety audit remains short-bound and is not used as the medium postrun pass/fail gate except to verify that its failure mode is exactly the known envelope mismatch.",
        },
        "finalization_and_close": {
            "sentinel_required_outputs_passed": sentinel.get("required_output_checks_passed"),
            "sentinel_audit_checks_passed": sentinel.get("audit_checks_passed"),
            "sentinel_safe_to_terminate_after_close_timeout": sentinel.get("safe_to_terminate_after_close_timeout"),
            "close_hung_after_finalization": shutdown.get("close_hung_after_finalization"),
            "manual_termination_clean": manual_term.get("still_running_after_final_check") is False,
            "live_stage4a714_processes": live_processes,
        },
        "negative_scope": {
            "training": False,
            "bc": False,
            "optimizer_step": False,
            "checkpoint": False,
            "model_save": False,
            "rl": False,
            "gdpo": False,
            "ppo": False,
            "label_promotion": False,
            "long_rollout": False,
            "lambda48_primary": False,
            "prediction_uncertainty_writeback": False,
        },
        "next_stage_if_passed": "Stage 4A-7.15 final Pre-RL readiness packet",
        "actual_rl_training_allowed": False,
    }


def write_index(summary: dict[str, Any]) -> None:
    rows = [
        ("Completed", summary["completed"]),
        ("Passed", summary["passed"]),
        ("Blocked", summary["blocked"]),
        ("Main Blocker", summary["main_blocker"]),
        ("Runtime Output", summary["runtime_output_dir"]),
        ("Actions", summary["medium_runtime_counts"]["executed_actions"]),
        ("Captures", summary["medium_runtime_counts"]["captures"]),
        ("Actual RL Training Allowed", summary["actual_rl_training_allowed"]),
    ]
    table = "\n".join(f"<tr><th>{key}</th><td><code>{value}</code></td></tr>" for key, value in rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 4A-7.14b Medium Postrun Safety Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #1d242d; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1200px; }}
    th, td {{ border: 1px solid #c8d0da; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ width: 260px; background: #eef3f8; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
    .note {{ padding: 12px; background: #eef8f0; border: 1px solid #82b68a; max-width: 1200px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-7.14b Medium Postrun Safety Audit</h1>
  <p class="note">This is a source-only audit over existing Stage 4A-7.14 runtime outputs. It does not authorize actual RL training.</p>
  <table>{table}</table>
  <h2>Files</h2>
  <ul>
    <li><a href="stage4a714b_medium_postrun_safety_audit_summary.md">Summary</a></li>
    <li><a href="stage4a714b_medium_postrun_gate_matrix.md">Gate matrix</a></li>
  </ul>
</body>
</html>
"""
    (OUT / "stage4a714b_medium_postrun_safety_audit_index.html").write_text(html, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")
    summary = build_summary()
    report_pair(
        "stage4a714b_medium_postrun_safety_audit_summary",
        summary,
        "Stage 4A-7.14b Medium Postrun Safety Audit",
    )
    gate_matrix = {
        "stage": summary["stage"],
        "passed": summary["passed"],
        "blockers": summary["blockers"],
        "checks": summary["checks"],
        "legacy_short_bound_audit_handling": summary["legacy_short_bound_audit_handling"],
        "negative_scope": summary["negative_scope"],
    }
    report_pair("stage4a714b_medium_postrun_gate_matrix", gate_matrix, "Stage 4A-7.14b Medium Postrun Gate Matrix")
    write_index(summary)
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
