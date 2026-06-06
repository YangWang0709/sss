from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/autonomous_pre_rl_bridge/stage4a715_final_pre_rl_readiness_packet"
SUMMARY = OUT / "stage4a715_final_pre_rl_readiness_summary.json"

REQUIRED_FILES = [
    "stage4a715_final_pre_rl_readiness_summary.json",
    "stage4a715_final_pre_rl_readiness_summary.md",
    "stage4a715_final_pre_rl_gate_matrix.json",
    "stage4a715_final_pre_rl_gate_matrix.md",
    "stage4a715_final_pre_rl_evidence_manifest.json",
    "stage4a715_final_pre_rl_evidence_manifest.md",
    "stage4a715_final_pre_rl_negative_scope.json",
    "stage4a715_final_pre_rl_negative_scope.md",
    "stage4a715_upload_to_web_review_file_list.md",
    "stage4a715_final_pre_rl_readiness_index.html",
    "git_status_before.txt",
    "git_status_after.txt",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def process_hits() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,pgid,stat,etime,cmd"], text=True, capture_output=True)
    hits = []
    for line in proc.stdout.splitlines():
        if "run_stage4a714_medium_uncertainty_bonus_rollout.py" in line:
            hits.append(line.strip())
        elif "run_with_isaac_close_guard.py" in line and "Stage4A-7.14" in line:
            hits.append(line.strip())
    return hits


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["output_dir_exists"] = OUT.is_dir()
    for name in REQUIRED_FILES:
        checks[f"required:{name}"] = (OUT / name).is_file()
    checks["summary_exists"] = SUMMARY.is_file()
    if not SUMMARY.is_file():
        blockers.append("missing_summary")
        print(json.dumps({"all_passed": False, "checks": checks, "blockers": blockers}, indent=2, sort_keys=True))
        return 1

    summary = load_json(SUMMARY)
    gate = load_json(OUT / "stage4a715_final_pre_rl_gate_matrix.json")
    evidence = load_json(OUT / "stage4a715_final_pre_rl_evidence_manifest.json")
    negative = load_json(OUT / "stage4a715_final_pre_rl_negative_scope.json")

    checks["packet_complete"] = summary.get("completed") is True and summary.get("pre_rl_readiness_packet_complete") is True
    checks["readiness_passed_without_blockers"] = (
        summary.get("pre_rl_readiness_passed") is True
        and summary.get("blocked") is False
        and summary.get("blockers") == []
        and summary.get("main_blocker") == ""
        and summary.get("pre_rl_readiness_decision") == "ready_for_future_explicit_pre_rl_design_review"
    )
    checks["all_evidence_exists"] = all(summary.get("evidence_exists", {}).values()) and all(evidence.get("evidence_exists", {}).values())
    checks["all_gates_passed"] = all(summary.get("gate_passed", {}).values())
    checks["gate_matrix_passed"] = gate.get("passed") is True and all(item.get("passed") is True for item in gate.get("gate_matrix", {}).values())
    label = summary.get("label_lineage", {})
    checks["label_lineage_beta8_lambda48_shadow"] = (
        label.get("primary_label_source") == "stage4a613_uncertainty_bonus_executed_primary"
        and label.get("primary_formula") == "uncertainty_bonus_composite_beta8"
        and label.get("lambda48_role") == "shadow/baseline only"
        and label.get("lambda48_primary_use") is False
        and label.get("no_recompute_from_lambda48") is True
    )
    dataset = summary.get("dataset_readiness", {})
    checks["dataset_counts_expected"] = (
        dataset.get("expanded_primary_samples") == 47
        and dataset.get("stage4a710_valid_primary_labels") is True
        and dataset.get("stage4a714_medium_actions") == 60
        and dataset.get("stage4a714_medium_captures") == 70
    )
    checks["negative_scope_no_training_or_rl"] = (
        summary.get("actual_rl_training_allowed") is False
        and negative.get("actual_rl_training_allowed") is False
        and negative.get("rl") is False
        and negative.get("gdpo") is False
        and negative.get("ppo") is False
        and negative.get("checkpoint_training") is False
        and negative.get("bc_training") is False
        and negative.get("optimizer_step") is False
        and negative.get("model_save") is False
        and negative.get("label_promotion") is False
        and negative.get("long_rollout") is False
        and negative.get("isaac_startup_in_stage715") is False
        and negative.get("map_predict_in_stage715") is False
        and negative.get("lambda48_primary") is False
    )
    checks["no_stage4a714_runtime_processes"] = summary.get("live_stage4a714_processes") == [] and process_hits() == []
    upload_text = (OUT / "stage4a715_upload_to_web_review_file_list.md").read_text(encoding="utf-8")
    checks["upload_list_has_runtime_html_mp4_and_no_rl_warning"] = (
        "short_rollout_uncertainty_bonus_index.html" in upload_text
        and "short_rollout_flythrough.mp4" in upload_text
        and "Actual RL/GDPO/PPO training remains unauthorized." in upload_text
    )
    tracked = git_files()
    checks["outputs_logs_checkpoints_not_tracked"] = (
        not any(path.startswith(("outputs/", "logs/")) for path in tracked)
        and not any(path.startswith("checkpoints/") or path.endswith((".pt", ".pth", ".ckpt", ".tar")) for path in tracked)
    )
    context = (ROOT / ".project_context/CURRENT_STATE.md").read_text(encoding="utf-8")[:2400]
    todo = (ROOT / ".project_context/TODO.md").read_text(encoding="utf-8")[:2400]
    log = (ROOT / ".project_context/CODEX_LOG.md").read_text(encoding="utf-8")[:3200]
    checks["context_updated"] = (
        "Stage 4A-7.15 Final Pre-RL Readiness Packet Complete" in context
        and "Stage 4A-7.15 Final Pre-RL Readiness Review" in todo
        and "Stage 4A-7.15 final Pre-RL readiness packet actions" in log
    )

    for key, passed in checks.items():
        if not passed:
            blockers.append(key)
    result = {"all_passed": not blockers, "checks": checks, "blockers": blockers}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
