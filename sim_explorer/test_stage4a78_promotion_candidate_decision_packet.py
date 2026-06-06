from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a78_promotion_candidate_decision_packet"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_files() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    required_files = [
        "stage4a78_promotion_candidate_decision_summary.json",
        "stage4a78_promotion_candidate_decision_summary.md",
        "loaded_stage4a77_review_evidence.json",
        "loaded_stage4a77_review_evidence.md",
        "clean_promotion_candidates.csv",
        "clean_promotion_candidates.json",
        "conflict_manual_recheck_rows.csv",
        "conflict_manual_recheck_rows.json",
        "rejected_rows.csv",
        "rejected_rows.json",
        "unsure_rows.csv",
        "unsure_rows.json",
        "promotion_candidate_policy.md",
        "promotion_candidate_policy.json",
        "why_no_promotion_yet.md",
        "why_no_promotion_yet.json",
        "future_stage4a79_no_training_promotion_implementation_sketch.md",
        "no_training_report.json",
        "no_training_report.md",
        "no_promotion_report.json",
        "no_promotion_report.md",
        "no_runtime_report.json",
        "no_runtime_report.md",
        "source_hash_report.json",
        "source_hash_report.md",
        "prior_dataset_hash_report.json",
        "prior_dataset_hash_report.md",
        "git_status_before.txt",
        "git_status_after.txt",
        "recommended_next_faithful_step.md",
    ]
    checks["output_dir_exists"] = OUT.is_dir()
    for name in required_files:
        checks[f"required:{name}"] = (OUT / name).is_file()

    summary = load_json(OUT / "stage4a78_promotion_candidate_decision_summary.json") if checks["required:stage4a78_promotion_candidate_decision_summary.json"] else {}
    policy = load_json(OUT / "promotion_candidate_policy.json") if checks["required:promotion_candidate_policy.json"] else {}
    no_promotion = load_json(OUT / "no_promotion_report.json") if checks["required:no_promotion_report.json"] else {}
    no_training = load_json(OUT / "no_training_report.json") if checks["required:no_training_report.json"] else {}
    no_runtime = load_json(OUT / "no_runtime_report.json") if checks["required:no_runtime_report.json"] else {}
    source_hash = load_json(OUT / "source_hash_report.json") if checks["required:source_hash_report.json"] else {}
    prior_hash = load_json(OUT / "prior_dataset_hash_report.json") if checks["required:prior_dataset_hash_report.json"] else {}
    clean = load_json(OUT / "clean_promotion_candidates.json") if checks["required:clean_promotion_candidates.json"] else []
    conflicts = load_json(OUT / "conflict_manual_recheck_rows.json") if checks["required:conflict_manual_recheck_rows.json"] else []

    checks["summary_completed"] = summary.get("completed") is True and summary.get("blocked") is False
    checks["clean_candidates_count_reported"] = isinstance(summary.get("clean_promotion_candidates"), int)
    checks["clean_candidates_count_matches"] = len(clean) == summary.get("clean_promotion_candidates")
    checks["warnings_count_from_7_7_zero"] = summary.get("stage4a77_warnings") == 0
    checks["conflict_count_reported"] = isinstance(summary.get("conflict_rows"), int)
    checks["conflict_count_matches"] = len(conflicts) == summary.get("conflict_rows")
    checks["no_expert_action_index_primary_created"] = summary.get("expert_action_index_primary_created") is False and no_promotion.get("expert_action_index_primary_created") is False
    checks["no_label_promotion"] = summary.get("label_promotion") is False and no_promotion.get("label_promotion") is False and policy.get("stage4a78_promotes_labels") is False
    checks["no_training"] = summary.get("bc_training") is False and no_training.get("training") is False
    checks["no_optimizer_step"] = summary.get("optimizer_step") is False and no_training.get("optimizer_step") is False
    checks["no_checkpoint"] = summary.get("checkpoint") is False and no_training.get("checkpoint") is False
    checks["no_isaac_startup"] = summary.get("isaac_startup") is False and no_runtime.get("isaac_startup") is False
    checks["no_map_predict"] = summary.get("map_predict") is False and no_runtime.get("map_predict") is False
    checks["no_rollout"] = summary.get("rollout") is False and no_runtime.get("rollout") is False
    checks["no_rl_gdpo_ppo"] = summary.get("rl_gdpo_ppo") is False and no_runtime.get("rl_gdpo_ppo") is False
    checks["lambda48_shadow_only"] = "shadow" in str(summary.get("lambda48_role", "")).lower()
    checks["future_sketch_first_line"] = (
        (OUT / "future_stage4a79_no_training_promotion_implementation_sketch.md").read_text(encoding="utf-8").splitlines()[0]
        == "DO NOT RUN IN STAGE 4A-7.8."
        if checks["required:future_stage4a79_no_training_promotion_implementation_sketch.md"]
        else False
    )
    checks["prior_datasets_unchanged"] = summary.get("prior_datasets_unchanged") is True and all(
        item.get("unchanged") is True for item in prior_hash.values() if item.get("exists")
    )
    checks["source_hashes_unchanged"] = summary.get("source_hashes_unchanged") is True and all(
        item.get("unchanged") is True for item in source_hash.values() if item.get("exists")
    )

    tracked = git_files()
    forbidden_tracked = [
        path
        for path in tracked
        if path.startswith(("outputs/", "logs/", "checkpoints/", "data/"))
        or path.lower().endswith((".npz", ".npy", ".png", ".mp4", ".usd", ".usda", ".usdc", ".pt", ".pth", ".ckpt"))
        or "checkpoint" in path.lower()
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
        "clean_promotion_candidates": summary.get("clean_promotion_candidates"),
        "conflict_rows": summary.get("conflict_rows"),
        "forbidden_tracked": forbidden_tracked,
        "large_tracked": large_tracked,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
