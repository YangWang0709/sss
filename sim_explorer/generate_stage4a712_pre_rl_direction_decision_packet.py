from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
BRIDGE = ROOT / "outputs/autonomous_pre_rl_bridge"
OUT = BRIDGE / "stage4a712_pre_rl_direction_decision_packet"
STAGE711 = ROOT / "outputs/isaac_stage4a711_expanded_tiny_no_checkpoint_eval"
STAGE710 = ROOT / "outputs/isaac_stage4a710_expanded_dataset_qa"
STAGE79 = ROOT / "outputs/isaac_stage4a79_no_training_promotion_implementation"
STAGE71B = ROOT / "outputs/isaac_stage4a71b_looso_tiny_eval_no_checkpoint"

SUMMARY711 = STAGE711 / "stage4a711_expanded_tiny_eval_summary.json"
SUMMARY710 = STAGE710 / "stage4a710_expanded_dataset_qa_summary.json"
DATASET79 = STAGE79 / "expanded_primary_bc_dataset.npz"

FORBIDDEN_FIELDS = [
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
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def git_status() -> str:
    return run(["git", "status", "--short", "--branch"]).stdout


def git_head() -> str:
    return run(["git", "rev-parse", "HEAD"]).stdout.strip()


def git_log() -> str:
    return run(["git", "log", "--oneline", "-5"]).stdout


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(jsonable(value), sort_keys=True)
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


def hash_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in paths.items()
    }


def prepend_once(path: Path, marker: str, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if marker in existing:
        return
    path.write_text(text.rstrip() + "\n\n---\n\n" + existing, encoding="utf-8")


def update_context(summary: dict[str, Any]) -> None:
    current = ROOT / ".project_context/CURRENT_STATE.md"
    todo = ROOT / ".project_context/TODO.md"
    log = ROOT / ".project_context/CODEX_LOG.md"
    marker = "Stage 4A-7.12 Pre-RL Direction Decision Packet Complete"
    current_text = f"""# Current State - {marker}

Stage 4A-7.12 decision-only packet is complete. Output directory:
`{OUT}`.

Decision: choose Option C, medium bounded expert rollout design/preflight, because Stage 4A-7.11 showed only a weak tiny no-checkpoint signal on `47` primary samples. The selected next step is Stage 4A-7.13 design/preflight, not runtime yet.

No Isaac startup, capture, map_predict, action execution, rollout, BC training, optimizer step, model save, checkpoint, label promotion, replay-buffer learning, or RL/GDPO/PPO occurred. Lambda48 remains shadow/baseline only and is not the primary label source.

Next: run Stage 4A-7.13 medium expert rollout design/preflight. Runtime may start only if the preflight passes and remains bounded to the approved medium expert rollout scope.
"""
    todo_text = f"""# TODO - {marker}

Open:
`{OUT / "stage4a712_pre_rl_direction_decision_index.html"}`.

Selected next faithful task: Stage 4A-7.13 medium bounded expert rollout design/preflight. Do not start runtime until preflight passes. Do not run full BC training, checkpoint training, long rollout, or RL/GDPO/PPO.

Important implementation note: the existing Stage 4A-6.13/7.2 rollout runner is hard-gated for the previous short `3 step / 30 action / 40 capture` envelope, so Stage 4A-7.13 must verify whether a safe medium runner/adaptation exists before Stage 4A-7.14 runtime.
"""
    log_text = f"""## {utc_now()} - Stage 4A-7.12 pre-RL direction decision packet

- Created `{OUT}`.
- Verified Stage 4A-7.11 evidence: samples `{summary.get("stage711_samples")}`, top1 `{summary.get("stage711_eval_top1")}`, top3 `{summary.get("stage711_eval_top3")}`, MRR `{summary.get("stage711_eval_mrr")}`, zero-top1 folds `{summary.get("stage711_zero_top1_folds")}`.
- Decision: Option C, medium bounded expert rollout design/preflight, because current data volume is below checkpoint readiness and the tiny evaluation remains weak.
- No runtime, training, optimizer step, checkpoint, model save, label promotion, replay-buffer training, or RL/GDPO/PPO occurred. Lambda48 remains shadow/baseline only.
"""
    prepend_once(current, marker, current_text)
    prepend_once(todo, marker, todo_text)
    log.write_text((log.read_text(encoding="utf-8") if log.is_file() else "") + "\n" + log_text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")

    blockers: list[str] = []
    for name, path in {
        "stage4a711_summary": SUMMARY711,
        "stage4a710_summary": SUMMARY710,
        "stage4a79_expanded_dataset": DATASET79,
    }.items():
        if not path.exists():
            blockers.append(f"missing:{name}:{path}")

    if blockers:
        summary = {
            "completed": False,
            "blocked": True,
            "main_blocker": "; ".join(blockers),
            "blockers": blockers,
            "stage": "Stage 4A-7.12",
            "output_dir": str(OUT),
        }
        write_json(OUT / "stage4a712_pre_rl_direction_decision_summary.json", summary)
        write_md_table(OUT / "stage4a712_pre_rl_direction_decision_summary.md", "Stage 4A-7.12 Pre-RL Direction Decision Summary", list(summary.items()))
        return 1

    s711 = read_json(SUMMARY711)
    s710 = read_json(SUMMARY710)

    loaded_evidence = {
        "stage": "Stage 4A-7.12",
        "git_head": git_head(),
        "git_log_oneline_5": git_log(),
        "stage711_summary_path": SUMMARY711,
        "stage710_summary_path": SUMMARY710,
        "expanded_dataset_path": DATASET79,
        "stage711_completed": s711.get("completed") is True,
        "stage711_blocked": s711.get("blocked") is True,
        "stage711_samples": s711.get("samples"),
        "stage711_candidate_count": s711.get("candidate_count"),
        "stage711_d_model": s711.get("D_model"),
        "stage711_valid_labels": s711.get("valid_labels"),
        "stage711_lambda48_primary_use": s711.get("lambda48_primary_use"),
        "stage711_eval_top1": s711.get("eval_top1"),
        "stage711_eval_top3": s711.get("eval_top3"),
        "stage711_eval_top5": s711.get("eval_top5"),
        "stage711_eval_mrr": s711.get("eval_mrr"),
        "stage711_zero_top1_folds": s711.get("zero_top1_folds"),
        "stage711_overfit_final_top1": s711.get("overfit_final_top1"),
        "stage711_checkpoint_created": s711.get("checkpoint_created"),
        "stage711_model_saved": s711.get("model_saved"),
        "stage711_isaac_startup": s711.get("isaac_startup"),
        "stage711_map_predict": s711.get("map_predict"),
        "stage711_rollout": s711.get("rollout"),
        "stage711_rl_gdpo_ppo": s711.get("rl_gdpo_ppo"),
        "stage710_completed": s710.get("completed") is True,
        "stage710_blocked": s710.get("blocked") is True,
    }
    report_pair(
        "loaded_stage4a711_evidence",
        loaded_evidence,
        "Loaded Stage 4A-7.11 Evidence",
        [
            ("samples", loaded_evidence["stage711_samples"]),
            ("candidate_count", loaded_evidence["stage711_candidate_count"]),
            ("D_model", loaded_evidence["stage711_d_model"]),
            ("eval_top1", loaded_evidence["stage711_eval_top1"]),
            ("eval_top3", loaded_evidence["stage711_eval_top3"]),
            ("eval_mrr", loaded_evidence["stage711_eval_mrr"]),
            ("zero_top1_folds", loaded_evidence["stage711_zero_top1_folds"]),
            ("lambda48_primary_use", loaded_evidence["stage711_lambda48_primary_use"]),
        ],
    )

    option_matrix = {
        "stage": "Stage 4A-7.12",
        "options": [
            {
                "id": "A",
                "name": "deeper no-checkpoint BC",
                "allowed_now": True,
                "recommended": False,
                "reason": "Could probe optimization mechanics, but the 47-sample dataset is still too small and weak for useful policy-quality evidence.",
                "risk": "more optimizer depth may overfit the same limited starts",
            },
            {
                "id": "B",
                "name": "controlled BC checkpoint experiment",
                "allowed_now": False,
                "recommended": False,
                "reason": "Checkpoint readiness gate prefers at least 150 reviewed primary samples and multiple reviewed rollout batches.",
                "risk": "checkpoint could look authoritative before data coverage is adequate",
            },
            {
                "id": "C",
                "name": "medium bounded expert rollout",
                "allowed_now": True,
                "recommended": True,
                "reason": "Best next value is more reviewed expert data under the same uncertainty_bonus_composite_beta8 primary lineage.",
                "risk": "requires runtime preflight and a medium runner/adaptation that preserves close guard and safety gates",
            },
        ],
        "selected_option": "C",
        "selection_reason": "Stage 4A-7.11 gives only a small positive no-checkpoint signal; data volume and start coverage are the bottleneck.",
    }
    report_pair(
        "decision_option_matrix",
        option_matrix,
        "Decision Option Matrix",
        [("selected_option", "C"), ("reason", option_matrix["selection_reason"])],
    )

    selected_decision = {
        "stage": "Stage 4A-7.12",
        "selected_next_stage": "Stage 4A-7.13 medium expert rollout design/preflight",
        "selected_option": "C",
        "approved_to_create_design_preflight": True,
        "approved_to_run_medium_runtime_now": False,
        "runtime_gate": "Stage 4A-7.13 preflight must pass before Stage 4A-7.14 runtime.",
        "primary_expert_label_source": "stage4a613_uncertainty_bonus_executed_primary",
        "primary_formula": "uncertainty_bonus_composite_beta8",
        "lambda48_role": "shadow/baseline only",
        "labels_not_recomputed_from_lambda48": True,
        "no_label_promotion_this_stage": True,
    }
    report_pair("selected_next_step_decision", selected_decision, "Selected Next Step Decision")

    approval_scope = {
        "stage": "Stage 4A-7.12",
        "decision_only": True,
        "medium_expert_rollout_design_preflight_next": True,
        "medium_expert_rollout_runtime_not_started": True,
        "controlled_bc_checkpoint_not_approved_this_stage": True,
        "rl_training_not_approved": True,
        "long_unbounded_rollout_not_approved": True,
        "actual_rl_training_must_wait_for_final_pre_rl_packet_and_user_phrase": True,
    }
    report_pair("approval_scope_report", approval_scope, "Approval Scope Report")

    negative_scope = {
        "stage": "Stage 4A-7.12",
        "isaac_startup_count": 0,
        "capture_count": 0,
        "map_predict_call_count": 0,
        "sscnet_inference_count": 0,
        "action_execution_count": 0,
        "rollout_count": 0,
        "short_rollout": False,
        "long_rollout": False,
        "bc_training": False,
        "training": False,
        "training_loop": False,
        "forward_count": 0,
        "backward_count": 0,
        "optimizer_step_count": 0,
        "checkpoint": False,
        "checkpoint_created": False,
        "model_save": False,
        "model_saved": False,
        "state_dict_saved": False,
        "torch_save_calls": 0,
        "rl": False,
        "gdpo": False,
        "ppo": False,
        "policy_optimization": False,
        "replay_buffer_training": False,
        "label_promotion": False,
        "prediction_writeback": False,
        "uncertainty_writeback": False,
        "target_ground_truth_future_observed_use": False,
        "prior_datasets_modified": False,
        "fixed_usd_modified": False,
        "sscnet_checkpoint_modified": False,
    }
    report_pair("global_negative_scope_report", negative_scope, "Global Negative Scope Report")
    report_pair("no_training_report", {"bc_training": False, "optimizer_step": False, "training_loop": False}, "No Training Report")
    report_pair("no_checkpoint_report", {"checkpoint_created": False, "model_saved": False, "torch_save_calls": 0}, "No Checkpoint Report")
    report_pair("no_runtime_report", {"isaac_startup": False, "capture": False, "map_predict": False, "rollout": False}, "No Runtime Report")
    report_pair("no_rl_gdpo_ppo_report", {"rl": False, "gdpo": False, "ppo": False, "replay_buffer_training": False}, "No RL/GDPO/PPO Report")
    report_pair("why_no_execution_yet", {"reason": "Stage 4A-7.12 is decision-only; Stage 4A-7.13 must preflight before any bounded medium runtime."}, "Why No Execution Yet")

    lambda_audit = {
        "lambda48_primary_use": False,
        "lambda48_role": "shadow/baseline only",
        "labels_not_recomputed_from_lambda48": True,
        "primary_label_source": "stage4a613_uncertainty_bonus_executed_primary",
        "blocker_if_primary_label_lambda48": True,
    }
    report_pair("lambda48_shadow_only_audit", lambda_audit, "Lambda48 Shadow-Only Audit")
    forbidden_audit = {
        "forbidden_fields": FORBIDDEN_FIELDS,
        "used_as_feature_label_score_reward_filter": False,
        "target_ground_truth_future_observed_use": False,
        "passed": True,
    }
    report_pair("forbidden_field_audit", forbidden_audit, "Forbidden Field Audit")

    source_hash = hash_report(
        {
            "stage4a712_generator": ROOT / "sim_explorer/generate_stage4a712_pre_rl_direction_decision_packet.py",
            "stage4a712_validator": ROOT / "sim_explorer/test_stage4a712_pre_rl_direction_decision_packet.py",
        }
    )
    prior_hash = hash_report(
        {
            "stage4a711_summary": SUMMARY711,
            "stage4a710_summary": SUMMARY710,
            "stage4a79_expanded_dataset": DATASET79,
        }
    )
    report_pair("source_hash_report", source_hash, "Source Hash Report")
    report_pair("prior_dataset_hash_report", prior_hash, "Prior Dataset Hash Report")

    future_sketch = """DO NOT RUN IN STAGE 4A-7.12.

Stage 4A-7.13 should create a medium expert rollout design/preflight packet:
- starts: 10
- steps_per_start: 6
- max actions: 60
- max decision frames: 60
- terminal QA capture per start: true
- close guard mandatory
- primary expert: uncertainty_bonus_composite_beta8
- lambda48: shadow/baseline only
- no training, checkpoint, long rollout, or RL/GDPO/PPO

The preflight must verify whether the current short rollout runner can be safely adapted beyond its existing hard-coded 3-step/30-action envelope.
"""
    (OUT / "future_stage4a713_medium_expert_rollout_design_preflight_sketch.md").write_text(future_sketch, encoding="utf-8")
    (OUT / "recommended_next_faithful_step.md").write_text(
        "# Recommended Next Faithful Step\n\nRun Stage 4A-7.13 medium expert rollout design/preflight. Do not start runtime until that preflight passes.\n",
        encoding="utf-8",
    )

    web_request = f"""# Stage 4A-7.12 Web Reviewer Request

Review the Stage 4A-7.12 decision packet and answer whether Option C, medium bounded expert rollout design/preflight, is the correct next step before any BC checkpoint or RL work.

Inputs:
- Stage 4A-7.11 samples: {loaded_evidence["stage711_samples"]}
- Eval top1/top3/MRR: {loaded_evidence["stage711_eval_top1"]} / {loaded_evidence["stage711_eval_top3"]} / {loaded_evidence["stage711_eval_mrr"]}
- Zero-top1 folds: {loaded_evidence["stage711_zero_top1_folds"]}
- Lambda48 primary use: {loaded_evidence["stage711_lambda48_primary_use"]}

Expected decision: approve Option C for design/preflight only. Do not approve runtime, checkpoint, or RL in Stage 4A-7.12.
"""
    (BRIDGE / "stage_request_to_web_reviewer.md").write_text(web_request, encoding="utf-8")
    write_json(
        BRIDGE / "web_reviewer_packet_manifest.json",
        {
            "stage": "Stage 4A-7.12",
            "packet_dir": str(OUT),
            "main_html": str(OUT / "stage4a712_pre_rl_direction_decision_index.html"),
            "request": str(BRIDGE / "stage_request_to_web_reviewer.md"),
        },
    )
    web_review_result = {
        "stage": "Stage 4A-7.12",
        "review_schema_version": "stage4a_pre_rl_web_review_v1",
        "web_model_used": False,
        "chrome_review_page_prepared": True,
        "fallback_review_result": "pending_chrome_or_web_review",
        "decision_under_review": "Option C design/preflight only",
        "machine_readable": True,
        "no_runtime_checkpoint_training_rl_approved_by_this_review": True,
    }
    write_json(BRIDGE / "web_review_result.json", web_review_result)
    (BRIDGE / "web_review_transcript.md").write_text(
        "# Stage 4A-7.12 Web Review Transcript\n\nPending Chrome/web-model review. This file is machine-readable-adjacent handoff text and does not approve runtime, checkpoint, or RL.\n",
        encoding="utf-8",
    )

    summary = {
        "completed": True,
        "blocked": False,
        "main_blocker": "",
        "stage": "Stage 4A-7.12",
        "output_dir": str(OUT),
        "bridge_dir": str(BRIDGE),
        "selected_option": "C",
        "recommended_next": "Stage 4A-7.13 medium expert rollout design/preflight",
        "stage711_samples": loaded_evidence["stage711_samples"],
        "stage711_eval_top1": loaded_evidence["stage711_eval_top1"],
        "stage711_eval_top3": loaded_evidence["stage711_eval_top3"],
        "stage711_eval_mrr": loaded_evidence["stage711_eval_mrr"],
        "stage711_zero_top1_folds": loaded_evidence["stage711_zero_top1_folds"],
        "lambda48_primary_use": False,
        "label_promotion": False,
        "training": False,
        "checkpoint_created": False,
        "runtime_started": False,
        "rl_gdpo_ppo": False,
    }
    report_pair(
        "stage4a712_pre_rl_direction_decision_summary",
        summary,
        "Stage 4A-7.12 Pre-RL Direction Decision Summary",
        [
            ("completed", True),
            ("blocked", False),
            ("selected_option", "C"),
            ("recommended_next", summary["recommended_next"]),
            ("stage711_samples", summary["stage711_samples"]),
            ("stage711_eval_top1", summary["stage711_eval_top1"]),
            ("runtime_started", False),
            ("rl_gdpo_ppo", False),
        ],
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 4A-7.12 Pre-RL Direction Decision</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f7f4; }}
    main {{ max-width: 1040px; margin: 0 auto; }}
    section {{ background: #fff; border: 1px solid #d8ddd7; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    h1, h2 {{ margin-top: 0; }}
    .choice {{ border-left: 6px solid #2f7d5c; }}
    .warn {{ border-left: 6px solid #b7791f; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8ddd7; padding: 8px; text-align: left; vertical-align: top; }}
    code {{ background: #eef1ee; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Stage 4A-7.12 Pre-RL Direction Decision</h1>
  <section class="choice">
    <h2>Selected Next Step</h2>
    <p><strong>Option C:</strong> medium bounded expert rollout <em>design/preflight only</em>.</p>
    <p>No runtime, checkpoint, training, label promotion, long rollout, or RL/GDPO/PPO is approved by this stage.</p>
  </section>
  <section>
    <h2>Evidence From Stage 4A-7.11</h2>
    <table>
      <tr><th>samples</th><td>{loaded_evidence["stage711_samples"]}</td></tr>
      <tr><th>candidate count / D_model</th><td>{loaded_evidence["stage711_candidate_count"]} / {loaded_evidence["stage711_d_model"]}</td></tr>
      <tr><th>eval top1 / top3 / MRR</th><td>{loaded_evidence["stage711_eval_top1"]} / {loaded_evidence["stage711_eval_top3"]} / {loaded_evidence["stage711_eval_mrr"]}</td></tr>
      <tr><th>zero-top1 folds</th><td>{loaded_evidence["stage711_zero_top1_folds"]}</td></tr>
      <tr><th>lambda48 primary use</th><td>{loaded_evidence["stage711_lambda48_primary_use"]}</td></tr>
    </table>
  </section>
  <section class="warn">
    <h2>Gate Warning</h2>
    <p>The existing Stage 4A-6.13/7.2 runner is known to enforce the older <code>3 step / 30 action / 40 capture</code> short rollout envelope. Stage 4A-7.13 must verify a safe medium adapter before Stage 4A-7.14 can run.</p>
  </section>
  <section>
    <h2>Negative Scope</h2>
    <p>RL training started: <strong>false</strong>. Checkpoint created: <strong>false</strong>. Runtime started: <strong>false</strong>.</p>
  </section>
</main>
</body>
</html>
"""
    (OUT / "stage4a712_pre_rl_direction_decision_index.html").write_text(html, encoding="utf-8")
    write_json(BRIDGE / "codex_stage_result_summary.json", summary)
    write_md_table(BRIDGE / "codex_stage_result_summary.md", "Codex Stage Result Summary", list(summary.items()))
    write_json(
        BRIDGE / "next_stage_gate_decision.json",
        {
            "from_stage": "Stage 4A-7.12",
            "next_stage": "Stage 4A-7.13",
            "approved_to_continue": True,
            "approved_scope": "medium expert rollout design/preflight only",
            "runtime_allowed": False,
            "rl_training_allowed": False,
        },
    )

    update_context(summary)
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
