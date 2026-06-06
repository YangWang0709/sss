from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
ADAPTER = ROOT / "sim_explorer/run_stage4a714_medium_uncertainty_bonus_rollout.py"
BASE = ROOT / "sim_explorer/run_stage4a613_uncertainty_bonus_short_rollout_pilot.py"


REQUIRED_STRINGS = [
    "Stage 4A-7.14-medium-bounded-uncertainty-bonus-expert-rollout",
    "MEDIUM_NUM_STARTS = 10",
    "MEDIUM_STEPS_PER_START = 6",
    "MEDIUM_TOTAL_ACTIONS = 60",
    "MEDIUM_TOTAL_DECISION_FRAMES = 60",
    "MEDIUM_TOTAL_CAPTURES = 70",
    "no_long_rollout",
    "no_full_expert_dataset",
    "no_training",
    "no_rl_gdpo",
    "write_finalization_sentinel_before_close",
    "bounded_medium_rollout",
    "close_guard_run_id",
    "finalization_sentinel_path",
    "base.enforce_args = enforce_stage4a714_medium_args",
    "base.main()",
]


def main() -> int:
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    checks["adapter_exists"] = ADAPTER.is_file()
    checks["base_runner_exists"] = BASE.is_file()
    if not checks["adapter_exists"]:
        blockers.append("adapter_missing")
        print(json.dumps({"all_passed": False, "checks": checks, "blockers": blockers}, indent=2, sort_keys=True))
        return 1

    text = ADAPTER.read_text(encoding="utf-8")
    checks["python_ast_parse"] = isinstance(ast.parse(text), ast.Module)
    for item in REQUIRED_STRINGS:
        checks[f"contains:{item}"] = item in text

    checks["does_not_define_training_loop"] = "optimizer.step(" not in text and ".backward(" not in text
    checks["does_not_save_checkpoint"] = "torch.save" not in text and "state_dict" not in text
    lowered = text.lower()
    checks["does_not_import_rl"] = all(
        token not in lowered
        for token in [
            "import ppo",
            "from ppo",
            "import gdpo",
            "from gdpo",
            "replay_buffer.append",
            "policy_gradient",
            "optimizer.step(",
        ]
    )
    checks["lambda48_shadow_doc"] = "Lambda48 remains" in text and "shadow/baseline only" in text
    checks["primary_beta8_doc"] = "uncertainty_bonus_composite_beta8" in text

    base_text = BASE.read_text(encoding="utf-8") if BASE.is_file() else ""
    checks["base_runner_unchanged_short_gate_present"] = (
        "Stage 4A-6.13 requires max_decision_steps_per_start=3" in base_text
        and "actions=30, decision_frames=30, captures=40" in base_text
    )

    for key, ok in checks.items():
        if not ok:
            blockers.append(key)

    result = {
        "all_passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "adapter": str(ADAPTER),
        "base_runner": str(BASE),
        "runtime_started": False,
        "training": False,
        "checkpoint_created": False,
        "rl_gdpo_ppo": False,
        "lambda48_primary_use": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
