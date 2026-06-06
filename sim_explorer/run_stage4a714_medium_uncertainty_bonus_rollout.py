#!/usr/bin/env python3
"""Stage 4A-7.14 medium bounded uncertainty-bonus expert rollout adapter.

This file is a narrow opt-in adapter around the Stage 4A-6.13 uncertainty
bonus runner. It leaves the historical short-rollout runner unchanged and
only relaxes the runtime envelope for the Stage 4A-7.14 medium expert rollout
entry point:

- 10 starts
- 6 decision steps per start
- 60 executed actions
- 60 decision frames
- 70 captures, counting one initial/terminal capture per start

The primary expert remains uncertainty_bonus_composite_beta8. Lambda48 remains
shadow/baseline only. The adapter does not train, checkpoint, save a model, or
run RL/GDPO/PPO.
"""

from __future__ import annotations

from pathlib import Path

import run_stage4a613_uncertainty_bonus_short_rollout_pilot as base


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
STAGE = "Stage 4A-7.14-medium-bounded-uncertainty-bonus-expert-rollout"
OUTPUT_NAME = "isaac_stage4a714_medium_bounded_expert_rollout_runtime"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs" / OUTPUT_NAME

MEDIUM_NUM_STARTS = 10
MEDIUM_STEPS_PER_START = 6
MEDIUM_TOTAL_ACTIONS = 60
MEDIUM_TOTAL_DECISION_FRAMES = 60
MEDIUM_TOTAL_CAPTURES = 70


def enforce_stage4a714_medium_args(args) -> None:
    required_flags = {
        "terminal_capture_per_start": args.terminal_capture_per_start,
        "save_dense_uncertainty_artifacts": args.save_dense_uncertainty_artifacts,
        "save_expert_quality_viz": args.save_expert_quality_viz,
        "compare_to_measured_only_pilot": args.compare_to_measured_only_pilot,
        "compare_to_lambda48_pilot": args.compare_to_lambda48_pilot,
        "compare_to_confidence_gated_pilot": args.compare_to_confidence_gated_pilot,
        "compare_to_uncertainty_bonus_decision_pilot": args.compare_to_uncertainty_bonus_decision_pilot,
        "save_viz": args.save_viz,
        "no_long_rollout": args.no_long_rollout,
        "no_full_expert_dataset": args.no_full_expert_dataset,
        "no_training": args.no_training,
        "no_rl_gdpo": args.no_rl_gdpo,
        "write_finalization_sentinel_before_close": args.write_finalization_sentinel_before_close,
    }
    missing = [key for key, value in required_flags.items() if not bool(value)]
    if missing:
        raise ValueError(f"Missing required Stage 4A-7.14 boundary flags: {missing}")

    if int(args.num_starts) != MEDIUM_NUM_STARTS:
        raise ValueError("Stage 4A-7.14 medium rollout requires exactly 10 starts")
    if int(args.max_decision_steps_per_start) != MEDIUM_STEPS_PER_START:
        raise ValueError("Stage 4A-7.14 medium rollout requires max_decision_steps_per_start=6")
    if (
        int(args.max_total_actions) != MEDIUM_TOTAL_ACTIONS
        or int(args.max_total_decision_frames) != MEDIUM_TOTAL_DECISION_FRAMES
        or int(args.max_total_captures) != MEDIUM_TOTAL_CAPTURES
    ):
        raise ValueError("Stage 4A-7.14 medium totals must be actions=60, decision_frames=60, captures=70")
    if int(args.num_candidates) != base.MAX_CANDIDATES or int(args.top_n) != 16:
        raise ValueError("Stage 4A-7.14 uses num_candidates=64 and top_n=16")
    if str(args.primary_formula) != base.PRIMARY_FORMULA:
        raise ValueError(f"Unsupported primary formula: {args.primary_formula}")
    if float(args.lambda_sc) != base.LAMBDA_SC or float(args.beta_uncertainty) != base.BETA_UNCERTAINTY:
        raise ValueError("Stage 4A-7.14 requires lambda_sc=48 and beta_uncertainty=8")
    if [float(v) for v in args.uncertainty_composite_weights] != [0.4, 0.4, 0.2]:
        raise ValueError("Stage 4A-7.14 requires uncertainty composite weights 0.4 0.4 0.2")
    if str(args.motion_mode) != "bounded_medium_rollout":
        raise ValueError("Stage 4A-7.14 requires motion_mode=bounded_medium_rollout")
    if not str(args.finalization_sentinel_path):
        raise ValueError("Stage 4A-7.14 requires an explicit close-guard finalization sentinel path")
    if not str(args.close_guard_run_id):
        raise ValueError("Stage 4A-7.14 requires an explicit close_guard_run_id")


def main() -> None:
    base.STAGE = STAGE
    base.OUTPUT_NAME = OUTPUT_NAME
    base.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    base.enforce_args = enforce_stage4a714_medium_args
    base.main()


if __name__ == "__main__":
    main()
