# WebGPT Brief

## Current Phase

Phase 5 fallback long rollout passed.

## Completed

- Confirmed PI_WORKSPACE: /home/ubuntu22/pi
- Preserved original primary USD: /home/ubuntu22/pi/building_scene.usd
- Documented primary USD blocker: Isaac/Omniverse USD context crashes when opening building_scene.usd.
- Built fallback scene: scenes/minimal_indoor_smoke.usda
- Passed scene load, RGB-D camera, point-cloud metadata, robot pose/control, partial-map, candidate-gain, closed-loop, and fallback long-rollout smoke stages.
- Generated human-review-ready fallback rollout metadata packet.

## Key Results

- fallback_scene: scenes/minimal_indoor_smoke.usda
- primary_usd_used: false
- start_count: 10
- total_action_count: 143
- total_step_rows: 149
- candidate_rows: 3576
- average_final_known_ratio: 0.8947
- starts_with_failures: 6
- training: false
- RL: false
- checkpoint: false

## Workspace

- PI_WORKSPACE: /home/ubuntu22/pi
- conda_env: env_isaaclab
- env_activation_command: source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab

## Key Paths

- run_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529
- rollout_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout
- rollout_summary: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/rollout_summary.json
- rollout_steps_csv: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/rollout_steps.csv
- candidate_summary_csv: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/candidate_summary.csv
- failure_summary_csv: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/failure_summary.csv
- human_review_checklist: /home/ubuntu22/pi/runs/HUMAN_REVIEW_CHECKLIST.md
- inspect_script: /home/ubuntu22/pi/scripts/inspect_phase5_rollout.py

## Metrics

- start_count: 10
- total_action_count: 143
- total_step_rows: 149
- candidate_rows: 3576
- average_final_known_ratio: 0.8947
- starts_with_failures: 6

## Current Issues

- The fallback pipeline passed and is ready for schema/pipeline sanity review.
- The data should not be used as final primary-scene training data.
- The original building_scene.usd still needs repair, conversion, or replacement before collecting final primary-scene rollout data.

## Critic Review Conclusion

- continue: wait for human review of fallback packet and repair/replace primary USD before primary-scene rollout.
- deviated_from_goal: no
- human_review_needed: yes, review the fallback rollout schema and action sanity.

## Questions For Web ChatGPT

1. Are the fallback rollout metadata fields sufficient for human review and later primary-scene collection?
2. Should the next engineering step prioritize repairing building_scene.usd or replacing it with a new stable primary indoor USD scene?
