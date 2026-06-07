# Human Review Checklist

- packet: Phase 5 fallback long rollout
- PI_WORKSPACE: /home/ubuntu22/pi
- rollout_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout
- primary_usd_used: false
- fallback_scene: scenes/minimal_indoor_smoke.usda
- training: false
- RL: false
- checkpoint: false

## Required Review Items

1. For each start directory start_000 through start_009, inspect trajectory.csv and confirm the pose/action sequence is continuous and reasonable.
2. Inspect candidates.csv for each start and confirm candidate viewpoints land in plausible free-space regions, not walls, obstacles, or out-of-bounds areas.
3. Confirm selected viewpoints generally move toward or near unknown/frontier regions.
4. Confirm known_ratio generally increases with actions in trajectory.csv and rollout_steps.csv.
5. Inspect failure_summary.csv and decide whether the 6 starts with failures are acceptable, expected end-of-exploration cases, or bad selector behavior.
6. Check for repeated selection of the same point, wall collisions, unreachable targets, and local spinning/stuck behavior.
7. Compare candidate score with actual coverage/newly_known_cells and decide whether the score approximately matches realized coverage gain.
8. Confirm the packet is only used for schema and pipeline sanity review, not final primary-scene training.

## Files To Open

- /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/rollout_summary.json
- /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/rollout_steps.csv
- /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/candidate_summary.csv
- /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/failure_summary.csv
- /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/start_000/trajectory.csv through start_009/trajectory.csv
- /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/start_000/candidates.csv through start_009/candidates.csv

## Review Decision Template

- approve_schema_for_primary_scene_collection: yes/no/unsure
- selector_behavior_reasonable: yes/no/unsure
- failure_cases_acceptable: yes/no/unsure
- primary_blocker_acknowledged: building_scene.usd still needs repair or replacement
- reviewer_notes:
