# Primary Scene Mapping Smoke Report

- phase: Phase 8 primary-scene mapping smoke
- result: passed
- project_root: /home/ubuntu22/pi
- scene_path: /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd
- scene_bundle_size: 490M
- scene_bundle_ignored_by_git: yes
- script_path: /home/ubuntu22/pi/scripts/phase8_primary_scene_mapping_smoke.py
- run_dir: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019
- stage_load_exit_code: 0
- mapping_smoke_exit_code: 0
- core_dump_count: 0

## Scope

This was a bounded mapping smoke only. It did not train, start RL, fine-tune PI/openpi/VLM, create checkpoints, run map_predict, or run primary long rollout. The output is smoke validation evidence, not training data.

## Stage And Pose

- open_stage_result: true
- stage_available: true
- prim_count: 1324
- Cube count: 279
- Mesh count: 127
- Material count: 124
- robot_spawned: true
- robot_pose_readable: true
- robot_initial_pose: x=-0.75, y=-6.0, z=0.35
- pose_selection_strategy: largest_floor_visibility_grid

## Mapping Metrics

- step_count: 10
- valid_observation_steps: 10
- final_known_ratio: 0.48057726
- final_occupied_cells: 124
- final_free_cells: 4305
- final_unknown_cells: 4787
- total_new_known_cells: 4429
- known_ratio_monotonic_non_decreasing: true
- map_snapshots_saved: 10
- plots_saved: 4
- collision_count: 0
- stuck_count: 0
- map_bounds: x=[-12.75, 11.25], y=[-18.0, 6.0]
- map_resolution: 0.25
- known_ratio_by_step: 0.24207899, 0.33007812, 0.35199653, 0.38476562, 0.41297743, 0.42838542, 0.44303385, 0.46191406, 0.46506076, 0.48057726
- new_known_cells_first_steps: 2231, 811, 202, 302

## Required Channel Checks

- known free / traversed proxy cells: passed, final_free_cells=4305
- occupied cells: passed, final_occupied_cells=124
- unknown cells: passed, final_unknown_cells=4787
- observed_count: passed, total_new_known_cells=4429
- robot_pose layer / trace: passed, robot_xy_trace.png generated
- map expands with actions: passed, known_ratio increased from 0.24207899 to 0.48057726
- non-empty/non-all-known map: passed
- coordinate/scale sanity: passed for bounded BEV window x=[-12.75, 11.25], y=[-18.0, 6.0]

## Plots

- known_ratio_curve: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/known_ratio_curve.png
- occupied_free_unknown_by_step: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/occupied_free_unknown_by_step.png
- bev_map_final: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/bev_map_final.png
- robot_xy_trace: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/robot_xy_trace.png
- ascii map fallback: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/summary/bev_map_final_ascii.txt

## Evidence Files

- mapping_summary: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/summary/mapping_summary.json
- mapping_steps: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/summary/mapping_steps.csv
- stage_probe: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/probes/stage_load_probe.json
- map_snapshots: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/maps/map_step_000.npz ... map_step_009.npz
- logs: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/logs/

## Decision

- phase8_passed: true
- allow_phase9_candidate_viewpoint_information_gain_smoke: true
- allow_primary_long_rollout: false
- allow_training: false
- allow_RL: false
- allow_PI_finetuning: false
- allow_checkpoint: false

## Caveat

This is still smoke validation. The map is produced from the Phase 7 lightweight USD geometry depth/pointcloud proxy, not from a final rendered sensor training-data pipeline. It should be used to validate coordinate consistency, partial-map update logic, and review visualization only.
