# WebGPT Brief

## Current Phase

Phase 8 primary-scene mapping smoke passed.

## Primary Scene

- scene: /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd
- scene bundle size: 490M
- dependencies present: yes
- scene bundle committed to Git: false
- scene bundle ignored by local Git exclude: true
- original /home/ubuntu22/pi/building_scene.usd preserved: true

## Phase 8 Result

- run_dir: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019
- script: /home/ubuntu22/pi/scripts/phase8_primary_scene_mapping_smoke.py
- stage_load_exit_code: 0
- mapping_smoke_exit_code: 0
- open_stage_result: true
- stage_available: true
- robot_pose_readable: true
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

## Scope Status

- training: false
- RL: false
- checkpoint: false
- PI/openpi/VLM fine-tuning: false
- primary long rollout: not started
- map_predict: false

## Reports And Review Artifacts

- primary mapping smoke report: /home/ubuntu22/pi/runs/PRIMARY_SCENE_MAPPING_SMOKE_REPORT.md
- mapping summary: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/summary/mapping_summary.json
- mapping steps: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/summary/mapping_steps.csv
- final BEV map: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/bev_map_final.png
- known ratio curve: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/known_ratio_curve.png
- occupancy/free/unknown curve: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/occupied_free_unknown_by_step.png
- robot trace: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/robot_xy_trace.png

## Next Phase

Phase 9 candidate viewpoint + information gain smoke may begin only as a bounded smoke. Do not run primary long rollout, training, RL, checkpointing, or PI/openpi/VLM fine-tuning.
