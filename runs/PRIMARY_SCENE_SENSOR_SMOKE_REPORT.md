# Primary Scene Sensor Smoke Report

- phase: Phase 7 primary-scene sensor smoke
- result: passed
- created_at: 2026-06-07
- project_root: /home/ubuntu22/pi
- run_dir: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109
- scene_path: /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd
- scene_bundle_size: 490M
- dependencies_present: yes
- original_building_scene_preserved: true

## Scope

This was a bounded primary-scene sensor smoke. It did not run training, RL, PI/openpi/VLM fine-tuning, checkpoint creation, primary long rollout, or map_predict.

## Git Safety

- scene bundle ignored by git: yes
- scene bundle tracked by git: no
- large tracked files over 50MB: 0
- forbidden artifacts tracked: 0
- committed outputs/logs/checkpoints/assets/large files: false

## Stage Load Smoke

- stage_load_exit_code: 0
- open_stage_result: true
- stage_available: true
- prim_count: 1324
- Cube count: 279
- Mesh count: 127
- Material count: 124
- Camera count: 4
- up_axis: Z
- meters_per_unit: 1.0
- core_dump: false
- sha256: 20cb78c9e4b562ba4264954a4617e0c5c482298d677f72c045b86738eb7f3b2a

## Robot Spawn Smoke

- smoke_exit_code: 0
- robot_spawned: true
- robot_pose_readable: true
- robot_initial_pose: x=-0.75, y=-6.0, z=0.35
- pose_selection_strategy: largest_floor_visibility_grid
- visible_structure_count_at_start: 78
- clearance_at_start: 3.599239972544992
- robot_fell: false
- large_scale_collision: false
- initial_collision_prim: none

## Sensor Smoke

- frame_count: 8
- sensor_metadata_mode: usd_geometry_visibility_proxy
- RGB result: passed, 8/8 frames had visible scene geometry and were not all-black by metadata proxy
- RGB visible prim count range: 3 to 142
- depth result: passed, 8/8 frames had finite depth values
- depth min range: 0.783938 to 3.756399
- depth max range: 115.457795 to 118.317380
- depth mean range: 7.412378 to 9.777415
- pointcloud/lidar result: passed, 8/8 frames had non-empty finite point samples
- point count range: 1287 to 1341

Note: this smoke uses a lightweight USD geometry visibility/depth/pointcloud proxy rather than committing raw rendered RGB PNG, NPZ, or HDF5 outputs. That is intentional for this bounded gate and keeps Git clean. It validates that the stage geometry is visible from the robot pose, depth values are finite, and point samples are non-empty. If final visual fidelity is required, add a separate raster camera capture gate before using the scene for training data.

## Short Control Smoke

- short_control_action_count: 8
- moved_actions: 8
- collision_count: 0
- stuck_count: 0
- trajectory_rows: 8
- control_smoke_result: passed

## MDL And Material Warnings

Isaac/Kit logs still contain MDL/material related warnings. They did not block stage load, geometry traversal, robot pose updates, sensor metadata generation, or short control execution. They are acceptable for this smoke gate, but material fidelity should still be treated as unresolved for any future rendered-visual dataset gate.

## Decision

- allow_phase_8_mapping_smoke: true
- allow_primary_long_rollout: false
- allow_training: false
- allow_RL: false
- allow_PI_finetuning: false
- allow_checkpoint: false

## Evidence Files

- stage probe JSON: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/probes/stage_load_probe.json
- stage probe log: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/logs/stage_load_probe.log
- sensor smoke summary: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/smoke/phase7_primary_scene_sensor_smoke_summary.json
- trajectory CSV: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/smoke/trajectory.csv
- sensor frame stats CSV: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/smoke/sensor_frame_stats.csv
- core check: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/core_file_check.txt
