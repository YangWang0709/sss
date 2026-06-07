# WebGPT Brief

## Current Phase

Phase 7 primary-scene sensor smoke passed.

## Primary Scene

- scene: /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd
- scene bundle size: 490M
- dependencies present: yes
- original /home/ubuntu22/pi/building_scene.usd preserved: true
- scene bundle committed to Git: false
- scene bundle ignored by local Git exclude: true

## Phase 7 Result

- run_dir: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109
- stage_load_exit_code: 0
- smoke_exit_code: 0
- open_stage_result: true
- stage_available: true
- prim_count: 1324
- Cube count: 279
- Mesh count: 127
- Material count: 124
- core_dump: false
- robot_spawned: true
- robot_pose_readable: true
- robot_fell: false
- large_scale_collision: false
- RGB metadata/proxy visible frames: 8/8
- finite depth frames: 8/8
- non-empty pointcloud/lidar frames: 8/8
- short control actions: 8
- moved actions: 8
- collision actions: 0
- stuck actions: 0

## Scope Status

- training: false
- RL: false
- checkpoint: false
- PI/openpi/VLM fine-tuning: false
- primary long rollout: false
- map_predict: false

## Caveat

Phase 7 used a lightweight USD geometry visibility/depth/pointcloud proxy and did not commit raw PNG, NPZ, HDF5, or large sensor assets. This is enough for the bounded sensor smoke gate. A future rendered camera gate can be added before any training-data use if visual fidelity matters.

## Reports

- primary scene sensor smoke report: /home/ubuntu22/pi/runs/PRIMARY_SCENE_SENSOR_SMOKE_REPORT.md
- sensor smoke run summary: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/smoke/phase7_primary_scene_sensor_smoke_summary.json
- trajectory CSV: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/smoke/trajectory.csv
- sensor frame stats CSV: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/smoke/sensor_frame_stats.csv

## Next Step

Proceed to Phase 8 mapping smoke on /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd. Do not run primary long rollout until mapping smoke, candidate gain smoke, and closed-loop smoke pass. Do not train, start RL, checkpoint, or fine-tune PI/openpi/VLM.
