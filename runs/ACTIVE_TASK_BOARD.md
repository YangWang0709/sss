# Active Task Board

- PI_WORKSPACE: /home/ubuntu22/pi
- current_phase: Phase 7 primary-scene sensor smoke passed
- current_goal: Prepare Phase 8 mapping smoke on the repaired primary scene
- latest_phase7_run_dir: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109
- latest_primary_scene: /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd
- scene_bundle_size: 490M
- latest_decision: proceed_to_phase_8_mapping_smoke

| Agent | Status | Task | Notes |
| --- | --- | --- | --- |
| Main Coordinator | done | Validate Phase 7 bounded sensor smoke | No training/RL/PI fine-tuning/checkpoint/long rollout. |
| Git Safety | done | Confirm 490M scene bundle stays local-only | Scene bundle ignored by .git/info/exclude and not tracked. |
| Isaac Stage Probe | done | Headless open repaired primary scene | open_stage_result=true, prim_count=1324. |
| Robot Proxy | done | Spawn minimal robot and read pose | Pose readable; no fall or large-scale collision. |
| Sensor Smoke | done | Save lightweight RGB/depth/pointcloud metadata | 8/8 frames visible, finite depth, non-empty point samples. |
| Short Control | done | Run 8 guarded pose actions | 8 moved, 0 collision, 0 stuck. |
| Phase 8 Mapping Smoke | ready | Run bounded mapping smoke next | Do not run primary long rollout yet. |
| Training/RL/PI | forbidden | No training, RL, checkpoint, or PI/openpi/VLM fine-tuning | Remains false. |
