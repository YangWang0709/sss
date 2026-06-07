# Critic Report

- phase: Phase 7 primary-scene sensor smoke
- result: passed
- scope_check: passed
- original_USD_deleted: false
- original_USD_overwritten: false
- scene_bundle_committed: false
- training: false
- RL: false
- checkpoint: false
- PI_finetuning: false

## Conclusion

The repaired/copied primary scene bundle at /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd passed the bounded Phase 7 sensor smoke. Isaac headless stage load exited cleanly with prim_count=1324, Cube count=279, Mesh count=127, and Material count=124. A minimal robot proxy spawned in the scene, pose was readable, eight short guarded actions completed without collision or stuck flags, and all eight frames produced finite geometry-depth and non-empty pointcloud/lidar proxy stats.

This validates the primary-scene load, pose, lightweight observation, and short-control chain enough to enter Phase 8 mapping smoke. It does not approve training data, primary long rollout, RL, checkpointing, or PI/openpi/VLM fine-tuning.

## Gates

- proceed_to_primary_scene_sensor_smoke: complete
- proceed_to_primary_mapping_smoke: true
- proceed_to_primary_candidate_gain_smoke: false until mapping smoke passes
- proceed_to_primary_closed_loop_smoke: false until candidate gain smoke passes
- proceed_to_primary_long_rollout: false until all smoke gates pass
- proceed_to_training: false
- proceed_to_RL: false
- proceed_to_PI_finetuning: false

## Risks And Caveats

- The 490M scene bundle is local-only and must remain excluded from Git.
- The sensor smoke uses USD geometry visibility/depth/pointcloud proxy stats, not committed rendered PNG/NPZ/HDF5 assets.
- MDL/material warnings remain acceptable for this smoke gate because stage load and geometry traversal passed, but rendered visual fidelity is not yet approved.
- Fallback Phase 5 rollout data remains schema/pipeline sanity data only and is not approved training data.

## Evidence

- report: /home/ubuntu22/pi/runs/PRIMARY_SCENE_SENSOR_SMOKE_REPORT.md
- run_dir: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109
- stage probe: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/probes/stage_load_probe.json
- smoke summary: /home/ubuntu22/pi/runs/primary_scene_sensor_smoke_20260607_165109/smoke/phase7_primary_scene_sensor_smoke_summary.json
