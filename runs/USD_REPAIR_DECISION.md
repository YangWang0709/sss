# USD Repair Decision

- conclusion: replacement_primary_scene_created
- repair_run_dir: /home/ubuntu22/pi/runs/usd_repair_building_scene_20260607_162058
- current_task: primary USD repair
- ready_for_primary_scene_sensor_smoke: true
- ready_for_primary_long_rollout: false
- ready_for_training: false
- ready_for_RL: false
- ready_for_PI_finetuning: false

## Required Decision Fields

1. original_file_path: /home/ubuntu22/pi/building_scene.usd
2. original_sha256: 11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b
3. original_file_type: USD crate, version 0.8.0
4. original_Isaac_open_stage_still_crashes: true, exit code 134
5. usdchecker_result: unavailable, usdchecker not found in PATH
6. usdcat_conversion_success: false, usdcat not found in PATH
7. Python_USD_API_can_open: false, standalone pxr module unavailable in env_isaaclab
8. Isaac_clean_flattened_sanitized_open: not available; clean/flattened/sanitized variants could not be produced safely
9. final_usable_primary_scene_path: /home/ubuntu22/pi/scenes/primary_indoor_scene.usda
10. original_USD_repaired: false
11. replacement_primary_scene_created: true
12. recommendation: enter primary-scene sensor smoke only; do not run primary long rollout until all primary-scene smoke gates pass

## Evidence

- original probe: /home/ubuntu22/pi/runs/usd_repair_building_scene_20260607_162058/probes/isaac_open_original.json
- replacement probe: /home/ubuntu22/pi/runs/usd_repair_building_scene_20260607_162058/probes/isaac_open_primary_indoor_scene.json
- replacement prim_count: 52
- replacement cube_count: 37
- replacement open_stage_result: True
- replacement stage_available: True

## Negative Scope

- training: false
- RL: false
- checkpoint: false
- PI fine-tuning: false
- openpi fine-tuning: false
- VLM fine-tuning: false
- fallback data promoted to final training data: false
- original building_scene.usd deleted: false
- original building_scene.usd overwritten: false
