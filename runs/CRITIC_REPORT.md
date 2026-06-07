# Critic Report

- phase: Phase 8 primary-scene mapping smoke
- result: passed
- scope_check: passed
- scene_bundle_committed: false
- training: false
- RL: false
- checkpoint: false
- PI_finetuning: false
- primary_long_rollout: false

## Conclusion

The repaired primary scene at /home/ubuntu22/pi/scenes/primary_building_scene_repaired/home_like_scene_v1.usd passed the bounded mapping smoke. The run completed 10 mapping steps with 10 valid depth/pointcloud proxy observations. The BEV partial map contains occupied, free/traversed, and unknown cells; known_ratio increased monotonically to 0.48057726; and no collision or stuck system failure was recorded.

This supports moving to Phase 9 candidate viewpoint + information gain smoke. It does not approve training data, primary long rollout, RL, checkpointing, or PI/openpi/VLM fine-tuning.

## Gates

- proceed_to_primary_scene_sensor_smoke: complete
- proceed_to_primary_mapping_smoke: complete
- proceed_to_phase9_candidate_viewpoint_information_gain_smoke: true
- proceed_to_primary_closed_loop_smoke: false until candidate gain smoke passes
- proceed_to_primary_long_rollout: false until all smoke gates pass
- proceed_to_training: false
- proceed_to_RL: false
- proceed_to_PI_finetuning: false

## Risks And Caveats

- The 490M primary scene bundle remains local-only and must not be committed.
- Phase 8 uses USD geometry depth/pointcloud proxy observations, not final rendered training sensors.
- Mapping artifacts are smoke/review evidence only, not a training dataset.
- Candidate scoring still needs a dedicated Phase 9 check before any closed-loop or long rollout.

## Evidence

- report: /home/ubuntu22/pi/runs/PRIMARY_SCENE_MAPPING_SMOKE_REPORT.md
- run_dir: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019
- mapping summary: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/summary/mapping_summary.json
- mapping steps: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/summary/mapping_steps.csv
- plots: /home/ubuntu22/pi/runs/primary_scene_mapping_smoke_20260607_170019/plots/
